#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <torch/all.h>

namespace {

__device__ __forceinline__ float decode_e4m3fn(uint8_t u) {
  const int sign = u >> 7;
  const int exp_bits = (u >> 3) & 0x0F;
  const int mant = u & 0x07;
  const bool is_normal = exp_bits != 0;
  const float sign_f = sign ? -1.0f : 1.0f;
  const float mant_f =
      is_normal ? static_cast<float>(8 + mant) * 0.125f
                : static_cast<float>(mant) * 0.125f;
  const int eff_exp = is_normal ? exp_bits : 1;
  return sign_f * ldexpf(mant_f, eff_exp - 7);
}

__global__ void fp8_mqa_dequant_q_kernel(
    const uint8_t* __restrict__ q, __nv_bfloat16* __restrict__ q_bf16,
    int64_t M, int64_t H, int64_t D, int64_t stride_q_m, int64_t stride_q_h,
    int64_t stride_q_d) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = M * H * D;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % D;
  const int64_t mh = idx / D;
  const int64_t m = mh / H;
  const int64_t h = mh - m * H;
  const uint8_t byte = q[m * stride_q_m + h * stride_q_h + d * stride_q_d];

  // Store as [H, M, D] so each head is a contiguous row-major [M, D]
  // matrix for cuBLAS.
  q_bf16[(h * M + m) * D + d] = __float2bfloat16(decode_e4m3fn(byte));
}

__global__ void fp8_mqa_dequant_k_kernel(
    const uint8_t* __restrict__ k, const float* __restrict__ k_scales,
    __nv_bfloat16* __restrict__ k_bf16, int64_t N, int64_t D,
    int64_t stride_k_n, int64_t stride_k_d, int64_t stride_scale_n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = N * D;
  if (idx >= total) {
    return;
  }

  const int64_t d = idx % D;
  const int64_t n = idx / D;
  const uint8_t byte = k[n * stride_k_n + d * stride_k_d];
  const float scale = k_scales[n * stride_scale_n];
  k_bf16[n * D + d] = __float2bfloat16(decode_e4m3fn(byte) * scale);
}

__global__ void fp8_mqa_accumulate_logits_kernel(
    const float* __restrict__ scores, const float* __restrict__ weights,
    float* __restrict__ logits, int64_t total, int64_t N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head, bool first_head) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t m = idx / N;
  const int64_t n = idx - m * N;
  const float score = scores[idx];
  const float weight = weights[m * stride_w_m + head * stride_w_h];
  const float value = score > 0.0f ? score * weight : 0.0f;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (first_head) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

__global__ void fp8_mqa_accumulate_logits_group_kernel(
    const float* __restrict__ scores, const float* __restrict__ weights,
    float* __restrict__ logits, int64_t total, int64_t N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head_start, int64_t group_heads,
    bool first_group) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t m = idx / N;
  const int64_t n = idx - m * N;
  float value = 0.0f;
#pragma unroll
  for (int64_t g = 0; g < 4; ++g) {
    if (g >= group_heads) {
      break;
    }
    const float score = scores[g * total + idx];
    const float weight =
        weights[m * stride_w_m + (head_start + g) * stride_w_h];
    value += score > 0.0f ? score * weight : 0.0f;
  }
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (first_group) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

__global__ void fp8_mqa_apply_mask_kernel(
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke, float* __restrict__ logits,
    int64_t total, int64_t N, int64_t stride_l_m, int64_t stride_l_n) {
  const int64_t idx =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int64_t m = idx / N;
  const int64_t n = idx - m * N;
  const int32_t ks = cu_seqlen_ks[m];
  const int32_t ke = cu_seqlen_ke[m];
  if (n < ks || n >= ke) {
    logits[m * stride_l_m + n * stride_l_n] = -INFINITY;
  }
}

void check_fp8_mqa_logits_inputs(const torch::Tensor& q, const torch::Tensor& k,
                                 const torch::Tensor& k_scales,
                                 const torch::Tensor& weights,
                                 const torch::Tensor& cu_seqlen_ks,
                                 const torch::Tensor& cu_seqlen_ke,
                                 const torch::Tensor& logits) {
  TORCH_CHECK(q.is_cuda(), "fp8_mqa_logits_cuda: q must be a CUDA tensor");
  TORCH_CHECK(k.is_cuda(), "fp8_mqa_logits_cuda: k must be a CUDA tensor");
  TORCH_CHECK(k_scales.is_cuda(),
              "fp8_mqa_logits_cuda: k_scales must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(),
              "fp8_mqa_logits_cuda: weights must be a CUDA tensor");
  TORCH_CHECK(cu_seqlen_ks.is_cuda(),
              "fp8_mqa_logits_cuda: cu_seqlen_ks must be a CUDA tensor");
  TORCH_CHECK(cu_seqlen_ke.is_cuda(),
              "fp8_mqa_logits_cuda: cu_seqlen_ke must be a CUDA tensor");
  TORCH_CHECK(logits.is_cuda(),
              "fp8_mqa_logits_cuda: logits must be a CUDA tensor");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_mqa_logits_cuda: q must be float8_e4m3fn");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_mqa_logits_cuda: k must be float8_e4m3fn");
  TORCH_CHECK(k_scales.scalar_type() == torch::kFloat32,
              "fp8_mqa_logits_cuda: k_scales must be float32");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "fp8_mqa_logits_cuda: weights must be float32");
  TORCH_CHECK(cu_seqlen_ks.scalar_type() == torch::kInt32,
              "fp8_mqa_logits_cuda: cu_seqlen_ks must be int32");
  TORCH_CHECK(cu_seqlen_ke.scalar_type() == torch::kInt32,
              "fp8_mqa_logits_cuda: cu_seqlen_ke must be int32");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "fp8_mqa_logits_cuda: logits must be float32");
  TORCH_CHECK(q.dim() == 3, "fp8_mqa_logits_cuda: q must be [M, H, D]");
  TORCH_CHECK(k.dim() == 2, "fp8_mqa_logits_cuda: k must be [N, D]");
  TORCH_CHECK(k_scales.dim() == 1,
              "fp8_mqa_logits_cuda: k_scales must be 1-D");
  TORCH_CHECK(weights.dim() == 2,
              "fp8_mqa_logits_cuda: weights must be [M, H]");
  TORCH_CHECK(cu_seqlen_ks.dim() == 1,
              "fp8_mqa_logits_cuda: cu_seqlen_ks must be 1-D");
  TORCH_CHECK(cu_seqlen_ke.dim() == 1,
              "fp8_mqa_logits_cuda: cu_seqlen_ke must be 1-D");
  TORCH_CHECK(logits.dim() == 2,
              "fp8_mqa_logits_cuda: logits must be [M, N]");

  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  const int64_t N = k.size(0);
  TORCH_CHECK(k.size(1) == D, "fp8_mqa_logits_cuda: k head_dim mismatch");
  TORCH_CHECK(k_scales.size(0) == N,
              "fp8_mqa_logits_cuda: k_scales length mismatch");
  TORCH_CHECK(weights.size(0) == M && weights.size(1) == H,
              "fp8_mqa_logits_cuda: weights shape mismatch");
  TORCH_CHECK(cu_seqlen_ks.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ks length mismatch");
  TORCH_CHECK(cu_seqlen_ke.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ke length mismatch");
  TORCH_CHECK(logits.size(0) == M && logits.size(1) == N,
              "fp8_mqa_logits_cuda: logits shape mismatch");
  TORCH_CHECK(logits.stride(1) == 1,
              "fp8_mqa_logits_cuda: logits must be row-major contiguous");
  TORCH_CHECK(D <= std::numeric_limits<int>::max() &&
                  M <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max(),
              "fp8_mqa_logits_cuda: dimensions exceed cuBLAS int limits");
}

int64_t get_env_int64(const char* name, int64_t default_value) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return default_value;
  }

  char* end = nullptr;
  const long long parsed = std::strtoll(value, &end, 10);
  if (end == value || parsed < 0) {
    return default_value;
  }
  return static_cast<int64_t>(parsed);
}

void fp8_mqa_logits_cuda_impl(const torch::Tensor& q, const torch::Tensor& k,
                              const torch::Tensor& k_scales,
                              const torch::Tensor& weights,
                              const torch::Tensor& cu_seqlen_ks,
                              const torch::Tensor& cu_seqlen_ke,
                              torch::Tensor& logits, int64_t group_size) {
  check_fp8_mqa_logits_inputs(q, k, k_scales, weights, cu_seqlen_ks,
                              cu_seqlen_ke, logits);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  const int64_t N = k.size(0);
  if (M == 0 || N == 0) {
    return;
  }

  auto bf16_options = q.options().dtype(torch::kBFloat16);
  auto fp32_options = q.options().dtype(torch::kFloat32);
  auto q_bf16 = torch::empty({H, M, D}, bf16_options);
  auto k_bf16 = torch::empty({N, D}, bf16_options);
  TORCH_CHECK(group_size == 1 || group_size == 2 || group_size == 4,
              "fp8_mqa_logits_cuda: group_size must be 1, 2, or 4");
  auto scores = torch::empty({group_size, M, N}, fp32_options);

  constexpr int threads = 256;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t q_total = M * H * D;
  const int64_t k_total = N * D;
  const int64_t logits_total = M * N;

  fp8_mqa_dequant_q_kernel<<<(q_total + threads - 1) / threads, threads, 0,
                              stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_bf16.data_ptr()), M, H, D, q.stride(0),
      q.stride(1), q.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_mqa_dequant_k_kernel<<<(k_total + threads - 1) / threads, threads, 0,
                              stream>>>(
      reinterpret_cast<const uint8_t*>(k.data_ptr()),
      k_scales.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(k_bf16.data_ptr()), N, D, k.stride(0),
      k.stride(1), k_scales.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  for (int64_t head_start = 0; head_start < H; head_start += group_size) {
    const int64_t group_heads = std::min(group_size, H - head_start);
    const void* q_head_group = static_cast<const char*>(q_bf16.data_ptr()) +
                               head_start * M * D * sizeof(at::BFloat16);
    if (group_heads == 1) {
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), q_head_group, CUDA_R_16BF,
          static_cast<int>(D), &beta, scores.data_ptr(), CUDA_R_32F,
          static_cast<int>(N), CUBLAS_COMPUTE_32F,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), 0LL, q_head_group, CUDA_R_16BF,
          static_cast<int>(D), M * D, &beta, scores.data_ptr(), CUDA_R_32F,
          static_cast<int>(N), M * N, static_cast<int>(group_heads),
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (group_size == 1) {
      fp8_mqa_accumulate_logits_kernel<<<
          (logits_total + threads - 1) / threads, threads, 0, stream>>>(
          scores.data_ptr<float>(), weights.data_ptr<float>(),
          logits.data_ptr<float>(), logits_total, N, weights.stride(0),
          weights.stride(1), logits.stride(0), logits.stride(1), head_start,
          head_start == 0);
    } else {
      fp8_mqa_accumulate_logits_group_kernel<<<
          (logits_total + threads - 1) / threads, threads, 0, stream>>>(
          scores.data_ptr<float>(), weights.data_ptr<float>(),
          logits.data_ptr<float>(), logits_total, N, weights.stride(0),
          weights.stride(1), logits.stride(0), logits.stride(1), head_start,
          group_heads, head_start == 0);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }

  fp8_mqa_apply_mask_kernel<<<(logits_total + threads - 1) / threads, threads,
                               0, stream>>>(
      cu_seqlen_ks.data_ptr<int32_t>(), cu_seqlen_ke.data_ptr<int32_t>(),
      logits.data_ptr<float>(), logits_total, N, logits.stride(0),
      logits.stride(1));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void fp8_mqa_logits_cuda(const torch::Tensor& q, const torch::Tensor& k,
                         const torch::Tensor& k_scales,
                         const torch::Tensor& weights,
                         const torch::Tensor& cu_seqlen_ks,
                         const torch::Tensor& cu_seqlen_ke,
                         torch::Tensor& logits) {
  fp8_mqa_logits_cuda_impl(q, k, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke,
                           logits, 1);
}

void fp8_mqa_logits_cuda_v5(const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& k_scales,
                            const torch::Tensor& weights,
                            const torch::Tensor& cu_seqlen_ks,
                            const torch::Tensor& cu_seqlen_ke,
                            torch::Tensor& logits) {
  const int64_t M = q.size(0);
  const int64_t N = k.size(0);
  const int64_t elements = M * N;
  const int64_t group4_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V5_GROUP4_MAX_ELEMENTS", 128LL * 1024 * 1024);
  const int64_t group2_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V5_GROUP2_MAX_ELEMENTS", 512LL * 1024 * 1024);
  int64_t group_size = 1;
  if (elements <= group4_max_elements) {
    group_size = 4;
  } else if (elements <= group2_max_elements) {
    group_size = 2;
  }
  fp8_mqa_logits_cuda_impl(q, k, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke,
                           logits, group_size);
}
