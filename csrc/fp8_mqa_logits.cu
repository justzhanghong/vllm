#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
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

__device__ __forceinline__ float score_to_float(float value) {
  return value;
}

__device__ __forceinline__ float score_to_float(__half value) {
  return __half2float(value);
}

__device__ __forceinline__ float score_to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
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

template <typename score_t>
__global__ void fp8_mqa_accumulate_logits_kernel(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int total, int N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head, bool first_head) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int m = idx / N;
  const int n = idx - m * N;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (n < cu_seqlen_ks[m] || n >= cu_seqlen_ke[m]) {
    if (first_head) {
      logits[logits_offset] = -INFINITY;
    }
    return;
  }
  const float score = score_to_float(scores[idx]);
  const float weight = weights[m * stride_w_m + head * stride_w_h];
  const float value = fmaxf(score, 0.0f) * weight;
  if (first_head) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

template <typename score_t>
__global__ void fp8_mqa_accumulate_logits_group_kernel(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int total, int N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head_start, int64_t group_heads,
    bool first_group) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int m = idx / N;
  const int n = idx - m * N;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (n < cu_seqlen_ks[m] || n >= cu_seqlen_ke[m]) {
    if (first_group) {
      logits[logits_offset] = -INFINITY;
    }
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int64_t g = 0; g < 32; ++g) {
    if (g >= group_heads) {
      break;
    }
    const float score = score_to_float(scores[g * total + idx]);
    const float weight =
        weights[m * stride_w_m + (head_start + g) * stride_w_h];
    value += fmaxf(score, 0.0f) * weight;
  }
  if (first_group) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

template <typename score_t, int GROUP_SIZE, bool FIRST_GROUP, bool CHECK_MASK>
__global__ void fp8_mqa_accumulate_logits_group_fixed_kernel(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int total, int N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head_start) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int m = idx / N;
  const int n = idx - m * N;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if constexpr (CHECK_MASK) {
    if (n < cu_seqlen_ks[m] || n >= cu_seqlen_ke[m]) {
      if constexpr (FIRST_GROUP) {
        logits[logits_offset] = -INFINITY;
      }
      return;
    }
  }

  const int64_t weight_base = m * stride_w_m + head_start * stride_w_h;
  float value = 0.0f;
#pragma unroll
  for (int g = 0; g < GROUP_SIZE; ++g) {
    const float score = score_to_float(scores[g * total + idx]);
    const float weight = weights[weight_base + g * stride_w_h];
    value += fmaxf(score, 0.0f) * weight;
  }
  if constexpr (FIRST_GROUP) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

template <typename score_t, int GROUP_SIZE>
__global__ void fp8_mqa_accumulate_logits_group_tiled_kernel(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int tile_total, int tile_N,
    int global_n_start, int64_t stride_w_m, int64_t stride_w_h,
    int64_t stride_l_m, int64_t stride_l_n, int64_t head_start) {
  const int m = blockIdx.x;
  const int n_local = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (n_local >= tile_N) {
    return;
  }

  __shared__ float weights_s[GROUP_SIZE];
  if (threadIdx.x < GROUP_SIZE) {
    weights_s[threadIdx.x] =
        weights[m * stride_w_m + (head_start + threadIdx.x) * stride_w_h];
  }
  __syncthreads();

  const int idx = m * tile_N + n_local;
  const int n = global_n_start + n_local;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  const int row_start = cu_seqlen_ks[m];
  const int row_end = cu_seqlen_ke[m];
  const int block_n_start =
      global_n_start + static_cast<int>(blockIdx.y) * blockDim.x;
  const int block_n_end = min(global_n_start + tile_N,
                              block_n_start + static_cast<int>(blockDim.x));
  const bool block_all_valid =
      block_n_start >= row_start && block_n_end <= row_end;
  if (!block_all_valid) {
    if (n < row_start || n >= row_end) {
      logits[logits_offset] = -INFINITY;
      return;
    }
  }

  float value = 0.0f;
#pragma unroll
  for (int g = 0; g < GROUP_SIZE; ++g) {
    const float score = score_to_float(scores[g * tile_total + idx]);
    value += fmaxf(score, 0.0f) * weights_s[g];
  }
  logits[logits_offset] = value;
}

template <typename score_t, int GROUP_SIZE>
__global__ void fp8_mqa_accumulate_logits_group_tiled_contig_kernel(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int tile_total, int tile_N, int N_work,
    int global_n_start) {
  const int m = blockIdx.x;
  const int n_local = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (n_local >= tile_N) {
    return;
  }

  __shared__ float weights_s[GROUP_SIZE];
  __shared__ int row_start_s;
  __shared__ int row_end_s;
  if (threadIdx.x < GROUP_SIZE) {
    weights_s[threadIdx.x] = weights[m * GROUP_SIZE + threadIdx.x];
  }
  if (threadIdx.x == 0) {
    row_start_s = cu_seqlen_ks[m];
  } else if (threadIdx.x == 1) {
    row_end_s = cu_seqlen_ke[m];
  }
  __syncthreads();

  const int idx = m * tile_N + n_local;
  const int n = global_n_start + n_local;
  const int row_start = row_start_s;
  const int row_end = row_end_s;
  const int block_n_start =
      global_n_start + static_cast<int>(blockIdx.y) * blockDim.x;
  const int block_n_end = min(global_n_start + tile_N,
                              block_n_start + static_cast<int>(blockDim.x));
  const bool block_all_valid =
      block_n_start >= row_start && block_n_end <= row_end;
  if (!block_all_valid) {
    if (n < row_start || n >= row_end) {
      logits[m * N_work + n] = -INFINITY;
      return;
    }
  }

  float value = 0.0f;
#pragma unroll
  for (int g = 0; g < GROUP_SIZE; ++g) {
    const float score = score_to_float(scores[g * tile_total + idx]);
    value += fmaxf(score, 0.0f) * weights_s[g];
  }
  logits[m * N_work + n] = value;
}

template <typename score_t>
__global__ void fp8_mqa_accumulate_logits_kernel_2d(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int N, int64_t stride_w_m,
    int64_t stride_w_h, int64_t stride_l_m, int64_t stride_l_n, int64_t head,
    bool first_head) {
  const int m = blockIdx.x;
  const int n = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  __shared__ float weight_s;
  if (threadIdx.x == 0) {
    weight_s = weights[m * stride_w_m + head * stride_w_h];
  }
  __syncthreads();

  const int64_t idx = m * N + n;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (n < cu_seqlen_ks[m] || n >= cu_seqlen_ke[m]) {
    if (first_head) {
      logits[logits_offset] = -INFINITY;
    }
    return;
  }
  const float score = score_to_float(scores[idx]);
  const float value = fmaxf(score, 0.0f) * weight_s;
  if (first_head) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

template <typename score_t>
__global__ void fp8_mqa_accumulate_logits_group_kernel_2d(
    const score_t* __restrict__ scores, const float* __restrict__ weights,
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke,
    float* __restrict__ logits, int total, int N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head_start, int64_t group_heads,
    bool first_group) {
  const int m = blockIdx.x;
  const int n = static_cast<int>(blockIdx.y) * blockDim.x + threadIdx.x;
  if (n >= N) {
    return;
  }

  __shared__ float weights_s[32];
  if (threadIdx.x < group_heads) {
    weights_s[threadIdx.x] =
        weights[m * stride_w_m + (head_start + threadIdx.x) * stride_w_h];
  }
  __syncthreads();

  const int64_t idx = m * N + n;
  const int64_t logits_offset = m * stride_l_m + n * stride_l_n;
  if (n < cu_seqlen_ks[m] || n >= cu_seqlen_ke[m]) {
    if (first_group) {
      logits[logits_offset] = -INFINITY;
    }
    return;
  }
  float value = 0.0f;
#pragma unroll
  for (int64_t g = 0; g < 32; ++g) {
    if (g >= group_heads) {
      break;
    }
    const float score = score_to_float(scores[g * total + idx]);
    value += fmaxf(score, 0.0f) * weights_s[g];
  }
  if (first_group) {
    logits[logits_offset] = value;
  } else {
    logits[logits_offset] += value;
  }
}

__global__ void fp8_mqa_apply_mask_kernel(
    const int32_t* __restrict__ cu_seqlen_ks,
    const int32_t* __restrict__ cu_seqlen_ke, float* __restrict__ logits,
    int total, int N, int64_t stride_l_m, int64_t stride_l_n) {
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int m = idx / N;
  const int n = idx - m * N;
  const int32_t ks = cu_seqlen_ks[m];
  const int32_t ke = cu_seqlen_ke[m];
  if (n < ks || n >= ke) {
    logits[m * stride_l_m + n * stride_l_n] = -INFINITY;
  }
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
  const int64_t logits_N = logits.size(1);
  const bool allow_padded_logits =
      get_env_int64("VLLM_MQA_CUDA_V7_PAD_N", 0) != 0 && logits_N >= N;
  TORCH_CHECK(k.size(1) == D, "fp8_mqa_logits_cuda: k head_dim mismatch");
  TORCH_CHECK(k_scales.size(0) == N,
              "fp8_mqa_logits_cuda: k_scales length mismatch");
  TORCH_CHECK(weights.size(0) == M && weights.size(1) == H,
              "fp8_mqa_logits_cuda: weights shape mismatch");
  TORCH_CHECK(cu_seqlen_ks.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ks length mismatch");
  TORCH_CHECK(cu_seqlen_ke.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ke length mismatch");
  TORCH_CHECK(logits.size(0) == M && (logits_N == N || allow_padded_logits),
              "fp8_mqa_logits_cuda: logits shape mismatch");
  TORCH_CHECK(logits_N == N || (N >= 32768 && logits_N % 128 == 0),
              "fp8_mqa_logits_cuda: padded logits must align long N to 128");
  TORCH_CHECK(logits.stride(1) == 1,
              "fp8_mqa_logits_cuda: logits must be row-major contiguous");
  TORCH_CHECK(D <= std::numeric_limits<int>::max() &&
                  M <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max() &&
                  logits_N <= std::numeric_limits<int>::max(),
              "fp8_mqa_logits_cuda: dimensions exceed cuBLAS int limits");
}

void check_fp8_mqa_logits_bf16_k_inputs(
    const torch::Tensor& q, const torch::Tensor& k_bf16,
    const torch::Tensor& weights, const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke, const torch::Tensor& logits) {
  TORCH_CHECK(q.is_cuda(), "fp8_mqa_logits_cuda: q must be a CUDA tensor");
  TORCH_CHECK(k_bf16.is_cuda(),
              "fp8_mqa_logits_cuda: k_bf16 must be a CUDA tensor");
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
  TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16,
              "fp8_mqa_logits_cuda: k_bf16 must be bfloat16");
  TORCH_CHECK(weights.scalar_type() == torch::kFloat32,
              "fp8_mqa_logits_cuda: weights must be float32");
  TORCH_CHECK(cu_seqlen_ks.scalar_type() == torch::kInt32,
              "fp8_mqa_logits_cuda: cu_seqlen_ks must be int32");
  TORCH_CHECK(cu_seqlen_ke.scalar_type() == torch::kInt32,
              "fp8_mqa_logits_cuda: cu_seqlen_ke must be int32");
  TORCH_CHECK(logits.scalar_type() == torch::kFloat32,
              "fp8_mqa_logits_cuda: logits must be float32");
  TORCH_CHECK(q.dim() == 3, "fp8_mqa_logits_cuda: q must be [M, H, D]");
  TORCH_CHECK(k_bf16.dim() == 2,
              "fp8_mqa_logits_cuda: k_bf16 must be [N, D]");
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
  const int64_t N_work = logits.size(1);
  TORCH_CHECK(k_bf16.size(0) == N_work && k_bf16.size(1) == D,
              "fp8_mqa_logits_cuda: k_bf16/logits shape mismatch");
  TORCH_CHECK(weights.size(0) == M && weights.size(1) == H,
              "fp8_mqa_logits_cuda: weights shape mismatch");
  TORCH_CHECK(cu_seqlen_ks.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ks length mismatch");
  TORCH_CHECK(cu_seqlen_ke.size(0) == M,
              "fp8_mqa_logits_cuda: cu_seqlen_ke length mismatch");
  TORCH_CHECK(logits.size(0) == M,
              "fp8_mqa_logits_cuda: logits row mismatch");
  TORCH_CHECK(k_bf16.stride(1) == 1,
              "fp8_mqa_logits_cuda: k_bf16 must be row-major");
  TORCH_CHECK(logits.stride(1) == 1,
              "fp8_mqa_logits_cuda: logits must be row-major contiguous");
  TORCH_CHECK(D <= std::numeric_limits<int>::max() &&
                  M <= std::numeric_limits<int>::max() &&
                  N_work <= std::numeric_limits<int>::max(),
              "fp8_mqa_logits_cuda: dimensions exceed cuBLAS int limits");
}

template <typename score_t>
void launch_fp8_mqa_accumulate_logits_group_1d(
    const score_t* scores, const float* weights, const int32_t* cu_seqlen_ks,
    const int32_t* cu_seqlen_ke, float* logits, int total, int N,
    int64_t stride_w_m, int64_t stride_w_h, int64_t stride_l_m,
    int64_t stride_l_n, int64_t head_start, int64_t group_size,
    int64_t group_heads, bool first_group, bool skip_later_group_mask,
    int threads, cudaStream_t stream) {
  const int blocks = (total + threads - 1) / threads;
  if (group_heads == group_size) {
#define LAUNCH_FIXED_GROUP(GROUP_SIZE_VALUE)                                      \
  do {                                                                            \
    if (first_group) {                                                            \
      fp8_mqa_accumulate_logits_group_fixed_kernel<score_t, GROUP_SIZE_VALUE,     \
                                                   true, true>                    \
          <<<blocks, threads, 0, stream>>>(                                       \
              scores, weights, cu_seqlen_ks, cu_seqlen_ke, logits, total, N,      \
              stride_w_m, stride_w_h, stride_l_m, stride_l_n, head_start);        \
    } else if (skip_later_group_mask) {                                           \
      fp8_mqa_accumulate_logits_group_fixed_kernel<score_t, GROUP_SIZE_VALUE,     \
                                                   false, false>                  \
          <<<blocks, threads, 0, stream>>>(                                       \
              scores, weights, cu_seqlen_ks, cu_seqlen_ke, logits, total, N,      \
              stride_w_m, stride_w_h, stride_l_m, stride_l_n, head_start);        \
    } else {                                                                      \
      fp8_mqa_accumulate_logits_group_fixed_kernel<score_t, GROUP_SIZE_VALUE,     \
                                                   false, true>                   \
          <<<blocks, threads, 0, stream>>>(                                       \
              scores, weights, cu_seqlen_ks, cu_seqlen_ke, logits, total, N,      \
              stride_w_m, stride_w_h, stride_l_m, stride_l_n, head_start);        \
    }                                                                             \
    return;                                                                       \
  } while (false)

    switch (group_size) {
      case 32:
        LAUNCH_FIXED_GROUP(32);
      case 16:
        LAUNCH_FIXED_GROUP(16);
      case 8:
        LAUNCH_FIXED_GROUP(8);
      case 4:
        LAUNCH_FIXED_GROUP(4);
      case 2:
        LAUNCH_FIXED_GROUP(2);
      default:
        break;
    }

#undef LAUNCH_FIXED_GROUP
  }

  fp8_mqa_accumulate_logits_group_kernel<<<blocks, threads, 0, stream>>>(
      scores, weights, cu_seqlen_ks, cu_seqlen_ke, logits, total, N,
      stride_w_m, stride_w_h, stride_l_m, stride_l_n, head_start, group_heads,
      first_group);
}

void fp8_mqa_logits_cuda_compute_bf16_qk(
    const torch::Tensor& q_bf16, const torch::Tensor& k_bf16,
    const torch::Tensor& weights, const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke, torch::Tensor& logits,
    int64_t group_size, bool bf16_scores, bool two_d_accum) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q_bf16));
  const int64_t H = q_bf16.size(0);
  const int64_t M = q_bf16.size(1);
  const int64_t D = q_bf16.size(2);
  const int64_t N_work = logits.size(1);
  if (M == 0 || N_work == 0) {
    return;
  }
  TORCH_CHECK(q_bf16.scalar_type() == torch::kBFloat16,
              "fp8_mqa_logits_cuda: q_bf16 must be bfloat16");
  TORCH_CHECK(q_bf16.dim() == 3,
              "fp8_mqa_logits_cuda: q_bf16 must be [H, M, D]");
  TORCH_CHECK(q_bf16.stride(2) == 1 && q_bf16.stride(1) == D,
              "fp8_mqa_logits_cuda: q_bf16 token rows must be contiguous");
  TORCH_CHECK(k_bf16.size(1) == D,
              "fp8_mqa_logits_cuda: q_bf16/k_bf16 head_dim mismatch");
  TORCH_CHECK(weights.size(0) == M && weights.size(1) == H,
              "fp8_mqa_logits_cuda: weights shape mismatch");

  auto bf16_options = q_bf16.options().dtype(torch::kBFloat16);
  auto fp32_options = q_bf16.options().dtype(torch::kFloat32);
  TORCH_CHECK(group_size == 1 || group_size == 2 || group_size == 4 ||
                  group_size == 8 || group_size == 16 || group_size == 32,
              "fp8_mqa_logits_cuda: group_size must be 1, 2, 4, 8, 16, or 32");

  const int threads =
      static_cast<int>(get_env_int64("VLLM_MQA_CUDA_THREADS", 256));
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "fp8_mqa_logits_cuda: VLLM_MQA_CUDA_THREADS must be 128, "
              "256, 512, or 1024");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool skip_later_group_mask =
      get_env_int64("VLLM_MQA_CUDA_V7_SKIP_LATER_GROUP_MASK", 0) != 0;
  const bool flat_group_gemm =
      get_env_int64("VLLM_MQA_CUDA_V7_FLAT_GROUP_GEMM", 0) != 0;
  const bool fast_bf16_gemm =
      get_env_int64("VLLM_MQA_CUDA_V7_FAST_BF16_GEMM", 0) != 0;
  const int64_t q_head_stride = q_bf16.stride(0);
  const int64_t logits_total = M * N_work;
  const bool group32_tiled =
      group_size == 32 && H == 32 &&
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILED", 0) != 0;
  int64_t group32_tile_n =
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILE_N", 0);
  if (group32_tiled && group32_tile_n == 0) {
    group32_tile_n = (N_work + 1) / 2;
    if (group32_tile_n >= 128) {
      group32_tile_n = ((group32_tile_n + 127) / 128) * 128;
    }
  }
  if (group32_tiled) {
    group32_tile_n = std::min(group32_tile_n, N_work);
    TORCH_CHECK(group32_tile_n > 0,
                "fp8_mqa_logits_cuda: group32 tile size must be positive");
  }
  const int64_t score_N = group32_tiled ? group32_tile_n : N_work;
  const bool group32_dual_stream =
      group32_tiled &&
      get_env_int64("VLLM_MQA_CUDA_V7_DUAL_STREAM", 0) != 0 &&
      N_work > group32_tile_n;
  const int64_t score_buffers = group32_dual_stream ? 2 : 1;
  const int64_t score_buffer_elements = group_size * M * score_N;
  auto scores = torch::empty({score_buffers * group_size, M, score_N},
                             bf16_scores ? bf16_options : fp32_options);
  const cudaDataType_t score_type = bf16_scores ? CUDA_R_16BF : CUDA_R_32F;
  const int logits_total_i = static_cast<int>(logits_total);
  const int N_i = static_cast<int>(N_work);
  const dim3 accum_grid_2d(
      static_cast<unsigned int>(M),
      static_cast<unsigned int>((N_work + threads - 1) / threads));

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasComputeType_t gemm_compute_type =
      fast_bf16_gemm ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
  if (group32_tiled) {
    TORCH_CHECK(!bf16_scores,
                "fp8_mqa_logits_cuda: group32 tiling requires fp32 scores");
    at::cuda::CUDAStream side_stream_obj = at::cuda::getCurrentCUDAStream();
    cudaStream_t side_stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    cudaEvent_t side_done_event = nullptr;
    if (group32_dual_stream) {
      side_stream_obj = at::cuda::getStreamFromPool(false, q_bf16.get_device());
      side_stream = side_stream_obj.stream();
      c10::cuda::CUDACachingAllocator::recordStream(
          scores.storage().data_ptr(), side_stream_obj);
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event,
                                              cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(ready_event, stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(side_stream, ready_event, 0));
    }

    const void* q_head_group = q_bf16.data_ptr();
    int64_t tile_id = 0;
    for (int64_t n_start = 0; n_start < N_work; n_start += group32_tile_n) {
      const int64_t tile_N = std::min(group32_tile_n, N_work - n_start);
      const int tile_N_i = static_cast<int>(tile_N);
      const int tile_total_i = static_cast<int>(M * tile_N);
      const void* k_tile = static_cast<const char*>(k_bf16.data_ptr()) +
                           n_start * D * sizeof(at::BFloat16);
      const int64_t score_buffer_id =
          group32_dual_stream ? (tile_id & 1) : 0;
      auto* score_tile_ptr =
          scores.data_ptr<float>() + score_buffer_id * score_buffer_elements;
      const cudaStream_t tile_stream =
          (group32_dual_stream && score_buffer_id == 1) ? side_stream : stream;
      TORCH_CUDABLAS_CHECK(cublasSetStream(handle, tile_stream));
      TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, tile_N_i, static_cast<int>(M),
          static_cast<int>(D), &alpha, k_tile, CUDA_R_16BF,
          static_cast<int>(D), 0LL, q_head_group, CUDA_R_16BF,
          static_cast<int>(D), q_head_stride, &beta, score_tile_ptr, score_type,
          tile_N_i, M * tile_N, 32, gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      const dim3 tile_grid(static_cast<unsigned int>(M),
                           static_cast<unsigned int>(
                               (tile_N + threads - 1) / threads));
      fp8_mqa_accumulate_logits_group_tiled_kernel<float, 32>
          <<<tile_grid, threads, 0, tile_stream>>>(
              score_tile_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              tile_total_i, tile_N_i, static_cast<int>(n_start),
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      ++tile_id;
    }
    if (group32_dual_stream) {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&side_done_event,
                                              cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(side_done_event, side_stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, side_done_event, 0));
      C10_CUDA_CHECK(cudaEventDestroy(ready_event));
      C10_CUDA_CHECK(cudaEventDestroy(side_done_event));
      TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    }
    return;
  }

  for (int64_t head_start = 0; head_start < H; head_start += group_size) {
    const int64_t group_heads = std::min(group_size, H - head_start);
    const void* q_head_group = static_cast<const char*>(q_bf16.data_ptr()) +
                               head_start * q_head_stride * sizeof(at::BFloat16);
    if (group_heads == 1) {
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), q_head_group, CUDA_R_16BF,
          static_cast<int>(D), &beta, scores.data_ptr(), score_type,
          static_cast<int>(N_work), gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (flat_group_gemm) {
      TORCH_CHECK(q_head_stride == M * D,
                  "fp8_mqa_logits_cuda: flat group GEMM requires contiguous "
                  "head-major q_bf16");
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M * group_heads), static_cast<int>(D), &alpha,
          k_bf16.data_ptr(), CUDA_R_16BF, static_cast<int>(D), q_head_group,
          CUDA_R_16BF, static_cast<int>(D), &beta, scores.data_ptr(),
          score_type, static_cast<int>(N_work), gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), 0LL, q_head_group, CUDA_R_16BF,
          static_cast<int>(D), q_head_stride, &beta, scores.data_ptr(), score_type,
          static_cast<int>(N_work), M * N_work, static_cast<int>(group_heads),
          gemm_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (bf16_scores) {
      const auto* scores_ptr =
          reinterpret_cast<const __nv_bfloat16*>(
              scores.data_ptr<at::BFloat16>());
      if (group_size == 1) {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_kernel_2d<<<accum_grid_2d, threads, 0,
                                                stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(), N_i,
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), head_start, head_start == 0);
        } else {
          fp8_mqa_accumulate_logits_kernel<<<
              (logits_total + threads - 1) / threads, threads, 0, stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, head_start == 0);
        }
      } else {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_group_kernel_2d<<<accum_grid_2d, threads, 0,
                                                      stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_heads,
              head_start == 0);
        } else {
          launch_fp8_mqa_accumulate_logits_group_1d(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_size,
              group_heads, head_start == 0, skip_later_group_mask, threads,
              stream);
        }
      }
    } else {
      if (group_size == 1) {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_kernel_2d<<<accum_grid_2d, threads, 0,
                                                stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(), N_i,
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), head_start, head_start == 0);
        } else {
          fp8_mqa_accumulate_logits_kernel<<<
              (logits_total + threads - 1) / threads, threads, 0, stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, head_start == 0);
        }
      } else {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_group_kernel_2d<<<accum_grid_2d, threads, 0,
                                                      stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_heads,
              head_start == 0);
        } else {
          launch_fp8_mqa_accumulate_logits_group_1d(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_size,
              group_heads, head_start == 0, skip_later_group_mask, threads,
              stream);
        }
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void fp8_mqa_logits_cuda_impl(const torch::Tensor& q, const torch::Tensor& k,
                              const torch::Tensor& k_scales,
                              const torch::Tensor& weights,
                              const torch::Tensor& cu_seqlen_ks,
                              const torch::Tensor& cu_seqlen_ke,
                              torch::Tensor& logits, int64_t group_size,
                              bool bf16_scores, bool two_d_accum) {
  check_fp8_mqa_logits_inputs(q, k, k_scales, weights, cu_seqlen_ks,
                              cu_seqlen_ke, logits);

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  const int64_t N = k.size(0);
  const int64_t N_work = logits.size(1);
  if (M == 0 || N == 0) {
    return;
  }

  auto bf16_options = q.options().dtype(torch::kBFloat16);
  auto fp32_options = q.options().dtype(torch::kFloat32);
  auto q_bf16 = torch::empty({H, M, D}, bf16_options);
  auto k_bf16 = torch::empty({N_work, D}, bf16_options);
  TORCH_CHECK(group_size == 1 || group_size == 2 || group_size == 4 ||
                  group_size == 8 || group_size == 16 || group_size == 32,
              "fp8_mqa_logits_cuda: group_size must be 1, 2, 4, 8, 16, or 32");

  const int threads =
      static_cast<int>(get_env_int64("VLLM_MQA_CUDA_THREADS", 256));
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "fp8_mqa_logits_cuda: VLLM_MQA_CUDA_THREADS must be 128, "
              "256, 512, or 1024");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const bool skip_later_group_mask =
      get_env_int64("VLLM_MQA_CUDA_V7_SKIP_LATER_GROUP_MASK", 0) != 0;
  const bool flat_group_gemm =
      get_env_int64("VLLM_MQA_CUDA_V7_FLAT_GROUP_GEMM", 0) != 0;
  const bool fast_bf16_gemm =
      get_env_int64("VLLM_MQA_CUDA_V7_FAST_BF16_GEMM", 0) != 0;
  const int64_t q_total = M * H * D;
  const int64_t k_total = N * D;
  const int64_t logits_total = M * N_work;
  const bool group32_tiled =
      group_size == 32 && H == 32 &&
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILED", 0) != 0;
  int64_t group32_tile_n =
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILE_N", 0);
  if (group32_tiled && group32_tile_n == 0) {
    group32_tile_n = (N_work + 1) / 2;
    if (group32_tile_n >= 128) {
      group32_tile_n = ((group32_tile_n + 127) / 128) * 128;
    }
  }
  if (group32_tiled) {
    group32_tile_n = std::min(group32_tile_n, N_work);
    TORCH_CHECK(group32_tile_n > 0,
                "fp8_mqa_logits_cuda: group32 tile size must be positive");
  }
  const int64_t score_N = group32_tiled ? group32_tile_n : N_work;
  const bool group32_dual_stream =
      group32_tiled &&
      get_env_int64("VLLM_MQA_CUDA_V7_DUAL_STREAM", 0) != 0 &&
      N_work > group32_tile_n;
  const int64_t score_buffers = group32_dual_stream ? 2 : 1;
  const int64_t score_buffer_elements = group_size * M * score_N;
  auto scores = torch::empty({score_buffers * group_size, M, score_N},
                             bf16_scores ? bf16_options : fp32_options);
  const cudaDataType_t score_type = bf16_scores ? CUDA_R_16BF : CUDA_R_32F;
  const int logits_total_i = static_cast<int>(logits_total);
  const int N_i = static_cast<int>(N_work);
  const dim3 accum_grid_2d(
      static_cast<unsigned int>(M),
      static_cast<unsigned int>((N_work + threads - 1) / threads));

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

  if (N_work > N) {
    const int64_t pad_bytes = (N_work - N) * D * sizeof(at::BFloat16);
    auto* pad_ptr = static_cast<char*>(k_bf16.data_ptr()) +
                    N * D * sizeof(at::BFloat16);
    C10_CUDA_CHECK(cudaMemsetAsync(pad_ptr, 0, pad_bytes, stream));
  }

  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  const cublasComputeType_t gemm_compute_type =
      fast_bf16_gemm ? CUBLAS_COMPUTE_32F_FAST_16BF : CUBLAS_COMPUTE_32F;
  if (group32_tiled) {
    TORCH_CHECK(!bf16_scores,
                "fp8_mqa_logits_cuda: group32 tiling requires fp32 scores");
    at::cuda::CUDAStream side_stream_obj = at::cuda::getCurrentCUDAStream();
    cudaStream_t side_stream = nullptr;
    cudaEvent_t ready_event = nullptr;
    cudaEvent_t side_done_event = nullptr;
    if (group32_dual_stream) {
      side_stream_obj = at::cuda::getStreamFromPool(false, q.get_device());
      side_stream = side_stream_obj.stream();
      c10::cuda::CUDACachingAllocator::recordStream(
          scores.storage().data_ptr(), side_stream_obj);
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready_event,
                                              cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(ready_event, stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(side_stream, ready_event, 0));
    }

    const void* q_head_group = q_bf16.data_ptr();
    int64_t tile_id = 0;
    for (int64_t n_start = 0; n_start < N_work; n_start += group32_tile_n) {
      const int64_t tile_N = std::min(group32_tile_n, N_work - n_start);
      const int tile_N_i = static_cast<int>(tile_N);
      const int tile_total_i = static_cast<int>(M * tile_N);
      const void* k_tile = static_cast<const char*>(k_bf16.data_ptr()) +
                           n_start * D * sizeof(at::BFloat16);
      const int64_t score_buffer_id =
          group32_dual_stream ? (tile_id & 1) : 0;
      auto* score_tile_ptr =
          scores.data_ptr<float>() + score_buffer_id * score_buffer_elements;
      const cudaStream_t tile_stream =
          (group32_dual_stream && score_buffer_id == 1) ? side_stream : stream;
      TORCH_CUDABLAS_CHECK(cublasSetStream(handle, tile_stream));
      TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, tile_N_i, static_cast<int>(M),
          static_cast<int>(D), &alpha, k_tile, CUDA_R_16BF,
          static_cast<int>(D), 0LL, q_head_group, CUDA_R_16BF,
          static_cast<int>(D), M * D, &beta, score_tile_ptr, score_type,
          tile_N_i, M * tile_N, 32, gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));

      const dim3 tile_grid(static_cast<unsigned int>(M),
                           static_cast<unsigned int>(
                               (tile_N + threads - 1) / threads));
      fp8_mqa_accumulate_logits_group_tiled_kernel<float, 32>
          <<<tile_grid, threads, 0, tile_stream>>>(
              score_tile_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              tile_total_i, tile_N_i, static_cast<int>(n_start),
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), 0);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      ++tile_id;
    }
    if (group32_dual_stream) {
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&side_done_event,
                                              cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(side_done_event, side_stream));
      C10_CUDA_CHECK(cudaStreamWaitEvent(stream, side_done_event, 0));
      C10_CUDA_CHECK(cudaEventDestroy(ready_event));
      C10_CUDA_CHECK(cudaEventDestroy(side_done_event));
      TORCH_CUDABLAS_CHECK(cublasSetStream(handle, stream));
    }
    return;
  }

  for (int64_t head_start = 0; head_start < H; head_start += group_size) {
    const int64_t group_heads = std::min(group_size, H - head_start);
    const void* q_head_group = static_cast<const char*>(q_bf16.data_ptr()) +
                               head_start * M * D * sizeof(at::BFloat16);
    if (group_heads == 1) {
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), q_head_group, CUDA_R_16BF,
          static_cast<int>(D), &beta, scores.data_ptr(), score_type,
          static_cast<int>(N_work), gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (flat_group_gemm) {
      TORCH_CUDABLAS_CHECK(cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M * group_heads), static_cast<int>(D), &alpha,
          k_bf16.data_ptr(), CUDA_R_16BF, static_cast<int>(D), q_head_group,
          CUDA_R_16BF, static_cast<int>(D), &beta, scores.data_ptr(),
          score_type, static_cast<int>(N_work), gemm_compute_type,
          CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
      TORCH_CUDABLAS_CHECK(cublasGemmStridedBatchedEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(N_work),
          static_cast<int>(M), static_cast<int>(D), &alpha, k_bf16.data_ptr(),
          CUDA_R_16BF, static_cast<int>(D), 0LL, q_head_group, CUDA_R_16BF,
          static_cast<int>(D), M * D, &beta, scores.data_ptr(), score_type,
          static_cast<int>(N_work), M * N_work, static_cast<int>(group_heads),
          gemm_compute_type, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    if (bf16_scores) {
      const auto* scores_ptr =
          reinterpret_cast<const __nv_bfloat16*>(
              scores.data_ptr<at::BFloat16>());
      if (group_size == 1) {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_kernel_2d<<<accum_grid_2d, threads, 0,
                                                stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(), N_i,
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), head_start, head_start == 0);
        } else {
          fp8_mqa_accumulate_logits_kernel<<<
              (logits_total + threads - 1) / threads, threads, 0, stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, head_start == 0);
        }
      } else {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_group_kernel_2d<<<accum_grid_2d, threads, 0,
                                                      stream>>>(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_heads,
              head_start == 0);
        } else {
          launch_fp8_mqa_accumulate_logits_group_1d(
              scores_ptr, weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_size,
              group_heads, head_start == 0, skip_later_group_mask, threads,
              stream);
        }
      }
    } else {
      if (group_size == 1) {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_kernel_2d<<<accum_grid_2d, threads, 0,
                                                stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(), N_i,
              weights.stride(0), weights.stride(1), logits.stride(0),
              logits.stride(1), head_start, head_start == 0);
        } else {
          fp8_mqa_accumulate_logits_kernel<<<
              (logits_total + threads - 1) / threads, threads, 0, stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, head_start == 0);
        }
      } else {
        if (two_d_accum) {
          fp8_mqa_accumulate_logits_group_kernel_2d<<<accum_grid_2d, threads, 0,
                                                      stream>>>(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_heads,
              head_start == 0);
        } else {
          launch_fp8_mqa_accumulate_logits_group_1d(
              scores.data_ptr<float>(), weights.data_ptr<float>(),
              cu_seqlen_ks.data_ptr<int32_t>(),
              cu_seqlen_ke.data_ptr<int32_t>(), logits.data_ptr<float>(),
              logits_total_i, N_i, weights.stride(0), weights.stride(1),
              logits.stride(0), logits.stride(1), head_start, group_size,
              group_heads, head_start == 0, skip_later_group_mask, threads,
              stream);
        }
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

void fp8_mqa_logits_cuda_compute_bf16_k(
    const torch::Tensor& q, const torch::Tensor& k_bf16,
    const torch::Tensor& weights, const torch::Tensor& cu_seqlen_ks,
    const torch::Tensor& cu_seqlen_ke, torch::Tensor& logits,
    int64_t group_size, bool bf16_scores, bool two_d_accum) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  if (M == 0 || logits.size(1) == 0) {
    return;
  }

  auto q_bf16 = torch::empty({H, M, D}, q.options().dtype(torch::kBFloat16));
  const int threads =
      static_cast<int>(get_env_int64("VLLM_MQA_CUDA_THREADS", 256));
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "fp8_mqa_logits_cuda: VLLM_MQA_CUDA_THREADS must be 128, "
              "256, 512, or 1024");
  const int64_t q_total = M * H * D;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fp8_mqa_dequant_q_kernel<<<(q_total + threads - 1) / threads, threads, 0,
                              stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_bf16.data_ptr()), M, H, D, q.stride(0),
      q.stride(1), q.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  fp8_mqa_logits_cuda_compute_bf16_qk(
      q_bf16, k_bf16, weights, cu_seqlen_ks, cu_seqlen_ke, logits, group_size,
      bf16_scores, two_d_accum);
}

}  // namespace

void fp8_mqa_logits_cuda(const torch::Tensor& q, const torch::Tensor& k,
                         const torch::Tensor& k_scales,
                         const torch::Tensor& weights,
                         const torch::Tensor& cu_seqlen_ks,
                         const torch::Tensor& cu_seqlen_ke,
                         torch::Tensor& logits) {
  fp8_mqa_logits_cuda_impl(q, k, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke,
                           logits, 1, false, false);
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
                           logits, group_size, false, false);
}

void fp8_mqa_logits_cuda_v7(const torch::Tensor& q, const torch::Tensor& k,
                            const torch::Tensor& k_scales,
                            const torch::Tensor& weights,
                            const torch::Tensor& cu_seqlen_ks,
                            const torch::Tensor& cu_seqlen_ke,
                            torch::Tensor& logits) {
  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t N = k.size(0);
  const int64_t elements = M * N;
  const int64_t group32_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP32_MAX_ELEMENTS", 0);
  const bool group32_tiled =
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILED", 0) != 0;
  const int64_t group16_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP16_MAX_ELEMENTS", 0);
  const int64_t group8_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP8_MAX_ELEMENTS", 64LL * 1024 * 1024);
  const int64_t group4_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP4_MAX_ELEMENTS", 128LL * 1024 * 1024);
  const int64_t group2_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP2_MAX_ELEMENTS", 512LL * 1024 * 1024);
  int64_t group_size = 1;
  if (group32_tiled && H == 32 && elements <= group16_max_elements) {
    group_size = 32;
  } else if (elements <= group32_max_elements) {
    group_size = 32;
  } else if (elements <= group16_max_elements) {
    group_size = 16;
  } else if (elements <= group8_max_elements) {
    group_size = 8;
  } else if (elements <= group4_max_elements) {
    group_size = 4;
  } else if (elements <= group2_max_elements) {
    group_size = 2;
  }
  const bool bf16_scores =
      get_env_int64("VLLM_MQA_CUDA_V7_BF16_SCORES", 1) != 0;
  const bool two_d_accum =
      get_env_int64("VLLM_MQA_CUDA_V7_2D_ACCUM", 0) != 0;
  fp8_mqa_logits_cuda_impl(q, k, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke,
                           logits, group_size, bf16_scores, two_d_accum);
}

void fp8_mqa_dequant_k_cuda(const torch::Tensor& k,
                            const torch::Tensor& k_scales,
                            torch::Tensor& k_bf16) {
  TORCH_CHECK(k.is_cuda(), "fp8_mqa_dequant_k_cuda: k must be CUDA");
  TORCH_CHECK(k_scales.is_cuda(),
              "fp8_mqa_dequant_k_cuda: k_scales must be CUDA");
  TORCH_CHECK(k_bf16.is_cuda(),
              "fp8_mqa_dequant_k_cuda: k_bf16 must be CUDA");
  TORCH_CHECK(k.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_mqa_dequant_k_cuda: k must be float8_e4m3fn");
  TORCH_CHECK(k_scales.scalar_type() == torch::kFloat32,
              "fp8_mqa_dequant_k_cuda: k_scales must be float32");
  TORCH_CHECK(k_bf16.scalar_type() == torch::kBFloat16,
              "fp8_mqa_dequant_k_cuda: k_bf16 must be bfloat16");
  TORCH_CHECK(k.dim() == 2, "fp8_mqa_dequant_k_cuda: k must be [N, D]");
  TORCH_CHECK(k_scales.dim() == 1,
              "fp8_mqa_dequant_k_cuda: k_scales must be 1-D");
  TORCH_CHECK(k_bf16.dim() == 2,
              "fp8_mqa_dequant_k_cuda: k_bf16 must be [N, D]");

  const int64_t N = k.size(0);
  const int64_t D = k.size(1);
  TORCH_CHECK(k_scales.size(0) == N,
              "fp8_mqa_dequant_k_cuda: k_scales length mismatch");
  TORCH_CHECK(k_bf16.size(0) >= N && k_bf16.size(1) == D,
              "fp8_mqa_dequant_k_cuda: k_bf16 shape mismatch");
  TORCH_CHECK(k_bf16.stride(1) == 1,
              "fp8_mqa_dequant_k_cuda: k_bf16 must be row-major");
  TORCH_CHECK(D <= std::numeric_limits<int>::max() &&
                  N <= std::numeric_limits<int>::max(),
              "fp8_mqa_dequant_k_cuda: dimensions exceed int limits");
  if (N == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(k));
  const int threads =
      static_cast<int>(get_env_int64("VLLM_MQA_CUDA_THREADS", 256));
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "fp8_mqa_dequant_k_cuda: VLLM_MQA_CUDA_THREADS must be 128, "
              "256, 512, or 1024");
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  const int64_t k_total = N * D;
  fp8_mqa_dequant_k_kernel<<<(k_total + threads - 1) / threads, threads, 0,
                              stream>>>(
      reinterpret_cast<const uint8_t*>(k.data_ptr()),
      k_scales.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(k_bf16.data_ptr()), N, D, k.stride(0),
      k.stride(1), k_scales.stride(0));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_mqa_dequant_q_cuda(const torch::Tensor& q, torch::Tensor& q_bf16) {
  TORCH_CHECK(q.is_cuda(), "fp8_mqa_dequant_q_cuda: q must be CUDA");
  TORCH_CHECK(q_bf16.is_cuda(),
              "fp8_mqa_dequant_q_cuda: q_bf16 must be CUDA");
  TORCH_CHECK(q.scalar_type() == at::ScalarType::Float8_e4m3fn,
              "fp8_mqa_dequant_q_cuda: q must be float8_e4m3fn");
  TORCH_CHECK(q_bf16.scalar_type() == torch::kBFloat16,
              "fp8_mqa_dequant_q_cuda: q_bf16 must be bfloat16");
  TORCH_CHECK(q.dim() == 3, "fp8_mqa_dequant_q_cuda: q must be [M, H, D]");
  TORCH_CHECK(q_bf16.dim() == 3,
              "fp8_mqa_dequant_q_cuda: q_bf16 must be [H, M, D]");

  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t D = q.size(2);
  TORCH_CHECK(q_bf16.size(0) == H && q_bf16.size(1) == M &&
                  q_bf16.size(2) == D,
              "fp8_mqa_dequant_q_cuda: q_bf16 shape mismatch");
  TORCH_CHECK(q_bf16.stride(2) == 1 && q_bf16.stride(1) == D &&
                  q_bf16.stride(0) == M * D,
              "fp8_mqa_dequant_q_cuda: q_bf16 must be contiguous [H, M, D]");
  if (M == 0) {
    return;
  }

  const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
  const int threads =
      static_cast<int>(get_env_int64("VLLM_MQA_CUDA_THREADS", 256));
  TORCH_CHECK(threads == 128 || threads == 256 || threads == 512 ||
                  threads == 1024,
              "fp8_mqa_dequant_q_cuda: VLLM_MQA_CUDA_THREADS must be 128, "
              "256, 512, or 1024");
  const int64_t q_total = M * H * D;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  fp8_mqa_dequant_q_kernel<<<(q_total + threads - 1) / threads, threads, 0,
                              stream>>>(
      reinterpret_cast<const uint8_t*>(q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(q_bf16.data_ptr()), M, H, D, q.stride(0),
      q.stride(1), q.stride(2));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void fp8_mqa_logits_cuda_v7_bf16_k(const torch::Tensor& q,
                                   const torch::Tensor& k_bf16,
                                   const torch::Tensor& weights,
                                   const torch::Tensor& cu_seqlen_ks,
                                   const torch::Tensor& cu_seqlen_ke,
                                   torch::Tensor& logits) {
  check_fp8_mqa_logits_bf16_k_inputs(q, k_bf16, weights, cu_seqlen_ks,
                                     cu_seqlen_ke, logits);

  const int64_t M = q.size(0);
  const int64_t H = q.size(1);
  const int64_t N_work = logits.size(1);
  const int64_t elements = M * N_work;
  const int64_t group32_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP32_MAX_ELEMENTS", 0);
  const bool group32_tiled =
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILED", 0) != 0;
  const int64_t group16_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP16_MAX_ELEMENTS", 0);
  const int64_t group8_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP8_MAX_ELEMENTS", 64LL * 1024 * 1024);
  const int64_t group4_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP4_MAX_ELEMENTS", 128LL * 1024 * 1024);
  const int64_t group2_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP2_MAX_ELEMENTS", 512LL * 1024 * 1024);
  int64_t group_size = 1;
  if (group32_tiled && H == 32 && elements <= group16_max_elements) {
    group_size = 32;
  } else if (elements <= group32_max_elements) {
    group_size = 32;
  } else if (elements <= group16_max_elements) {
    group_size = 16;
  } else if (elements <= group8_max_elements) {
    group_size = 8;
  } else if (elements <= group4_max_elements) {
    group_size = 4;
  } else if (elements <= group2_max_elements) {
    group_size = 2;
  }
  const bool bf16_scores =
      get_env_int64("VLLM_MQA_CUDA_V7_BF16_SCORES", 1) != 0;
  const bool two_d_accum =
      get_env_int64("VLLM_MQA_CUDA_V7_2D_ACCUM", 0) != 0;
  fp8_mqa_logits_cuda_compute_bf16_k(q, k_bf16, weights, cu_seqlen_ks,
                                     cu_seqlen_ke, logits, group_size,
                                     bf16_scores, two_d_accum);
}

void fp8_mqa_logits_cuda_v7_bf16_qk(const torch::Tensor& q_bf16,
                                    const torch::Tensor& k_bf16,
                                    const torch::Tensor& weights,
                                    const torch::Tensor& cu_seqlen_ks,
                                    const torch::Tensor& cu_seqlen_ke,
                                    torch::Tensor& logits) {
  const int64_t H = q_bf16.size(0);
  const int64_t M = q_bf16.size(1);
  const int64_t N_work = logits.size(1);
  const int64_t elements = M * N_work;
  const int64_t group32_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP32_MAX_ELEMENTS", 0);
  const bool group32_tiled =
      get_env_int64("VLLM_MQA_CUDA_V7_GROUP32_TILED", 0) != 0;
  const int64_t group16_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP16_MAX_ELEMENTS", 0);
  const int64_t group8_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP8_MAX_ELEMENTS", 64LL * 1024 * 1024);
  const int64_t group4_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP4_MAX_ELEMENTS", 128LL * 1024 * 1024);
  const int64_t group2_max_elements = get_env_int64(
      "VLLM_MQA_CUDA_V7_GROUP2_MAX_ELEMENTS", 512LL * 1024 * 1024);
  int64_t group_size = 1;
  if (group32_tiled && H == 32 && elements <= group16_max_elements) {
    group_size = 32;
  } else if (elements <= group32_max_elements) {
    group_size = 32;
  } else if (elements <= group16_max_elements) {
    group_size = 16;
  } else if (elements <= group8_max_elements) {
    group_size = 8;
  } else if (elements <= group4_max_elements) {
    group_size = 4;
  } else if (elements <= group2_max_elements) {
    group_size = 2;
  }
  const bool bf16_scores =
      get_env_int64("VLLM_MQA_CUDA_V7_BF16_SCORES", 1) != 0;
  const bool two_d_accum =
      get_env_int64("VLLM_MQA_CUDA_V7_2D_ACCUM", 0) != 0;
  fp8_mqa_logits_cuda_compute_bf16_qk(
      q_bf16, k_bf16, weights, cu_seqlen_ks, cu_seqlen_ke, logits, group_size,
      bf16_scores, two_d_accum);
}
