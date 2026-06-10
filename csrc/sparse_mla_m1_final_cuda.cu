#include <torch/all.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#ifndef USE_ROCM

namespace {

constexpr int kQNopeDim = 512;
constexpr int kQPeDim = 64;
constexpr int kVDim = 512;
constexpr int kQKDim = kQNopeDim + kQPeDim;
constexpr int kThreads = 256;
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

inline void check_cuda_launch(const char* kernel_name) {
  const cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, kernel_name,
              " failed: ", cudaGetErrorString(err));
}

__global__ void sparse_mla_m1_qk_scores_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ kv,
    const int32_t* __restrict__ indices, float* __restrict__ scores,
    int64_t q_nope_stride_head, int64_t q_nope_stride_d,
    int64_t q_pe_stride_head, int64_t q_pe_stride_d, int64_t kv_stride_token,
    int64_t kv_stride_d, int64_t indices_stride_k, int64_t scores_stride_head,
    int topk, int seq_kv, float sm_scale) {
  const int head = blockIdx.x;
  const int k_pos = blockIdx.y;
  const int tid = threadIdx.x;

  __shared__ float partial[kThreads];

  const int32_t kv_idx = indices[k_pos * indices_stride_k];
  float acc = 0.0f;
  if (kv_idx >= 0 && kv_idx < seq_kv) {
    for (int d = tid; d < kQNopeDim; d += blockDim.x) {
      const float qv =
          bf16_to_float(q_nope[head * q_nope_stride_head +
                               static_cast<int64_t>(d) * q_nope_stride_d]);
      const float kvv =
          bf16_to_float(kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
                           d * kv_stride_d]);
      acc += qv * kvv;
    }
    for (int d = tid; d < kQPeDim; d += blockDim.x) {
      const float qv =
          bf16_to_float(q_pe[head * q_pe_stride_head +
                             static_cast<int64_t>(d) * q_pe_stride_d]);
      const float kvv = bf16_to_float(
          kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
             static_cast<int64_t>(kQNopeDim + d) * kv_stride_d]);
      acc += qv * kvv;
    }
  }

  partial[tid] = acc;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    scores[head * scores_stride_head + k_pos] =
        (kv_idx >= 0 && kv_idx < seq_kv) ? partial[0] * sm_scale
                                         : kNegInf;
  }
}

__global__ void sparse_mla_m1_softmax_norm_kernel(
    const float* __restrict__ scores, float* __restrict__ norm,
    int64_t scores_stride_head, int64_t norm_stride_head, int topk) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float partial[kThreads];

  float local_max = kNegInf;
  for (int k = tid; k < topk; k += blockDim.x) {
    local_max = fmaxf(local_max, scores[head * scores_stride_head + k]);
  }
  partial[tid] = local_max;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
    }
    __syncthreads();
  }
  const float row_max = partial[0];

  float local_sum = 0.0f;
  if (row_max != kNegInf) {
    for (int k = tid; k < topk; k += blockDim.x) {
      local_sum += __expf(scores[head * scores_stride_head + k] - row_max);
    }
  }
  partial[tid] = local_sum;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    const float sum = partial[0];
    norm[head * norm_stride_head + 0] = row_max;
    norm[head * norm_stride_head + 1] = sum > 0.0f ? 1.0f / sum : 0.0f;
  }
}

__global__ void sparse_mla_m1_output_kernel(
    const __nv_bfloat16* __restrict__ kv,
    const int32_t* __restrict__ indices, const float* __restrict__ scores,
    const float* __restrict__ norm, __nv_bfloat16* __restrict__ out,
    int64_t kv_stride_token, int64_t kv_stride_d, int64_t indices_stride_k,
    int64_t scores_stride_head, int64_t norm_stride_head,
    int64_t out_stride_head, int64_t out_stride_d, int topk, int seq_kv) {
  const int head = blockIdx.x;
  const int dv = blockIdx.y;
  const int tid = threadIdx.x;
  __shared__ float partial[kThreads];

  const float row_max = norm[head * norm_stride_head + 0];
  const float inv_sum = norm[head * norm_stride_head + 1];
  float acc = 0.0f;
  if (inv_sum > 0.0f && row_max != kNegInf) {
    for (int k = tid; k < topk; k += blockDim.x) {
      const int32_t kv_idx = indices[k * indices_stride_k];
      if (kv_idx >= 0 && kv_idx < seq_kv) {
        const float prob =
            __expf(scores[head * scores_stride_head + k] - row_max) * inv_sum;
        const float vv =
            bf16_to_float(kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
                             static_cast<int64_t>(dv) * kv_stride_d]);
        acc += prob * vv;
      }
    }
  }

  partial[tid] = acc;
  __syncthreads();
  for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      partial[tid] += partial[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    out[head * out_stride_head + static_cast<int64_t>(dv) * out_stride_d] =
        __float2bfloat16_rn(partial[0]);
  }
}

void check_sparse_mla_m1_final_inputs(
    const torch::Tensor& q_nope, const torch::Tensor& q_pe,
    const torch::Tensor& kv, const torch::Tensor& indices,
    const torch::Tensor& scores, const torch::Tensor& norm,
    const torch::Tensor& out) {
  TORCH_CHECK(q_nope.is_cuda(), "q_nope must be a CUDA tensor");
  TORCH_CHECK(q_pe.is_cuda(), "q_pe must be a CUDA tensor");
  TORCH_CHECK(kv.is_cuda(), "kv must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(scores.is_cuda(), "scores must be a CUDA tensor");
  TORCH_CHECK(norm.is_cuda(), "norm must be a CUDA tensor");
  TORCH_CHECK(out.is_cuda(), "out must be a CUDA tensor");
  TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16,
              "q_nope must be bfloat16");
  TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16,
              "q_pe must be bfloat16");
  TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(scores.scalar_type() == torch::kFloat32,
              "scores must be float32");
  TORCH_CHECK(norm.scalar_type() == torch::kFloat32, "norm must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(q_nope.dim() == 3, "q_nope must be [1, H, 512]");
  TORCH_CHECK(q_pe.dim() == 3, "q_pe must be [1, H, 64]");
  TORCH_CHECK(kv.dim() == 3, "kv must be [seq_kv, 1, 576]");
  TORCH_CHECK(indices.dim() == 3, "indices must be [1, 1, topk]");
  TORCH_CHECK(out.dim() == 3, "out must be [1, H, 512]");
  TORCH_CHECK(q_nope.size(0) == 1, "q_nope M dimension must be 1");
  TORCH_CHECK(q_pe.size(0) == 1, "q_pe M dimension must be 1");
  TORCH_CHECK(out.size(0) == 1, "out M dimension must be 1");
  TORCH_CHECK(q_nope.size(2) == kQNopeDim, "q_nope dim must be 512");
  TORCH_CHECK(q_pe.size(2) == kQPeDim, "q_pe dim must be 64");
  TORCH_CHECK(kv.size(1) == 1 && kv.size(2) == kQKDim,
              "kv must be [seq_kv, 1, 576]");
  TORCH_CHECK(q_nope.size(1) == q_pe.size(1) &&
                  q_nope.size(1) == out.size(1),
              "head dimension mismatch");
  TORCH_CHECK(out.size(2) == kVDim, "out dim must be 512");
  TORCH_CHECK(indices.size(0) == 1 && indices.size(1) == 1,
              "indices must be [1, 1, topk]");
  TORCH_CHECK(scores.dim() == 2, "scores must be [H, topk]");
  TORCH_CHECK(norm.dim() == 2, "norm must be [H, 2]");
  TORCH_CHECK(scores.size(0) >= q_nope.size(1) &&
                  scores.size(1) >= indices.size(2),
              "scores workspace is too small");
  TORCH_CHECK(norm.size(0) >= q_nope.size(1) && norm.size(1) >= 2,
              "norm workspace is too small");
  TORCH_CHECK(q_nope.size(1) == 16,
              "prototype currently supports exactly 16 query heads");
  TORCH_CHECK(indices.size(2) == 2048,
              "prototype currently supports exactly topK=2048");
  TORCH_CHECK(q_nope.stride(2) == 1 && q_pe.stride(2) == 1 &&
                  kv.stride(2) == 1 && indices.stride(2) == 1 &&
                  scores.stride(1) == 1 && norm.stride(1) == 1 &&
                  out.stride(2) == 1,
              "prototype requires contiguous innermost dimensions");
}

}  // namespace

#endif  // USE_ROCM

void sparse_mla_m1_final_cuda(const torch::Tensor& q_nope,
                              const torch::Tensor& q_pe,
                              const torch::Tensor& kv,
                              const torch::Tensor& indices,
                              torch::Tensor& scores, torch::Tensor& norm,
                              torch::Tensor& out, double sm_scale) {
#ifndef USE_ROCM
  check_sparse_mla_m1_final_inputs(q_nope, q_pe, kv, indices, scores, norm,
                                   out);

  const int heads = static_cast<int>(q_nope.size(1));
  const int topk = static_cast<int>(indices.size(2));
  const int seq_kv = static_cast<int>(kv.size(0));
  cudaStream_t stream = 0;

  const auto* q_nope_ptr =
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
  const auto* q_pe_ptr =
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
  const auto* kv_ptr =
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>());
  const auto* indices_ptr = indices.data_ptr<int32_t>();
  auto* scores_ptr = scores.data_ptr<float>();
  auto* norm_ptr = norm.data_ptr<float>();
  auto* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

  sparse_mla_m1_qk_scores_kernel<<<dim3(heads, topk), kThreads, 0, stream>>>(
      q_nope_ptr, q_pe_ptr, kv_ptr, indices_ptr, scores_ptr,
      q_nope.stride(1), q_nope.stride(2), q_pe.stride(1), q_pe.stride(2),
      kv.stride(0), kv.stride(2), indices.stride(2), scores.stride(0), topk,
      seq_kv, static_cast<float>(sm_scale));
  check_cuda_launch("sparse_mla_m1_qk_scores_kernel");

  sparse_mla_m1_softmax_norm_kernel<<<heads, kThreads, 0, stream>>>(
      scores_ptr, norm_ptr, scores.stride(0), norm.stride(0), topk);
  check_cuda_launch("sparse_mla_m1_softmax_norm_kernel");

  sparse_mla_m1_output_kernel<<<dim3(heads, kVDim), kThreads, 0, stream>>>(
      kv_ptr, indices_ptr, scores_ptr, norm_ptr, out_ptr, kv.stride(0),
      kv.stride(2), indices.stride(2), scores.stride(0), norm.stride(0),
      out.stride(1), out.stride(2), topk, seq_kv);
  check_cuda_launch("sparse_mla_m1_output_kernel");
#else
  TORCH_CHECK(false, "sparse_mla_m1_final_cuda is not supported on ROCm");
#endif
}
