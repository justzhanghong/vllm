#include <torch/all.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#ifndef USE_ROCM

namespace {

namespace cg = cooperative_groups;

constexpr int kQNopeDim = 512;
constexpr int kQPeDim = 64;
constexpr int kVDim = 512;
constexpr int kQKDim = kQNopeDim + kQPeDim;
constexpr int kThreads = 128;
constexpr int kDVsPerThread = kVDim / kThreads;
constexpr int kMergeTile = 128;
constexpr int kMergeTiles = kVDim / kMergeTile;
constexpr float kNegInf = -3.4028234663852886e38f;

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
  return __bfloat162float(v);
}

__global__ void sparse_mla_m1_coop_final_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ kv,
    const int32_t* __restrict__ indices, float* __restrict__ partial_acc,
    float* __restrict__ partial_meta, __nv_bfloat16* __restrict__ out,
    int64_t q_nope_stride_head, int64_t q_nope_stride_d,
    int64_t q_pe_stride_head, int64_t q_pe_stride_d, int64_t kv_stride_token,
    int64_t kv_stride_d, int64_t indices_stride_k,
    int64_t partial_acc_stride_head, int64_t partial_acc_stride_split,
    int64_t partial_meta_stride_head, int64_t partial_meta_stride_split,
    int64_t out_stride_head, int64_t out_stride_d, int num_heads, int topk,
    int seq_kv, int num_splits, float sm_scale) {
  __shared__ float reduce[kThreads];
  const int tid = threadIdx.x;
  const int phase_blocks = num_heads * num_splits;

  if (blockIdx.x < phase_blocks) {
    const int head = blockIdx.x / num_splits;
    const int split = blockIdx.x - head * num_splits;
    const int split_topk = topk / num_splits;
    const int split_start = split * split_topk;
    const int split_end = split_start + split_topk;

    float e_max = kNegInf;
    float e_sum = 0.0f;
    float acc[kDVsPerThread];
#pragma unroll
    for (int i = 0; i < kDVsPerThread; ++i) {
      acc[i] = 0.0f;
    }

    for (int k_pos = split_start; k_pos < split_end; ++k_pos) {
      const int32_t kv_idx = indices[k_pos * indices_stride_k];
      if (kv_idx < 0 || kv_idx >= seq_kv) {
        continue;
      }

      float qk_part = 0.0f;
      for (int d = tid; d < kQNopeDim; d += blockDim.x) {
        const float qv =
            bf16_to_float(q_nope[head * q_nope_stride_head +
                                 static_cast<int64_t>(d) * q_nope_stride_d]);
        const float kvv =
            bf16_to_float(kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
                             static_cast<int64_t>(d) * kv_stride_d]);
        qk_part += qv * kvv;
      }
      for (int d = tid; d < kQPeDim; d += blockDim.x) {
        const float qv =
            bf16_to_float(q_pe[head * q_pe_stride_head +
                               static_cast<int64_t>(d) * q_pe_stride_d]);
        const float kvv = bf16_to_float(
            kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
               static_cast<int64_t>(kQNopeDim + d) * kv_stride_d]);
        qk_part += qv * kvv;
      }

      reduce[tid] = qk_part;
      __syncthreads();
      for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
          reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
      }

      const float qk = reduce[0] * sm_scale;
      const float n_e_max = fmaxf(e_max, qk);
      const float old_scale = (e_sum > 0.0f) ? __expf(e_max - n_e_max) : 0.0f;
      const float p = __expf(qk - n_e_max);

#pragma unroll
      for (int i = 0; i < kDVsPerThread; ++i) {
        const int dv = tid + i * kThreads;
        acc[i] *= old_scale;
        const float vv =
            bf16_to_float(kv[static_cast<int64_t>(kv_idx) * kv_stride_token +
                             static_cast<int64_t>(dv) * kv_stride_d]);
        acc[i] += p * vv;
      }
      e_sum = e_sum * old_scale + p;
      e_max = n_e_max;
      __syncthreads();
    }

    float* acc_base = partial_acc + head * partial_acc_stride_head +
                      split * partial_acc_stride_split;
#pragma unroll
    for (int i = 0; i < kDVsPerThread; ++i) {
      const int dv = tid + i * kThreads;
      acc_base[dv] = acc[i];
    }
    if (tid == 0) {
      float* meta = partial_meta + head * partial_meta_stride_head +
                    split * partial_meta_stride_split;
      meta[0] = e_sum > 0.0f ? e_max : kNegInf;
      meta[1] = e_sum;
    }
  }

  cg::this_grid().sync();

  const int merge_block = blockIdx.x;
  const int merge_blocks = num_heads * kMergeTiles;
  if (merge_block >= merge_blocks) {
    return;
  }

  const int head = merge_block / kMergeTiles;
  const int tile = merge_block - head * kMergeTiles;
  const int lane = tid;
  const int dv = tile * kMergeTile + lane;

  float global_max = kNegInf;
  for (int split = 0; split < num_splits; ++split) {
    const float* meta = partial_meta + head * partial_meta_stride_head +
                        split * partial_meta_stride_split;
    global_max = fmaxf(global_max, meta[0]);
  }

  float global_sum = 0.0f;
  float out_acc = 0.0f;
  if (global_max != kNegInf) {
    for (int split = 0; split < num_splits; ++split) {
      const float* meta = partial_meta + head * partial_meta_stride_head +
                          split * partial_meta_stride_split;
      const float split_sum = meta[1];
      if (split_sum <= 0.0f) {
        continue;
      }
      const float coeff = __expf(meta[0] - global_max);
      global_sum += coeff * split_sum;
      if (lane < kMergeTile && dv < kVDim) {
        const float* acc_base = partial_acc + head * partial_acc_stride_head +
                                split * partial_acc_stride_split;
        out_acc += coeff * acc_base[dv];
      }
    }
  }

  if (lane < kMergeTile && dv < kVDim) {
    const float result = global_sum > 0.0f ? out_acc / global_sum : 0.0f;
    out[head * out_stride_head + static_cast<int64_t>(dv) * out_stride_d] =
        __float2bfloat16_rn(result);
  }
}

inline void check_cuda_call(cudaError_t err, const char* name) {
  TORCH_CHECK(err == cudaSuccess, name, " failed: ", cudaGetErrorString(err));
}

void check_sparse_mla_m1_coop_inputs(
    const torch::Tensor& q_nope, const torch::Tensor& q_pe,
    const torch::Tensor& kv, const torch::Tensor& indices,
    const torch::Tensor& partial_acc, const torch::Tensor& partial_meta,
    const torch::Tensor& out, int64_t num_splits) {
  TORCH_CHECK(q_nope.is_cuda(), "q_nope must be CUDA");
  TORCH_CHECK(q_pe.is_cuda(), "q_pe must be CUDA");
  TORCH_CHECK(kv.is_cuda(), "kv must be CUDA");
  TORCH_CHECK(indices.is_cuda(), "indices must be CUDA");
  TORCH_CHECK(partial_acc.is_cuda(), "partial_acc must be CUDA");
  TORCH_CHECK(partial_meta.is_cuda(), "partial_meta must be CUDA");
  TORCH_CHECK(out.is_cuda(), "out must be CUDA");
  TORCH_CHECK(q_nope.scalar_type() == torch::kBFloat16,
              "q_nope must be bfloat16");
  TORCH_CHECK(q_pe.scalar_type() == torch::kBFloat16,
              "q_pe must be bfloat16");
  TORCH_CHECK(kv.scalar_type() == torch::kBFloat16, "kv must be bfloat16");
  TORCH_CHECK(indices.scalar_type() == torch::kInt32, "indices must be int32");
  TORCH_CHECK(partial_acc.scalar_type() == torch::kFloat32,
              "partial_acc must be float32");
  TORCH_CHECK(partial_meta.scalar_type() == torch::kFloat32,
              "partial_meta must be float32");
  TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(q_nope.dim() == 3 && q_nope.size(0) == 1 &&
                  q_nope.size(1) > 0 && q_nope.size(1) <= 16 &&
                  q_nope.size(2) == kQNopeDim,
              "q_nope must be [1, heads<=16, 512]");
  const int64_t num_heads = q_nope.size(1);
  TORCH_CHECK(q_pe.dim() == 3 && q_pe.size(0) == 1 &&
                  q_pe.size(1) == num_heads && q_pe.size(2) == kQPeDim,
              "q_pe must be [1, heads, 64]");
  TORCH_CHECK(kv.dim() == 3 && kv.size(1) == 1 && kv.size(2) == kQKDim,
              "kv must be [seq_kv, 1, 576]");
  TORCH_CHECK(indices.dim() == 3 && indices.size(0) == 1 &&
                  indices.size(1) == 1 && indices.size(2) == 2048,
              "indices must be [1, 1, 2048]");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == 1 && out.size(1) == num_heads &&
                  out.size(2) == kVDim,
              "out must be [1, heads, 512]");
  TORCH_CHECK(num_splits == 4 || num_splits == 8 || num_splits == 16 ||
                  num_splits == 32,
              "num_splits must be 4, 8, 16, or 32");
  TORCH_CHECK(indices.size(2) % num_splits == 0,
              "topk must be divisible by num_splits");
  TORCH_CHECK(partial_acc.dim() == 3 && partial_acc.size(0) >= num_heads &&
                  partial_acc.size(1) >= num_splits &&
                  partial_acc.size(2) >= kVDim,
              "partial_acc must be at least [heads, num_splits, 512]");
  TORCH_CHECK(partial_meta.dim() == 3 && partial_meta.size(0) >= num_heads &&
                  partial_meta.size(1) >= num_splits &&
                  partial_meta.size(2) >= 2,
              "partial_meta must be at least [heads, num_splits, 2]");
  TORCH_CHECK(q_nope.stride(2) == 1 && q_pe.stride(2) == 1 &&
                  kv.stride(2) == 1 && indices.stride(2) == 1 &&
                  partial_acc.stride(2) == 1 && partial_meta.stride(2) == 1 &&
                  out.stride(2) == 1,
              "prototype requires contiguous innermost dimensions");
}

}  // namespace

#endif  // USE_ROCM

void sparse_mla_m1_coop_final_cuda(const torch::Tensor& q_nope,
                                   const torch::Tensor& q_pe,
                                   const torch::Tensor& kv,
                                   const torch::Tensor& indices,
                                   torch::Tensor& partial_acc,
                                   torch::Tensor& partial_meta,
                                   torch::Tensor& out, double sm_scale,
                                   int64_t num_splits) {
#ifndef USE_ROCM
  check_sparse_mla_m1_coop_inputs(q_nope, q_pe, kv, indices, partial_acc,
                                  partial_meta, out, num_splits);
  const c10::cuda::CUDAGuard device_guard(q_nope.device());
  const int device = q_nope.get_device();
  int coop = 0;
  check_cuda_call(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch,
                                         device),
                  "cudaDeviceGetAttribute(cooperativeLaunch)");
  TORCH_CHECK(coop != 0, "device does not support cooperative launch");

  int sm_count = 0;
  check_cuda_call(cudaDeviceGetAttribute(&sm_count,
                                         cudaDevAttrMultiProcessorCount,
                                         device),
                  "cudaDeviceGetAttribute(multiprocessorCount)");
  int active_blocks_per_sm = 0;
  check_cuda_call(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                      &active_blocks_per_sm,
                      sparse_mla_m1_coop_final_kernel, kThreads, 0),
                  "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

  int num_heads = static_cast<int>(q_nope.size(1));
  const int phase_blocks = num_heads * static_cast<int>(num_splits);
  const int merge_blocks = num_heads * kMergeTiles;
  const int grid_blocks = std::max(phase_blocks, merge_blocks);
  TORCH_CHECK(grid_blocks <= active_blocks_per_sm * sm_count,
              "cooperative grid too large: grid=", grid_blocks,
              " active_capacity=", active_blocks_per_sm * sm_count);

  const auto* q_nope_ptr =
      reinterpret_cast<const __nv_bfloat16*>(q_nope.data_ptr<at::BFloat16>());
  const auto* q_pe_ptr =
      reinterpret_cast<const __nv_bfloat16*>(q_pe.data_ptr<at::BFloat16>());
  const auto* kv_ptr =
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr<at::BFloat16>());
  const auto* indices_ptr = indices.data_ptr<int32_t>();
  auto* partial_acc_ptr = partial_acc.data_ptr<float>();
  auto* partial_meta_ptr = partial_meta.data_ptr<float>();
  auto* out_ptr =
      reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
  int topk = static_cast<int>(indices.size(2));
  int seq_kv = static_cast<int>(kv.size(0));
  int splits = static_cast<int>(num_splits);
  float scale = static_cast<float>(sm_scale);
  int64_t q_nope_stride_head = q_nope.stride(1);
  int64_t q_nope_stride_d = q_nope.stride(2);
  int64_t q_pe_stride_head = q_pe.stride(1);
  int64_t q_pe_stride_d = q_pe.stride(2);
  int64_t kv_stride_token = kv.stride(0);
  int64_t kv_stride_d = kv.stride(2);
  int64_t indices_stride_k = indices.stride(2);
  int64_t partial_acc_stride_head = partial_acc.stride(0);
  int64_t partial_acc_stride_split = partial_acc.stride(1);
  int64_t partial_meta_stride_head = partial_meta.stride(0);
  int64_t partial_meta_stride_split = partial_meta.stride(1);
  int64_t out_stride_head = out.stride(1);
  int64_t out_stride_d = out.stride(2);
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream(device).stream();

  void* kernel_args[] = {
      const_cast<__nv_bfloat16**>(&q_nope_ptr),
      const_cast<__nv_bfloat16**>(&q_pe_ptr),
      const_cast<__nv_bfloat16**>(&kv_ptr),
      const_cast<int32_t**>(&indices_ptr),
      &partial_acc_ptr,
      &partial_meta_ptr,
      &out_ptr,
      &q_nope_stride_head,
      &q_nope_stride_d,
      &q_pe_stride_head,
      &q_pe_stride_d,
      &kv_stride_token,
      &kv_stride_d,
      &indices_stride_k,
      &partial_acc_stride_head,
      &partial_acc_stride_split,
      &partial_meta_stride_head,
      &partial_meta_stride_split,
      &out_stride_head,
      &out_stride_d,
      &num_heads,
      &topk,
      &seq_kv,
      &splits,
      &scale,
  };

  check_cuda_call(cudaLaunchCooperativeKernel(
                      reinterpret_cast<void*>(sparse_mla_m1_coop_final_kernel),
                      dim3(grid_blocks), dim3(kThreads), kernel_args, 0,
                      stream),
                  "cudaLaunchCooperativeKernel");
#else
  TORCH_CHECK(false, "sparse_mla_m1_coop_final_cuda is not supported on ROCm");
#endif
}
