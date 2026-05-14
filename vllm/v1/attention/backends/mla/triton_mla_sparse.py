# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for GPUs without FlashMLA Sparse (SM90+)
or FlashInfer MLA Sparse (SM100+), e.g. SM80 (A100) and SM121 (GB10)."""

import os
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    _BLOCK_DV,
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_sparse_mla_attention,
)

# DeepSeek-V3.2 / GLM-5.1 indexer shape, the only model family this backend
# serves. Used only for autotune priming — if a future model differs, the
# kernel simply re-tunes on first real use (same as pre-warmup behavior).
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    """Metadata builder advertising cudagraph support for the CUDA/Triton
    sparse MLA path. The XPU base keeps `AttentionCGSupport.NEVER` because
    its kernel has not been validated under cudagraph capture; this subclass
    is the only place the capability is claimed."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Overrides XPU sparse impl to use the split-KV kernel, which is
    3–7× faster for single-query decode on SM80 (A100/A30) and SM120 (GB10).
    """

    can_return_lse_for_decode: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Cached device SM count; passed into the kernel dispatch each forward
        # so the hot path doesn't re-query `q.device.index` → dict lookup.
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first user request
        does not pay the ~24 config-sweep cost inline."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        q = torch.empty(1, self.num_heads, _DIM_QK, dtype=torch.bfloat16, device=device)
        kv = torch.empty(64, 1, _DIM_QK, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_sparse_mla_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )
        extra_warmup_shapes = os.getenv("VLLM_SPARSE_MLA_WARMUP_NUM_TOKENS")
        if extra_warmup_shapes:
            for shape in extra_warmup_shapes.split(","):
                num_tokens = int(shape.strip())
                if num_tokens <= 1:
                    continue
                q = torch.empty(
                    num_tokens,
                    self.num_heads,
                    _DIM_QK,
                    dtype=torch.bfloat16,
                    device=device,
                )
                indices = torch.zeros(
                    num_tokens, 1, topk, dtype=torch.int32, device=device
                )
                triton_sparse_mla_attention(
                    q,
                    kv,
                    indices,
                    sm_scale=self.softmax_scale,
                    num_kv_splits=1,
                    sm_count=self._sm_count,
                )
                if os.getenv("VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK", "0") == "1":
                    triton_sparse_mla_attention(
                        q,
                        kv,
                        indices,
                        sm_scale=self.softmax_scale,
                        num_kv_splits=1,
                        sm_count=self._sm_count,
                        assume_valid_indices=True,
                    )
                    if (
                        os.getenv(
                            "VLLM_SPARSE_MLA_ASSUME_VALID_AFTER_TOPK_NOMASK",
                            "0",
                        )
                        == "1"
                    ):
                        triton_sparse_mla_attention(
                            q,
                            kv,
                            indices,
                            sm_scale=self.softmax_scale,
                            num_kv_splits=1,
                            sm_count=self._sm_count,
                            valid_index_base_seq_len=0,
                        )
            del q, indices
            torch.cuda.empty_cache()
        # The indexer's fp8 MQA logits kernels live on a separate autotune
        # cache. Prime them here so cold TTFT doesn't include their sweep.
        warmup_fp8_mqa_logits_triton(
            num_heads=_INDEXER_NUM_HEADS, head_dim=_INDEXER_HEAD_DIM, device=device
        )
        cfg = get_current_vllm_config_or_none()
        if cfg is not None:
            warmup_fp8_paged_mqa_logits_triton(
                num_heads=_INDEXER_NUM_HEADS,
                head_dim=_INDEXER_HEAD_DIM,
                block_size=cfg.cache_config.block_size,
                device=device,
            )

    @staticmethod
    def _slice_q(
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor], item: slice
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if isinstance(q, tuple):
            return q[0][item], q[1][item]
        return q[item]

    def _forward_bf16_kv(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
        return_lse: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        q_nope = q[0] if isinstance(q, tuple) else q
        num_tokens = q_nope.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        out_heads = q_nope.shape[1] if return_lse else self.num_heads
        if (
            os.getenv("VLLM_SPARSE_MLA_ASSUME_VALID_DYNAMIC", "0") == "1"
            and attn_metadata.num_reqs == 1
        ):
            full_topk_start = attn_metadata.full_topk_start
            if full_topk_start <= 0:
                result = triton_sparse_mla_attention(
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    assume_valid_indices=True,
                    return_lse=return_lse,
                )
                if return_lse:
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                return result[:, :out_heads, :]
            if full_topk_start < num_tokens:
                result = triton_sparse_mla_attention(
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    valid_index_base_seq_len=attn_metadata.base_seq_len,
                    return_lse=return_lse,
                )
                if return_lse:
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                return result[:, :out_heads, :]
        if (
            os.getenv("VLLM_SPARSE_MLA_ASSUME_VALID_SPLIT", "0") == "1"
            and attn_metadata.num_reqs == 1
        ):
            full_topk_start = attn_metadata.full_topk_start
            if full_topk_start <= 0:
                result = triton_sparse_mla_attention(
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    assume_valid_indices=True,
                    return_lse=return_lse,
                )
                if return_lse:
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                return result[:, :out_heads, :]
            if full_topk_start < num_tokens:
                output = torch.empty(
                    (num_tokens, self.num_heads, _BLOCK_DV),
                    dtype=torch.bfloat16,
                    device=q_nope.device,
                )
                # This split shortcut is only used in the non-DCP path; DCP
                # needs all gathered heads plus LSE for the downstream combine.
                if return_lse:
                    result = triton_sparse_mla_attention(
                        q,
                        kv_c_and_k_pe_cache,
                        topk_indices,
                        sm_scale=self.softmax_scale,
                        sm_count=self._sm_count,
                        valid_index_base_seq_len=attn_metadata.base_seq_len,
                        return_lse=True,
                    )
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                triton_sparse_mla_attention(
                    self._slice_q(q, slice(None, full_topk_start)),
                    kv_c_and_k_pe_cache,
                    topk_indices[:full_topk_start],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    out=output[:full_topk_start],
                )
                triton_sparse_mla_attention(
                    self._slice_q(q, slice(full_topk_start, None)),
                    kv_c_and_k_pe_cache,
                    topk_indices[full_topk_start:],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    out=output[full_topk_start:],
                    assume_valid_indices=True,
                )
                return output[:, :out_heads, :]
        result = triton_sparse_mla_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
            return_lse=return_lse,
        )
        if return_lse:
            output, lse = result
            return output[:, :out_heads, :], lse[:, :out_heads]
        return result[:, :out_heads, :]

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 kv is not supported with MLA Sparse yet")

        q_shape = q[0] if isinstance(q, tuple) else q
        num_actual_toks = q_shape.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
        )

        result = self._forward_bf16_kv(
            q,
            kv_c_and_k_pe_cache,
            topk_indices_global,
            attn_metadata,
            return_lse=self.need_to_return_lse_for_decode,
        )
        if self.need_to_return_lse_for_decode:
            attn_out, lse = result
            return attn_out, lse
        return result, None


class TritonMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_metadata_cls() -> type[XPUMLASparseMetadata]:
        return XPUMLASparseMetadata

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl

    @classmethod
    def is_mla(cls) -> bool:
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [_DIM_QK]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True
