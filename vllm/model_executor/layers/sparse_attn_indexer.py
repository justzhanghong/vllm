# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import os
import time

import torch

import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    is_deep_gemm_supported,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)
from vllm.v1.attention.backends.mla.indexer import (
    DeepseekV32IndexerMetadata,
)
from vllm.v1.attention.ops.common import pack_seq_triton, unpack_seq_triton
from vllm.v1.attention.ops.mqa_logits_triton import (
    fp8_mqa_dequant_k_cuda,
    fp8_mqa_dequant_q_cuda,
    fp8_mqa_logits_cuda,
    fp8_mqa_logits_cuda_v5,
    fp8_mqa_logits_cuda_v7,
    fp8_mqa_logits_cuda_v7_bf16_k,
    fp8_mqa_logits_cuda_v7_bf16_qk,
    fp8_mqa_logits_cuda_v7_bf16_qk_fused_triton,
    fp8_mqa_logits_cuda_v7_fused_triton,
    fp8_mqa_logits_triton,
    fp8_paged_mqa_logits_triton,
)
from vllm.v1.worker.workspace import current_workspace_manager

if current_platform.is_cuda_alike():
    from vllm import _custom_ops as ops
elif current_platform.is_xpu():
    from vllm._xpu_ops import xpu_ops

logger = init_logger(__name__)

RADIX_TOPK_WORKSPACE_SIZE = 1024 * 1024
TOPK_HIST_BINS = 2048


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: str = "0") -> int:
    return int(os.getenv(name, default) or default)


# Stage TPOT services restart per candidate, so these sparse DSA decode
# controls are fixed for the process lifetime.
_DECODE_LOGITS_WORKSPACE = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_LOGITS_WORKSPACE"
)
_DECODE_PREDEQUANT_Q = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_PREDEQUANT_Q"
)
_DECODE_TRIM_LOGITS = _env_flag("VLLM_SPARSE_INDEXER_DECODE_TRIM_LOGITS")
_DECODE_LOGITS_BUCKET_SIZE = _env_int(
    "VLLM_SPARSE_INDEXER_DECODE_LOGITS_BUCKET_SIZE",
    "8192",
)
_DECODE_TRIM_BLOCK_TABLE = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TRIM_BLOCK_TABLE"
)
_DECODE_TOPK_BACKEND = os.getenv(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BACKEND",
    "persistent",
)
_DECODE_TOPK_PAD_LOGITS_LEN = _env_int(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_PAD_LOGITS_LEN",
    "0",
)
_DECODE_TOPK_HIST_FUSION = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_HIST_FUSION"
)
_DECODE_TOPK_BIN_FUSION = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_BIN_FUSION"
)
_DECODE_TOPK_TILE_SELECT = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT"
)
_DECODE_TOPK_TILE_SELECT_NO_STORE = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_NO_STORE"
)
_DECODE_TOPK_TILE_SELECT_NO_STORE_CHECK = _env_flag(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_NO_STORE_CHECK",
    "1",
)
_DECODE_TOPK_TILE_SELECT_CANDIDATES = _env_int(
    "VLLM_SPARSE_INDEXER_DECODE_TOPK_TILE_SELECT_CANDIDATES",
    "16",
)
_DECODE_LOGITS_BLOCK_PAGES = _env_int(
    "VLLM_SPARSE_INDEXER_DECODE_LOGITS_BLOCK_PAGES",
    "1",
)
_SKIP_PREFILL_TOPK_CLEAR = _env_flag(
    "VLLM_SPARSE_INDEXER_SKIP_PREFILL_TOPK_CLEAR"
)
_SKIP_DECODE_TOPK_CLEAR = _env_flag("VLLM_SPARSE_INDEXER_SKIP_DECODE_TOPK_CLEAR")
_PREFILL_SHAPE_BUCKET_TRACE = _env_flag("VLLM_PREFILL_SHAPE_BUCKET_TRACE")
_PREFILL_MQA_CANONICAL_M = _env_int(
    "VLLM_MQA_CUDA_V7_FUSED_TRITON_PREFILL_CANONICAL_M",
    "0",
)


def _shape_bucket_trace_sync(device: torch.device) -> None:
    if _PREFILL_SHAPE_BUCKET_TRACE and device.type == "cuda":
        torch.cuda.synchronize(device)


def _shape_bucket_trace_ms(start: float, device: torch.device) -> float:
    _shape_bucket_trace_sync(device)
    return (time.perf_counter() - start) * 1000.0


def sparse_attn_indexer(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor,
) -> torch.Tensor:
    # careful! this will be None in dummy run
    attn_metadata = get_forward_context().attn_metadata
    fp8_dtype = current_platform.fp8_dtype()
    k_cache_prefix = _resolve_layer_name(k_cache_prefix)

    # assert isinstance(attn_metadata, dict)
    if not isinstance(attn_metadata, dict):
        # Reserve workspace for indexer during profiling run
        workspace_shapes: list[tuple[tuple[int, ...], torch.dtype]] = [
            ((total_seq_lens, head_dim), torch.float8_e4m3fn),
            ((total_seq_lens, 4), torch.uint8),
            ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
        ]
        max_logits_elems = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
        if _DECODE_LOGITS_WORKSPACE:
            workspace_shapes.append(((max_logits_elems,), torch.uint8))
        if _DECODE_PREDEQUANT_Q:
            workspace_shapes.append(
                (
                    (q_fp8.shape[1], hidden_states.shape[0], q_fp8.shape[2]),
                    torch.bfloat16,
                )
            )
        if _DECODE_TOPK_HIST_FUSION:
            workspace_shapes.append(
                ((hidden_states.shape[0], TOPK_HIST_BINS), torch.int32)
            )
        if _DECODE_TOPK_BIN_FUSION:
            workspace_shapes.append(
                ((hidden_states.shape[0], max_model_len), torch.int16)
            )
        if _DECODE_TOPK_TILE_SELECT:
            tile_width = max(1, _DECODE_LOGITS_BLOCK_PAGES) * 64
            tile_groups = (max_model_len + tile_width - 1) // tile_width
            tile_candidates = max(1, _DECODE_TOPK_TILE_SELECT_CANDIDATES)
            workspace_shapes.extend(
                (
                    (
                        (hidden_states.shape[0], tile_groups * tile_candidates),
                        torch.float32,
                    ),
                    (
                        (hidden_states.shape[0], tile_groups * tile_candidates),
                        torch.int32,
                    ),
                    ((hidden_states.shape[0], tile_groups), torch.float32),
                    ((hidden_states.shape[0],), torch.int32),
                )
            )
        if (
            envs.VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND == "cuda_v7"
            and os.getenv("VLLM_MQA_CUDA_V7_PREDEQUANT_K", "0") == "1"
            and os.getenv("VLLM_MQA_CUDA_V7_PREDEQUANT_K_WORKSPACE", "0") == "1"
        ):
            padded_total_seq_lens = total_seq_lens
            if (
                os.getenv("VLLM_MQA_CUDA_V7_PAD_N", "0") == "1"
                and padded_total_seq_lens >= 32768
                and padded_total_seq_lens % 128 != 0
            ):
                padded_total_seq_lens = (
                    (padded_total_seq_lens + 127) // 128
                ) * 128
            workspace_shapes.append(
                ((padded_total_seq_lens, head_dim), torch.bfloat16)
            )
        current_workspace_manager().get_simultaneous(*workspace_shapes)

        # Dummy allocation to simulate for peak logits tensor memory during inference.
        # FP8 elements so elements == bytes
        if not _DECODE_LOGITS_WORKSPACE:
            _ = torch.empty(
                max_logits_elems, dtype=torch.uint8, device=hidden_states.device
            )

        return sparse_attn_indexer_fake(
            hidden_states,
            k_cache_prefix,
            kv_cache,
            q_fp8,
            k,
            weights,
            quant_block_size,
            scale_fmt,
            topk_tokens,
            head_dim,
            max_model_len,
            total_seq_lens,
            topk_indices_buffer,
        )
    attn_metadata_narrowed = attn_metadata[k_cache_prefix]
    assert isinstance(attn_metadata_narrowed, DeepseekV32IndexerMetadata)
    slot_mapping = attn_metadata_narrowed.slot_mapping
    has_decode = attn_metadata_narrowed.num_decodes > 0
    has_prefill = attn_metadata_narrowed.num_prefills > 0
    num_decode_tokens = attn_metadata_narrowed.num_decode_tokens

    # During speculative decoding, k may be padded to the CUDA graph batch
    # size while slot_mapping only covers actual tokens. Truncate k to avoid
    # out-of-bounds reads in the kernel.
    num_tokens = slot_mapping.shape[0]
    k = k[:num_tokens]

    # scale_fmt can be None, but the function expects str
    assert scale_fmt is not None
    if _PREFILL_SHAPE_BUCKET_TRACE:
        _shape_bucket_trace_sync(hidden_states.device)
        trace_start = time.perf_counter()
    ops.indexer_k_quant_and_cache(
        k,
        kv_cache,
        slot_mapping,
        quant_block_size,
        scale_fmt,
    )
    if _PREFILL_SHAPE_BUCKET_TRACE:
        logger.info(
            "PREFILL_SHAPE_BUCKET_INDEXER_TIMING op=indexer_k_quant_cache "
            "hidden=%s slot_mapping=%s ms=%.3f",
            tuple(hidden_states.shape),
            tuple(slot_mapping.shape),
            _shape_bucket_trace_ms(trace_start, hidden_states.device),
        )

    skip_prefill_topk_clear = (
        has_prefill
        and not has_decode
        and _SKIP_PREFILL_TOPK_CLEAR
    )
    skip_decode_topk_clear = (
        has_decode
        and not has_prefill
        and _SKIP_DECODE_TOPK_CLEAR
    )
    if not (skip_prefill_topk_clear or skip_decode_topk_clear):
        topk_indices_buffer[: hidden_states.shape[0]] = -1
    if has_prefill:
        prefill_metadata = attn_metadata_narrowed.prefill
        assert prefill_metadata is not None
        if _PREFILL_SHAPE_BUCKET_TRACE:
            logger.info(
                "PREFILL_SHAPE_BUCKET_INDEXER_META hidden=%s q_fp8=%s k=%s "
                "weights=%s slot_mapping=%s num_tokens=%s max_query=%s "
                "max_seq=%s chunks=%s",
                tuple(hidden_states.shape),
                tuple(q_fp8.shape),
                tuple(k.shape),
                tuple(weights.shape),
                tuple(slot_mapping.shape),
                attn_metadata_narrowed.num_actual_tokens,
                attn_metadata_narrowed.max_query_len,
                attn_metadata_narrowed.max_seq_len,
                len(prefill_metadata.chunks),
            )

        # Get the full shared workspace buffers once (will allocate on first use)
        workspace_manager = current_workspace_manager()
        mqa_backend = envs.VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND
        predequant_k = (
            mqa_backend == "cuda_v7"
            and os.getenv("VLLM_MQA_CUDA_V7_PREDEQUANT_K", "0") == "1"
        )
        predequant_q = (
            predequant_k
            and os.getenv("VLLM_MQA_CUDA_V7_PREDEQUANT_Q", "0") == "1"
        )
        fused_triton = (
            predequant_k
            and os.getenv("VLLM_MQA_CUDA_V7_FUSED_TRITON", "0") == "1"
        )
        q_workspace = (
            predequant_k
            and not predequant_q
            and (
                fused_triton
                or os.getenv("VLLM_MQA_CUDA_V7_Q_WORKSPACE", "0") == "1"
            )
        )
        use_predequant_workspace = (
            predequant_k
            and os.getenv("VLLM_MQA_CUDA_V7_PREDEQUANT_K_WORKSPACE", "0") == "1"
        )
        k_bf16_workspace: torch.Tensor | None = None
        if use_predequant_workspace:

            def padded_total_n(n: int) -> int:
                if (
                    os.getenv("VLLM_MQA_CUDA_V7_PAD_N", "0") == "1"
                    and n >= 32768
                    and n % 128 != 0
                ):
                    return ((n + 127) // 128) * 128
                return n

            max_k_bf16_rows = max(
                padded_total_n(chunk.total_seq_lens)
                for chunk in prefill_metadata.chunks
            )
            k_fp8_full, k_scale_full, k_bf16_workspace = (
                workspace_manager.get_simultaneous(
                    ((total_seq_lens, head_dim), fp8_dtype),
                    ((total_seq_lens, 4), torch.uint8),
                    ((max_k_bf16_rows, head_dim), torch.bfloat16),
                )
            )
        else:
            k_fp8_full, k_scale_full = workspace_manager.get_simultaneous(
                ((total_seq_lens, head_dim), fp8_dtype),
                ((total_seq_lens, 4), torch.uint8),
            )
        k_bf16_full: torch.Tensor | None = None
        q_bf16_full: torch.Tensor | None = None
        if predequant_q:
            q_bf16_full = torch.empty(
                (q_fp8.shape[1], q_fp8.shape[0], q_fp8.shape[2]),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
            fp8_mqa_dequant_q_cuda(q_fp8, q_bf16_full)
        max_actual_chunk_m = max(
            chunk.token_end - chunk.token_start
            for chunk in prefill_metadata.chunks
        )
        canonical_chunk_m = max_actual_chunk_m
        if fused_triton and _PREFILL_MQA_CANONICAL_M > 0:
            if max_actual_chunk_m > _PREFILL_MQA_CANONICAL_M:
                logger.warning_once(
                    "Prefill fused Triton MQA actual chunk M %s exceeds "
                    "configured canonical M %s; check split_indexer_prefill_chunks.",
                    max_actual_chunk_m,
                    _PREFILL_MQA_CANONICAL_M,
                )
            canonical_chunk_m = max(
                canonical_chunk_m, _PREFILL_MQA_CANONICAL_M
            )
        q_bf16_workspace: torch.Tensor | None = None
        if q_workspace:
            q_bf16_workspace = torch.empty(
                (q_fp8.shape[1], canonical_chunk_m, q_fp8.shape[2]),
                dtype=torch.bfloat16,
                device=hidden_states.device,
            )
        logits_workspace: torch.Tensor | None = None
        reuse_logits = (
            predequant_k
            and (
                fused_triton
                or os.getenv("VLLM_MQA_CUDA_V7_REUSE_LOGITS", "0") == "1"
            )
        )
        if reuse_logits:
            pad_logits_n = os.getenv("VLLM_MQA_CUDA_V7_PAD_N", "0") == "1"

            def padded_n(n: int) -> int:
                if pad_logits_n and n >= 32768 and n % 128 != 0:
                    return ((n + 127) // 128) * 128
                return n

            max_chunk_n = max(
                padded_n(chunk.active_seq_lens)
                for chunk in prefill_metadata.chunks
            )
            logits_workspace = torch.empty(
                (canonical_chunk_m, max_chunk_n),
                dtype=torch.float32,
                device=hidden_states.device,
            )
        for chunk in prefill_metadata.chunks:
            actual_total_seq_lens = (
                chunk.actual_total_seq_lens
                if chunk.actual_total_seq_lens is not None
                else chunk.total_seq_lens
            )
            actual_active_seq_lens = (
                chunk.actual_active_seq_lens
                if chunk.actual_active_seq_lens is not None
                else chunk.active_seq_lens
            )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_INDEXER_CHUNK token_start=%s "
                    "token_end=%s q_m=%s canonical_m=%s total_seq_lens=%s "
                    "active_seq_lens=%s "
                    "actual_total_seq_lens=%s actual_active_seq_lens=%s "
                    "num_reqs=%s skip_kv_gather=%s",
                    chunk.token_start,
                    chunk.token_end,
                    chunk.token_end - chunk.token_start,
                    canonical_chunk_m,
                    chunk.total_seq_lens,
                    chunk.active_seq_lens,
                    chunk.actual_total_seq_lens,
                    chunk.actual_active_seq_lens,
                    chunk.num_reqs,
                    chunk.skip_kv_gather,
                )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                _shape_bucket_trace_sync(hidden_states.device)
                trace_gather_start = time.perf_counter()
            if not chunk.skip_kv_gather:
                k_fp8_gather = k_fp8_full[:actual_total_seq_lens]
                k_scale_gather = k_scale_full[:actual_total_seq_lens]
                ops.cp_gather_indexer_k_quant_cache(
                    kv_cache,
                    k_fp8_gather,
                    k_scale_gather,
                    chunk.block_table,
                    chunk.cu_seq_lens,
                )
                if predequant_k:
                    padded_total_seq_lens = chunk.total_seq_lens
                    if (
                        os.getenv("VLLM_MQA_CUDA_V7_PAD_N", "0") == "1"
                        and padded_total_seq_lens >= 32768
                        and padded_total_seq_lens % 128 != 0
                    ):
                        padded_total_seq_lens = (
                            (padded_total_seq_lens + 127) // 128
                        ) * 128
                    if use_predequant_workspace:
                        assert k_bf16_workspace is not None
                        k_bf16_full = k_bf16_workspace[:padded_total_seq_lens]
                    else:
                        k_bf16_full = torch.empty(
                            (padded_total_seq_lens, head_dim),
                            dtype=torch.bfloat16,
                            device=hidden_states.device,
                        )
                    fp8_mqa_dequant_k_cuda(
                        k_fp8_gather,
                        k_scale_gather.view(torch.float32).flatten(),
                        k_bf16_full[:actual_total_seq_lens],
                    )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_INDEXER_TIMING op=gather_dequant "
                    "token_start=%s token_end=%s q_m=%s active_n=%s "
                    "total_n=%s skip_kv_gather=%s ms=%.3f",
                    chunk.token_start,
                    chunk.token_end,
                    chunk.token_end - chunk.token_start,
                    chunk.active_seq_lens,
                    chunk.total_seq_lens,
                    chunk.skip_kv_gather,
                    _shape_bucket_trace_ms(trace_gather_start, hidden_states.device),
                )

            k_fp8 = k_fp8_full[: chunk.active_seq_lens]
            k_scale = k_scale_full[: chunk.active_seq_lens]

            if _PREFILL_SHAPE_BUCKET_TRACE:
                _shape_bucket_trace_sync(hidden_states.device)
                trace_mqa_start = time.perf_counter()
            if is_deep_gemm_supported():
                logits = fp8_mqa_logits(
                    q_fp8[chunk.token_start : chunk.token_end],
                    (k_fp8, k_scale.view(torch.float32).flatten()),
                    weights[chunk.token_start : chunk.token_end],
                    chunk.cu_seqlen_ks,
                    chunk.cu_seqlen_ke,
                    clean_logits=False,
                )
            else:
                mqa_backend = envs.VLLM_SPARSE_INDEXER_MQA_LOGITS_BACKEND
                q_chunk = q_fp8[chunk.token_start : chunk.token_end]
                w_chunk = weights[chunk.token_start : chunk.token_end]
                k_scales = k_scale.view(torch.float32).flatten()
                topk_indices = topk_indices_buffer[
                    chunk.token_start : chunk.token_end, :topk_tokens
                ]
                cu_seqlen_ks = chunk.cu_seqlen_ks
                cu_seqlen_ke = chunk.cu_seqlen_ke
                token_start = chunk.token_start
                row_start_zero = (
                    chunk.num_reqs == 1
                    and os.getenv(
                        "VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_START_ZERO", "0"
                    )
                    == "1"
                    and q_chunk.shape[0]
                    >= int(
                        os.getenv(
                            "VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_START_MIN_M",
                            "0",
                        )
                    )
                )
                row_end_base = None
                if (
                    row_start_zero
                    and os.getenv(
                        "VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_END_CONTIGUOUS", "0"
                    )
                    == "1"
                    and q_chunk.shape[0]
                    >= int(
                        os.getenv(
                            "VLLM_MQA_CUDA_V7_FUSED_TRITON_ROW_END_MIN_M",
                            "1024",
                        )
                    )
                ):
                    row_end_base = actual_active_seq_lens - q_chunk.shape[0] + 1
                if mqa_backend == "triton":
                    logger.info_once(
                        "Sparse indexer prefill MQA logits backend: Triton"
                    )
                    logits = fp8_mqa_logits_triton(
                        q_chunk,
                        (k_fp8, k_scales),
                        w_chunk,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                    )
                elif mqa_backend == "cuda_v5":
                    logger.info_once(
                        "Sparse indexer prefill MQA logits backend: CUDA/cuBLAS v5"
                    )
                    logits = fp8_mqa_logits_cuda_v5(
                        q_chunk,
                        (k_fp8, k_scales),
                        w_chunk,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        fallback=False,
                    )
                elif mqa_backend == "cuda_v7":
                    if fused_triton:
                        logger.info_once(
                            "Sparse indexer prefill MQA logits backend: "
                            "CUDA/cuBLAS v7 + fused Triton logits"
                        )
                    else:
                        logger.info_once(
                            "Sparse indexer prefill MQA logits backend: "
                            "CUDA/cuBLAS v7"
                        )
                    if predequant_k:
                        assert k_bf16_full is not None
                        if fused_triton:
                            if predequant_q:
                                assert q_bf16_full is not None
                                logits = fp8_mqa_logits_cuda_v7_bf16_qk_fused_triton(
                                    q_bf16_full[
                                        :,
                                        token_start : chunk.token_end,
                                        :,
                                    ],
                                    k_bf16_full,
                                    w_chunk,
                                    cu_seqlen_ks,
                                    cu_seqlen_ke,
                                    actual_m=q_chunk.shape[0],
                                    canonical_m=canonical_chunk_m,
                                    actual_n=actual_active_seq_lens,
                                    canonical_n=chunk.active_seq_lens,
                                    logits_out=logits_workspace,
                                    row_start_zero=row_start_zero,
                                    row_end_base=row_end_base,
                                )
                            else:
                                q_bf16_out = None
                                if q_workspace:
                                    assert q_bf16_workspace is not None
                                    q_bf16_out = q_bf16_workspace.as_strided(
                                        (
                                            q_fp8.shape[1],
                                            q_chunk.shape[0],
                                            q_fp8.shape[2],
                                        ),
                                        (
                                            q_chunk.shape[0] * q_fp8.shape[2],
                                            q_fp8.shape[2],
                                            1,
                                        ),
                                    )
                                logits = fp8_mqa_logits_cuda_v7_fused_triton(
                                    q_chunk,
                                    k_bf16_full,
                                    w_chunk,
                                    cu_seqlen_ks,
                                    cu_seqlen_ke,
                                    actual_m=q_chunk.shape[0],
                                    canonical_m=canonical_chunk_m,
                                    actual_n=actual_active_seq_lens,
                                    canonical_n=chunk.active_seq_lens,
                                    q_bf16_out=q_bf16_out,
                                    logits_out=logits_workspace,
                                    row_start_zero=row_start_zero,
                                    row_end_base=row_end_base,
                                    fallback=False,
                                )
                        elif predequant_q:
                            assert q_bf16_full is not None
                            logits = fp8_mqa_logits_cuda_v7_bf16_qk(
                                q_bf16_full[
                                    :,
                                    token_start : chunk.token_end,
                                    :,
                                ],
                                k_bf16_full,
                                w_chunk,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                actual_n=actual_active_seq_lens,
                                logits_out=logits_workspace,
                                fallback=False,
                            )
                        elif q_workspace:
                            assert q_bf16_workspace is not None
                            q_bf16 = q_bf16_workspace.as_strided(
                                (
                                    q_fp8.shape[1],
                                    q_chunk.shape[0],
                                    q_fp8.shape[2],
                                ),
                                (
                                    q_chunk.shape[0] * q_fp8.shape[2],
                                    q_fp8.shape[2],
                                    1,
                                ),
                            )
                            fp8_mqa_dequant_q_cuda(q_chunk, q_bf16)
                            logits = fp8_mqa_logits_cuda_v7_bf16_qk(
                                q_bf16,
                                k_bf16_full,
                                w_chunk,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                actual_n=actual_active_seq_lens,
                                logits_out=logits_workspace,
                                fallback=False,
                            )
                        else:
                            logits = fp8_mqa_logits_cuda_v7_bf16_k(
                                q_chunk,
                                k_bf16_full,
                                w_chunk,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                actual_n=actual_active_seq_lens,
                                logits_out=logits_workspace,
                                fallback=False,
                            )
                    else:
                        logits = fp8_mqa_logits_cuda_v7(
                            q_chunk,
                            (k_fp8, k_scales),
                            w_chunk,
                            cu_seqlen_ks,
                            cu_seqlen_ke,
                            fallback=False,
                        )
                else:
                    logger.info_once(
                        "Sparse indexer prefill MQA logits backend: CUDA/cuBLAS"
                    )
                    logits = fp8_mqa_logits_cuda(
                        q_chunk,
                        (k_fp8, k_scales),
                        w_chunk,
                        cu_seqlen_ks,
                        cu_seqlen_ke,
                        fallback=mqa_backend == "auto",
                    )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_INDEXER_TIMING op=mqa_logits "
                    "token_start=%s token_end=%s q_m=%s canonical_m=%s "
                    "active_n=%s logits=%s ms=%.3f",
                    chunk.token_start,
                    chunk.token_end,
                    chunk.token_end - chunk.token_start,
                    canonical_chunk_m,
                    chunk.active_seq_lens,
                    tuple(logits.shape),
                    _shape_bucket_trace_ms(trace_mqa_start, hidden_states.device),
                )
            num_rows = logits.shape[0]

            if _PREFILL_SHAPE_BUCKET_TRACE:
                _shape_bucket_trace_sync(hidden_states.device)
                trace_topk_start = time.perf_counter()
            if (
                current_platform.is_cuda()
                and chunk.num_reqs == 1
                and os.getenv("VLLM_SPARSE_INDEXER_PREFILL_DECODE_TOPK", "0")
                == "1"
            ):
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    1,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            elif (
                current_platform.is_cuda()
                and chunk.num_reqs == 1
                and os.getenv("VLLM_SPARSE_INDEXER_PREFILL_PERSISTENT_TOPK", "0")
                == "1"
            ):
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
                torch.ops._C.persistent_topk(
                    logits,
                    cu_seqlen_ke,
                    topk_indices,
                    topk_workspace,
                    topk_tokens,
                    attn_metadata_narrowed.max_seq_len,
                )
            elif current_platform.is_xpu():
                xpu_ops.top_k_per_row_prefill(  # type: ignore[attr-defined]
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_prefill(
                    logits,
                    cu_seqlen_ks,
                    cu_seqlen_ke,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_INDEXER_TIMING op=topk "
                    "token_start=%s token_end=%s q_m=%s active_n=%s "
                    "num_rows=%s topk=%s ms=%.3f",
                    chunk.token_start,
                    chunk.token_end,
                    chunk.token_end - chunk.token_start,
                    chunk.active_seq_lens,
                    num_rows,
                    topk_tokens,
                    _shape_bucket_trace_ms(trace_topk_start, hidden_states.device),
                )
    if has_decode:
        decode_metadata = attn_metadata_narrowed.decode
        assert decode_metadata is not None
        # kv_cache shape [
        # kv_cache size requirement [num_block, block_size, n_head, head_dim],
        # we only have [num_block, block_size, head_dim],
        kv_cache = kv_cache.unsqueeze(-2)
        decode_lens = decode_metadata.decode_lens
        if decode_metadata.requires_padding:
            # pad in edge case where we have short chunked prefill length <
            # decode_threshold since we unstrictly split
            # prefill and decode by decode_threshold
            # (currently set to 1 + speculative tokens)
            padded_q_fp8_decode_tokens = pack_seq_triton(
                q_fp8[:num_decode_tokens], decode_lens
            )
        else:
            padded_q_fp8_decode_tokens = q_fp8[:num_decode_tokens].reshape(
                decode_lens.shape[0], -1, *q_fp8.shape[1:]
            )
        # TODO: move and optimize below logic with triton kernels
        batch_size = padded_q_fp8_decode_tokens.shape[0]
        next_n = padded_q_fp8_decode_tokens.shape[1]
        num_padded_tokens = batch_size * next_n
        seq_lens = decode_metadata.seq_lens[:batch_size]
        # seq_lens is (B, next_n) for native spec decode, (B,) otherwise.
        # fp8_paged_mqa_logits and all topk kernels accept both shapes.
        decode_block_table = decode_metadata.block_table
        decode_logits_len = max_model_len
        if _DECODE_TRIM_LOGITS:
            bucket_size = _DECODE_LOGITS_BUCKET_SIZE
            active_len = max(int(attn_metadata_narrowed.max_seq_len), topk_tokens)
            if bucket_size > 0:
                active_len = (
                    (active_len + bucket_size - 1) // bucket_size
                ) * bucket_size
            decode_logits_len = min(max_model_len, active_len)
            block_size = int(kv_cache.shape[1])
            max_blocks = (decode_logits_len + block_size - 1) // block_size
            decode_block_table = decode_block_table[:, :max_blocks]
        if (
            _DECODE_TRIM_BLOCK_TABLE
            and attn_metadata_narrowed.max_seq_len < max_model_len
        ):
            block_size = int(kv_cache.shape[1])
            max_blocks = (
                attn_metadata_narrowed.max_seq_len + block_size - 1
            ) // block_size
            decode_block_table = decode_block_table[:, :max_blocks]
        if _PREFILL_SHAPE_BUCKET_TRACE:
            logger.info(
                "PREFILL_SHAPE_BUCKET_DECODE_META q=%s batch=%s next_n=%s "
                "seq_lens=%s max_seq=%s logits_len=%s block_table=%s "
                "topk_backend=%s",
                tuple(q_fp8[:num_decode_tokens].shape),
                batch_size,
                next_n,
                tuple(seq_lens.shape),
                attn_metadata_narrowed.max_seq_len,
                decode_logits_len,
                tuple(decode_block_table.shape),
                _DECODE_TOPK_BACKEND,
            )
        topk_workspace: torch.Tensor | None = None
        decode_topk_backend = _DECODE_TOPK_BACKEND
        decode_topk_logits_len = decode_logits_len
        topk_pad_logits_len = _DECODE_TOPK_PAD_LOGITS_LEN
        if (
            current_platform.is_cuda()
            and decode_topk_backend == "legacy"
            and topk_pad_logits_len > decode_logits_len
        ):
            # Widen only the topK view so legacy topK can use its multi-block
            # branch. MQA logits still computes decode_logits_len columns, and
            # topK only scans rowEnd <= seq_len <= decode_logits_len.
            decode_topk_logits_len = min(max_model_len, topk_pad_logits_len)
        use_topk_hist_fusion = (
            current_platform.is_cuda()
            and _DECODE_TOPK_HIST_FUSION
            and decode_topk_backend == "legacy"
            and topk_tokens == TOPK_HIST_BINS
            and next_n == 1
            and seq_lens.dim() == 1
            and not decode_metadata.requires_padding
            and decode_topk_logits_len == decode_logits_len
        )
        use_topk_bin_fusion = (
            current_platform.is_cuda()
            and _DECODE_TOPK_BIN_FUSION
            and not use_topk_hist_fusion
            and decode_topk_backend == "legacy"
            and topk_tokens == TOPK_HIST_BINS
            and next_n == 1
            and seq_lens.dim() == 1
            and not decode_metadata.requires_padding
            and decode_topk_logits_len == decode_logits_len
        )
        tile_select_candidates = max(1, _DECODE_TOPK_TILE_SELECT_CANDIDATES)
        tile_select_block_pages = max(1, _DECODE_LOGITS_BLOCK_PAGES)
        tile_select_block_size = int(kv_cache.shape[1])
        tile_select_groups = (
            decode_block_table.shape[1] + tile_select_block_pages - 1
        ) // tile_select_block_pages
        use_topk_tile_select = (
            current_platform.is_cuda()
            and _DECODE_TOPK_TILE_SELECT
            and not use_topk_hist_fusion
            and not use_topk_bin_fusion
            and decode_topk_backend == "legacy"
            and topk_tokens == TOPK_HIST_BINS
            and next_n == 1
            and seq_lens.dim() == 1
            and not decode_metadata.requires_padding
            and decode_topk_logits_len == decode_logits_len
            and tile_select_groups * tile_select_candidates >= topk_tokens
            and tile_select_block_size > 0
        )
        use_topk_tile_select_no_store = (
            use_topk_tile_select and _DECODE_TOPK_TILE_SELECT_NO_STORE
        )
        if use_topk_tile_select_no_store:
            logger.info_once(
                "Sparse indexer decode TopK tile-select no-store is enabled; "
                "fallback-free candidate proof is required before deployment."
            )
        topk_histogram: torch.Tensor | None = None
        topk_bins: torch.Tensor | None = None
        topk_tile_candidate_logits: torch.Tensor | None = None
        topk_tile_candidate_indices: torch.Tensor | None = None
        topk_tile_candidate_cutoffs: torch.Tensor | None = None
        topk_tile_fallback_flags: torch.Tensor | None = None
        if is_deep_gemm_supported():
            if _PREFILL_SHAPE_BUCKET_TRACE:
                _shape_bucket_trace_sync(hidden_states.device)
                trace_decode_mqa_start = time.perf_counter()
            logits = fp8_paged_mqa_logits(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_block_table,
                decode_metadata.schedule_metadata,
                max_model_len=decode_logits_len,
                clean_logits=False,
            )
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_DECODE_TIMING op=paged_mqa_logits "
                    "logits=%s max_seq=%s logits_len=%s ms=%.3f",
                    tuple(logits.shape),
                    attn_metadata_narrowed.max_seq_len,
                    decode_logits_len,
                    _shape_bucket_trace_ms(trace_decode_mqa_start,
                                           hidden_states.device),
                )
        else:
            logits_out: torch.Tensor | None = None
            q_bf16_decode: torch.Tensor | None = None
            if current_platform.is_cuda():
                workspace_manager = current_workspace_manager()
                workspace_shapes: list[tuple[tuple[int, ...], torch.dtype]] = []
                if _DECODE_PREDEQUANT_Q:
                    workspace_shapes.append(
                        (
                            (
                                padded_q_fp8_decode_tokens.shape[2],
                                num_padded_tokens,
                                padded_q_fp8_decode_tokens.shape[3],
                            ),
                            torch.bfloat16,
                        )
                    )
                if use_topk_hist_fusion:
                    workspace_shapes.append(
                        ((num_padded_tokens, TOPK_HIST_BINS), torch.int32)
                    )
                if use_topk_bin_fusion:
                    workspace_shapes.append(
                        ((num_padded_tokens, decode_logits_len), torch.int16)
                    )
                if use_topk_tile_select:
                    workspace_shapes.extend(
                        (
                            (
                                (
                                    num_padded_tokens,
                                    tile_select_groups * tile_select_candidates,
                                ),
                                torch.float32,
                            ),
                            (
                                (
                                    num_padded_tokens,
                                    tile_select_groups * tile_select_candidates,
                                ),
                                torch.int32,
                            ),
                            (
                                (num_padded_tokens, tile_select_groups),
                                torch.float32,
                            ),
                            ((num_padded_tokens,), torch.int32),
                        )
                    )
                if _DECODE_LOGITS_WORKSPACE:
                    if decode_topk_backend == "legacy":
                        workspace_shapes.append(
                            (
                                (num_padded_tokens, decode_topk_logits_len),
                                torch.float32,
                            )
                        )
                    else:
                        workspace_shapes.extend(
                            (
                                (
                                    (num_padded_tokens, decode_logits_len),
                                    torch.float32,
                                ),
                                ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                            )
                        )
                if workspace_shapes:
                    workspace_buffers = workspace_manager.get_simultaneous(
                        *workspace_shapes
                    )
                    buffer_idx = 0
                    if _DECODE_PREDEQUANT_Q:
                        q_bf16_decode = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                        q_decode_flat = padded_q_fp8_decode_tokens.reshape(
                            num_padded_tokens,
                            padded_q_fp8_decode_tokens.shape[2],
                            padded_q_fp8_decode_tokens.shape[3],
                        )
                        fp8_mqa_dequant_q_cuda(q_decode_flat, q_bf16_decode)
                    if use_topk_hist_fusion:
                        topk_histogram = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                        topk_histogram.zero_()
                    if use_topk_bin_fusion:
                        topk_bins = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                    if use_topk_tile_select:
                        topk_tile_candidate_logits = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                        topk_tile_candidate_indices = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                        topk_tile_candidate_cutoffs = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                        topk_tile_fallback_flags = workspace_buffers[buffer_idx]
                        buffer_idx += 1
                    if _DECODE_LOGITS_WORKSPACE:
                        if decode_topk_backend == "legacy":
                            logits_out = workspace_buffers[buffer_idx]
                        else:
                            logits_out = workspace_buffers[buffer_idx]
                            topk_workspace = workspace_buffers[buffer_idx + 1]
            if _PREFILL_SHAPE_BUCKET_TRACE:
                _shape_bucket_trace_sync(hidden_states.device)
                trace_decode_mqa_start = time.perf_counter()
            logits = fp8_paged_mqa_logits_triton(
                padded_q_fp8_decode_tokens,
                kv_cache,
                weights[:num_padded_tokens],
                seq_lens,
                decode_block_table,
                max_model_len=decode_logits_len,
                logits_out=logits_out,
                q_bf16=q_bf16_decode,
                topk_histogram=topk_histogram,
                topk_bins=topk_bins,
                topk_tile_candidate_logits=topk_tile_candidate_logits,
                topk_tile_candidate_indices=topk_tile_candidate_indices,
                topk_tile_candidate_cutoffs=topk_tile_candidate_cutoffs,
                store_logits=not use_topk_tile_select_no_store,
            )
            if (
                logits_out is not None
                and decode_topk_logits_len > decode_logits_len
            ):
                logits = logits_out[:num_padded_tokens, :decode_topk_logits_len]
            if _PREFILL_SHAPE_BUCKET_TRACE:
                logger.info(
                    "PREFILL_SHAPE_BUCKET_DECODE_TIMING "
                    "op=paged_mqa_logits_triton logits=%s max_seq=%s "
                    "logits_len=%s topk_logits_len=%s ms=%.3f",
                    tuple(logits.shape),
                    attn_metadata_narrowed.max_seq_len,
                    decode_logits_len,
                    decode_topk_logits_len,
                    _shape_bucket_trace_ms(trace_decode_mqa_start,
                                           hidden_states.device),
                )
        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        if _PREFILL_SHAPE_BUCKET_TRACE:
            _shape_bucket_trace_sync(hidden_states.device)
            trace_decode_topk_start = time.perf_counter()
        if current_platform.is_cuda() and decode_topk_backend != "legacy":
            if topk_workspace is None:
                workspace_manager = current_workspace_manager()
                (topk_workspace,) = workspace_manager.get_simultaneous(
                    ((RADIX_TOPK_WORKSPACE_SIZE,), torch.uint8),
                )
            torch.ops._C.persistent_topk(
                logits,
                seq_lens,
                topk_indices,
                topk_workspace,
                topk_tokens,
                decode_logits_len,
            )
        elif current_platform.is_cuda():
            if (
                use_topk_tile_select
                and topk_tile_candidate_logits is not None
                and topk_tile_candidate_indices is not None
                and topk_tile_candidate_cutoffs is not None
                and topk_tile_fallback_flags is not None
            ):
                torch.ops._C.top_k_per_row_decode_from_candidates(
                    logits,
                    next_n,
                    seq_lens,
                    topk_tile_candidate_logits,
                    topk_tile_candidate_indices,
                    topk_tile_candidate_cutoffs,
                    topk_tile_fallback_flags,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
                if (
                    use_topk_tile_select_no_store
                    and _DECODE_TOPK_TILE_SELECT_NO_STORE_CHECK
                    and int(topk_tile_fallback_flags.sum().item()) != 0
                ):
                    raise RuntimeError(
                        "Decode TopK tile-select no-store fallback was "
                        "required. Full logits were not stored, so this "
                        "candidate must be rejected before service use."
                    )
            elif use_topk_hist_fusion and topk_histogram is not None:
                torch.ops._C.top_k_per_row_decode_from_hist(
                    logits,
                    next_n,
                    seq_lens,
                    topk_histogram,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            elif use_topk_bin_fusion and topk_bins is not None:
                torch.ops._C.top_k_per_row_decode_from_bins(
                    logits,
                    next_n,
                    seq_lens,
                    topk_bins,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
        else:
            if current_platform.is_xpu():
                xpu_ops.top_k_per_row_decode(  # type: ignore[attr-defined]
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
            else:
                torch.ops._C.top_k_per_row_decode(
                    logits,
                    next_n,
                    seq_lens,
                    topk_indices,
                    num_rows,
                    logits.stride(0),
                    logits.stride(1),
                    topk_tokens,
                )
        if _PREFILL_SHAPE_BUCKET_TRACE:
            logger.info(
                "PREFILL_SHAPE_BUCKET_DECODE_TIMING op=topk "
                "backend=%s logits=%s max_seq=%s topk_shape_len=%s "
                "topk=%s ms=%.3f",
                decode_topk_backend,
                tuple(logits.shape),
                attn_metadata_narrowed.max_seq_len,
                decode_logits_len
                if decode_topk_backend != "legacy"
                else decode_topk_logits_len,
                topk_tokens,
                _shape_bucket_trace_ms(trace_decode_topk_start,
                                       hidden_states.device),
            )
        if decode_metadata.requires_padding:
            # if padded, we need to unpack
            # the topk indices removing padded tokens
            topk_indices = unpack_seq_triton(
                topk_indices.reshape(batch_size, -1, topk_indices.shape[-1]),
                decode_lens,
            )
            topk_indices_buffer[: topk_indices.shape[0], : topk_indices.shape[-1]] = (
                topk_indices
            )

    return topk_indices_buffer


def sparse_attn_indexer_fake(
    hidden_states: torch.Tensor,
    k_cache_prefix: LayerNameType,
    kv_cache: torch.Tensor,
    q_fp8: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    quant_block_size: int,
    scale_fmt: str | None,
    topk_tokens: int,
    head_dim: int,
    max_model_len: int,
    total_seq_lens: int,
    topk_indices_buffer: torch.Tensor | None,
) -> torch.Tensor:
    return topk_indices_buffer


direct_register_custom_op(
    op_name="sparse_attn_indexer",
    op_func=sparse_attn_indexer,
    mutates_args=["topk_indices_buffer"],
    fake_impl=sparse_attn_indexer_fake,
    dispatch_key=current_platform.dispatch_key,
)


@CustomOp.register("sparse_attn_indexer")
class SparseAttnIndexer(CustomOp):
    """Sparse Attention Indexer Custom Op Layer. This layer is extracted as a
    separate custom op since it involves heavy custom kernels like `mqa_logits`,
    `paged_mqa_logits` and `top_k_per_row`, etc. Those kernels maybe requires
    specific memory layout or implementation for different hardware backends to
    achieve optimal performance.

    For now, the default native path will use CUDA backend path. Other platform
    may requires add the corresponding Custom Op name `sparse_attn_indexer` to
    `custom_ops` in `CompilationConfig` to enable the platform specific path.
    """

    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super().__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer
        if current_platform.is_cuda() and not is_deep_gemm_supported():
            logger.warning_once(
                "DeepGEMM not supported on this platform; "
                "using Triton fallback for sparse attention indexer."
            )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if current_platform.is_cuda() or current_platform.is_xpu():
            return self.forward_cuda(hidden_states, q_fp8, k, weights)
        elif current_platform.is_rocm():
            return self.forward_hip(hidden_states, q_fp8, k, weights)
        else:
            raise NotImplementedError(
                "SparseAttnIndexer native forward is only implemented for "
                "CUDA, ROCm and XPU platforms."
            )

    def forward_cuda(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return torch.ops.vllm.sparse_attn_indexer(
            hidden_states,
            _encode_layer_name(self.k_cache.prefix),
            self.k_cache.kv_cache,
            q_fp8,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_hip(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if rocm_aiter_ops.is_enabled():
            return torch.ops.vllm.rocm_aiter_sparse_attn_indexer(
                hidden_states,
                _encode_layer_name(self.k_cache.prefix),
                self.k_cache.kv_cache,
                q_fp8,
                k,
                weights,
                self.quant_block_size,
                self.scale_fmt,
                self.topk_tokens,
                self.head_dim,
                self.max_model_len,
                self.max_total_seq_len,
                self.topk_indices_buffer,
            )
        else:
            raise RuntimeError(
                "Sparse attention indexer ROCm custom op requires ROCm "
                "Aiter ops to be enabled."
            )
