# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for GPUs without FlashMLA Sparse (SM90+)
or FlashInfer MLA Sparse (SM100+), e.g. SM80 (A100) and SM121 (GB10)."""

import os
import time
from dataclasses import replace
from typing import ClassVar

import torch

from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.utils.platform_utils import num_compute_units
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionLayer,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.ops.mqa_logits_triton import (
    warmup_fp8_mqa_logits_triton,
    warmup_fp8_paged_mqa_logits_triton,
)
from vllm.v1.attention.ops.triton_oscar_mla_decode import (
    oscar_mla_sparse_prefill,
    prepare_grouped_h4_score_workspace,
)
from vllm.v1.attention.ops.triton_oscar_mla_materialize import (
    OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
    OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
    allocate_oscar_mtp_direct_attention_cache,
    allocate_oscar_mtp_row576_temporal_cache,
    allocate_oscar_mtp_temporal_cache,
    allocate_oscar_mtp_temporal_cache_with_direct_storage,
    allocate_oscar_mtp_temporal_cache_with_split_direct_storage,
    can_use_oscar_bf16_materialized_read,
    commit_oscar_mla_direct_attention_misses,
    commit_oscar_mla_dual_source_attention_misses,
    materialize_oscar_mla_bf16_rows,
    materialize_oscar_mla_bf16_rows_direct_attention,
    materialize_oscar_mla_bf16_rows_temporal,
    merge_oscar_chunked_attention_states,
    prepare_oscar_bf16_materialization_workspace,
    prepare_oscar_mtp_temporal_workspace,
    reset_oscar_mtp_temporal_cache,
    restore_oscar_mla_hp_rows,
    restore_oscar_topk_after_chunk,
    save_and_remap_oscar_topk_for_chunk,
    seed_oscar_mtp_temporal_cache_recent,
    seed_oscar_mtp_temporal_cache_rows,
)
from vllm.v1.attention.ops.triton_oscar_mla_store import (
    allocate_oscar_demotion_ksplit_workspace,
    oscar_mla_demote_recent,
    oscar_mla_rotate_quantize_store,
    oscar_mla_rotate_quantize_store_decode,
    oscar_mla_store_bf16,
    oscar_mla_store_rope,
)
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    _BLOCK_DV,
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_sparse_mla_attention,
    triton_sparse_mla_attention_dual_source,
)
from vllm.v1.worker.oscar_mla_cache import OscarMLACacheTensors

logger = init_logger(__name__)

# DeepSeek-V3.2 / GLM-5.1 indexer shape, the only model family this backend
# serves. Used only for autotune priming — if a future model differs, the
# kernel simply re-tunes on first real use (same as pre-warmup behavior).
_INDEXER_NUM_HEADS = 64
_INDEXER_HEAD_DIM = 128


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


_FUSED_REQ_TO_GLOBAL = _env_flag("VLLM_SPARSE_MLA_FUSED_REQ_TO_GLOBAL")
_FUSED_REQ_TO_GLOBAL_MAX_TOKENS = _env_int(
    "VLLM_SPARSE_MLA_FUSED_REQ_TO_GLOBAL_MAX_TOKENS",
    1,
)
_REQ_TO_GLOBAL_BLOCK_N = _env_int("VLLM_SPARSE_MLA_REQ_TO_GLOBAL_BLOCK_N", 128)
_ASSUME_VALID_DYNAMIC = _env_flag("VLLM_SPARSE_MLA_ASSUME_VALID_DYNAMIC")
_ASSUME_VALID_SPLIT = _env_flag("VLLM_SPARSE_MLA_ASSUME_VALID_SPLIT")
_ASSUME_VALID_NOMASK = _env_flag("VLLM_SPARSE_MLA_ASSUME_VALID_NOMASK")
_ASSUME_VALID_AFTER_TOPK_NOMASK = _env_flag(
    "VLLM_SPARSE_MLA_ASSUME_VALID_AFTER_TOPK_NOMASK"
)
_SPARSE_MLA_WARMUP_NUM_TOKENS = os.getenv("VLLM_SPARSE_MLA_WARMUP_NUM_TOKENS")
_FORCE_PREFIX_MASK_DECODE = _env_flag("VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE")
_FORCE_PREFIX_MASK_DECODE_MAX_TOKENS = _env_int(
    "VLLM_SPARSE_MLA_FORCE_PREFIX_MASK_DECODE_MAX_TOKENS",
    4,
)
_PREFILL_SHAPE_BUCKET_TRACE = _env_flag("VLLM_PREFILL_SHAPE_BUCKET_TRACE")
_OSCAR_MTP_TEMPORAL_CACHE_ENABLED = _env_flag("VLLM_OSCAR_MTP_TEMPORAL_CACHE")
_OSCAR_MTP_PREFILL_SEED_ENABLED = _env_flag("VLLM_OSCAR_MTP_PREFILL_SEED")
_OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
)
_OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION"
)
_OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED = _env_flag("VLLM_OSCAR_MTP_TEMPORAL_TWO_WAY")
_OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_PREQUANT_DEMOTION_CACHE"
)
_OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY"
)
_OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY"
)
_OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY"
)
_OSCAR_MTP_DIRECT_RESET_EACH_STEP_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_RESET_EACH_STEP"
)
_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_ENABLED = _env_flag(
    "VLLM_OSCAR_MTP_DIRECT_COMPARE_REFERENCE"
)
_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS = _env_int(
    "VLLM_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS",
    1,
)
logger = init_logger(__name__)


def _log_oscar_mtp_direct_reference_diff(
    *,
    layer_name: str,
    compare_step: int,
    direct_output: torch.Tensor,
    direct_lse: torch.Tensor,
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
) -> None:
    """Log one eager-only direct/reference attention comparison."""
    direct_fp32 = direct_output.float()
    reference_fp32 = reference_output.float()
    output_diff = (direct_fp32 - reference_fp32).abs()
    lse_diff = (direct_lse.float() - reference_lse.float()).abs()
    dot = torch.sum(direct_fp32 * reference_fp32)
    norm_product = torch.linalg.vector_norm(direct_fp32) * torch.linalg.vector_norm(
        reference_fp32
    )
    cosine = dot / torch.clamp_min(norm_product, 1e-30)
    summary = torch.stack(
        (
            output_diff.max(),
            output_diff.mean(),
            torch.count_nonzero(direct_output != reference_output).float(),
            lse_diff.max(),
            lse_diff.mean(),
            cosine,
        )
    ).cpu()
    (
        output_max_abs,
        output_mean_abs,
        output_neq,
        lse_max_abs,
        lse_mean_abs,
        output_cosine,
    ) = summary.tolist()
    logger.warning(
        "OSCAR MTP direct/reference layer=%s step=%d output_max_abs=%.9g "
        "output_mean_abs=%.9g output_neq=%d lse_max_abs=%.9g "
        "lse_mean_abs=%.9g output_cosine=%.12g",
        layer_name,
        compare_step,
        output_max_abs,
        output_mean_abs,
        int(output_neq),
        lse_max_abs,
        lse_mean_abs,
        output_cosine,
    )


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    """Metadata builder advertising cudagraph support for the CUDA/Triton
    sparse MLA path. The XPU base keeps `AttentionCGSupport.NEVER` because
    its kernel has not been validated under cudagraph capture; this subclass
    is the only place the capability is claimed."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> XPUMLASparseMetadata:
        metadata = super().build_for_drafting(common_attn_metadata, draft_index)
        oscar = metadata.oscar_mla
        if draft_index <= 0 or oscar is None:
            return metadata

        metadata.oscar_mla = replace(
            oscar,
            demotion_hp_rows=oscar.demotion_hp_rows[:0],
            demotion_positions=oscar.demotion_positions[:0],
            demotion_page_ids=oscar.demotion_page_ids[:0],
            demotion_page_offsets=oscar.demotion_page_offsets[:0],
        )
        metadata.oscar_mla_draft_step = True
        return metadata


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Overrides XPU sparse impl to use the split-KV kernel, which is
    3–7× faster for single-query decode on SM80 (A100/A30) and SM120 (GB10).
    """

    can_return_lse_for_decode: ClassVar[bool] = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if (
            _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED
            and not _OSCAR_MTP_TEMPORAL_CACHE_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION requires "
                "VLLM_OSCAR_MTP_TEMPORAL_CACHE"
            )
        if (
            _OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED
            and not _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_TEMPORAL_TWO_WAY requires "
                "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION"
            )
        if _OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED and not (
            _OSCAR_MTP_TEMPORAL_CACHE_ENABLED
            and _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED
            and _OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_PREQUANT_DEMOTION_CACHE requires temporal "
                "cache, dual-source attention, and temporal two-way"
            )
        if (
            _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED
            and _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION is incompatible with "
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
            )
        if _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED and _OSCAR_MTP_PREFILL_SEED_ENABLED:
            raise ValueError(
                "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION is incompatible with "
                "VLLM_OSCAR_MTP_PREFILL_SEED"
            )
        if _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED and (
            _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED
            or _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED
            or _OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DUAL_SOURCE_ATTENTION is incompatible with "
                "allocation-only diagnostics"
            )
        if (
            _OSCAR_MTP_DIRECT_RESET_EACH_STEP_ENABLED
            and not _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_RESET_EACH_STEP requires "
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
            )
        if (
            _OSCAR_MTP_DIRECT_COMPARE_REFERENCE_ENABLED
            and not _OSCAR_MTP_DIRECT_RESET_EACH_STEP_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_COMPARE_REFERENCE requires "
                "VLLM_OSCAR_MTP_DIRECT_RESET_EACH_STEP"
            )
        if (
            _OSCAR_MTP_DIRECT_COMPARE_REFERENCE_ENABLED
            and _OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS <= 0
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS must be positive"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED
            and _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY is incompatible "
                "with VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED
            and not _OSCAR_MTP_TEMPORAL_CACHE_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY requires "
                "VLLM_OSCAR_MTP_TEMPORAL_CACHE"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED
            and _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY is "
                "incompatible with VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED
            and _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY is "
                "incompatible with VLLM_OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED
            and not _OSCAR_MTP_TEMPORAL_CACHE_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY requires "
                "VLLM_OSCAR_MTP_TEMPORAL_CACHE"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED
            and _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY is "
                "incompatible with VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION"
            )
        if _OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED and (
            _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED
            or _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY is "
                "incompatible with other allocation-only diagnostics"
            )
        if (
            _OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED
            and not _OSCAR_MTP_TEMPORAL_CACHE_ENABLED
        ):
            raise ValueError(
                "VLLM_OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY requires "
                "VLLM_OSCAR_MTP_TEMPORAL_CACHE"
            )
        if _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED:
            if not _OSCAR_MTP_TEMPORAL_CACHE_ENABLED:
                raise ValueError(
                    "VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION requires "
                    "VLLM_OSCAR_MTP_TEMPORAL_CACHE"
                )
            if _OSCAR_MTP_PREFILL_SEED_ENABLED:
                raise ValueError(
                    "VLLM_OSCAR_MTP_DIRECT_CACHE_ATTENTION is incompatible with "
                    "VLLM_OSCAR_MTP_PREFILL_SEED"
                )
        self.oscar_write_calls = 0
        self.oscar_demotion_calls = 0
        self.oscar_read_calls = 0
        self.oscar_restore_calls = 0
        self._oscar_grouped_h4_score_workspace: torch.Tensor | None = None
        self._oscar_bf16_materialization_workspace: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None
        self._oscar_demotion_ksplit_workspace: torch.Tensor | None = None
        self._oscar_mtp_temporal_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._oscar_mtp_direct_compare_count = 0
        self._oscar_mtp_temporal_workspace: (
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
            | None
        ) = None
        self._oscar_capability_major = 0
        if (
            self.kv_cache_dtype == "oscar_mla_int2"
            and self.topk_indices_buffer is not None
            and self.num_heads == 8
            and self.topk_indices_buffer.shape[-1] == 2048
        ):
            self._oscar_grouped_h4_score_workspace = prepare_grouped_h4_score_workspace(
                self.topk_indices_buffer
            )
            self._oscar_capability_major = torch.cuda.get_device_capability(
                self.topk_indices_buffer.device
            )[0]
            if self._oscar_capability_major == 8 and self.kv_lora_rank == 512:
                self._oscar_demotion_ksplit_workspace = (
                    allocate_oscar_demotion_ksplit_workspace(self.topk_indices_buffer)
                )
                self._oscar_bf16_materialization_workspace = (
                    prepare_oscar_bf16_materialization_workspace(
                        self.topk_indices_buffer
                    )
                )
                if _OSCAR_MTP_TEMPORAL_CACHE_ENABLED:
                    if _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED:
                        self._oscar_mtp_temporal_cache = (
                            allocate_oscar_mtp_direct_attention_cache(
                                self.topk_indices_buffer
                            )
                        )
                    elif _OSCAR_MTP_DIRECT_CACHE_ROW576_ALLOCATION_ONLY_ENABLED:
                        row576_cache = allocate_oscar_mtp_row576_temporal_cache(
                            self.topk_indices_buffer
                        )
                        self._oscar_mtp_temporal_cache = row576_cache
                    elif _OSCAR_MTP_DIRECT_CACHE_SPLIT_ALLOCATION_ONLY_ENABLED:
                        self._oscar_mtp_temporal_cache = (
                            allocate_oscar_mtp_temporal_cache_with_split_direct_storage(
                                self.topk_indices_buffer
                            )
                        )
                    elif _OSCAR_MTP_DIRECT_CACHE_ALLOCATION_ONLY_ENABLED:
                        self._oscar_mtp_temporal_cache = (
                            allocate_oscar_mtp_temporal_cache_with_direct_storage(
                                self.topk_indices_buffer
                            )
                        )
                    else:
                        self._oscar_mtp_temporal_cache = (
                            allocate_oscar_mtp_temporal_cache(self.topk_indices_buffer)
                        )
                    self._oscar_mtp_temporal_workspace = (
                        prepare_oscar_mtp_temporal_workspace(self.topk_indices_buffer)
                    )
        # Cached device SM count; passed into the kernel dispatch each forward
        # so the hot path doesn't re-query `q.device.index` → dict lookup.
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        self._warmup_autotune()

    @staticmethod
    def _oscar_query_positions(
        attn_metadata: XPUMLASparseMetadata,
        num_tokens: int,
    ) -> torch.Tensor:
        token_rows = torch.arange(
            num_tokens,
            dtype=torch.int32,
            device=attn_metadata.seq_lens.device,
        )
        requests = attn_metadata.req_id_per_token[:num_tokens].long()
        query_ends = attn_metadata.query_start_loc[requests + 1]
        return attn_metadata.seq_lens[requests] - (query_ends - token_rows)

    def do_oscar_kv_cache_update(
        self,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: OscarMLACacheTensors,
        attn_metadata: XPUMLASparseMetadata,
        rotation: torch.Tensor,
        *,
        clip_ratio: float,
    ) -> None:
        """Apply demotion before writing this batch's final three-pool partition."""
        oscar = attn_metadata.oscar_mla
        if oscar is None:
            raise RuntimeError("oscar_mla_int2 attention metadata is missing")
        if not isinstance(kv_cache, OscarMLACacheTensors):
            raise TypeError("oscar_mla_int2 requires OSCAR MLA cache views")
        num_tokens = attn_metadata.num_actual_tokens
        latent = kv_c_normed[:num_tokens]
        rope_values = k_pe[:num_tokens]
        is_decode = (
            attn_metadata.max_query_len == 1 and num_tokens == attn_metadata.num_reqs
        )
        is_incremental_mtp_target = (
            not is_decode
            and attn_metadata.num_reqs == 1
            and attn_metadata.base_seq_len > 0
            and attn_metadata.full_topk_start <= 0
            and 1 < num_tokens <= 6
            and attn_metadata.max_query_len == num_tokens
        )
        use_mtp_prefill_seed = (
            _OSCAR_MTP_PREFILL_SEED_ENABLED
            and self._oscar_mtp_temporal_cache is not None
            and not is_decode
            and not is_incremental_mtp_target
            and attn_metadata.num_reqs == 1
        )
        if (
            self._oscar_mtp_temporal_cache is not None
            and not is_decode
            and not is_incremental_mtp_target
            and (not use_mtp_prefill_seed or attn_metadata.base_seq_len == 0)
        ):
            reset_oscar_mtp_temporal_cache(self._oscar_mtp_temporal_cache)
        if is_decode:
            if attn_metadata.oscar_mla_draft_step:
                final_seq_lens = attn_metadata.seq_lens[:num_tokens]
                query_positions = final_seq_lens - 1
            else:
                query_positions = oscar.decode_positions[:num_tokens]
                final_seq_lens = oscar.final_seq_lens[:num_tokens]
            token_hp_rows = oscar.hp_rows[:num_tokens]
        else:
            request_indices = attn_metadata.req_id_per_token[:num_tokens].long()
            query_positions = self._oscar_query_positions(attn_metadata, num_tokens)
            final_seq_lens = attn_metadata.seq_lens[request_indices]
            token_hp_rows = oscar.hp_rows[request_indices]

        oscar_mla_store_rope(
            rope_values,
            kv_cache.rope,
            attn_metadata.slot_mapping[:num_tokens],
        )

        # Prefix caching reuses standard vLLM physical blocks after the
        # originating request has released its private BF16 row. Persist every
        # new latent in canonical INT2 form so a later hit can restore its
        # fixed prefix/recent windows. Keep this extra write fully disabled for
        # the frozen C135 no-prefix runtime.
        if attn_metadata.enable_prefix_caching:
            slots = attn_metadata.slot_mapping[:num_tokens]
            if (
                is_decode
                and num_tokens == 1
                and self._oscar_demotion_ksplit_workspace is not None
            ):
                oscar_mla_rotate_quantize_store_decode(
                    latent,
                    rotation,
                    kv_cache.history_data,
                    kv_cache.history_scale,
                    kv_cache.history_zero,
                    slots,
                    clip_ratio=clip_ratio,
                    partial_workspace=self._oscar_demotion_ksplit_workspace,
                )
            else:
                valid_slots = slots >= 0
                page_ids = torch.where(
                    valid_slots,
                    torch.div(
                        slots,
                        kv_cache.history_data.shape[1],
                        rounding_mode="floor",
                    ),
                    -1,
                )
                page_offsets = torch.where(
                    valid_slots,
                    slots % kv_cache.history_data.shape[1],
                    0,
                )
                oscar_mla_rotate_quantize_store(
                    latent,
                    rotation,
                    kv_cache.history_data,
                    kv_cache.history_scale,
                    kv_cache.history_zero,
                    page_ids,
                    page_offsets,
                    clip_ratio=clip_ratio,
                )
            if self.oscar_write_calls == 0:
                logger.info_once(
                    "OSCAR MLA prefix-cache canonical INT2 writes active"
                )

        if oscar.demotion_positions.numel():
            if self.oscar_demotion_calls == 0:
                logger.info_once(
                    "OSCAR MLA recent-to-INT2 demotion active; first batch=%d tokens",
                    oscar.demotion_positions.numel(),
                )
            oscar_mla_demote_recent(
                kv_cache.recent,
                rotation,
                kv_cache.history_data,
                kv_cache.history_scale,
                kv_cache.history_zero,
                oscar.demotion_positions,
                oscar.demotion_hp_rows,
                oscar.demotion_page_ids,
                oscar.demotion_page_offsets,
                prefix_tokens=kv_cache.prefix.shape[1],
                clip_ratio=clip_ratio,
                partial_workspace=(
                    self._oscar_demotion_ksplit_workspace
                    if (
                        is_decode
                        and num_tokens == 1
                        and oscar.demotion_positions.numel() == 1
                    )
                    else None
                ),
                prequant_temporal_cache=(
                    self._oscar_mtp_temporal_cache
                    if (
                        _OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED
                        and is_incremental_mtp_target
                    )
                    else None
                ),
                temporal_two_way=(
                    _OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED
                    and is_incremental_mtp_target
                ),
            )
            if use_mtp_prefill_seed:
                assert self._oscar_mtp_temporal_cache is not None
                seed_oscar_mtp_temporal_cache_recent(
                    kv_cache.recent,
                    oscar.demotion_positions,
                    oscar.demotion_hp_rows,
                    oscar.demotion_page_ids,
                    self._oscar_mtp_temporal_cache,
                    prefix_tokens=kv_cache.prefix.shape[1],
                )
            self.oscar_demotion_calls += 1

        if not is_decode and not is_incremental_mtp_target:
            history_end = torch.maximum(
                torch.full_like(final_seq_lens, kv_cache.prefix.shape[1]),
                final_seq_lens - kv_cache.recent_tokens,
            )
            current_history = (query_positions >= kv_cache.prefix.shape[1]) & (
                query_positions < history_end
            )
            history_indices = query_positions - kv_cache.prefix.shape[1]
            logical_pages = torch.div(
                history_indices,
                kv_cache.history_data.shape[1],
                rounding_mode="floor",
            )
            valid_history = (
                current_history
                & (logical_pages >= 0)
                & (logical_pages < oscar.history_page_table.shape[1])
            )
            safe_logical_pages = torch.clamp(
                logical_pages,
                min=0,
                max=oscar.history_page_table.shape[1] - 1,
            )
            page_offsets = history_indices % kv_cache.history_data.shape[1]
            page_ids = oscar.history_page_table[
                request_indices,
                safe_logical_pages.long(),
            ]
            page_ids = torch.where(valid_history, page_ids, -1)
            oscar_mla_rotate_quantize_store(
                latent,
                rotation,
                kv_cache.history_data,
                kv_cache.history_scale,
                kv_cache.history_zero,
                page_ids,
                page_offsets,
                clip_ratio=clip_ratio,
            )
            if use_mtp_prefill_seed:
                assert self._oscar_mtp_temporal_cache is not None
                seed_oscar_mtp_temporal_cache_rows(
                    latent,
                    query_positions,
                    valid_history,
                    self._oscar_mtp_temporal_cache,
                )

        store_recent_tokens = (
            kv_cache.recent.shape[1]
            if attn_metadata.oscar_mla_draft_step
            else kv_cache.recent_tokens
        )
        oscar_mla_store_bf16(
            latent,
            kv_cache.prefix,
            kv_cache.recent,
            query_positions,
            final_seq_lens,
            token_hp_rows,
            store_recent_tokens,
        )
        if self.oscar_write_calls == 0:
            logger.info_once(
                "OSCAR MLA three-pool write active; no full BF16 latent history"
            )
        self.oscar_write_calls += 1

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
        if _FUSED_REQ_TO_GLOBAL:
            req_id = torch.zeros(1, dtype=torch.int32, device=device)
            block_table = torch.zeros(1, 1, dtype=torch.int32, device=device)
            triton_sparse_mla_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=1,
                sm_count=self._sm_count,
                assume_valid_indices=True,
                req_id=req_id,
                block_table=block_table,
                block_size=64,
            )
        if _FORCE_PREFIX_MASK_DECODE:
            triton_sparse_mla_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=1,
                sm_count=self._sm_count,
                valid_index_base_seq_len=topk,
            )
        extra_warmup_shapes = _SPARSE_MLA_WARMUP_NUM_TOKENS
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
                if _ASSUME_VALID_NOMASK:
                    triton_sparse_mla_attention(
                        q,
                        kv,
                        indices,
                        sm_scale=self.softmax_scale,
                        num_kv_splits=1,
                        sm_count=self._sm_count,
                        assume_valid_indices=True,
                    )
                    if _ASSUME_VALID_AFTER_TOPK_NOMASK:
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
            torch.accelerator.empty_cache()
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
        trace_enabled = _PREFILL_SHAPE_BUCKET_TRACE and num_tokens > 1
        if trace_enabled:
            logger.info(
                "PREFILL_SHAPE_BUCKET_TRITON_MLA q=%s topk=%s "
                "num_tokens=%s meta_tokens=%s max_query=%s max_seq=%s "
                "num_prefills=%s num_decodes=%s full_topk_start=%s base_seq_len=%s",
                tuple(q_nope.shape),
                tuple(topk_indices.shape),
                num_tokens,
                getattr(attn_metadata, "num_actual_tokens", None),
                getattr(attn_metadata, "max_query_len", None),
                getattr(attn_metadata, "max_seq_len", None),
                getattr(attn_metadata, "num_prefills", None),
                getattr(attn_metadata, "num_decodes", None),
                getattr(attn_metadata, "full_topk_start", None),
                getattr(attn_metadata, "base_seq_len", None),
            )

        def _trace_sync() -> None:
            if trace_enabled and q_nope.device.type == "cuda":
                torch.accelerator.synchronize(device=q_nope.device)

        def _sparse_mla_call(label: str, *args, **kwargs):
            start = 0.0
            if trace_enabled:
                _trace_sync()
                start = time.perf_counter()
            result = triton_sparse_mla_attention(*args, **kwargs)
            if trace_enabled:
                _trace_sync()
                logger.info(
                    "PREFILL_SHAPE_BUCKET_TRITON_MLA_ATTENTION_TIMING "
                    "label=%s q=%s topk=%s num_tokens=%s max_query=%s "
                    "max_seq=%s num_reqs=%s return_lse=%s ms=%.3f",
                    label,
                    tuple(q_nope.shape),
                    tuple(topk_indices.shape),
                    num_tokens,
                    getattr(attn_metadata, "max_query_len", None),
                    getattr(attn_metadata, "max_seq_len", None),
                    getattr(attn_metadata, "num_reqs", None),
                    return_lse,
                    (time.perf_counter() - start) * 1000.0,
                )
            return result

        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        use_fused_req_to_global = (
            _FUSED_REQ_TO_GLOBAL
            and attn_metadata.full_topk_start <= 0
            and num_tokens <= _FUSED_REQ_TO_GLOBAL_MAX_TOKENS
        )
        if use_fused_req_to_global:
            topk_indices = topk_indices.view(num_tokens, 1, -1)
            sparse_kwargs = {
                "req_id": attn_metadata.req_id_per_token,
                "block_table": attn_metadata.block_table,
                "block_size": attn_metadata.block_size,
            }
        else:
            convert_start = 0.0
            if trace_enabled:
                _trace_sync()
                convert_start = time.perf_counter()
            topk_indices = triton_convert_req_index_to_global_index(
                attn_metadata.req_id_per_token,
                attn_metadata.block_table,
                topk_indices,
                BLOCK_SIZE=attn_metadata.block_size,
                NUM_TOPK_TOKENS=attn_metadata.topk_tokens,
                BLOCK_N=_REQ_TO_GLOBAL_BLOCK_N,
            ).view(num_tokens, 1, -1)
            if trace_enabled:
                _trace_sync()
                logger.info(
                    "PREFILL_SHAPE_BUCKET_TRITON_MLA_CONVERT_TIMING "
                    "q=%s topk=%s num_tokens=%s max_query=%s max_seq=%s "
                    "num_reqs=%s block_n=%s ms=%.3f",
                    tuple(q_nope.shape),
                    tuple(topk_indices.shape),
                    num_tokens,
                    getattr(attn_metadata, "max_query_len", None),
                    getattr(attn_metadata, "max_seq_len", None),
                    getattr(attn_metadata, "num_reqs", None),
                    _REQ_TO_GLOBAL_BLOCK_N,
                    (time.perf_counter() - convert_start) * 1000.0,
                )
            sparse_kwargs = {}
        out_heads = q_nope.shape[1] if return_lse else self.num_heads
        if _ASSUME_VALID_DYNAMIC and attn_metadata.num_reqs == 1:
            full_topk_start = attn_metadata.full_topk_start
            if full_topk_start <= 0:
                if (
                    _FORCE_PREFIX_MASK_DECODE
                    and num_tokens <= _FORCE_PREFIX_MASK_DECODE_MAX_TOKENS
                ):
                    result = _sparse_mla_call(
                        "dynamic_prefix_mask_decode",
                        q,
                        kv_c_and_k_pe_cache,
                        topk_indices,
                        sm_scale=self.softmax_scale,
                        sm_count=self._sm_count,
                        valid_index_base_seq_len=attn_metadata.base_seq_len,
                        return_lse=return_lse,
                        **sparse_kwargs,
                    )
                    if return_lse:
                        output, lse = result
                        return output[:, :out_heads, :], lse[:, :out_heads]
                    return result[:, :out_heads, :]
                result = _sparse_mla_call(
                    "dynamic_assume_valid",
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    assume_valid_indices=True,
                    return_lse=return_lse,
                    **sparse_kwargs,
                )
                if return_lse:
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                return result[:, :out_heads, :]
            if full_topk_start < num_tokens:
                result = _sparse_mla_call(
                    "dynamic_prefix_mask",
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    valid_index_base_seq_len=attn_metadata.base_seq_len,
                    return_lse=return_lse,
                    **sparse_kwargs,
                )
                if return_lse:
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                return result[:, :out_heads, :]
        if _ASSUME_VALID_SPLIT and attn_metadata.num_reqs == 1:
            full_topk_start = attn_metadata.full_topk_start
            if full_topk_start <= 0:
                result = _sparse_mla_call(
                    "split_assume_valid",
                    q,
                    kv_c_and_k_pe_cache,
                    topk_indices,
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    assume_valid_indices=True,
                    return_lse=return_lse,
                    **sparse_kwargs,
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
                    result = _sparse_mla_call(
                        "split_return_lse_prefix_mask",
                        q,
                        kv_c_and_k_pe_cache,
                        topk_indices,
                        sm_scale=self.softmax_scale,
                        sm_count=self._sm_count,
                        valid_index_base_seq_len=attn_metadata.base_seq_len,
                        return_lse=True,
                        **sparse_kwargs,
                    )
                    output, lse = result
                    return output[:, :out_heads, :], lse[:, :out_heads]
                _sparse_mla_call(
                    "split_prefix_mask",
                    self._slice_q(q, slice(None, full_topk_start)),
                    kv_c_and_k_pe_cache,
                    topk_indices[:full_topk_start],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    out=output[:full_topk_start],
                )
                _sparse_mla_call(
                    "split_assume_valid_tail",
                    self._slice_q(q, slice(full_topk_start, None)),
                    kv_c_and_k_pe_cache,
                    topk_indices[full_topk_start:],
                    sm_scale=self.softmax_scale,
                    sm_count=self._sm_count,
                    out=output[full_topk_start:],
                    assume_valid_indices=True,
                )
                return output[:, :out_heads, :]
        result = _sparse_mla_call(
            "default",
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
            return_lse=return_lse,
            **sparse_kwargs,
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
        if self.kv_cache_dtype == "oscar_mla_int2":
            if not isinstance(q, tuple):
                raise TypeError(
                    "oscar_mla_int2 requires separate latent and RoPE query"
                )
            if not isinstance(kv_c_and_k_pe_cache, OscarMLACacheTensors):
                raise TypeError("oscar_mla_int2 requires OSCAR MLA cache views")
            oscar = attn_metadata.oscar_mla
            if oscar is None:
                raise RuntimeError("oscar_mla_int2 attention metadata is missing")
            q_nope, q_pe = q
            num_actual_toks = q_nope.shape[0]
            assert self.topk_indices_buffer is not None
            if oscar.num_restore_rows:
                if self._oscar_bf16_materialization_workspace is None:
                    raise RuntimeError(
                        "OSCAR MLA cache-hit restore workspace is missing"
                    )
                history_rotated, _, restored, *_ = (
                    self._oscar_bf16_materialization_workspace
                )
                restore_oscar_mla_hp_rows(
                    positions=oscar.restore_positions,
                    hp_rows=oscar.restore_hp_rows,
                    page_ids=oscar.restore_page_ids,
                    page_offsets=oscar.restore_page_offsets,
                    num_rows=oscar.num_restore_rows,
                    history_data=kv_c_and_k_pe_cache.history_data,
                    history_scale=kv_c_and_k_pe_cache.history_scale,
                    history_zero=kv_c_and_k_pe_cache.history_zero,
                    prefix=kv_c_and_k_pe_cache.prefix,
                    recent=kv_c_and_k_pe_cache.recent,
                    inverse_rotation=layer._oscar_inverse_rotation_bf16,
                    history_rotated=history_rotated,
                    restored=restored,
                )
                if self.oscar_restore_calls == 0:
                    logger.info_once(
                        "OSCAR MLA prefix-cache BF16 window restore active; rows=%d",
                        oscar.num_restore_rows,
                    )
                self.oscar_restore_calls += 1
            is_decode = (
                attn_metadata.max_query_len == 1
                and num_actual_toks == attn_metadata.num_reqs
            )
            topk_width = attn_metadata.topk_tokens
            if not is_decode:
                topk_width = min(topk_width, attn_metadata.max_seq_len)
            use_selected_incremental_materialization = (
                not is_decode
                and attn_metadata.num_reqs == 1
                and attn_metadata.base_seq_len > 0
                and attn_metadata.full_topk_start <= 0
                and 1 < num_actual_toks <= 6
                and attn_metadata.max_query_len == num_actual_toks
                and num_actual_toks * topk_width <= OSCAR_BF16_MATERIALIZATION_MAX_ROWS
            )
            use_mtp_temporal_materialization = (
                use_selected_incremental_materialization
                and self._oscar_mtp_temporal_cache is not None
                and self._oscar_mtp_temporal_workspace is not None
                and attn_metadata.max_seq_len <= OSCAR_MTP_TEMPORAL_MAX_POSITIONS
            )
            use_mtp_direct_cache_attention = (
                use_mtp_temporal_materialization
                and _OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED
            )
            use_mtp_dual_source_attention = (
                use_mtp_temporal_materialization
                and _OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED
            )
            group_size = self.kv_lora_rank // kv_c_and_k_pe_cache.history_scale.shape[2]
            read_recent_tokens = (
                kv_c_and_k_pe_cache.recent.shape[1]
                if attn_metadata.oscar_mla_draft_step
                else kv_c_and_k_pe_cache.recent_tokens
            )
            use_bf16_materialized_read = (
                self._oscar_bf16_materialization_workspace is not None
                and can_use_oscar_bf16_materialized_read(
                    capability_major=self._oscar_capability_major,
                    num_requests=attn_metadata.num_reqs,
                    num_heads=self.num_heads,
                    latent_rank=self.kv_lora_rank,
                    rope_head_size=kv_c_and_k_pe_cache.rope.shape[2],
                    group_size=group_size,
                    prefix_tokens=kv_c_and_k_pe_cache.prefix.shape[1],
                    recent_tokens=kv_c_and_k_pe_cache.recent_tokens,
                    topk=topk_width,
                )
            )
            if use_bf16_materialized_read:
                assert self._oscar_bf16_materialization_workspace is not None
                selected = self.topk_indices_buffer[:num_actual_toks, :topk_width]
                (
                    history_rotated,
                    history_mask,
                    materialized_kv,
                    remapped,
                    partial_output,
                    accumulated_lse,
                    partial_lse,
                ) = self._oscar_bf16_materialization_workspace
                dual_miss_values: torch.Tensor | None = None
                use_selected_materialization = (
                    is_decode or use_selected_incremental_materialization
                )
                if (
                    use_selected_materialization
                    or attn_metadata.max_seq_len <= OSCAR_BF16_MATERIALIZATION_MAX_ROWS
                ):
                    positions = (
                        selected.reshape(-1) if use_selected_materialization else None
                    )
                    if use_selected_incremental_materialization:
                        materialized_rows = num_actual_toks * topk_width
                    else:
                        materialized_rows = (
                            topk_width if is_decode else attn_metadata.max_seq_len
                        )
                    if use_mtp_temporal_materialization:
                        assert self._oscar_mtp_temporal_cache is not None
                        assert self._oscar_mtp_temporal_workspace is not None
                        assert positions is not None
                        if use_mtp_direct_cache_attention:
                            if _OSCAR_MTP_DIRECT_RESET_EACH_STEP_ENABLED:
                                reset_oscar_mtp_temporal_cache(
                                    self._oscar_mtp_temporal_cache
                                )
                            kv, remapped_indices = (
                                materialize_oscar_mla_bf16_rows_direct_attention(
                                    positions=positions,
                                    num_rows=materialized_rows,
                                    num_requests=attn_metadata.num_reqs,
                                    prefix=kv_c_and_k_pe_cache.prefix,
                                    recent=kv_c_and_k_pe_cache.recent,
                                    rope=kv_c_and_k_pe_cache.rope,
                                    rope_block_table=attn_metadata.block_table,
                                    history_data=kv_c_and_k_pe_cache.history_data,
                                    history_scale=kv_c_and_k_pe_cache.history_scale,
                                    history_zero=kv_c_and_k_pe_cache.history_zero,
                                    history_page_table=oscar.history_page_table,
                                    hp_rows=oscar.hp_rows,
                                    seq_lens=attn_metadata.seq_lens,
                                    inverse_rotation=layer._oscar_inverse_rotation_bf16,
                                    history_rotated=history_rotated,
                                    remapped_indices=remapped,
                                    temporal_workspace=(
                                        self._oscar_mtp_temporal_workspace
                                    ),
                                    direct_cache=self._oscar_mtp_temporal_cache,
                                    recent_tokens=read_recent_tokens,
                                )
                            )
                        else:
                            kv, remapped_indices = (
                                materialize_oscar_mla_bf16_rows_temporal(
                                    positions=positions,
                                    num_rows=materialized_rows,
                                    num_requests=attn_metadata.num_reqs,
                                    prefix=kv_c_and_k_pe_cache.prefix,
                                    recent=kv_c_and_k_pe_cache.recent,
                                    rope=kv_c_and_k_pe_cache.rope,
                                    rope_block_table=attn_metadata.block_table,
                                    history_data=kv_c_and_k_pe_cache.history_data,
                                    history_scale=kv_c_and_k_pe_cache.history_scale,
                                    history_zero=kv_c_and_k_pe_cache.history_zero,
                                    history_page_table=oscar.history_page_table,
                                    hp_rows=oscar.hp_rows,
                                    seq_lens=attn_metadata.seq_lens,
                                    inverse_rotation=(
                                        layer._oscar_inverse_rotation_bf16
                                    ),
                                    history_rotated=history_rotated,
                                    history_mask=history_mask,
                                    output_kv=materialized_kv,
                                    remapped_indices=remapped,
                                    temporal_workspace=(
                                        self._oscar_mtp_temporal_workspace
                                    ),
                                    temporal_cache=self._oscar_mtp_temporal_cache,
                                    recent_tokens=read_recent_tokens,
                                    dual_source_attention=(
                                        use_mtp_dual_source_attention
                                    ),
                                    two_way=_OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED,
                                )
                            )
                            if use_mtp_dual_source_attention:
                                dual_miss_values = kv
                    else:
                        kv, remapped_indices = materialize_oscar_mla_bf16_rows(
                            positions=positions,
                            num_rows=materialized_rows,
                            num_requests=attn_metadata.num_reqs,
                            prefix=kv_c_and_k_pe_cache.prefix,
                            recent=kv_c_and_k_pe_cache.recent,
                            rope=kv_c_and_k_pe_cache.rope,
                            rope_block_table=attn_metadata.block_table,
                            history_data=kv_c_and_k_pe_cache.history_data,
                            history_scale=kv_c_and_k_pe_cache.history_scale,
                            history_zero=kv_c_and_k_pe_cache.history_zero,
                            history_page_table=oscar.history_page_table,
                            hp_rows=oscar.hp_rows,
                            seq_lens=attn_metadata.seq_lens,
                            inverse_rotation=layer._oscar_inverse_rotation_bf16,
                            history_rotated=history_rotated,
                            history_mask=history_mask,
                            output_kv=materialized_kv,
                            remapped_indices=remapped,
                            recent_tokens=read_recent_tokens,
                        )
                    indices = (
                        remapped_indices.view(num_actual_toks, 1, topk_width)
                        if use_selected_materialization
                        else selected.view(num_actual_toks, 1, topk_width)
                    )
                    if use_mtp_dual_source_attention:
                        assert positions is not None
                        assert self._oscar_mtp_temporal_cache is not None
                        assert dual_miss_values is not None
                        source_positions = positions.view(
                            num_actual_toks, 1, topk_width
                        )
                        output, lse = triton_sparse_mla_attention_dual_source(
                            q,
                            self._oscar_mtp_temporal_cache[0],
                            dual_miss_values,
                            kv_c_and_k_pe_cache.rope,
                            attn_metadata.block_table,
                            source_positions,
                            indices,
                            sm_scale=self.softmax_scale,
                            num_kv_splits=16,
                        )
                        assert self._oscar_mtp_temporal_workspace is not None
                        commit_oscar_mla_dual_source_attention_misses(
                            positions=positions,
                            num_rows=materialized_rows,
                            miss_values=dual_miss_values,
                            temporal_workspace=self._oscar_mtp_temporal_workspace,
                            temporal_cache=self._oscar_mtp_temporal_cache,
                            two_way=_OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED,
                        )
                    else:
                        output, lse = triton_sparse_mla_attention(
                            q,
                            kv,
                            indices,
                            sm_scale=self.softmax_scale,
                            num_kv_splits=16 if use_selected_materialization else 1,
                            sm_count=self._sm_count,
                            assume_valid_indices=False,
                            return_lse=True,
                        )
                    if use_mtp_direct_cache_attention:
                        assert positions is not None
                        assert self._oscar_mtp_temporal_workspace is not None
                        assert self._oscar_mtp_temporal_cache is not None
                        commit_oscar_mla_direct_attention_misses(
                            positions=positions,
                            num_rows=materialized_rows,
                            temporal_workspace=self._oscar_mtp_temporal_workspace,
                            direct_cache=self._oscar_mtp_temporal_cache,
                        )
                        if (
                            _OSCAR_MTP_DIRECT_COMPARE_REFERENCE_ENABLED
                            and self._oscar_mtp_direct_compare_count
                            < _OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS
                        ):
                            reference_kv, reference_remapped_indices = (
                                materialize_oscar_mla_bf16_rows(
                                    positions=positions,
                                    num_rows=materialized_rows,
                                    num_requests=attn_metadata.num_reqs,
                                    prefix=kv_c_and_k_pe_cache.prefix,
                                    recent=kv_c_and_k_pe_cache.recent,
                                    rope=kv_c_and_k_pe_cache.rope,
                                    rope_block_table=attn_metadata.block_table,
                                    history_data=kv_c_and_k_pe_cache.history_data,
                                    history_scale=kv_c_and_k_pe_cache.history_scale,
                                    history_zero=kv_c_and_k_pe_cache.history_zero,
                                    history_page_table=oscar.history_page_table,
                                    hp_rows=oscar.hp_rows,
                                    seq_lens=attn_metadata.seq_lens,
                                    inverse_rotation=(
                                        layer._oscar_inverse_rotation_bf16
                                    ),
                                    history_rotated=history_rotated,
                                    history_mask=history_mask,
                                    output_kv=materialized_kv,
                                    remapped_indices=remapped,
                                    recent_tokens=read_recent_tokens,
                                )
                            )
                            reference_indices = reference_remapped_indices.view(
                                num_actual_toks, 1, topk_width
                            )
                            reference_output, reference_lse = (
                                triton_sparse_mla_attention(
                                    q,
                                    reference_kv,
                                    reference_indices,
                                    sm_scale=self.softmax_scale,
                                    num_kv_splits=16,
                                    sm_count=self._sm_count,
                                    assume_valid_indices=False,
                                    return_lse=True,
                                )
                            )
                            _log_oscar_mtp_direct_reference_diff(
                                layer_name=getattr(layer, "layer_name", "unknown"),
                                compare_step=self._oscar_mtp_direct_compare_count,
                                direct_output=output,
                                direct_lse=lse,
                                reference_output=reference_output,
                                reference_lse=reference_lse,
                            )
                            self._oscar_mtp_direct_compare_count += 1
                else:
                    output = None
                    lse = accumulated_lse[:num_actual_toks]
                    for row_offset in range(
                        0,
                        attn_metadata.max_seq_len,
                        OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
                    ):
                        materialized_rows = min(
                            OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
                            attn_metadata.max_seq_len - row_offset,
                        )
                        kv, _ = materialize_oscar_mla_bf16_rows(
                            positions=None,
                            num_rows=materialized_rows,
                            row_offset=row_offset,
                            num_requests=attn_metadata.num_reqs,
                            prefix=kv_c_and_k_pe_cache.prefix,
                            recent=kv_c_and_k_pe_cache.recent,
                            rope=kv_c_and_k_pe_cache.rope,
                            rope_block_table=attn_metadata.block_table,
                            history_data=kv_c_and_k_pe_cache.history_data,
                            history_scale=kv_c_and_k_pe_cache.history_scale,
                            history_zero=kv_c_and_k_pe_cache.history_zero,
                            history_page_table=oscar.history_page_table,
                            hp_rows=oscar.hp_rows,
                            seq_lens=attn_metadata.seq_lens,
                            inverse_rotation=layer._oscar_inverse_rotation_bf16,
                            history_rotated=history_rotated,
                            history_mask=history_mask,
                            output_kv=materialized_kv,
                            remapped_indices=remapped,
                            recent_tokens=read_recent_tokens,
                        )
                        saved_indices = save_and_remap_oscar_topk_for_chunk(
                            selected,
                            history_rotated,
                            row_offset,
                            materialized_rows,
                        )
                        chunk_output_buffer: torch.Tensor | None = (
                            None if output is None else partial_output[:num_actual_toks]
                        )
                        chunk_lse_buffer = (
                            lse if output is None else partial_lse[:num_actual_toks]
                        )
                        chunk_output, chunk_lse = triton_sparse_mla_attention(
                            q,
                            kv,
                            selected.view(num_actual_toks, 1, topk_width),
                            sm_scale=self.softmax_scale,
                            num_kv_splits=1,
                            sm_count=self._sm_count,
                            out=chunk_output_buffer,
                            lse_out=chunk_lse_buffer,
                            assume_valid_indices=False,
                            return_lse=True,
                        )
                        restore_oscar_topk_after_chunk(selected, saved_indices)
                        if output is None:
                            output = chunk_output
                        else:
                            merge_oscar_chunked_attention_states(
                                output,
                                lse,
                                chunk_output,
                                chunk_lse,
                            )
                    assert output is not None
                if self.oscar_read_calls == 0:
                    logger.info_once("OSCAR MLA BF16-materialized sparse read active")
                self.oscar_read_calls += 1
                output = output.to(q_nope.dtype)
                return (
                    (output, lse)
                    if self.need_to_return_lse_for_decode
                    else (output, None)
                )
            query_positions = self._oscar_query_positions(
                attn_metadata,
                num_actual_toks,
            )
            output, lse = oscar_mla_sparse_prefill(
                q_nope,
                q_pe,
                self.topk_indices_buffer[:num_actual_toks, :topk_width],
                attn_metadata.req_id_per_token[:num_actual_toks],
                query_positions,
                kv_c_and_k_pe_cache.prefix,
                kv_c_and_k_pe_cache.recent,
                kv_c_and_k_pe_cache.rope,
                attn_metadata.block_table,
                kv_c_and_k_pe_cache.history_data,
                kv_c_and_k_pe_cache.history_scale,
                kv_c_and_k_pe_cache.history_zero,
                oscar.history_page_table,
                oscar.hp_rows,
                attn_metadata.seq_lens,
                layer._oscar_rotation,
                inverse_rotation=layer._oscar_inverse_rotation,
                attention_scale=self.softmax_scale,
                num_splits=16 if is_decode else 1,
                recent_tokens=read_recent_tokens,
                score_workspace=self._oscar_grouped_h4_score_workspace,
                group_decode_h4=is_decode,
            )
            if self.oscar_read_calls == 0:
                logger.info_once(
                    "OSCAR MLA DSA-selected mixed prefix/recent/INT2 read active"
                )
            self.oscar_read_calls += 1
            output = output.to(q_nope.dtype)
            return (
                (output, lse) if self.need_to_return_lse_for_decode else (output, None)
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError("FP8 kv is not supported with MLA Sparse yet")

        q_shape = q[0] if isinstance(q, tuple) else q
        num_actual_toks = q_shape.shape[0]
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        result = self._forward_bf16_kv(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
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
        "oscar_mla_int2",
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
