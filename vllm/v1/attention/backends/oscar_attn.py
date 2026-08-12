# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR mixed BF16/INT2 KV-cache attention backend for vLLM.

Prefill: run standard attention on the current BF16 K/V, then rotate and store
         prefix/recent tokens in a bounded BF16 arena and history as clipped
         INT2.
Decode:  rotate the query by ``R_k`` and run one online softmax over BF16 and
         dequantized INT2 tiers, then map the weighted value sum back with
         ``R_v^T``.

Each layer receives three separately indexed tensors: paged INT2 history,
BF16 prefix, and BF16 recent. INT2 slots follow the request block table; the
BF16 pools use stable request-owned rows supplied by the scheduler.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import torch
import torch.nn.functional as F

from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.model_executor.layers.quantization.oscar.rotation import get_layer_rotation
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
from vllm.v1.attention.ops.triton_oscar_bulk_flush import (
    OscarBulkFlushPlan,
    is_oscar_bulk_flush_supported,
    is_oscar_bulk_flush_target,
    oscar_bulk_flush,
    prepare_oscar_bulk_flush_plan,
)
from vllm.v1.attention.ops.triton_oscar_decode import (
    materialize_oscar_slot_ids,
    oscar_decode_attention,
)
from vllm.v1.attention.ops.triton_oscar_mixed_store import (
    oscar_demote_hp,
    oscar_store_hp,
)
from vllm.v1.attention.ops.triton_oscar_prefill import (
    oscar_cached_prefill_attention,
    oscar_materialize_prefill_kv,
)
from vllm.v1.attention.ops.triton_oscar_store import oscar_store
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    is_workspace_manager_initialized,
)

logger = init_logger(__name__)

_HAS_FLASH_ATTN = is_flash_attn_varlen_func_available()
if _HAS_FLASH_ATTN:
    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func


def _materialize_token_capacity(
    vllm_config, num_kv_heads: int, head_size: int, dtype: torch.dtype
) -> int:
    scheduler_config = vllm_config.scheduler_config
    cache_config = vllm_config.cache_config
    if (
        not _HAS_FLASH_ATTN
        or not scheduler_config.enable_chunked_prefill
        or scheduler_config.max_num_seqs != 1
        or cache_config.enable_prefix_caching
        or vllm_config.speculative_config is not None
        or vllm_config.parallel_config.use_ubatching
    ):
        return 0
    max_model_len = vllm_config.model_config.max_model_len
    if (
        not isinstance(max_model_len, int)
        or isinstance(max_model_len, bool)
        or max_model_len <= 0
    ):
        return 0
    return max_model_len


class OscarAttentionBackend(AttentionBackend):
    """Attention backend using OSCAR INT2 KV-cache compression."""

    accept_output_buffer: bool = True
    forward_includes_kv_cache_update: bool = True

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["oscar_int2"]

    @staticmethod
    def get_name() -> str:
        return "OSCAR"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [16, 32, 64, 128]

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:
        return False

    @staticmethod
    def get_impl_cls() -> type["OscarAttentionImpl"]:
        return OscarAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["OscarMetadataBuilder"]:
        return OscarMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "oscar_int2",
    ) -> tuple[int, ...]:
        cfg = OscarConfig.from_cache_dtype(cache_dtype_str, head_size)
        return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "oscar_int2"

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size > 0


@dataclass
class OscarMetadata(AttentionMetadata):
    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    hp_row_ids: torch.Tensor
    prefix_page_ids: torch.Tensor
    shared_hit_tokens: torch.Tensor
    token_to_req_indices: torch.Tensor
    physical_slot_ids: torch.Tensor | None = None
    seq_start_loc: torch.Tensor | None = None
    cached_lens: torch.Tensor | None = None
    num_actual_tokens: int = 0
    max_query_len: int = 0
    max_seq_len: int = 0
    is_prefill: bool = False
    num_decodes: int = 0
    num_decode_tokens: int = 0
    query_start_loc_cpu: torch.Tensor | None = None
    seq_lens_cpu: torch.Tensor | None = None
    bulk_flush_plan: OscarBulkFlushPlan | None = None
    bulk_flush_enabled: bool = False


class OscarMetadataBuilder(AttentionMetadataBuilder[OscarMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    @classmethod
    def get_cudagraph_support(
        cls, vllm_config, kv_cache_spec
    ) -> AttentionCGSupport:
        scheduler_config = vllm_config.scheduler_config
        cache_config = vllm_config.cache_config
        if (
            scheduler_config.max_num_seqs == 1
            and not cache_config.enable_prefix_caching
            and vllm_config.speculative_config is None
        ):
            if scheduler_config.enable_chunked_prefill:
                return AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE
            return AttentionCGSupport.ALWAYS
        return AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self._init_reorder_batch_threshold(1, supports_spec_as_decode=False)
        self.token_to_req_indices = torch.zeros(
            vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=device,
        )
        self.seq_start_loc = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device=device,
        )
        self.cached_lens = torch.zeros(
            vllm_config.scheduler_config.max_num_seqs,
            dtype=torch.int32,
            device=device,
        )
        self._block_size = kv_cache_spec.block_size
        self._bulk_flush_enabled = is_oscar_bulk_flush_supported(
            vllm_config
        ) and is_oscar_bulk_flush_target(kv_cache_spec, layer_names)
        self._flush_interval = getattr(kv_cache_spec, "flush_interval", 8)
        self._prefix_tokens = getattr(kv_cache_spec, "prefix_tokens", 64)
        self._recent_tokens = getattr(kv_cache_spec, "recent_tokens", 256)
        self._recent_capacity = getattr(
            kv_cache_spec,
            "recent_row_capacity",
            self._recent_tokens,
        )
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self.flush_phase = torch.zeros(max_num_seqs, dtype=torch.int32, device=device)
        self.row_generations = torch.full(
            (max_num_seqs,), -1, dtype=torch.int64, device=device
        )
        self.recent_extra = torch.zeros(
            max_num_seqs, dtype=torch.int32, device=device
        )
        plan_shape = (max_num_seqs, self._flush_interval)
        self.flush_positions = torch.full(
            plan_shape, -1, dtype=torch.int32, device=device
        )
        self.flush_src_slots = torch.full(
            plan_shape, -1, dtype=torch.int64, device=device
        )
        self.flush_dst_slots = torch.full(
            plan_shape, -1, dtype=torch.int64, device=device
        )
        self.flush_valid = torch.zeros(plan_shape, dtype=torch.bool, device=device)
        self._building_for_capture = False
        self.physical_slot_ids = torch.full(
            (
                vllm_config.scheduler_config.max_num_seqs,
                vllm_config.model_config.max_model_len,
            ),
            -1,
            dtype=torch.int64,
            device=device,
        )
        max_materialized_tokens = _materialize_token_capacity(
            vllm_config,
            kv_cache_spec.num_kv_heads,
            kv_cache_spec.head_size,
            vllm_config.model_config.dtype,
        )
        if max_materialized_tokens and is_workspace_manager_initialized():
            workspace_shape = (
                max_materialized_tokens,
                kv_cache_spec.num_kv_heads,
                kv_cache_spec.head_size,
            )
            current_workspace_manager().get_simultaneous(
                (workspace_shape, vllm_config.model_config.dtype),
                (workspace_shape, vllm_config.model_config.dtype),
            )

    def build_for_cudagraph_capture(
        self, common_attn_metadata: CommonAttentionMetadata
    ) -> OscarMetadata:
        self._building_for_capture = True
        try:
            m = self.build(0, common_attn_metadata)
        finally:
            self._building_for_capture = False
        if common_attn_metadata.max_query_len <= 1:
            m.seq_lens.fill_(1)
        return m

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        cam = common_attn_metadata
        if cam.oscar_hp_row_ids is None:
            raise RuntimeError("OSCAR attention requires scheduler-owned HP row IDs")
        if cam.oscar_prefix_page_ids is None:
            raise RuntimeError(
                "OSCAR attention requires scheduler-owned prefix page IDs"
            )
        if cam.oscar_shared_hit_tokens is None:
            raise RuntimeError(
                "OSCAR attention requires scheduler-owned shared hit lengths"
            )
        token_to_req_indices = cam.token_to_req_indices(self.token_to_req_indices)
        num_reqs = cam.seq_lens.shape[0]
        torch.cumsum(
            cam.seq_lens,
            dim=0,
            out=self.seq_start_loc[1 : num_reqs + 1],
        )
        query_lens = (
            cam.query_start_loc[1 : num_reqs + 1] - cam.query_start_loc[:num_reqs]
        )
        torch.sub(
            cam.seq_lens,
            query_lens,
            out=self.cached_lens[:num_reqs],
        )
        assert self.reorder_batch_threshold is not None
        num_decodes, num_prefills, num_decode_tokens, _ = split_decodes_and_prefills(
            cam, decode_threshold=self.reorder_batch_threshold
        )
        physical_slot_ids = None
        if num_decodes:
            physical_slot_ids = self.physical_slot_ids[:num_decodes]
            # V2 metadata preparation and full-graph replay use the same
            # current stream, so this update precedes every layer's read.
            materialize_oscar_slot_ids(
                cam.block_table_tensor[:num_decodes],
                cam.seq_lens[:num_decodes],
                physical_slot_ids,
                self._block_size,
            )
        bulk_flush_plan = None
        if self._bulk_flush_enabled and num_decodes:
            if self._building_for_capture:
                self.recent_extra[:num_decodes].zero_()
                self.flush_positions[:num_decodes].fill_(-1)
                self.flush_src_slots[:num_decodes].fill_(-1)
                self.flush_dst_slots[:num_decodes].fill_(-1)
                self.flush_valid[:num_decodes].zero_()
                bulk_flush_plan = OscarBulkFlushPlan(
                    next_phase=self.recent_extra[:num_decodes],
                    recent_extra=self.recent_extra[:num_decodes],
                    positions=self.flush_positions[:num_decodes],
                    src_recent_slots=self.flush_src_slots[:num_decodes],
                    dst_slots=self.flush_dst_slots[:num_decodes],
                    valid=self.flush_valid[:num_decodes],
                )
            else:
                if cam.oscar_reset_mask is None or cam.oscar_row_generations is None:
                    raise RuntimeError(
                        "OSCAR bulk flush requires reset mask and row generations"
                    )
                bulk_flush_plan = prepare_oscar_bulk_flush_plan(
                    phase=self.flush_phase,
                    row_generations=self.row_generations,
                    reset_mask=cam.oscar_reset_mask[:num_decodes],
                    request_generations=cam.oscar_row_generations[:num_decodes],
                    cached_lens=self.cached_lens[:num_decodes],
                    hp_row_ids=cam.oscar_hp_row_ids[:num_decodes],
                    shared_hit_tokens=cam.oscar_shared_hit_tokens[:num_decodes],
                    block_table=cam.block_table_tensor[:num_decodes],
                    recent_extra=self.recent_extra,
                    positions=self.flush_positions,
                    src_recent_slots=self.flush_src_slots,
                    dst_slots=self.flush_dst_slots,
                    valid=self.flush_valid,
                    prefix_tokens=self._prefix_tokens,
                    recent_tokens=self._recent_tokens,
                    recent_capacity=self._recent_capacity,
                    block_size=self._block_size,
                    flush_interval=self._flush_interval,
                )
        return OscarMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            hp_row_ids=cam.oscar_hp_row_ids,
            prefix_page_ids=cam.oscar_prefix_page_ids,
            shared_hit_tokens=cam.oscar_shared_hit_tokens,
            token_to_req_indices=token_to_req_indices,
            physical_slot_ids=physical_slot_ids,
            seq_start_loc=self.seq_start_loc[: num_reqs + 1],
            cached_lens=self.cached_lens[:num_reqs],
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=(cam.max_query_len > 1),
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            query_start_loc_cpu=cam.query_start_loc_cpu,
            seq_lens_cpu=cam.seq_lens_cpu_upper_bound,
            bulk_flush_plan=bulk_flush_plan,
            bulk_flush_enabled=self._bulk_flush_enabled,
        )


class OscarAttentionImpl(AttentionImpl["OscarMetadata"]):
    supports_quant_query_input: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.cfg = OscarConfig.from_cache_dtype(kv_cache_dtype, head_size)
        self.fa_version = get_flash_attn_version(head_size=head_size)
        vllm_config = get_current_vllm_config()
        self.max_num_kv_splits = (
            vllm_config.attention_config.tq_max_kv_splits_for_cuda_graph
        )
        self._max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        self._materialize_max_tokens = _materialize_token_capacity(
            vllm_config,
            self.num_kv_heads,
            self.head_size,
            vllm_config.model_config.dtype,
        )

    # ---- rotation setup (one-time per layer) ------------------------------
    def _v_rotation_absorbed(self, layer: Any) -> bool:
        absorbed = bool(getattr(layer, "oscar_v_rotation_absorbed", False))
        if absorbed != self.cfg.absorb_v_rotation:
            raise RuntimeError(
                "OSCAR V rotation absorption config does not match the loaded "
                f"layer state: configured={self.cfg.absorb_v_rotation}, "
                f"layer={absorbed}"
            )
        return absorbed

    def _ensure_rotations(self, layer: Any, device: torch.device, dtype: torch.dtype):
        self._v_rotation_absorbed(layer)
        if getattr(layer, "_oscar_ready", False):
            return
        D = self.head_size
        rk = get_layer_rotation(
            self.cfg.k_rotation_path, layer.layer_name, D, device, torch.float32
        )
        rv = get_layer_rotation(
            self.cfg.v_rotation_path, layer.layer_name, D, device, torch.float32
        )
        layer._oscar_Rk = rk
        layer._oscar_RkT = rk.t().contiguous()
        layer._oscar_Rv = rv
        layer._oscar_RvT = rv.t().contiguous()
        layer._oscar_Rk_fast = rk.to(dtype)
        layer._oscar_RvT_fast = layer._oscar_RvT.to(dtype)
        layer._oscar_ready = True
        logger.info_once(
            "OSCAR calibration active: K rotation=%s, V rotation=%s, head_dim=%d",
            self.cfg.k_rotation_path,
            self.cfg.v_rotation_path,
            D,
        )

    def _update_mixed_cache(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        attn_metadata: OscarMetadata,
        *,
        demote_recent: bool,
        rotated_key: torch.Tensor | None = None,
        rotated_value: torch.Tensor | None = None,
    ) -> None:
        quant_cache, prefix_cache, recent_cache = kv_cache
        num_tokens = attn_metadata.num_actual_tokens
        if num_tokens <= 0:
            return
        k = key[:num_tokens].view(num_tokens, self.num_kv_heads, self.head_size)
        v = value[:num_tokens].view(num_tokens, self.num_kv_heads, self.head_size)
        if rotated_key is None:
            k_rot = torch.matmul(k.float(), layer._oscar_Rk)
        else:
            k_rot = rotated_key
        if rotated_value is None:
            v_rot = (
                v
                if self._v_rotation_absorbed(layer)
                else torch.matmul(v.float(), layer._oscar_Rv)
            )
        else:
            v_rot = rotated_value

        bulk_flush = (
            attn_metadata.bulk_flush_enabled
            and demote_recent
            and attn_metadata.bulk_flush_plan is not None
        )
        if bulk_flush:
            state = getattr(layer, "_oscar_bulk_flush_state", None)
            if state is None or attn_metadata.bulk_flush_plan is None:
                raise RuntimeError("OSCAR bulk flush cache state is not registered")
            if getattr(layer, "_oscar_bulk_flush_owner", False):
                oscar_bulk_flush(
                    state,
                    attn_metadata.bulk_flush_plan,
                    key_levels=self.cfg.key_levels,
                    value_levels=self.cfg.value_levels,
                    key_packed_size=self.cfg.key_packed_size,
                    data_bytes=self.cfg.key_data_bytes,
                    k_clip_ratio=self.cfg.k_clip_ratio,
                    v_clip_ratio=self.cfg.v_clip_ratio,
                )
        else:
            oscar_store(
                k_rot,
                v_rot,
                quant_cache,
                attn_metadata.slot_mapping[:num_tokens],
                key_levels=self.cfg.key_levels,
                value_levels=self.cfg.value_levels,
                key_packed_size=self.cfg.key_packed_size,
                data_bytes=self.cfg.key_data_bytes,
                k_clip_ratio=self.cfg.k_clip_ratio,
                v_clip_ratio=self.cfg.v_clip_ratio,
                token_to_req_indices=attn_metadata.token_to_req_indices,
                query_start_loc=attn_metadata.query_start_loc,
                seq_lens=attn_metadata.seq_lens,
                prefix_tokens=self.cfg.prefix_tokens,
                recent_tokens=self.cfg.recent_tokens,
            )
        if demote_recent and not bulk_flush:
            oscar_demote_hp(
                recent_cache,
                quant_cache,
                attn_metadata.block_table,
                attn_metadata.seq_lens,
                attn_metadata.hp_row_ids,
                shared_hit_tokens=attn_metadata.shared_hit_tokens,
                query_start_loc=attn_metadata.query_start_loc,
                max_query_len=attn_metadata.max_query_len,
                prefix_tokens=self.cfg.prefix_tokens,
                recent_tokens=self.cfg.recent_tokens,
                recent_capacity=recent_cache.shape[0] // self._max_num_seqs,
                key_levels=self.cfg.key_levels,
                value_levels=self.cfg.value_levels,
                key_packed_size=self.cfg.key_packed_size,
                data_bytes=self.cfg.key_data_bytes,
                k_clip_ratio=self.cfg.k_clip_ratio,
                v_clip_ratio=self.cfg.v_clip_ratio,
            )

        oscar_store_hp(
            k_rot,
            v_rot,
            prefix_cache,
            recent_cache,
            token_to_req_indices=attn_metadata.token_to_req_indices,
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            hp_row_ids=attn_metadata.hp_row_ids,
            prefix_page_ids=attn_metadata.prefix_page_ids,
            prefix_block_size=quant_cache.shape[1],
            prefix_tokens=self.cfg.prefix_tokens,
            recent_tokens=self.cfg.recent_tokens,
            recent_capacity=recent_cache.shape[0] // self._max_num_seqs,
        )
        logger.info_once(
            "OSCAR Triton KV write active: prefix=%d BF16, recent=%d BF16, "
            "history=clip+INT2 (K %.2f, V %.2f)",
            self.cfg.prefix_tokens,
            self.cfg.recent_tokens,
            self.cfg.k_clip_ratio,
            self.cfg.v_clip_ratio,
        )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: list[torch.Tensor],
        attn_metadata: "OscarMetadata",
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        if output is None:
            output = torch.zeros(
                num_tokens,
                self.num_heads * self.head_size,
                dtype=query.dtype,
                device=query.device,
            )
        if attn_metadata is None:
            return output.fill_(0)
        N = attn_metadata.num_actual_tokens
        if N <= 0:
            return output.fill_(0)

        self._ensure_rotations(layer, query.device, query.dtype)
        q = query[:N].view(N, self.num_heads, self.head_size)

        num_decodes = attn_metadata.num_decodes
        num_decode_tokens = attn_metadata.num_decode_tokens

        if not attn_metadata.is_prefill:
            self._update_mixed_cache(
                layer,
                key,
                value,
                kv_cache,
                attn_metadata,
                demote_recent=True,
            )
            attn_out = self._decode_attention(q, kv_cache, attn_metadata, layer)
        elif num_decodes == 0:
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            has_cached_context = self._has_cached_context(attn_metadata)
            materialize_tokens = self._materialize_tokens(attn_metadata)
            v_rotation_absorbed = self._v_rotation_absorbed(layer)
            k_rot = v_rot = None
            if has_cached_context and (materialize_tokens or v_rotation_absorbed):
                prefill_oscar_layer: Any = layer
                k_rotation = prefill_oscar_layer._oscar_Rk
                v_rotation = prefill_oscar_layer._oscar_Rv
                k_rot = torch.matmul(k.float(), k_rotation)
                v_rot = (
                    v if v_rotation_absorbed else torch.matmul(v.float(), v_rotation)
                )
            attn_out = self._prefill_attention(
                q,
                k,
                v,
                kv_cache,
                attn_metadata,
                layer,
                has_cached_context,
                materialize_tokens=materialize_tokens,
                rotated_key=k_rot,
                rotated_value=v_rot,
            )
            self._update_mixed_cache(
                layer,
                key,
                value,
                kv_cache,
                attn_metadata,
                demote_recent=has_cached_context,
                rotated_key=k_rot,
                rotated_value=v_rot,
            )
        else:
            attn_out = torch.empty(
                N, self.num_heads, self.head_size, device=q.device, dtype=q.dtype
            )
            decode_meta = OscarMetadata(
                seq_lens=attn_metadata.seq_lens[:num_decodes],
                slot_mapping=attn_metadata.slot_mapping[:num_decode_tokens],
                block_table=attn_metadata.block_table[:num_decodes],
                query_start_loc=attn_metadata.query_start_loc[: num_decodes + 1],
                hp_row_ids=attn_metadata.hp_row_ids[:num_decodes],
                prefix_page_ids=attn_metadata.prefix_page_ids[:num_decodes],
                shared_hit_tokens=attn_metadata.shared_hit_tokens[:num_decodes],
                token_to_req_indices=attn_metadata.token_to_req_indices[
                    :num_decode_tokens
                ],
                physical_slot_ids=(
                    attn_metadata.physical_slot_ids[:num_decodes]
                    if attn_metadata.physical_slot_ids is not None
                    else None
                ),
                seq_start_loc=None,
                cached_lens=attn_metadata.cached_lens[:num_decodes]
                if attn_metadata.cached_lens is not None
                else None,
                num_actual_tokens=num_decode_tokens,
                max_query_len=1,
                max_seq_len=attn_metadata.max_seq_len,
                is_prefill=False,
                bulk_flush_plan=attn_metadata.bulk_flush_plan,
                bulk_flush_enabled=attn_metadata.bulk_flush_enabled,
            )
            self._update_mixed_cache(
                layer,
                key[:num_decode_tokens],
                value[:num_decode_tokens],
                kv_cache,
                decode_meta,
                demote_recent=True,
            )
            attn_out[:num_decode_tokens] = self._decode_attention(
                q[:num_decode_tokens], kv_cache, decode_meta, layer
            )
            prefill_seq_lens = attn_metadata.seq_lens[num_decodes:]
            if attn_metadata.seq_lens_cpu is not None:
                prefill_max_seq = int(attn_metadata.seq_lens_cpu[num_decodes:].max())
            else:
                prefill_max_seq = attn_metadata.max_seq_len
            prefill_qsl = (
                attn_metadata.query_start_loc[num_decodes:] - num_decode_tokens
            )
            prefill_qsl_cpu = None
            if attn_metadata.query_start_loc_cpu is not None:
                prefill_qsl_cpu = (
                    attn_metadata.query_start_loc_cpu[num_decodes:] - num_decode_tokens
                )
            prefill_meta = OscarMetadata(
                seq_lens=prefill_seq_lens,
                slot_mapping=attn_metadata.slot_mapping[num_decode_tokens:N],
                block_table=attn_metadata.block_table[num_decodes:],
                query_start_loc=prefill_qsl,
                hp_row_ids=attn_metadata.hp_row_ids[num_decodes:],
                prefix_page_ids=attn_metadata.prefix_page_ids[num_decodes:],
                shared_hit_tokens=attn_metadata.shared_hit_tokens[num_decodes:],
                token_to_req_indices=(
                    attn_metadata.token_to_req_indices[num_decode_tokens:N]
                    - num_decodes
                ),
                seq_start_loc=(
                    attn_metadata.seq_start_loc[num_decodes:]
                    - attn_metadata.seq_start_loc[num_decodes]
                    if attn_metadata.seq_start_loc is not None
                    else None
                ),
                cached_lens=attn_metadata.cached_lens[num_decodes:]
                if attn_metadata.cached_lens is not None
                else None,
                num_actual_tokens=N - num_decode_tokens,
                max_query_len=attn_metadata.max_query_len,
                max_seq_len=prefill_max_seq,
                is_prefill=True,
                query_start_loc_cpu=prefill_qsl_cpu,
                seq_lens_cpu=attn_metadata.seq_lens_cpu[num_decodes:]
                if attn_metadata.seq_lens_cpu is not None
                else None,
            )
            k = key[:N].view(N, self.num_kv_heads, self.head_size)
            v = value[:N].view(N, self.num_kv_heads, self.head_size)
            has_cached_context = self._has_cached_context(prefill_meta)
            materialize_tokens = self._materialize_tokens(prefill_meta)
            v_rotation_absorbed = self._v_rotation_absorbed(layer)
            k_rot = v_rot = None
            if has_cached_context and (materialize_tokens or v_rotation_absorbed):
                oscar_layer: Any = layer
                k_rotation = oscar_layer._oscar_Rk
                v_rotation = oscar_layer._oscar_Rv
                k_rot = torch.matmul(k[num_decode_tokens:].float(), k_rotation)
                v_rot = (
                    v[num_decode_tokens:]
                    if v_rotation_absorbed
                    else torch.matmul(v[num_decode_tokens:].float(), v_rotation)
                )
            attn_out[num_decode_tokens:] = self._prefill_attention(
                q[num_decode_tokens:],
                k[num_decode_tokens:],
                v[num_decode_tokens:],
                kv_cache,
                prefill_meta,
                layer,
                has_cached_context,
                materialize_tokens=materialize_tokens,
                rotated_key=k_rot,
                rotated_value=v_rot,
            )
            self._update_mixed_cache(
                layer,
                key[num_decode_tokens:N],
                value[num_decode_tokens:N],
                kv_cache,
                prefill_meta,
                demote_recent=has_cached_context,
                rotated_key=k_rot,
                rotated_value=v_rot,
            )

        if output.ndim == 3:
            output[:N] = attn_out.to(output.dtype)
        else:
            output[:N] = attn_out.reshape(N, -1).to(output.dtype)
        return output

    def _flash_attn_varlen(
        self,
        q,
        k,
        v,
        cu_q,
        cu_k,
        max_q,
        max_k,
        *,
        return_lse=False,
    ):
        kwargs = dict(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_lse,
        )
        if self.fa_version is not None:
            kwargs["fa_version"] = self.fa_version
        return flash_attn_varlen_func(**kwargs)

    @staticmethod
    def _has_cached_context(attn_metadata: OscarMetadata) -> bool:
        qsl = attn_metadata.query_start_loc_cpu
        seq_lens = attn_metadata.seq_lens_cpu
        if qsl is None:
            qsl = attn_metadata.query_start_loc.cpu()
        if seq_lens is None:
            seq_lens = attn_metadata.seq_lens.cpu()
        query_lens = qsl[1:] - qsl[:-1]
        return bool(torch.any(seq_lens > query_lens))

    def _materialize_tokens(self, attn_metadata: OscarMetadata) -> int:
        max_tokens = getattr(self, "_materialize_max_tokens", 0)
        if (
            not _HAS_FLASH_ATTN
            or not max_tokens
            or attn_metadata.seq_lens_cpu is None
            or attn_metadata.seq_start_loc is None
            or not is_workspace_manager_initialized()
        ):
            return 0
        total_tokens = int(attn_metadata.seq_lens_cpu.sum())
        if total_tokens > max_tokens:
            logger.info_once(
                "OSCAR materialized prefill exceeds its bounded workspace; "
                "falling back to fused GQA2 cached attention"
            )
            return 0
        return total_tokens

    def _prefill_attention(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        layer,
        has_cached_context,
        *,
        materialize_tokens=0,
        rotated_key=None,
        rotated_value=None,
    ):
        if not has_cached_context:
            if _HAS_FLASH_ATTN:
                output = self._flash_attn_varlen(
                    query,
                    key,
                    value,
                    attn_metadata.query_start_loc,
                    attn_metadata.query_start_loc,
                    attn_metadata.max_query_len,
                    attn_metadata.max_query_len,
                )
            else:
                output = self._prefill_sdpa_fallback(query, key, value, attn_metadata)
            return (
                self._inverse_v_rotation(output, layer)
                if self._v_rotation_absorbed(layer)
                else output
            )

        if not _HAS_FLASH_ATTN:
            raise RuntimeError(
                "OSCAR chunked prefill requires FlashAttention LSE support"
            )

        if materialize_tokens:
            assert rotated_key is not None and rotated_value is not None
            return self._materialized_prefill_attention(
                query,
                rotated_key,
                rotated_value,
                kv_cache,
                attn_metadata,
                layer,
                materialize_tokens,
            )

        cached_lens_per_req = attn_metadata.cached_lens
        if cached_lens_per_req is None:
            query_lens = (
                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            )
            cached_lens_per_req = attn_metadata.seq_lens - query_lens
        quant_cache, prefix_cache, recent_cache = kv_cache
        v_rotation_absorbed = self._v_rotation_absorbed(layer)
        if v_rotation_absorbed:
            assert rotated_key is not None and rotated_value is not None
            q_rotation = getattr(layer, "_oscar_Rk_fast", None)
            if q_rotation is None or q_rotation.dtype != query.dtype:
                q_rotation = layer._oscar_Rk.to(query.dtype)
            q_rot = torch.matmul(query, q_rotation)
        else:
            q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        cached_output, cached_lse = oscar_cached_prefill_attention(
            q_rot,
            quant_cache,
            prefix_cache,
            recent_cache,
            attn_metadata.block_table,
            cached_lens_per_req,
            attn_metadata.query_start_loc,
            attn_metadata.hp_row_ids,
            attn_metadata.prefix_page_ids,
            attn_metadata.shared_hit_tokens,
            scale=self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            prefix_tokens=self.cfg.prefix_tokens,
            recent_tokens=self.cfg.recent_tokens,
            recent_capacity=recent_cache.shape[0] // self._max_num_seqs,
            max_query_len=attn_metadata.max_query_len,
            v_rotation_t=None if v_rotation_absorbed else layer._oscar_RvT,
            output_dtype=query.dtype if v_rotation_absorbed else None,
            current_key=rotated_key if v_rotation_absorbed else None,
            current_value=rotated_value if v_rotation_absorbed else None,
        )
        if v_rotation_absorbed:
            return self._inverse_v_rotation(cached_output, layer)

        suffix_output, suffix_lse = self._flash_attn_varlen(
            query,
            key,
            value,
            attn_metadata.query_start_loc,
            attn_metadata.query_start_loc,
            attn_metadata.max_query_len,
            attn_metadata.max_query_len,
            return_lse=True,
        )
        if cached_output.dtype != suffix_output.dtype:
            cached_output = cached_output.to(suffix_output.dtype)
        output = torch.empty_like(suffix_output)
        merge_attn_states(
            output=output,
            prefix_output=cached_output,
            prefix_lse=cached_lse,
            suffix_output=suffix_output,
            suffix_lse=suffix_lse,
        )
        return output

    def _materialized_prefill_attention(
        self,
        query,
        rotated_key,
        rotated_value,
        kv_cache,
        attn_metadata,
        layer,
        total_tokens,
    ):
        workspace_shape = (
            total_tokens,
            self.num_kv_heads,
            self.head_size,
        )
        materialized_key, materialized_value = (
            current_workspace_manager().get_simultaneous(
                (workspace_shape, query.dtype),
                (workspace_shape, query.dtype),
            )
        )
        quant_cache, prefix_cache, recent_cache = kv_cache
        cached_lens = attn_metadata.cached_lens
        if cached_lens is None:
            query_lens = (
                attn_metadata.query_start_loc[1:] - attn_metadata.query_start_loc[:-1]
            )
            cached_lens = attn_metadata.seq_lens - query_lens
        oscar_materialize_prefill_kv(
            rotated_key,
            rotated_value,
            quant_cache,
            prefix_cache,
            recent_cache,
            attn_metadata.block_table,
            cached_lens,
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            attn_metadata.hp_row_ids,
            attn_metadata.prefix_page_ids,
            attn_metadata.shared_hit_tokens,
            materialized_key,
            materialized_value,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            prefix_tokens=self.cfg.prefix_tokens,
            recent_tokens=self.cfg.recent_tokens,
            recent_capacity=recent_cache.shape[0] // self._max_num_seqs,
            max_seq_len=attn_metadata.max_seq_len,
        )
        q_rotation = getattr(layer, "_oscar_Rk_fast", None)
        if q_rotation is None or q_rotation.dtype != query.dtype:
            q_rotation = layer._oscar_Rk.to(query.dtype)
        q_rot = torch.matmul(query, q_rotation)
        output_rot = self._flash_attn_varlen(
            q_rot,
            materialized_key,
            materialized_value,
            attn_metadata.query_start_loc,
            attn_metadata.seq_start_loc,
            attn_metadata.max_query_len,
            attn_metadata.max_seq_len,
        )
        logger.info_once(
            "OSCAR bounded materialized prefill active: one packed mixed-KV "
            "workspace and one causal FlashAttention call"
        )
        return self._inverse_v_rotation(output_rot, layer)

    @staticmethod
    def _inverse_v_rotation(output, layer):
        v_rotation_t = getattr(layer, "_oscar_RvT_fast", None)
        if v_rotation_t is None or v_rotation_t.dtype != output.dtype:
            v_rotation_t = layer._oscar_RvT.to(output.dtype)
        return torch.matmul(output, v_rotation_t)

    def _prefill_sdpa_fallback(self, query, key, value, attn_metadata):
        N, Hq, D = query.shape
        Hk = key.shape[1]
        use_gqa = Hk < Hq
        if attn_metadata.query_start_loc_cpu is not None:
            qsl = attn_metadata.query_start_loc_cpu.tolist()
        else:
            qsl = attn_metadata.query_start_loc.tolist()
        if attn_metadata.seq_lens_cpu is not None:
            seq_lens_list = attn_metadata.seq_lens_cpu.tolist()
        else:
            seq_lens_list = attn_metadata.seq_lens.tolist()

        output = torch.zeros(N, Hq, D, device=query.device, dtype=query.dtype)
        num_reqs = len(qsl) - 1
        for i in range(num_reqs):
            q_start, q_end = qsl[i], qsl[i + 1]
            q_len = q_end - q_start
            if q_len <= 0:
                continue
            seq_len = seq_lens_list[i]
            q_seq = query[q_start:q_end]
            k_seq = key[q_start:q_end]
            v_seq = value[q_start:q_end]
            cached_len = seq_len - q_len
            out = self._sdpa(q_seq, k_seq, v_seq, cached_len, seq_len, use_gqa)
            output[q_start:q_end] = out.to(query.dtype)
        return output

    def _sdpa(self, q_seq, k_full, v_full, cached_len, seq_len, use_gqa):
        q_len = q_seq.shape[0]
        q_t = q_seq.transpose(0, 1).unsqueeze(0)
        k_t = k_full.transpose(0, 1).unsqueeze(0)
        v_t = v_full.transpose(0, 1).unsqueeze(0)
        device = q_seq.device
        if cached_len <= 0:
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.scale, enable_gqa=use_gqa
            )
        else:
            q_pos = torch.arange(q_len, device=device).unsqueeze(1) + cached_len
            k_pos = torch.arange(seq_len, device=device).unsqueeze(0)
            mask = k_pos <= q_pos
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask, scale=self.scale, enable_gqa=use_gqa
            )
        return out[0].transpose(0, 1)

    def _decode_attention(self, query, kv_cache, attn_metadata, layer):
        quant_cache, prefix_cache, recent_cache = kv_cache
        # Rotate the query into the same space as the rotated stored keys.
        q_rot = torch.matmul(query.float(), layer._oscar_Rk)
        logger.info_once(
            "OSCAR Triton mixed attention read active: fused BF16/INT2 "
            "read, dequantization, online softmax, and V inverse rotation"
        )
        out_rot = oscar_decode_attention(
            q_rot,
            quant_cache,
            attn_metadata.block_table,
            attn_metadata.seq_lens,
            self.scale,
            key_levels=self.cfg.key_levels,
            value_levels=self.cfg.value_levels,
            key_data_bytes=self.cfg.key_data_bytes,
            key_packed_size=self.cfg.key_packed_size,
            value_data_bytes=self.cfg.value_data_bytes,
            max_num_kv_splits=self.max_num_kv_splits,
            prefix_cache=prefix_cache,
            recent_cache=recent_cache,
            hp_row_ids=attn_metadata.hp_row_ids,
            prefix_page_ids=attn_metadata.prefix_page_ids,
            shared_hit_tokens=attn_metadata.shared_hit_tokens,
            physical_slot_ids=attn_metadata.physical_slot_ids,
            prefix_tokens=self.cfg.prefix_tokens,
            recent_tokens=self.cfg.recent_tokens,
            recent_capacity=recent_cache.shape[0] // self._max_num_seqs,
            recent_extra=(
                attn_metadata.bulk_flush_plan.recent_extra
                if attn_metadata.bulk_flush_plan is not None
                else None
            ),
            v_rotation_t=layer._oscar_RvT,
        )
        assert isinstance(out_rot, torch.Tensor)
        return out_rot.to(query.dtype)
