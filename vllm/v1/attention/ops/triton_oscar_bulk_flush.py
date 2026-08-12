# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-layer periodic demotion for OSCAR's BF16 recent rows."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_oscar_mixed_store import _clip_index


@dataclass(frozen=True)
class OscarBulkFlushPlan:
    """Persistent metadata consumed by the cross-layer flush kernel."""

    next_phase: torch.Tensor
    recent_extra: torch.Tensor
    positions: torch.Tensor
    src_recent_slots: torch.Tensor
    dst_slots: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class OscarBulkFlushCacheState:
    """Stable per-layer cache address table and uniform layout."""

    layer_names: tuple[str, ...]
    quant_ptrs: torch.Tensor
    recent_ptrs: torch.Tensor
    quant_sample: torch.Tensor
    recent_sample: torch.Tensor
    quant_stride: tuple[int, ...]
    recent_stride: tuple[int, ...]
    recent_capacity: int


def is_oscar_bulk_flush_supported(vllm_config: Any) -> bool:
    """Return whether the narrow batch-1 P1 path is safe to enable."""
    scheduler = vllm_config.scheduler_config
    cache = vllm_config.cache_config
    parallel = getattr(vllm_config, "parallel_config", None)
    transfer = getattr(vllm_config, "kv_transfer_config", None)
    offload = getattr(vllm_config, "offload_config", None)
    uva = getattr(offload, "uva", offload)
    prefetch = getattr(offload, "prefetch", None)
    cpu_offload_gb = getattr(uva, "cpu_offload_gb", 0)
    offload_group_size = getattr(prefetch, "offload_group_size", 0)
    return bool(
        getattr(scheduler, "max_num_seqs", 0) == 1
        and getattr(scheduler, "enable_chunked_prefill", False)
        and not getattr(cache, "enable_prefix_caching", True)
        and getattr(vllm_config, "speculative_config", object()) is None
        and parallel is not None
        and parallel.tensor_parallel_size == 1
        and parallel.pipeline_parallel_size == 1
        and (transfer is None or transfer.kv_connector is None)
        and cpu_offload_gb == 0
        and offload_group_size == 0
    )


def is_oscar_bulk_flush_target(
    kv_cache_spec: Any, layer_names: Sequence[str]
) -> bool:
    """Restrict P1 to the calibrated Qwen3-4B cache geometry."""
    return bool(
        len(layer_names) == 36
        and kv_cache_spec.block_size == 16
        and kv_cache_spec.num_kv_heads == 8
        and kv_cache_spec.head_size == 128
        and kv_cache_spec.head_size_v == 128
        and kv_cache_spec.quant_slot_size == 72
        and kv_cache_spec.group_size == 128
        and kv_cache_spec.prefix_tokens == 64
        and kv_cache_spec.recent_tokens == 256
        and getattr(kv_cache_spec, "prefix_cache_extra_tokens", 0) == 0
        and kv_cache_spec.flush_interval == 8
        and kv_cache_spec.recent_row_capacity == 272
        and kv_cache_spec.hp_dtype == torch.bfloat16
    )


def validate_oscar_bulk_flush_scheduler_output(
    vllm_config: Any, scheduler_output: Any, *, enabled: bool | None = None
) -> None:
    """Reject unsupported request lifecycle operations before mutation."""
    if enabled is None:
        enabled = is_oscar_bulk_flush_supported(vllm_config)
    if not enabled:
        return
    if (
        scheduler_output.preempted_req_ids
        or scheduler_output.scheduled_cached_reqs.resumed_req_ids
    ):
        raise RuntimeError(
            "OSCAR bulk flush does not support preemption or resume"
        )
    for request in scheduler_output.scheduled_new_reqs:
        sampling = request.sampling_params
        if sampling is not None and (
            getattr(sampling, "n", 1) != 1
            or type(sampling).__name__ == "BeamSearchParams"
        ):
            raise RuntimeError("OSCAR bulk flush does not support beam or fanout")
        if request.num_computed_tokens != 0:
            raise RuntimeError(
                "OSCAR bulk flush does not support copy or fork requests"
            )


def update_oscar_bulk_flush_row_generations(
    row_generations: np.ndarray,
    req_ids: Sequence[str],
    hp_rows: Sequence[int] | np.ndarray,
    new_req_ids: set[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Build reset/generation metadata after lifecycle validation succeeds."""
    if len(req_ids) != len(hp_rows):
        raise ValueError("OSCAR request and HP-row counts must match")
    rows = np.asarray(hp_rows, dtype=np.int64)
    if np.any(rows < 0) or np.any(rows >= row_generations.shape[0]):
        raise ValueError("OSCAR HP row is outside the generation table")
    if np.unique(rows).size != rows.size:
        raise ValueError("OSCAR HP rows must have unique live owners")

    reset = np.zeros_like(row_generations, dtype=np.bool_)
    request_generations = np.zeros_like(row_generations, dtype=np.int64)
    for index, (req_id, hp_row) in enumerate(zip(req_ids, rows)):
        if req_id in new_req_ids:
            row_generations[hp_row] += 1
            reset[index] = True
        request_generations[index] = row_generations[hp_row]
    return reset, request_generations


def build_oscar_bulk_flush_plan_cpu(
    *,
    phase: torch.Tensor,
    cached_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    shared_hit_tokens: torch.Tensor,
    block_table: torch.Tensor,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int,
    block_size: int,
    flush_interval: int,
) -> OscarBulkFlushPlan:
    """Build the reference plan used by CPU tests and CPU execution."""
    num_reqs = cached_lens.shape[0]
    next_phase = torch.remainder(phase[:num_reqs] + 1, flush_interval)
    recent_extra = next_phase.clone()
    offsets = torch.arange(flush_interval, dtype=torch.int32)
    positions = cached_lens[:, None] - recent_tokens - flush_interval + 1 + offsets
    lower = torch.maximum(
        shared_hit_tokens[:num_reqs],
        torch.full_like(shared_hit_tokens[:num_reqs], prefix_tokens),
    )
    valid = (next_phase[:, None] == 0) & (positions >= lower[:, None])
    safe_positions = positions.clamp_min(0)
    page_indices = torch.div(safe_positions, block_size, rounding_mode="floor")
    valid &= page_indices < block_table.shape[1]
    safe_pages = page_indices.clamp_max(block_table.shape[1] - 1).to(torch.long)
    blocks = torch.gather(block_table[:num_reqs], 1, safe_pages)
    valid &= blocks >= 0
    dst_slots = blocks.to(torch.int64) * block_size + safe_positions % block_size
    src_recent_slots = (
        hp_row_ids[:num_reqs, None].to(torch.int64) * recent_capacity
        + (safe_positions - prefix_tokens) % recent_capacity
    )
    invalid = torch.full_like(dst_slots, -1)
    return OscarBulkFlushPlan(
        next_phase=next_phase,
        recent_extra=recent_extra,
        positions=torch.where(valid, positions, invalid.to(torch.int32)),
        src_recent_slots=torch.where(valid, src_recent_slots, invalid),
        dst_slots=torch.where(valid, dst_slots, invalid),
        valid=valid,
    )


def register_oscar_bulk_flush_caches(
    kv_caches: dict[str, Sequence[torch.Tensor]],
    layer_names: Sequence[str],
    *,
    expected_recent_capacity: int,
    max_num_seqs: int,
) -> OscarBulkFlushCacheState:
    """Create stable pointer tables in the exact model layer order."""
    names = tuple(layer_names)
    if not names:
        raise ValueError("OSCAR bulk flush requires at least one layer")
    if len(set(names)) != len(names):
        raise ValueError("OSCAR bulk flush layer names must be unique")
    missing = [name for name in names if name not in kv_caches]
    if missing:
        raise ValueError(f"OSCAR bulk flush cache layers are missing: {missing}")
    if max_num_seqs <= 0 or expected_recent_capacity <= 0:
        raise ValueError("OSCAR bulk flush capacities must be positive")
    malformed = [name for name in names if len(kv_caches[name]) != 3]
    if malformed:
        raise ValueError(f"OSCAR mixed cache triples are malformed: {malformed}")
    quant = [kv_caches[name][0] for name in names]
    prefix = [kv_caches[name][1] for name in names]
    recent = [kv_caches[name][2] for name in names]
    if any(tensor.data_ptr() == 0 for tensor in (*quant, *prefix, *recent)):
        raise ValueError("OSCAR bulk flush cache pointers must be nonzero")
    device = quant[0].device
    if any(tensor.device != device for tensor in (*quant, *prefix, *recent)):
        raise ValueError("OSCAR mixed caches must share one device")
    if any(tensor.dtype != torch.uint8 for tensor in quant):
        raise ValueError("OSCAR quant caches must use uint8")
    if any(tensor.dtype != torch.bfloat16 for tensor in (*prefix, *recent)):
        raise ValueError("OSCAR BF16 caches must use bfloat16")
    if any(tensor.ndim != 4 for tensor in (*quant, *prefix, *recent)):
        raise ValueError("OSCAR mixed caches must be four-dimensional")
    if (
        quant[0].shape[1:] != (16, 8, 72)
        or prefix[0].shape[1:] != (8, 2, 128)
        or recent[0].shape[1:] != (8, 2, 128)
    ):
        raise ValueError("OSCAR quant/prefix/recent K/V shapes are inconsistent")
    quant_stride = quant[0].stride()
    prefix_stride = prefix[0].stride()
    recent_stride = recent[0].stride()
    recent_slots = recent[0].shape[0]
    if quant[0].shape[0] <= 0 or prefix[0].shape[0] != max_num_seqs * 64:
        raise ValueError("OSCAR quant/prefix cache capacities are inconsistent")
    if recent_slots != max_num_seqs * expected_recent_capacity:
        raise ValueError(
            "OSCAR recent cache does not match its physical row capacity"
        )
    recent_capacity = expected_recent_capacity
    if (
        quant_stride[-1] != 1
        or prefix_stride[-1] != 1
        or recent_stride[-1] != 1
    ):
        raise ValueError("OSCAR bulk flush requires contiguous inner dimensions")
    if (
        prefix_stride[2] != prefix[0].shape[3]
        or recent_stride[2] != recent[0].shape[3]
    ):
        raise ValueError("OSCAR combined K/V offset does not match head size")
    if (
        prefix_stride[1] != 2 * prefix_stride[2]
        or recent_stride[1] != 2 * recent_stride[2]
    ):
        raise ValueError("OSCAR BF16 head stride is not tightly packed")
    if (
        prefix_stride[0] != prefix[0].shape[1] * prefix_stride[1]
        or recent_stride[0] != recent[0].shape[1] * recent_stride[1]
    ):
        raise ValueError("OSCAR BF16 slot stride is not tightly packed")
    if (
        quant_stride[2] != quant[0].shape[3]
        or quant_stride[1] != quant[0].shape[2] * quant_stride[2]
        or quant_stride[0] != quant[0].shape[1] * quant_stride[1]
    ):
        raise ValueError("OSCAR bulk flush requires a linear paged quant cache")
    for layer_quant, layer_prefix, layer_recent in zip(quant, prefix, recent):
        if layer_quant.stride() != quant_stride or layer_quant.shape != quant[0].shape:
            raise ValueError("OSCAR quant cache layouts must be uniform across layers")
        if (
            layer_prefix.stride() != prefix[0].stride()
            or layer_prefix.shape != prefix[0].shape
        ):
            raise ValueError("OSCAR prefix cache layouts must be uniform across layers")
        if (
            layer_recent.stride() != recent_stride
            or layer_recent.shape != recent[0].shape
        ):
            raise ValueError("OSCAR recent cache layouts must be uniform across layers")
    state = OscarBulkFlushCacheState(
        layer_names=names,
        quant_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in quant], dtype=torch.int64, device=device
        ),
        recent_ptrs=torch.tensor(
            [tensor.data_ptr() for tensor in recent], dtype=torch.int64, device=device
        ),
        quant_sample=quant[0],
        recent_sample=recent[0],
        quant_stride=quant_stride,
        recent_stride=recent_stride,
        recent_capacity=recent_capacity,
    )
    if torch.any(state.quant_ptrs == 0) or torch.any(state.recent_ptrs == 0):
        raise ValueError("OSCAR pointer tables contain a null address")
    return state


def bind_oscar_bulk_flush_state(
    forward_context: dict[str, Any], state: OscarBulkFlushCacheState
) -> None:
    """Attach one shared state and exactly one owner in model layer order."""
    missing = [name for name in state.layer_names if name not in forward_context]
    if missing:
        raise ValueError(f"OSCAR attention layers are missing: {missing}")
    for name in state.layer_names:
        layer = forward_context[name]
        layer._oscar_bulk_flush_state = state
        layer._oscar_bulk_flush_owner = False
    forward_context[state.layer_names[0]]._oscar_bulk_flush_owner = True
    owners = sum(
        bool(forward_context[name]._oscar_bulk_flush_owner)
        for name in state.layer_names
    )
    if owners != 1:
        raise RuntimeError("OSCAR bulk flush requires exactly one owner")


def clear_oscar_bulk_flush_state(forward_context: dict[str, Any]) -> None:
    """Release profiling-only cache pointers before final KV allocation."""
    for layer in forward_context.values():
        if hasattr(layer, "_oscar_bulk_flush_state"):
            layer._oscar_bulk_flush_state = None
        if hasattr(layer, "_oscar_bulk_flush_owner"):
            layer._oscar_bulk_flush_owner = False


@triton.jit
def _prepare_bulk_flush_plan_kernel(
    Phase_ptr,
    Row_generations_ptr,
    Reset_mask_ptr,
    Request_generations_ptr,
    Cached_lens_ptr,
    HP_rows_ptr,
    Shared_hit_ptr,
    Block_table_ptr,
    Recent_extra_ptr,
    Positions_ptr,
    Src_slots_ptr,
    Dst_slots_ptr,
    Valid_ptr,
    stride_bt_req: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    FLUSH_INTERVAL: tl.constexpr,
):
    req = tl.program_id(0)
    hp_row = tl.load(HP_rows_ptr + req)
    phase = tl.load(Phase_ptr + hp_row)
    stored_generation = tl.load(Row_generations_ptr + hp_row)
    request_generation = tl.load(Request_generations_ptr + req)
    reset = tl.load(Reset_mask_ptr + req) | (
        stored_generation != request_generation
    )
    phase = tl.where(reset, 0, phase)
    tl.store(Row_generations_ptr + hp_row, request_generation)
    next_phase = (phase + 1) % FLUSH_INTERVAL
    tl.store(Phase_ptr + hp_row, next_phase)
    tl.store(Recent_extra_ptr + req, next_phase)
    cached_len = tl.load(Cached_lens_ptr + req)
    shared_hit = tl.load(Shared_hit_ptr + req)
    lower = tl.maximum(PREFIX_TOKENS, shared_hit)
    for index in tl.static_range(FLUSH_INTERVAL):
        out = req * FLUSH_INTERVAL + index
        pos = cached_len - RECENT_TOKENS - FLUSH_INTERVAL + 1 + index
        page = pos // BLOCK_SIZE
        valid = (
            (next_phase == 0)
            & (pos >= lower)
            & (page >= 0)
            & (page < NUM_BLOCKS)
        )
        block = tl.load(
            Block_table_ptr + req * stride_bt_req + page,
            mask=valid,
            other=-1,
        )
        valid &= block >= 0
        src = hp_row * RECENT_CAPACITY + (pos - PREFIX_TOKENS) % RECENT_CAPACITY
        dst = block * BLOCK_SIZE + pos % BLOCK_SIZE
        tl.store(Positions_ptr + out, tl.where(valid, pos, -1))
        tl.store(Src_slots_ptr + out, tl.where(valid, src, -1))
        tl.store(Dst_slots_ptr + out, tl.where(valid, dst, -1))
        tl.store(Valid_ptr + out, valid)


def prepare_oscar_bulk_flush_plan(
    *,
    phase: torch.Tensor,
    row_generations: torch.Tensor,
    reset_mask: torch.Tensor,
    request_generations: torch.Tensor,
    cached_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    shared_hit_tokens: torch.Tensor,
    block_table: torch.Tensor,
    recent_extra: torch.Tensor,
    positions: torch.Tensor,
    src_recent_slots: torch.Tensor,
    dst_slots: torch.Tensor,
    valid: torch.Tensor,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int,
    block_size: int,
    flush_interval: int,
) -> OscarBulkFlushPlan:
    """Populate persistent plan buffers on the current execution device."""
    num_reqs = cached_lens.shape[0]
    if cached_lens.device.type == "cpu":
        rows = hp_row_ids[:num_reqs].to(torch.long)
        reset = reset_mask[:num_reqs] | (
            row_generations[rows] != request_generations[:num_reqs]
        )
        phase[rows] = torch.where(reset, 0, phase[rows])
        row_generations[rows] = request_generations[:num_reqs]
        reference = build_oscar_bulk_flush_plan_cpu(
            phase=phase[rows],
            cached_lens=cached_lens,
            hp_row_ids=hp_row_ids,
            shared_hit_tokens=shared_hit_tokens,
            block_table=block_table,
            prefix_tokens=prefix_tokens,
            recent_tokens=recent_tokens,
            recent_capacity=recent_capacity,
            block_size=block_size,
            flush_interval=flush_interval,
        )
        phase[rows] = reference.next_phase
        recent_extra[:num_reqs].copy_(reference.recent_extra)
        positions[:num_reqs].copy_(reference.positions)
        src_recent_slots[:num_reqs].copy_(reference.src_recent_slots)
        dst_slots[:num_reqs].copy_(reference.dst_slots)
        valid[:num_reqs].copy_(reference.valid)
    else:
        _prepare_bulk_flush_plan_kernel[(num_reqs,)](
            phase,
            row_generations,
            reset_mask,
            request_generations,
            cached_lens,
            hp_row_ids,
            shared_hit_tokens,
            block_table,
            recent_extra,
            positions,
            src_recent_slots,
            dst_slots,
            valid,
            stride_bt_req=block_table.stride(0),
            NUM_BLOCKS=block_table.shape[1],
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            RECENT_CAPACITY=recent_capacity,
            BLOCK_SIZE=block_size,
            FLUSH_INTERVAL=flush_interval,
            num_warps=1,
        )
    return OscarBulkFlushPlan(
        next_phase=recent_extra[:num_reqs],
        recent_extra=recent_extra[:num_reqs],
        positions=positions[:num_reqs],
        src_recent_slots=src_recent_slots[:num_reqs],
        dst_slots=dst_slots[:num_reqs],
        valid=valid[:num_reqs],
    )


@triton.jit
def _bulk_quantize_pack_int2_tile(
    vec,
    Quant_ptr,
    Quant_meta_ptr,
    region_base,
    dst_base,
    active,
    D: tl.constexpr,
    LEVELS: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
):
    """Quantize and pack one ``[BLOCK_TOK, D]`` tile row-wise."""
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D
    row_mask = active[:, None] & d_mask[None, :]
    if CLIP_INDEX >= 0:
        sorted_abs = tl.sort(tl.abs(vec))
        pick = d_offs == CLIP_INDEX
        threshold = tl.sum(
            tl.where(pick[None, :], sorted_abs, 0.0), axis=1
        )
        vec = tl.minimum(
            tl.maximum(vec, -threshold[:, None]), threshold[:, None]
        )

    vmin = tl.min(tl.where(row_mask, vec, float("inf")), axis=1)
    vmax = tl.max(tl.where(row_mask, vec, -float("inf")), axis=1)
    scale = tl.maximum(vmax - vmin, 1e-8) / (LEVELS - 1)
    zero = -vmin / scale
    q = tl.minimum(
        tl.maximum(
            (vec / scale[:, None] + zero[:, None] + 0.5).to(tl.int32), 0
        ),
        LEVELS - 1,
    )
    shifts = tl.arange(0, 4) * 2
    if D == 128:
        q_grouped = tl.reshape(q, [BLOCK_TOK, 4, BLOCK_D // 4])
        packed = tl.sum(
            (q_grouped & 0x3) << shifts[None, :, None], axis=1
        ).to(tl.uint8)
    else:
        q_grouped = tl.reshape(q, [BLOCK_TOK, BLOCK_D // 4, 4])
        packed = tl.sum(
            (q_grouped & 0x3) << shifts[None, None, :], axis=2
        ).to(tl.uint8)

    pack_offs = tl.arange(0, BLOCK_PACK)
    pack_mask = active[:, None] & (pack_offs[None, :] < DATA_BYTES)
    tl.store(
        Quant_ptr + dst_base[:, None] + region_base + pack_offs[None, :],
        packed,
        mask=pack_mask,
    )
    meta_offset = (dst_base + region_base + DATA_BYTES) // 2
    tl.store(Quant_meta_ptr + meta_offset, scale, mask=active)
    tl.store(Quant_meta_ptr + meta_offset + 1, zero, mask=active)


@triton.jit
def _bulk_flush_quant_kernel(
    Recent_ptrs,
    Quant_ptrs,
    Recent_sample,
    Quant_sample,
    Src_slots,
    Dst_slots,
    Valid,
    num_flush_tokens,
    stride_recent_slot: tl.constexpr,
    stride_recent_head: tl.constexpr,
    stride_recent_kv: tl.constexpr,
    stride_quant_pos: tl.constexpr,
    stride_quant_head: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NUM_LAYERS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    BLOCK_TOK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,
):
    token_tile = tl.program_id(0)
    head = tl.program_id(1)
    layer = tl.program_id(2)
    if head >= NUM_HEADS or layer >= NUM_LAYERS:
        return
    offsets = token_tile * BLOCK_TOK + tl.arange(0, BLOCK_TOK)
    token_mask = offsets < num_flush_tokens
    active = tl.load(Valid + offsets, mask=token_mask, other=0)
    if tl.max(active, axis=0) == 0:
        return
    recent_base = tl.load(Recent_ptrs + layer).to(
        tl.pointer_type(Recent_sample.dtype.element_ty)
    )
    quant_base = tl.load(Quant_ptrs + layer).to(
        tl.pointer_type(Quant_sample.dtype.element_ty)
    )
    quant_meta = quant_base.to(tl.pointer_type(tl.bfloat16))
    src = tl.load(Src_slots + offsets, mask=token_mask, other=0).to(tl.int64)
    dst_slot = tl.load(Dst_slots + offsets, mask=token_mask, other=0).to(tl.int64)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    hp_offset = src[:, None] * stride_recent_slot + head * stride_recent_head
    key = tl.load(
        recent_base + hp_offset + d_offs[None, :],
        mask=active[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    value = tl.load(
        recent_base + hp_offset + stride_recent_kv + d_offs[None, :],
        mask=active[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    dst = dst_slot * stride_quant_pos + head * stride_quant_head
    _bulk_quantize_pack_int2_tile(
        key,
        quant_base,
        quant_meta,
        0,
        dst,
        active,
        D=HEAD_DIM,
        LEVELS=KEY_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_TOK=BLOCK_TOK,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=K_CLIP_INDEX,
    )
    _bulk_quantize_pack_int2_tile(
        value,
        quant_base,
        quant_meta,
        KEY_PACKED,
        dst,
        active,
        D=HEAD_DIM,
        LEVELS=VALUE_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_TOK=BLOCK_TOK,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=V_CLIP_INDEX,
    )


def oscar_bulk_flush(
    state: OscarBulkFlushCacheState,
    plan: OscarBulkFlushPlan,
    *,
    key_levels: int,
    value_levels: int,
    key_packed_size: int,
    data_bytes: int,
    k_clip_ratio: float,
    v_clip_ratio: float,
) -> None:
    """Quantize one 8-token plan for all layers in one Triton launch."""
    num_heads = state.quant_sample.shape[2]
    head_dim = data_bytes * 4
    flush_interval = plan.valid.shape[1]
    if flush_interval != 8:
        raise ValueError("OSCAR P1 bulk flush requires an 8-token interval")
    block_tok = 2
    num_flush_tokens = plan.valid.numel()
    if num_flush_tokens != 8:
        raise ValueError("OSCAR P1 bulk flush requires one batch-1 plan")
    grid = (4, num_heads, len(state.layer_names))
    _bulk_flush_quant_kernel[grid](
        state.recent_ptrs,
        state.quant_ptrs,
        state.recent_sample,
        state.quant_sample,
        plan.src_recent_slots.reshape(-1),
        plan.dst_slots.reshape(-1),
        plan.valid.reshape(-1),
        num_flush_tokens,
        stride_recent_slot=state.recent_stride[0],
        stride_recent_head=state.recent_stride[1],
        stride_recent_kv=state.recent_stride[2],
        stride_quant_pos=state.quant_stride[1],
        stride_quant_head=state.quant_stride[2],
        NUM_HEADS=num_heads,
        NUM_LAYERS=len(state.layer_names),
        HEAD_DIM=head_dim,
        KEY_PACKED=key_packed_size,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        DATA_BYTES=data_bytes,
        K_CLIP_INDEX=_clip_index(k_clip_ratio, head_dim),
        V_CLIP_INDEX=_clip_index(v_clip_ratio, head_dim),
        BLOCK_TOK=block_tok,
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_PACK=triton.next_power_of_2(data_bytes),
        num_warps=1,
        num_stages=1,
    )
