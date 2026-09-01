# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM80 Triton writes and demotion for OSCAR shared-latent MLA caches."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_oscar_mla_materialize import (
    OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
    OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK,
    OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT,
    OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT,
)

_OSCAR_DEMOTION_KSPLIT_SPLITS = 8
_OSCAR_DEMOTION_KSPLIT_GROUPS = 4
_OSCAR_DEMOTION_KSPLIT_GROUP_SIZE = 128


def allocate_oscar_demotion_ksplit_workspace(
    reference: torch.Tensor,
) -> torch.Tensor:
    """Allocate the bounded batch-one decode demotion workspace."""
    return torch.empty(
        (
            1,
            _OSCAR_DEMOTION_KSPLIT_GROUPS,
            _OSCAR_DEMOTION_KSPLIT_SPLITS,
            _OSCAR_DEMOTION_KSPLIT_GROUP_SIZE,
        ),
        dtype=torch.float32,
        device=reference.device,
    )


@triton.jit
def _rotate_latent_kernel(
    latent_ptr,
    rotation_ptr,
    output_ptr,
    num_rows,
    latent_rank: tl.constexpr,
    stride_latent_row: tl.constexpr,
    stride_latent_dim: tl.constexpr,
    stride_rotation_row: tl.constexpr,
    stride_rotation_col: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute ``latent @ rotation`` with an FP32 accumulator."""
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(latent_rank, block_n)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    rows = pid_m * block_m + tl.arange(0, block_m)
    cols = pid_n * block_n + tl.arange(0, block_n)
    ks = tl.arange(0, block_k)
    latent_ptrs = (
        latent_ptr + rows[:, None] * stride_latent_row + ks[None, :] * stride_latent_dim
    )
    rotation_ptrs = (
        rotation_ptr
        + ks[:, None] * stride_rotation_row
        + cols[None, :] * stride_rotation_col
    )
    accumulator = tl.zeros((block_m, block_n), dtype=tl.float32)

    for k_start in range(0, latent_rank, block_k):
        k_mask = k_start + ks < latent_rank
        latent = tl.load(
            latent_ptrs,
            mask=(rows[:, None] < num_rows) & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rotation = tl.load(
            rotation_ptrs,
            mask=k_mask[:, None] & (cols[None, :] < latent_rank),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(
            latent,
            rotation,
            accumulator,
            input_precision="ieee",
        )
        latent_ptrs += block_k * stride_latent_dim
        rotation_ptrs += block_k * stride_rotation_row

    output_ptrs = (
        output_ptr
        + rows[:, None] * stride_output_row
        + cols[None, :] * stride_output_dim
    )
    tl.store(
        output_ptrs,
        accumulator,
        mask=(rows[:, None] < num_rows) & (cols[None, :] < latent_rank),
    )


@triton.jit
def _quantize_store_history_kernel(
    rotated_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    num_rows,
    stride_rotated_row: tl.constexpr,
    stride_rotated_dim: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    clip_index: tl.constexpr,
):
    """Clip, asymmetrically quantize and pack one latent group."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    if row >= num_rows:
        return
    page = tl.load(page_ids_ptr + row)
    if page < 0:
        return
    token_offset = tl.load(page_offsets_ptr + row)

    dims = tl.arange(0, group_size)
    values = tl.load(
        rotated_ptr
        + row * stride_rotated_row
        + (group * group_size + dims) * stride_rotated_dim
    ).to(tl.float32)
    if clip_index >= 0:
        sorted_abs = tl.sort(tl.abs(values))
        threshold = tl.sum(
            tl.where(dims == clip_index, sorted_abs, 0.0),
            axis=0,
        )
        values = tl.minimum(tl.maximum(values, -threshold), threshold)

    value_min = tl.min(values, axis=0)
    value_max = tl.max(values, axis=0)
    scale = tl.maximum(value_max - value_min, 1e-8) / 3.0
    zero = -value_min / scale
    quantized = tl.minimum(
        tl.maximum((values / scale + zero + 0.5).to(tl.int32), 0),
        3,
    )
    quantized = tl.reshape(quantized, (packed_group_bytes, 4))
    shifts = tl.arange(0, 4) * 2
    packed = tl.sum(
        (quantized & 0x3) << shifts[None, :],
        axis=1,
    ).to(tl.uint8)
    packed_offsets = tl.arange(0, packed_group_bytes)
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    tl.store(
        history_data_ptr + data_base + packed_offsets * stride_data_byte,
        packed,
    )
    scale_base = (
        page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    )
    zero_base = (
        page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    )
    tl.store(history_scale_ptr + scale_base, scale)
    tl.store(history_zero_ptr + zero_base, zero)


@triton.jit
def _demote_recent_rotate_quantize_store_kernel(
    recent_ptr,
    rotation_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    logical_positions_ptr,
    hp_rows_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    prequant_cache_values_ptr,
    prequant_cache_tags_ptr,
    num_rows,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    stride_rotation_row: tl.constexpr,
    stride_rotation_col: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_prequant_cache_row: tl.constexpr,
    stride_prequant_cache_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    clip_index: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
    write_prequant_cache: tl.constexpr,
    temporal_set_count: tl.constexpr,
    temporal_state_bit: tl.constexpr,
    temporal_position_mask: tl.constexpr,
):
    """Rotate one recent row and write one packed INT2 group."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    if row >= num_rows:
        return
    page = tl.load(page_ids_ptr + row)
    if page < 0:
        return

    logical_position = tl.load(logical_positions_ptr + row)
    hp_row = tl.load(hp_rows_ptr + row)
    token_offset = tl.load(page_offsets_ptr + row)
    recent_index = (logical_position - prefix_tokens) % recent_tokens
    valid_recent = (hp_row >= 0) & (logical_position >= prefix_tokens)
    m_offsets = tl.arange(0, block_m)
    columns = group * group_size + tl.arange(0, group_size)
    k_offsets = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, group_size), dtype=tl.float32)

    latent_ptrs = (
        recent_ptr
        + hp_row * stride_recent_row
        + recent_index * stride_recent_token
        + k_offsets[None, :] * stride_recent_dim
    )
    rotation_ptrs = (
        rotation_ptr
        + k_offsets[:, None] * stride_rotation_row
        + columns[None, :] * stride_rotation_col
    )
    for k_start in range(0, latent_rank, block_k):
        k_mask = k_start + k_offsets < latent_rank
        latent = tl.load(
            latent_ptrs,
            mask=valid_recent & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        latent = tl.broadcast_to(latent, (block_m, block_k))
        rotation = tl.load(
            rotation_ptrs,
            mask=k_mask[:, None] & (columns[None, :] < latent_rank),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(
            latent,
            rotation,
            accumulator,
            input_precision="ieee",
        )
        latent_ptrs += block_k * stride_recent_dim
        rotation_ptrs += block_k * stride_rotation_row

    values = tl.sum(
        tl.where(m_offsets[:, None] == 0, accumulator, 0.0),
        axis=0,
    )
    dims = tl.arange(0, group_size)
    if clip_index >= 0:
        sorted_abs = tl.sort(tl.abs(values))
        threshold = tl.sum(
            tl.where(dims == clip_index, sorted_abs, 0.0),
            axis=0,
        )
        values = tl.minimum(tl.maximum(values, -threshold), threshold)

    value_min = tl.min(values, axis=0)
    value_max = tl.max(values, axis=0)
    scale = tl.maximum(value_max - value_min, 1e-8) / 3.0
    zero = -value_min / scale
    quantized = tl.minimum(
        tl.maximum((values / scale + zero + 0.5).to(tl.int32), 0),
        3,
    )
    quantized = tl.reshape(quantized, (packed_group_bytes, 4))
    shifts = tl.arange(0, 4) * 2
    packed = tl.sum(
        (quantized & 0x3) << shifts[None, :],
        axis=1,
    ).to(tl.uint8)

    packed_offsets = tl.arange(0, packed_group_bytes)
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    tl.store(
        history_data_ptr + data_base + packed_offsets * stride_data_byte,
        packed,
    )
    scale_base = (
        page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    )
    zero_base = (
        page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    )
    tl.store(history_scale_ptr + scale_base, scale)
    tl.store(history_zero_ptr + zero_base, zero)

    if write_prequant_cache:
        cache_position = logical_position.to(tl.int32)
        set_slot = (cache_position % temporal_set_count) * 2
        cache_slot = set_slot
        resolved = cache_position < 0
        for _ in range(4):
            if not resolved:
                raw_tag0 = tl.load(prequant_cache_tags_ptr + set_slot)
                raw_tag1 = tl.load(prequant_cache_tags_ptr + set_slot + 1)
                valid0 = raw_tag0 >= 0
                valid1 = raw_tag1 >= 0
                hit0 = valid0 & ((raw_tag0 & temporal_position_mask) == cache_position)
                hit1 = valid1 & (raw_tag1 == cache_position)
                hit = hit0 | hit1
                hit_slot = tl.where(hit0, set_slot, set_slot + 1)
                next_victim_way1 = (raw_tag0 & temporal_state_bit) != 0
                selected_way1 = tl.where(
                    ~valid0,
                    False,
                    tl.where(~valid1, True, next_victim_way1),
                )
                selected_slot = set_slot + selected_way1.to(tl.int32)
                expected_tag = tl.where(selected_way1, raw_tag1, raw_tag0)
                replacement_tag = tl.where(
                    selected_way1,
                    cache_position,
                    cache_position | temporal_state_bit,
                )
                claimed = cache_position < 0
                if not hit:
                    observed_tag = tl.atomic_cas(
                        prequant_cache_tags_ptr + selected_slot,
                        expected_tag,
                        replacement_tag,
                    )
                    claimed = observed_tag == expected_tag
                    if claimed & selected_way1:
                        tl.atomic_and(
                            prequant_cache_tags_ptr + set_slot,
                            temporal_position_mask,
                        )
                cache_slot = tl.where(hit, hit_slot, selected_slot)
                resolved = hit | claimed

        prequant_values = tl.load(
            recent_ptr
            + hp_row * stride_recent_row
            + recent_index * stride_recent_token
            + columns * stride_recent_dim,
            mask=valid_recent & resolved & (columns < latent_rank),
            other=0.0,
        )
        tl.store(
            prequant_cache_values_ptr
            + cache_slot * stride_prequant_cache_row
            + columns * stride_prequant_cache_dim,
            prequant_values,
            mask=valid_recent & resolved & (columns < latent_rank),
        )


@triton.jit
def _demotion_ksplit_partial_kernel(
    recent_ptr,
    rotation_ptr,
    partial_ptr,
    logical_positions_ptr,
    hp_rows_ptr,
    page_ids_ptr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    stride_rotation_row: tl.constexpr,
    stride_rotation_col: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_group: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    split_k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute one independent K-slice of one rotated latent group."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    page = tl.load(page_ids_ptr + row)
    logical_position = tl.load(logical_positions_ptr + row)
    hp_row = tl.load(hp_rows_ptr + row)
    valid = (page >= 0) & (hp_row >= 0) & (logical_position >= prefix_tokens)
    recent_index = (logical_position - prefix_tokens) % recent_tokens
    columns = group * group_size + tl.arange(0, group_size)
    m_offsets = tl.arange(0, block_m)
    k_offsets = tl.arange(0, block_k)
    k_base = split * split_k
    accumulator = tl.zeros((block_m, group_size), dtype=tl.float32)

    latent_ptrs = (
        recent_ptr
        + hp_row * stride_recent_row
        + recent_index * stride_recent_token
        + (k_base + k_offsets[None, :]) * stride_recent_dim
    )
    rotation_ptrs = (
        rotation_ptr
        + (k_base + k_offsets[:, None]) * stride_rotation_row
        + columns[None, :] * stride_rotation_col
    )
    for k_start in range(0, split_k, block_k):
        k_mask = k_base + k_start + k_offsets < latent_rank
        latent = tl.load(
            latent_ptrs,
            mask=valid & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        latent = tl.broadcast_to(latent, (block_m, block_k))
        rotation = tl.load(
            rotation_ptrs,
            mask=k_mask[:, None] & (columns[None, :] < latent_rank),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(
            latent,
            rotation,
            accumulator,
            input_precision="ieee",
        )
        latent_ptrs += block_k * stride_recent_dim
        rotation_ptrs += block_k * stride_rotation_row

    values = tl.sum(
        tl.where(m_offsets[:, None] == 0, accumulator, 0.0),
        axis=0,
    )
    output = (
        partial_ptr
        + row * stride_partial_row
        + group * stride_partial_group
        + split * stride_partial_split
        + tl.arange(0, group_size) * stride_partial_dim
    )
    tl.store(output, values)


@triton.jit
def _latent_ksplit_partial_kernel(
    latent_ptr,
    rotation_ptr,
    partial_ptr,
    locations_ptr,
    stride_latent_row: tl.constexpr,
    stride_latent_dim: tl.constexpr,
    stride_rotation_row: tl.constexpr,
    stride_rotation_col: tl.constexpr,
    stride_partial_row: tl.constexpr,
    stride_partial_group: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_dim: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    split_k: tl.constexpr,
    block_m: tl.constexpr,
    block_k: tl.constexpr,
):
    """Compute one K-slice directly from a newly produced latent row."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    split = tl.program_id(2)
    location = tl.load(locations_ptr + row)
    valid = location >= 0
    columns = group * group_size + tl.arange(0, group_size)
    m_offsets = tl.arange(0, block_m)
    k_offsets = tl.arange(0, block_k)
    k_base = split * split_k
    accumulator = tl.zeros((block_m, group_size), dtype=tl.float32)

    latent_ptrs = (
        latent_ptr
        + row * stride_latent_row
        + (k_base + k_offsets[None, :]) * stride_latent_dim
    )
    rotation_ptrs = (
        rotation_ptr
        + (k_base + k_offsets[:, None]) * stride_rotation_row
        + columns[None, :] * stride_rotation_col
    )
    for k_start in range(0, split_k, block_k):
        k_mask = k_base + k_start + k_offsets < latent_rank
        latent = tl.load(
            latent_ptrs,
            mask=valid & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        latent = tl.broadcast_to(latent, (block_m, block_k))
        rotation = tl.load(
            rotation_ptrs,
            mask=k_mask[:, None] & (columns[None, :] < latent_rank),
            other=0.0,
        ).to(tl.float32)
        accumulator = tl.dot(latent, rotation, accumulator, input_precision="ieee")
        latent_ptrs += block_k * stride_latent_dim
        rotation_ptrs += block_k * stride_rotation_row

    values = tl.sum(
        tl.where(m_offsets[:, None] == 0, accumulator, 0.0),
        axis=0,
    )
    output = (
        partial_ptr
        + row * stride_partial_row
        + group * stride_partial_group
        + split * stride_partial_split
        + tl.arange(0, group_size) * stride_partial_dim
    )
    tl.store(output, values)


@triton.jit
def _demotion_ksplit_reduce_quantize_store_kernel(
    partial_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    stride_partial_row: tl.constexpr,
    stride_partial_group: tl.constexpr,
    stride_partial_split: tl.constexpr,
    stride_partial_dim: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    num_splits: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    clip_index: tl.constexpr,
    block_size: tl.constexpr,
    locations_are_slots: tl.constexpr,
):
    """Reduce K-slices, then clip, quantize, pack, and store."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    location = tl.load(page_ids_ptr + row)
    if location < 0:
        return
    if locations_are_slots:
        page = location // block_size
        token_offset = location % block_size
    else:
        page = location
        token_offset = tl.load(page_offsets_ptr + row)
    dims = tl.arange(0, group_size)
    values = tl.zeros((group_size,), dtype=tl.float32)
    base = (
        partial_ptr
        + row * stride_partial_row
        + group * stride_partial_group
        + dims * stride_partial_dim
    )
    for split in range(0, num_splits):
        values += tl.load(base + split * stride_partial_split).to(tl.float32)

    if clip_index >= 0:
        sorted_abs = tl.sort(tl.abs(values))
        threshold = tl.sum(
            tl.where(dims == clip_index, sorted_abs, 0.0),
            axis=0,
        )
        values = tl.minimum(tl.maximum(values, -threshold), threshold)

    value_min = tl.min(values, axis=0)
    value_max = tl.max(values, axis=0)
    scale = tl.maximum(value_max - value_min, 1e-8) / 3.0
    zero = -value_min / scale
    quantized = tl.minimum(
        tl.maximum((values / scale + zero + 0.5).to(tl.int32), 0),
        3,
    )
    quantized = tl.reshape(quantized, (packed_group_bytes, 4))
    shifts = tl.arange(0, 4) * 2
    packed = tl.sum(
        (quantized & 0x3) << shifts[None, :],
        axis=1,
    ).to(tl.uint8)
    packed_offsets = tl.arange(0, packed_group_bytes)
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    tl.store(
        history_data_ptr + data_base + packed_offsets * stride_data_byte,
        packed,
    )
    scale_base = (
        page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    )
    zero_base = (
        page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    )
    tl.store(history_scale_ptr + scale_base, scale)
    tl.store(history_zero_ptr + zero_base, zero)


@triton.jit
def _store_bf16_latent_kernel(
    latent_ptr,
    prefix_ptr,
    recent_ptr,
    logical_positions_ptr,
    final_seq_lens_ptr,
    hp_rows_ptr,
    num_rows,
    stride_latent_row: tl.constexpr,
    stride_latent_dim: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_dim: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    """Store only the final BF16 prefix/recent partition of each token."""
    row = tl.program_id(0)
    if row >= num_rows:
        return
    hp_row = tl.load(hp_rows_ptr + row)
    if hp_row < 0:
        return
    position = tl.load(logical_positions_ptr + row)
    seq_len = tl.load(final_seq_lens_ptr + row)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    is_prefix = position < prefix_tokens
    is_recent = position >= recent_start
    if not (is_prefix or is_recent):
        return

    dims = tl.arange(0, block_d)
    mask = dims < latent_rank
    values = tl.load(
        latent_ptr + row * stride_latent_row + dims * stride_latent_dim,
        mask=mask,
    ).to(tl.bfloat16)
    prefix_base = hp_row * stride_prefix_row + position * stride_prefix_token
    tl.store(
        prefix_ptr + prefix_base + dims * stride_prefix_dim,
        values,
        mask=mask & is_prefix,
    )
    recent_idx = (position - prefix_tokens) % recent_capacity_tokens
    recent_base = hp_row * stride_recent_row + recent_idx * stride_recent_token
    tl.store(
        recent_ptr + recent_base + dims * stride_recent_dim,
        values,
        mask=mask & is_recent & ~is_prefix,
    )


@triton.jit
def _store_rope_kernel(
    rope_values_ptr,
    rope_cache_ptr,
    slot_mapping_ptr,
    num_rows,
    stride_values_row: tl.constexpr,
    stride_values_dim: tl.constexpr,
    stride_cache_block: tl.constexpr,
    stride_cache_token: tl.constexpr,
    stride_cache_dim: tl.constexpr,
    block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    block_d: tl.constexpr,
):
    """Store original-precision RoPE rows through the standard block slots."""
    row = tl.program_id(0)
    if row >= num_rows:
        return
    slot = tl.load(slot_mapping_ptr + row)
    if slot < 0:
        return
    block = slot // block_size
    token_offset = slot % block_size
    dims = tl.arange(0, block_d)
    mask = dims < rope_head_size
    values = tl.load(
        rope_values_ptr + row * stride_values_row + dims * stride_values_dim,
        mask=mask,
    ).to(tl.bfloat16)
    tl.store(
        rope_cache_ptr
        + block * stride_cache_block
        + token_offset * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=mask,
    )


@triton.jit
def _dequantize_history_kernel(
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    output_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    num_rows,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_dim: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
):
    """Dequantize one packed history group into an FP32 oracle buffer."""
    row = tl.program_id(0)
    group = tl.program_id(1)
    if row >= num_rows:
        return
    page = tl.load(page_ids_ptr + row)
    if page < 0:
        return
    token_offset = tl.load(page_offsets_ptr + row)

    dims = tl.arange(0, group_size)
    byte_offsets = dims // 4
    shifts = (dims % 4) * 2
    data_base = (
        page * stride_data_page
        + token_offset * stride_data_token
        + group * packed_group_bytes * stride_data_byte
    )
    packed = tl.load(history_data_ptr + data_base + byte_offsets * stride_data_byte).to(
        tl.int32
    )
    quantized = ((packed >> shifts) & 0x3).to(tl.float32)
    scale = tl.load(
        history_scale_ptr
        + page * stride_scale_page
        + token_offset * stride_scale_token
        + group * stride_scale_group
    ).to(tl.float32)
    zero = tl.load(
        history_zero_ptr
        + page * stride_zero_page
        + token_offset * stride_zero_token
        + group * stride_zero_group
    ).to(tl.float32)
    restored = (quantized - zero) * scale
    tl.store(
        output_ptr
        + row * stride_output_row
        + (group * group_size + dims) * stride_output_dim,
        restored,
    )


def _require_cuda_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    ndim: int,
    dtype: torch.dtype | tuple[torch.dtype, ...],
) -> None:
    dtypes = (dtype,) if isinstance(dtype, torch.dtype) else dtype
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
    if tensor.dtype not in dtypes:
        raise TypeError(f"{name} has unsupported dtype {tensor.dtype}")


def _validate_history_tensors(
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
) -> tuple[int, int, int, int]:
    _require_cuda_tensor(
        history_data,
        name="history_data",
        ndim=3,
        dtype=torch.uint8,
    )
    _require_cuda_tensor(
        history_scale,
        name="history_scale",
        ndim=3,
        dtype=torch.float32,
    )
    _require_cuda_tensor(
        history_zero,
        name="history_zero",
        ndim=3,
        dtype=torch.float32,
    )
    if history_scale.shape != history_zero.shape:
        raise ValueError("history scale/zero shapes must match")
    if history_data.shape[:2] != history_scale.shape[:2]:
        raise ValueError("history data/metadata page geometry must match")
    num_groups = history_scale.shape[2]
    if num_groups <= 0:
        raise ValueError("history cache must contain at least one group")
    packed_bytes = history_data.shape[2]
    if packed_bytes % num_groups:
        raise ValueError("packed latent bytes must divide evenly across groups")
    packed_group_bytes = packed_bytes // num_groups
    group_size = packed_group_bytes * 4
    latent_rank = group_size * num_groups
    return num_groups, group_size, packed_group_bytes, latent_rank


def _validate_indices(
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    num_rows: int,
) -> None:
    for name, tensor in (("page_ids", page_ids), ("page_offsets", page_offsets)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} length must equal the number of rows")


def _clip_index(clip_ratio: float, group_size: int) -> int:
    if not 0 < clip_ratio <= 1:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
    return min(int(clip_ratio * group_size), group_size - 1)


def oscar_mla_rotate(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate shared latent rows with an SM80-compatible Triton matmul."""
    _require_cuda_tensor(
        latent,
        name="latent",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    _require_cuda_tensor(
        rotation,
        name="rotation",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    num_rows, latent_rank = latent.shape
    if rotation.shape != (latent_rank, latent_rank):
        raise ValueError("rotation must be square and match the latent rank")
    if latent.device != rotation.device:
        raise ValueError("latent and rotation must be on the same CUDA device")
    if output is None:
        output = torch.empty(
            (num_rows, latent_rank),
            dtype=torch.float32,
            device=latent.device,
        )
    else:
        _require_cuda_tensor(
            output,
            name="output",
            ndim=2,
            dtype=torch.float32,
        )
        if output.shape != latent.shape or output.device != latent.device:
            raise ValueError("rotation output shape/device must match latent")
    if num_rows == 0:
        return output

    block_m = 16
    block_n = 64
    block_k = 32
    grid = (triton.cdiv(num_rows, block_m) * triton.cdiv(latent_rank, block_n),)
    _rotate_latent_kernel[grid](
        latent,
        rotation,
        output,
        num_rows,
        latent_rank=latent_rank,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_rotation_row=rotation.stride(0),
        stride_rotation_col=rotation.stride(1),
        stride_output_row=output.stride(0),
        stride_output_dim=output.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=4,
        num_stages=2,
    )
    return output


def oscar_mla_quantize_store_history(
    rotated: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    clip_ratio: float,
) -> None:
    """Clip and store already-rotated shared latent rows as grouped INT2."""
    _require_cuda_tensor(
        rotated,
        name="rotated",
        ndim=2,
        dtype=torch.float32,
    )
    num_groups, group_size, packed_group_bytes, latent_rank = _validate_history_tensors(
        history_data, history_scale, history_zero
    )
    num_rows = rotated.shape[0]
    if rotated.shape[1] != latent_rank:
        raise ValueError("rotated latent rank does not match history cache geometry")
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    if not (
        rotated.device
        == history_data.device
        == history_scale.device
        == history_zero.device
        == page_ids.device
        == page_offsets.device
    ):
        raise ValueError("all OSCAR MLA history tensors must share one CUDA device")
    if num_rows == 0:
        return

    _quantize_store_history_kernel[(num_rows, num_groups)](
        rotated,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        num_rows,
        stride_rotated_row=rotated.stride(0),
        stride_rotated_dim=rotated.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        num_groups=num_groups,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        clip_index=_clip_index(clip_ratio, group_size),
        num_warps=4,
        num_stages=1,
    )


def oscar_mla_rotate_quantize_store(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    clip_ratio: float,
    rotated: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rotate, clip, INT2-pack and store shared latent history rows."""
    rotated = oscar_mla_rotate(latent, rotation, output=rotated)
    oscar_mla_quantize_store_history(
        rotated,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        clip_ratio=clip_ratio,
    )
    return rotated


def oscar_mla_rotate_quantize_store_decode(
    latent: torch.Tensor,
    rotation: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    clip_ratio: float,
    partial_workspace: torch.Tensor,
) -> None:
    """Fused K-split canonical INT2 store for one decode latent row."""
    _require_cuda_tensor(
        latent,
        name="latent",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    _require_cuda_tensor(
        rotation,
        name="rotation",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    num_groups, group_size, packed_group_bytes, latent_rank = _validate_history_tensors(
        history_data, history_scale, history_zero
    )
    _require_cuda_tensor(
        slot_mapping,
        name="slot_mapping",
        ndim=1,
        dtype=(torch.int32, torch.int64),
    )
    if slot_mapping.shape[0] != latent.shape[0]:
        raise ValueError("slot_mapping length must equal the number of rows")
    expected_shape = (
        1,
        _OSCAR_DEMOTION_KSPLIT_GROUPS,
        _OSCAR_DEMOTION_KSPLIT_SPLITS,
        _OSCAR_DEMOTION_KSPLIT_GROUP_SIZE,
    )
    if latent.shape != (1, latent_rank) or latent_rank != 512 or group_size != 128:
        raise ValueError("decode K-split requires one D512 row and group size 128")
    if rotation.shape != (latent_rank, latent_rank):
        raise ValueError("rotation must be square and match the latent rank")
    if (
        partial_workspace.shape != expected_shape
        or partial_workspace.dtype != torch.float32
    ):
        raise ValueError(f"partial_workspace must be FP32 with shape {expected_shape}")
    if not partial_workspace.is_contiguous():
        raise ValueError("partial_workspace must be contiguous")
    tensors = (
        latent,
        rotation,
        history_data,
        history_scale,
        history_zero,
        slot_mapping,
        partial_workspace,
    )
    if any(tensor.device != latent.device for tensor in tensors):
        raise ValueError("all OSCAR MLA decode-store tensors must share one device")

    _latent_ksplit_partial_kernel[(1, num_groups, _OSCAR_DEMOTION_KSPLIT_SPLITS)](
        latent,
        rotation,
        partial_workspace,
        slot_mapping,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_rotation_row=rotation.stride(0),
        stride_rotation_col=rotation.stride(1),
        stride_partial_row=partial_workspace.stride(0),
        stride_partial_group=partial_workspace.stride(1),
        stride_partial_split=partial_workspace.stride(2),
        stride_partial_dim=partial_workspace.stride(3),
        latent_rank=latent_rank,
        group_size=group_size,
        split_k=latent_rank // _OSCAR_DEMOTION_KSPLIT_SPLITS,
        block_m=16,
        block_k=64,
        num_warps=2,
        num_stages=2,
    )
    _demotion_ksplit_reduce_quantize_store_kernel[(1, num_groups)](
        partial_workspace,
        history_data,
        history_scale,
        history_zero,
        slot_mapping,
        slot_mapping,
        stride_partial_row=partial_workspace.stride(0),
        stride_partial_group=partial_workspace.stride(1),
        stride_partial_split=partial_workspace.stride(2),
        stride_partial_dim=partial_workspace.stride(3),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        num_splits=_OSCAR_DEMOTION_KSPLIT_SPLITS,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        clip_index=_clip_index(clip_ratio, group_size),
        block_size=history_data.shape[1],
        locations_are_slots=True,
        num_warps=1,
        num_stages=1,
    )


def oscar_mla_store_bf16(
    latent: torch.Tensor,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    logical_positions: torch.Tensor,
    final_seq_lens: torch.Tensor,
    hp_rows: torch.Tensor,
    recent_tokens: int | None = None,
) -> None:
    """Write rows belonging to the final BF16 prefix/recent partition."""
    _require_cuda_tensor(
        latent,
        name="latent",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    for name, tensor in (("prefix", prefix), ("recent", recent)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=3,
            dtype=torch.bfloat16,
        )
    num_rows, latent_rank = latent.shape
    if prefix.shape[0] != recent.shape[0]:
        raise ValueError("prefix/recent row capacities must match")
    if prefix.shape[2] != latent_rank or recent.shape[2] != latent_rank:
        raise ValueError("BF16 cache latent rank must match input")
    if prefix.shape[1] <= 0 or recent.shape[1] <= 0:
        raise ValueError("BF16 prefix/recent windows must be positive")
    if recent_tokens is None:
        recent_tokens = recent.shape[1]
    if not 0 < recent_tokens <= recent.shape[1]:
        raise ValueError("logical recent window must fit the physical recent pool")
    for name, tensor in (
        ("logical_positions", logical_positions),
        ("final_seq_lens", final_seq_lens),
        ("hp_rows", hp_rows),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} length must equal the number of latent rows")
    if not (
        latent.device
        == prefix.device
        == recent.device
        == logical_positions.device
        == final_seq_lens.device
        == hp_rows.device
    ):
        raise ValueError("all OSCAR MLA BF16 tensors must share one CUDA device")
    if num_rows == 0:
        return

    block_d = triton.next_power_of_2(latent_rank)
    _store_bf16_latent_kernel[(num_rows,)](
        latent,
        prefix,
        recent,
        logical_positions,
        final_seq_lens,
        hp_rows,
        num_rows,
        stride_latent_row=latent.stride(0),
        stride_latent_dim=latent.stride(1),
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_dim=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_dim=recent.stride(2),
        prefix_tokens=prefix.shape[1],
        recent_tokens=recent_tokens,
        recent_capacity_tokens=recent.shape[1],
        latent_rank=latent_rank,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )


def oscar_mla_store_rope(
    rope_values: torch.Tensor,
    rope_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Store BF16 RoPE keys using the standard vLLM physical slot mapping."""
    if rope_values.ndim == 3:
        if rope_values.shape[1] != 1:
            raise ValueError("OSCAR MLA RoPE values must have one KV head")
        rope_values = rope_values[:, 0, :]
    _require_cuda_tensor(
        rope_values,
        name="rope_values",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    _require_cuda_tensor(
        rope_cache,
        name="rope_cache",
        ndim=3,
        dtype=torch.bfloat16,
    )
    _require_cuda_tensor(
        slot_mapping,
        name="slot_mapping",
        ndim=1,
        dtype=(torch.int32, torch.int64),
    )
    num_rows, rope_head_size = rope_values.shape
    if slot_mapping.shape[0] != num_rows:
        raise ValueError("slot_mapping length must equal the number of RoPE rows")
    if rope_cache.shape[2] != rope_head_size:
        raise ValueError("RoPE cache head size does not match input")
    if rope_cache.shape[0] <= 0 or rope_cache.shape[1] <= 0:
        raise ValueError("RoPE cache must contain at least one non-empty block")
    if not (rope_values.device == rope_cache.device == slot_mapping.device):
        raise ValueError("all OSCAR MLA RoPE tensors must share one CUDA device")
    if num_rows == 0:
        return

    _store_rope_kernel[(num_rows,)](
        rope_values,
        rope_cache,
        slot_mapping,
        num_rows,
        stride_values_row=rope_values.stride(0),
        stride_values_dim=rope_values.stride(1),
        stride_cache_block=rope_cache.stride(0),
        stride_cache_token=rope_cache.stride(1),
        stride_cache_dim=rope_cache.stride(2),
        block_size=rope_cache.shape[1],
        rope_head_size=rope_head_size,
        block_d=triton.next_power_of_2(rope_head_size),
        num_warps=1,
        num_stages=1,
    )


def oscar_mla_demote_recent(
    recent: torch.Tensor,
    rotation: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    logical_positions: torch.Tensor,
    hp_rows: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    prefix_tokens: int,
    clip_ratio: float,
    partial_workspace: torch.Tensor | None = None,
    prequant_temporal_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    temporal_two_way: bool = False,
) -> None:
    """Rotate recent rows and store them directly as packed INT2 history."""
    _require_cuda_tensor(
        recent,
        name="recent",
        ndim=3,
        dtype=torch.bfloat16,
    )
    if prefix_tokens <= 0:
        raise ValueError("prefix_tokens must be positive")
    num_rows = logical_positions.shape[0]
    for name, tensor in (
        ("logical_positions", logical_positions),
        ("hp_rows", hp_rows),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_rows:
            raise ValueError(f"{name} lengths must match")
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    num_groups, group_size, packed_group_bytes, history_rank = (
        _validate_history_tensors(history_data, history_scale, history_zero)
    )
    latent_rank = recent.shape[2]
    _require_cuda_tensor(
        rotation,
        name="rotation",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    if rotation.shape != (latent_rank, latent_rank):
        raise ValueError("rotation must be square and match the latent rank")
    if history_rank != latent_rank:
        raise ValueError("recent latent rank does not match history cache geometry")
    if not (
        recent.device
        == rotation.device
        == history_data.device
        == history_scale.device
        == history_zero.device
        == logical_positions.device
        == hp_rows.device
        == page_ids.device
        == page_offsets.device
    ):
        raise ValueError("all OSCAR MLA demotion tensors must share one CUDA device")
    if num_rows == 0:
        return

    if temporal_two_way and prequant_temporal_cache is None:
        raise ValueError("temporal_two_way requires a prequant temporal cache")
    if prequant_temporal_cache is not None and not temporal_two_way:
        raise ValueError("prequant temporal cache requires temporal_two_way")
    if prequant_temporal_cache is not None:
        prequant_cache_values, prequant_cache_tags = prequant_temporal_cache
        _require_cuda_tensor(
            prequant_cache_values,
            name="prequant_cache_values",
            ndim=2,
            dtype=torch.bfloat16,
        )
        _require_cuda_tensor(
            prequant_cache_tags,
            name="prequant_cache_tags",
            ndim=1,
            dtype=torch.int32,
        )
        if (
            prequant_cache_values.shape
            != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, latent_rank)
            or prequant_cache_values.stride() != (latent_rank, 1)
            or prequant_cache_tags.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        ):
            raise ValueError("prequant temporal cache geometry is invalid")
        if not (
            prequant_cache_values.device == prequant_cache_tags.device == recent.device
        ):
            raise ValueError("prequant temporal cache must share the demotion device")
        if num_rows > 6 or latent_rank != 512 or group_size != 128:
            raise ValueError(
                "prequant temporal cache requires at most six D512/G128 rows"
            )
        if partial_workspace is not None:
            raise ValueError("prequant temporal cache is incompatible with K-split")
    else:
        prequant_cache_values = recent
        prequant_cache_tags = logical_positions

    if partial_workspace is not None:
        _require_cuda_tensor(
            partial_workspace,
            name="partial_workspace",
            ndim=4,
            dtype=torch.float32,
        )
        expected_shape = (
            1,
            _OSCAR_DEMOTION_KSPLIT_GROUPS,
            _OSCAR_DEMOTION_KSPLIT_SPLITS,
            _OSCAR_DEMOTION_KSPLIT_GROUP_SIZE,
        )
        if partial_workspace.shape != expected_shape:
            raise ValueError(f"partial_workspace must have shape {expected_shape}")
        if not partial_workspace.is_contiguous():
            raise ValueError("partial_workspace must be contiguous")
        if partial_workspace.device != recent.device:
            raise ValueError("partial_workspace must share the demotion device")
        if num_rows != 1 or latent_rank != 512 or group_size != 128:
            raise ValueError(
                "demotion K-split requires one row, latent rank 512, and group size 128"
            )
        _demotion_ksplit_partial_kernel[
            (num_rows, num_groups, _OSCAR_DEMOTION_KSPLIT_SPLITS)
        ](
            recent,
            rotation,
            partial_workspace,
            logical_positions,
            hp_rows,
            page_ids,
            stride_recent_row=recent.stride(0),
            stride_recent_token=recent.stride(1),
            stride_recent_dim=recent.stride(2),
            stride_rotation_row=rotation.stride(0),
            stride_rotation_col=rotation.stride(1),
            stride_partial_row=partial_workspace.stride(0),
            stride_partial_group=partial_workspace.stride(1),
            stride_partial_split=partial_workspace.stride(2),
            stride_partial_dim=partial_workspace.stride(3),
            prefix_tokens=prefix_tokens,
            recent_tokens=recent.shape[1],
            latent_rank=latent_rank,
            group_size=group_size,
            split_k=latent_rank // _OSCAR_DEMOTION_KSPLIT_SPLITS,
            block_m=16,
            block_k=64,
            num_warps=2,
            num_stages=2,
        )
        _demotion_ksplit_reduce_quantize_store_kernel[(num_rows, num_groups)](
            partial_workspace,
            history_data,
            history_scale,
            history_zero,
            page_ids,
            page_offsets,
            stride_partial_row=partial_workspace.stride(0),
            stride_partial_group=partial_workspace.stride(1),
            stride_partial_split=partial_workspace.stride(2),
            stride_partial_dim=partial_workspace.stride(3),
            stride_data_page=history_data.stride(0),
            stride_data_token=history_data.stride(1),
            stride_data_byte=history_data.stride(2),
            stride_scale_page=history_scale.stride(0),
            stride_scale_token=history_scale.stride(1),
            stride_scale_group=history_scale.stride(2),
            stride_zero_page=history_zero.stride(0),
            stride_zero_token=history_zero.stride(1),
            stride_zero_group=history_zero.stride(2),
            num_splits=_OSCAR_DEMOTION_KSPLIT_SPLITS,
            group_size=group_size,
            packed_group_bytes=packed_group_bytes,
            clip_index=_clip_index(clip_ratio, group_size),
            block_size=1,
            locations_are_slots=False,
            num_warps=1,
            num_stages=1,
        )
        return

    _demote_recent_rotate_quantize_store_kernel[(num_rows, num_groups)](
        recent,
        rotation,
        history_data,
        history_scale,
        history_zero,
        logical_positions,
        hp_rows,
        page_ids,
        page_offsets,
        prequant_cache_values,
        prequant_cache_tags,
        num_rows,
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_dim=recent.stride(2),
        stride_rotation_row=rotation.stride(0),
        stride_rotation_col=rotation.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_prequant_cache_row=prequant_cache_values.stride(-2),
        stride_prequant_cache_dim=prequant_cache_values.stride(-1),
        prefix_tokens=prefix_tokens,
        recent_tokens=recent.shape[1],
        latent_rank=latent_rank,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        clip_index=_clip_index(clip_ratio, group_size),
        block_m=16,
        block_k=64,
        write_prequant_cache=prequant_temporal_cache is not None,
        temporal_set_count=OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT,
        temporal_state_bit=OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT,
        temporal_position_mask=OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK,
        num_warps=2,
        num_stages=2,
    )


def oscar_mla_dequantize_history(
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize selected history slots to an FP32 oracle tensor."""
    num_groups, group_size, packed_group_bytes, latent_rank = _validate_history_tensors(
        history_data, history_scale, history_zero
    )
    num_rows = page_ids.shape[0]
    _validate_indices(page_ids, page_offsets, num_rows=num_rows)
    if output is None:
        output = torch.empty(
            (num_rows, latent_rank),
            dtype=torch.float32,
            device=history_data.device,
        )
    else:
        _require_cuda_tensor(
            output,
            name="output",
            ndim=2,
            dtype=torch.float32,
        )
        if output.shape != (num_rows, latent_rank):
            raise ValueError("history dequant output shape does not match cache")
    if not (
        history_data.device
        == history_scale.device
        == history_zero.device
        == page_ids.device
        == page_offsets.device
        == output.device
    ):
        raise ValueError("all OSCAR MLA dequant tensors must share one CUDA device")
    if num_rows == 0:
        return output

    _dequantize_history_kernel[(num_rows, num_groups)](
        history_data,
        history_scale,
        history_zero,
        output,
        page_ids,
        page_offsets,
        num_rows,
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_output_row=output.stride(0),
        stride_output_dim=output.stride(1),
        num_groups=num_groups,
        group_size=group_size,
        packed_group_bytes=packed_group_bytes,
        num_warps=4,
        num_stages=1,
    )
    return output
