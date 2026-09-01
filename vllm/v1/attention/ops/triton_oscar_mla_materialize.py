# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialize OSCAR three-pool MLA rows for the BF16 sparse read path."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

OSCAR_BF16_MATERIALIZATION_MAX_ROWS = 32769
OSCAR_MTP_TEMPORAL_CACHE_CAPACITY = 8192
OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT = 4096
OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT = 1 << 30
OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK = (1 << 30) - 1
_OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_BITS = 14
_OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_MASK = (1 << 14) - 1
OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY = 32768
OSCAR_MTP_TEMPORAL_MAX_POSITIONS = 65536
OSCAR_MTP_TEMPORAL_MAX_ROWS = 6 * 2048
_workspace_cache: dict[
    tuple[str, int | None, int],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}
_mtp_temporal_workspace_cache: dict[
    tuple[str, int | None],
    tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
] = {}


@triton.jit
def _gather_oscar_mla_rows_kernel(
    positions_ptr,
    prefix_ptr,
    recent_ptr,
    rope_ptr,
    rope_block_table_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    history_rotated_ptr,
    history_mask_ptr,
    output_kv_ptr,
    remapped_indices_ptr,
    num_rows,
    row_offset,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_table_b: tl.constexpr,
    stride_rope_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_history_table_b: tl.constexpr,
    stride_history_table_page: tl.constexpr,
    stride_history_row: tl.constexpr,
    stride_history_d: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_d: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    block_d: tl.constexpr,
    use_positions: tl.constexpr,
):
    row = tl.program_id(0)
    dim_block = tl.program_id(1)
    dims = dim_block * block_d + tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    row_valid = row < num_rows
    logical_position = row + row_offset
    if use_positions:
        logical_position = tl.load(
            positions_ptr + row,
            mask=row_valid,
            other=-1,
        )
    seq_len = tl.load(seq_lens_ptr)
    hp_row = tl.load(hp_rows_ptr)
    valid = (
        row_valid
        & (logical_position >= 0)
        & (logical_position < seq_len)
        & (hp_row >= 0)
    )
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    is_prefix = valid & (logical_position < prefix_tokens)
    is_recent = valid & (logical_position >= recent_start)
    is_history = valid & ~is_prefix & ~is_recent

    prefix_values = tl.load(
        prefix_ptr
        + hp_row * stride_prefix_row
        + logical_position * stride_prefix_token
        + dims * stride_prefix_d,
        mask=is_prefix & dim_mask,
        other=0.0,
    )
    recent_position = (logical_position - prefix_tokens) % recent_capacity_tokens
    recent_values = tl.load(
        recent_ptr
        + hp_row * stride_recent_row
        + recent_position * stride_recent_token
        + dims * stride_recent_d,
        mask=is_recent & dim_mask,
        other=0.0,
    )
    bf16_values = tl.where(is_prefix, prefix_values, recent_values)
    tl.store(
        output_kv_ptr + row * stride_output_row + dims * stride_output_d,
        tl.where(is_prefix | is_recent, bf16_values, 0.0),
        mask=row_valid & dim_mask,
    )

    history_index = logical_position - prefix_tokens
    logical_page = history_index // history_block_size
    page_offset = history_index % history_block_size
    physical_page = tl.load(
        history_page_table_ptr + logical_page * stride_history_table_page,
        mask=is_history,
        other=0,
    )
    packed = tl.load(
        history_data_ptr
        + physical_page * stride_data_page
        + page_offset * stride_data_token
        + (dims // 4) * stride_data_byte,
        mask=is_history & dim_mask,
        other=0,
    ).to(tl.int32)
    quantized = ((packed >> ((dims % 4) * 2)) & 0x3).to(tl.float32)
    group = dims // group_size
    scale = tl.load(
        history_scale_ptr
        + physical_page * stride_scale_page
        + page_offset * stride_scale_token
        + group * stride_scale_group,
        mask=is_history & dim_mask,
        other=0.0,
    ).to(tl.float32)
    zero = tl.load(
        history_zero_ptr
        + physical_page * stride_zero_page
        + page_offset * stride_zero_token
        + group * stride_zero_group,
        mask=is_history & dim_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        history_rotated_ptr + row * stride_history_row + dims * stride_history_d,
        tl.where(is_history, (quantized - zero) * scale, 0.0),
        mask=row_valid & dim_mask,
    )

    if dim_block == 0:
        rope_dims = tl.arange(0, 64)
        rope_logical_page = logical_position // rope_block_size
        rope_page_offset = logical_position % rope_block_size
        rope_physical_page = tl.load(
            rope_block_table_ptr + rope_logical_page * stride_rope_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + rope_physical_page * stride_rope_block
            + rope_page_offset * stride_rope_token
            + rope_dims * stride_rope_d,
            mask=valid & (rope_dims < rope_head_size),
            other=0.0,
        )
        tl.store(
            output_kv_ptr
            + row * stride_output_row
            + (latent_rank + rope_dims) * stride_output_d,
            rope_values,
            mask=row_valid & (rope_dims < rope_head_size),
        )
        tl.store(history_mask_ptr + row, is_history.to(tl.uint8), mask=row_valid)
        tl.store(
            remapped_indices_ptr + row,
            tl.where(valid, row, -1),
            mask=row_valid,
        )


@triton.jit
def _gather_canonical_restore_kernel(
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    page_ids_ptr,
    page_offsets_ptr,
    output_ptr,
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
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, block_d)
    valid = row < num_rows
    dim_mask = dims < latent_rank
    page = tl.load(page_ids_ptr + row, mask=valid, other=0)
    page_offset = tl.load(page_offsets_ptr + row, mask=valid, other=0)
    packed = tl.load(
        history_data_ptr
        + page * stride_data_page
        + page_offset * stride_data_token
        + (dims // 4) * stride_data_byte,
        mask=valid & dim_mask,
        other=0,
    ).to(tl.int32)
    quantized = ((packed >> ((dims % 4) * 2)) & 0x3).to(tl.float32)
    group = dims // group_size
    scale = tl.load(
        history_scale_ptr
        + page * stride_scale_page
        + page_offset * stride_scale_token
        + group * stride_scale_group,
        mask=valid & dim_mask,
        other=0.0,
    ).to(tl.float32)
    zero = tl.load(
        history_zero_ptr
        + page * stride_zero_page
        + page_offset * stride_zero_token
        + group * stride_zero_group,
        mask=valid & dim_mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + dims * stride_output_dim,
        (quantized - zero) * scale,
        mask=valid & dim_mask,
    )


@triton.jit
def _scatter_restored_hp_kernel(
    restored_ptr,
    positions_ptr,
    hp_rows_ptr,
    prefix_ptr,
    recent_ptr,
    num_rows,
    stride_restored_row: tl.constexpr,
    stride_restored_dim: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_dim: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_dim: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    latent_rank: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    dims = tl.arange(0, block_d)
    valid = row < num_rows
    position = tl.load(positions_ptr + row, mask=valid, other=-1)
    hp_row = tl.load(hp_rows_ptr + row, mask=valid, other=-1)
    values = tl.load(
        restored_ptr + row * stride_restored_row + dims * stride_restored_dim,
        mask=valid & (dims < latent_rank),
        other=0.0,
    )
    is_prefix = valid & (hp_row >= 0) & (position >= 0) & (position < prefix_tokens)
    tl.store(
        prefix_ptr
        + hp_row * stride_prefix_row
        + position * stride_prefix_token
        + dims * stride_prefix_dim,
        values,
        mask=is_prefix & (dims < latent_rank),
    )
    recent_position = (position - prefix_tokens) % recent_tokens
    is_recent = valid & (hp_row >= 0) & (position >= prefix_tokens)
    tl.store(
        recent_ptr
        + hp_row * stride_recent_row
        + recent_position * stride_recent_token
        + dims * stride_recent_dim,
        values,
        mask=is_recent & (dims < latent_rank),
    )


@triton.jit
def _seed_oscar_mtp_temporal_rows_kernel(
    values_ptr,
    positions_ptr,
    valid_rows_ptr,
    cache_values_ptr,
    cache_tags_ptr,
    stride_values_row: tl.constexpr,
    stride_values_d: tl.constexpr,
    stride_cache_row: tl.constexpr,
    stride_cache_d: tl.constexpr,
    num_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    latent_rank: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    valid = row_valid & tl.load(valid_rows_ptr + rows, mask=row_valid, other=0).to(
        tl.int1
    )
    slots = positions % cache_capacity
    values = tl.load(
        values_ptr
        + rows[:, None] * stride_values_row
        + dims[None, :] * stride_values_d,
        mask=valid[:, None] & (dims[None, :] < latent_rank),
        other=0.0,
    )
    tl.store(
        cache_values_ptr
        + slots[:, None] * stride_cache_row
        + dims[None, :] * stride_cache_d,
        values,
        mask=valid[:, None] & (dims[None, :] < latent_rank),
    )
    if tl.program_id(1) == 0:
        tl.store(cache_tags_ptr + slots, positions, mask=valid)


@triton.jit
def _seed_oscar_mtp_temporal_recent_kernel(
    recent_ptr,
    positions_ptr,
    hp_rows_ptr,
    page_ids_ptr,
    cache_values_ptr,
    cache_tags_ptr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_cache_row: tl.constexpr,
    stride_cache_d: tl.constexpr,
    num_rows: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_capacity: tl.constexpr,
    cache_capacity: tl.constexpr,
    latent_rank: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    hp_rows = tl.load(hp_rows_ptr + rows, mask=row_valid, other=-1)
    page_ids = tl.load(page_ids_ptr + rows, mask=row_valid, other=-1)
    valid = row_valid & (positions >= prefix_tokens) & (hp_rows >= 0) & (page_ids >= 0)
    recent_slots = (positions - prefix_tokens) % recent_capacity
    cache_slots = positions % cache_capacity
    values = tl.load(
        recent_ptr
        + hp_rows[:, None] * stride_recent_row
        + recent_slots[:, None] * stride_recent_token
        + dims[None, :] * stride_recent_d,
        mask=valid[:, None] & (dims[None, :] < latent_rank),
        other=0.0,
    )
    tl.store(
        cache_values_ptr
        + cache_slots[:, None] * stride_cache_row
        + dims[None, :] * stride_cache_d,
        values,
        mask=valid[:, None] & (dims[None, :] < latent_rank),
    )
    if tl.program_id(1) == 0:
        tl.store(cache_tags_ptr + cache_slots, positions, mask=valid)


@triton.jit
def _mark_oscar_mtp_temporal_miss_owners_kernel(
    positions_ptr,
    cache_tags_ptr,
    position_owners_ptr,
    cache_slot_owners_ptr,
    miss_flags_ptr,
    seq_lens_ptr,
    num_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    max_positions: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    block_size: tl.constexpr,
):
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    seq_len = tl.load(seq_lens_ptr)
    valid = (
        row_valid
        & (positions >= 0)
        & (positions < seq_len)
        & (positions < max_positions)
    )
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    cacheable = valid & (positions >= prefix_tokens) & (positions < recent_start)
    slots = positions % cache_capacity
    tags = tl.load(cache_tags_ptr + slots, mask=valid, other=-1)
    hits = cacheable & (tags == positions)
    misses = valid & ~hits
    tl.store(miss_flags_ptr + rows, misses.to(tl.uint8), mask=row_valid)
    tl.atomic_min(position_owners_ptr + positions, rows, mask=misses)
    tl.atomic_max(cache_slot_owners_ptr + slots, rows, mask=cacheable)


@triton.jit
def _mark_oscar_mtp_temporal_two_way_miss_owners_kernel(
    positions_ptr,
    cache_tags_ptr,
    position_owners_ptr,
    selected_slots_ptr,
    cache_slot_owners_ptr,
    miss_flags_ptr,
    seq_lens_ptr,
    num_rows: tl.constexpr,
    set_count: tl.constexpr,
    state_bit: tl.constexpr,
    position_mask: tl.constexpr,
    max_positions: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    block_size: tl.constexpr,
):
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    seq_len = tl.load(seq_lens_ptr)
    valid = (
        row_valid
        & (positions >= 0)
        & (positions < seq_len)
        & (positions < max_positions)
    )
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    cacheable = valid & (positions >= prefix_tokens) & (positions < recent_start)
    set_slots = (positions % set_count) * 2
    raw_tag0 = tl.load(cache_tags_ptr + set_slots, mask=valid, other=-1)
    raw_tag1 = tl.load(cache_tags_ptr + set_slots + 1, mask=valid, other=-1)
    valid0 = raw_tag0 >= 0
    valid1 = raw_tag1 >= 0
    tag0 = raw_tag0 & position_mask
    hit0 = cacheable & valid0 & (tag0 == positions)
    hit1 = cacheable & valid1 & (raw_tag1 == positions)
    hits = hit0 | hit1
    next_victim_way1 = (raw_tag0 & state_bit) != 0
    selected_way = tl.where(
        ~valid0,
        0,
        tl.where(~valid1, 1, next_victim_way1.to(tl.int32)),
    )
    selected_slots = set_slots + selected_way
    hit_slots = tl.where(hit0, set_slots, set_slots + 1)
    owner_slots = tl.where(hits, hit_slots, selected_slots)
    misses = valid & ~hits
    tl.store(miss_flags_ptr + rows, misses.to(tl.uint8), mask=row_valid)
    tl.store(selected_slots_ptr + positions, selected_slots, mask=misses)
    tl.atomic_min(position_owners_ptr + positions, rows, mask=misses)
    tl.atomic_max(cache_slot_owners_ptr + owner_slots, rows, mask=cacheable)


@triton.jit
def _assign_oscar_mtp_temporal_unique_misses_kernel(
    positions_ptr,
    position_owners_ptr,
    miss_flags_ptr,
    miss_count_ptr,
    miss_positions_ptr,
    position_to_miss_ptr,
    num_rows: tl.constexpr,
    block_size: tl.constexpr,
):
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=0)
    misses = tl.load(miss_flags_ptr + rows, mask=row_valid, other=0).to(tl.int1)
    owners = tl.load(position_owners_ptr + positions, mask=misses, other=-1)
    unique_owner = misses & (owners == rows)
    count_ptrs = miss_count_ptr + tl.zeros((block_size,), tl.int32)
    miss_ids = tl.atomic_add(count_ptrs, 1, mask=unique_owner)
    tl.store(miss_positions_ptr + miss_ids, positions, mask=unique_owner)
    tl.store(position_to_miss_ptr + positions, miss_ids, mask=unique_owner)


@triton.jit
def _assign_oscar_mtp_temporal_two_way_unique_misses_kernel(
    positions_ptr,
    position_owners_ptr,
    miss_flags_ptr,
    miss_count_ptr,
    miss_positions_ptr,
    position_to_miss_ptr,
    num_rows: tl.constexpr,
    miss_id_bits: tl.constexpr,
    block_size: tl.constexpr,
):
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=0)
    misses = tl.load(miss_flags_ptr + rows, mask=row_valid, other=0).to(tl.int1)
    owners = tl.load(position_owners_ptr + positions, mask=misses, other=-1)
    unique_owner = misses & (owners == rows)
    selected_slots = tl.load(
        position_to_miss_ptr + positions,
        mask=unique_owner,
        other=0,
    )
    count_ptrs = miss_count_ptr + tl.zeros((block_size,), tl.int32)
    miss_ids = tl.atomic_add(count_ptrs, 1, mask=unique_owner)
    packed = (selected_slots << miss_id_bits) | miss_ids
    tl.store(miss_positions_ptr + miss_ids, positions, mask=unique_owner)
    tl.store(position_to_miss_ptr + positions, packed, mask=unique_owner)


@triton.jit
def _gather_oscar_mtp_temporal_unique_misses_kernel(
    miss_positions_ptr,
    miss_count_ptr,
    prefix_ptr,
    recent_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    history_rotated_ptr,
    output_ptr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_history_table_page: tl.constexpr,
    stride_history_row: tl.constexpr,
    stride_history_d: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_d: tl.constexpr,
    max_rows: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    active_rows = tl.load(miss_count_ptr)
    if row < active_rows:
        dims = tl.arange(0, block_d)
        dim_valid = dims < latent_rank
        position = tl.load(miss_positions_ptr + row)
        seq_len = tl.load(seq_lens_ptr)
        hp_row = tl.load(hp_rows_ptr)
        valid = (position >= 0) & (position < seq_len) & (hp_row >= 0)
        recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
        is_prefix = valid & (position < prefix_tokens)
        is_recent = valid & (position >= recent_start)
        is_history = valid & ~is_prefix & ~is_recent
        prefix_values = tl.load(
            prefix_ptr
            + hp_row * stride_prefix_row
            + position * stride_prefix_token
            + dims * stride_prefix_d,
            mask=is_prefix & dim_valid,
            other=0.0,
        )
        recent_position = (position - prefix_tokens) % recent_capacity_tokens
        recent_values = tl.load(
            recent_ptr
            + hp_row * stride_recent_row
            + recent_position * stride_recent_token
            + dims * stride_recent_d,
            mask=is_recent & dim_valid,
            other=0.0,
        )
        high_precision = tl.where(is_prefix, prefix_values, recent_values)
        tl.store(
            output_ptr + row * stride_output_row + dims * stride_output_d,
            tl.where(is_prefix | is_recent, high_precision, 0.0),
            mask=dim_valid,
        )
        history_index = position - prefix_tokens
        logical_page = history_index // history_block_size
        page_offset = history_index % history_block_size
        physical_page = tl.load(
            history_page_table_ptr + logical_page * stride_history_table_page,
            mask=is_history,
            other=0,
        )
        packed = tl.load(
            history_data_ptr
            + physical_page * stride_data_page
            + page_offset * stride_data_token
            + (dims // 4) * stride_data_byte,
            mask=is_history & dim_valid,
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> ((dims % 4) * 2)) & 0x3).to(tl.float32)
        group = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_page * stride_scale_page
            + page_offset * stride_scale_token
            + group * stride_scale_group,
            mask=is_history & dim_valid,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_page * stride_zero_page
            + page_offset * stride_zero_token
            + group * stride_zero_group,
            mask=is_history & dim_valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            history_rotated_ptr + row * stride_history_row + dims * stride_history_d,
            tl.where(is_history, (quantized - zero) * scale, 0.0),
            mask=dim_valid,
        )


@triton.jit
def _gather_oscar_mtp_direct_unique_misses_kernel(
    miss_positions_ptr,
    miss_count_ptr,
    prefix_ptr,
    recent_ptr,
    rope_ptr,
    rope_block_table_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    history_rotated_ptr,
    direct_values_ptr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_history_table_page: tl.constexpr,
    stride_history_row: tl.constexpr,
    stride_history_d: tl.constexpr,
    stride_direct_row: tl.constexpr,
    stride_direct_d: tl.constexpr,
    max_rows: tl.constexpr,
    direct_cache_capacity: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    block_d: tl.constexpr,
):
    """Gather each unique miss directly into the cache's compact tail."""
    row = tl.program_id(0)
    active_rows = tl.load(miss_count_ptr)
    if row < active_rows:
        dims = tl.arange(0, block_d)
        dim_valid = dims < latent_rank
        position = tl.load(miss_positions_ptr + row)
        seq_len = tl.load(seq_lens_ptr)
        hp_row = tl.load(hp_rows_ptr)
        valid = (position >= 0) & (position < seq_len) & (hp_row >= 0)
        recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
        is_prefix = valid & (position < prefix_tokens)
        is_recent = valid & (position >= recent_start)
        is_history = valid & ~is_prefix & ~is_recent
        prefix_values = tl.load(
            prefix_ptr
            + hp_row * stride_prefix_row
            + position * stride_prefix_token
            + dims * stride_prefix_d,
            mask=is_prefix & dim_valid,
            other=0.0,
        )
        recent_position = (position - prefix_tokens) % recent_capacity_tokens
        recent_values = tl.load(
            recent_ptr
            + hp_row * stride_recent_row
            + recent_position * stride_recent_token
            + dims * stride_recent_d,
            mask=is_recent & dim_valid,
            other=0.0,
        )
        high_precision = tl.where(is_prefix, prefix_values, recent_values)
        direct_row = direct_cache_capacity + row
        tl.store(
            direct_values_ptr + direct_row * stride_direct_row + dims * stride_direct_d,
            tl.where(is_prefix | is_recent, high_precision, 0.0),
            mask=dim_valid,
        )
        history_index = position - prefix_tokens
        logical_page = history_index // history_block_size
        page_offset = history_index % history_block_size
        physical_page = tl.load(
            history_page_table_ptr + logical_page * stride_history_table_page,
            mask=is_history,
            other=0,
        )
        packed = tl.load(
            history_data_ptr
            + physical_page * stride_data_page
            + page_offset * stride_data_token
            + (dims // 4) * stride_data_byte,
            mask=is_history & dim_valid,
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> ((dims % 4) * 2)) & 0x3).to(tl.float32)
        group = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_page * stride_scale_page
            + page_offset * stride_scale_token
            + group * stride_scale_group,
            mask=is_history & dim_valid,
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_page * stride_zero_page
            + page_offset * stride_zero_token
            + group * stride_zero_group,
            mask=is_history & dim_valid,
            other=0.0,
        ).to(tl.float32)
        tl.store(
            history_rotated_ptr + row * stride_history_row + dims * stride_history_d,
            tl.where(is_history, (quantized - zero) * scale, 0.0),
            mask=dim_valid,
        )
        rope_dims = tl.arange(0, 64)
        logical_rope_page = position // rope_block_size
        rope_page_offset = position % rope_block_size
        physical_rope_page = tl.load(
            rope_block_table_ptr + logical_rope_page * stride_rope_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + physical_rope_page * stride_rope_block
            + rope_page_offset * stride_rope_token
            + rope_dims * stride_rope_d,
            mask=valid & (rope_dims < rope_head_size),
            other=0.0,
        )
        tl.store(
            direct_values_ptr
            + direct_row * stride_direct_row
            + (latent_rank + rope_dims) * stride_direct_d,
            rope_values,
            mask=rope_dims < rope_head_size,
        )


@triton.jit
def _oscar_mtp_temporal_runtime_rows_addmm_kernel(
    lhs_ptr,
    rhs_ptr,
    output_ptr,
    active_rows_ptr,
    stride_lhs_m: tl.constexpr,
    stride_lhs_k: tl.constexpr,
    stride_rhs_k: tl.constexpr,
    stride_rhs_n: tl.constexpr,
    stride_output_m: tl.constexpr,
    stride_output_n: tl.constexpr,
    max_rows: tl.constexpr,
    k_size: tl.constexpr,
    n_size: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    active_rows = tl.load(active_rows_ptr)
    block_start = pid_m * block_m
    if block_start < active_rows:
        rows = block_start + tl.arange(0, block_m)
        cols = pid_n * block_n + tl.arange(0, block_n)
        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k_start in range(0, k_size, block_k):
            ks = k_start + tl.arange(0, block_k)
            lhs = tl.load(
                lhs_ptr + rows[:, None] * stride_lhs_m + ks[None, :] * stride_lhs_k,
                mask=(rows[:, None] < active_rows) & (ks[None, :] < k_size),
                other=0.0,
            )
            rhs = tl.load(
                rhs_ptr + ks[:, None] * stride_rhs_k + cols[None, :] * stride_rhs_n,
                mask=(ks[:, None] < k_size) & (cols[None, :] < n_size),
                other=0.0,
            )
            acc += tl.dot(lhs, rhs)
        prior = tl.load(
            output_ptr
            + rows[:, None] * stride_output_m
            + cols[None, :] * stride_output_n,
            mask=(rows[:, None] < active_rows) & (cols[None, :] < n_size),
            other=0.0,
        ).to(tl.float32)
        tl.store(
            output_ptr
            + rows[:, None] * stride_output_m
            + cols[None, :] * stride_output_n,
            acc + prior,
            mask=(rows[:, None] < active_rows) & (cols[None, :] < n_size),
        )


@triton.jit
def _finalize_oscar_mtp_temporal_rows_kernel(
    positions_ptr,
    miss_flags_ptr,
    position_to_miss_ptr,
    miss_values_ptr,
    cache_values_ptr,
    rope_ptr,
    rope_block_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    output_ptr,
    history_mask_ptr,
    remapped_ptr,
    stride_miss_row: tl.constexpr,
    stride_miss_d: tl.constexpr,
    stride_cache_row: tl.constexpr,
    stride_cache_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_table_page: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_d: tl.constexpr,
    num_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    rope_head_size: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    row_valid = rows < num_rows
    dim_valid = dims < latent_rank
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    seq_len = tl.load(seq_lens_ptr)
    hp_row = tl.load(hp_rows_ptr)
    valid = row_valid & (positions >= 0) & (positions < seq_len) & (hp_row >= 0)
    misses = tl.load(miss_flags_ptr + rows, mask=row_valid, other=0).to(tl.int1)
    miss_ids = tl.load(position_to_miss_ptr + positions, mask=misses, other=0)
    slots = positions % cache_capacity
    miss_values = tl.load(
        miss_values_ptr
        + miss_ids[:, None] * stride_miss_row
        + dims[None, :] * stride_miss_d,
        mask=misses[:, None] & dim_valid[None, :],
        other=0.0,
    )
    cached_values = tl.load(
        cache_values_ptr
        + slots[:, None] * stride_cache_row
        + dims[None, :] * stride_cache_d,
        mask=(valid & ~misses)[:, None] & dim_valid[None, :],
        other=0.0,
    )
    values = tl.where(misses[:, None], miss_values, cached_values)
    tl.store(
        output_ptr
        + rows[:, None] * stride_output_row
        + dims[None, :] * stride_output_d,
        tl.where(valid[:, None], values, 0.0),
        mask=row_valid[:, None] & dim_valid[None, :],
    )
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)
    if tl.program_id(1) == 0:
        history = valid & (positions >= prefix_tokens) & (positions < recent_start)
        tl.store(history_mask_ptr + rows, history.to(tl.uint8), mask=row_valid)
        tl.store(remapped_ptr + rows, tl.where(valid, rows, -1), mask=row_valid)
        rope_dims = tl.arange(0, 64)
        logical_pages = positions // rope_block_size
        page_offsets = positions % rope_block_size
        physical_pages = tl.load(
            rope_block_table_ptr + logical_pages * stride_rope_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + physical_pages[:, None] * stride_rope_block
            + page_offsets[:, None] * stride_rope_token
            + rope_dims[None, :] * stride_rope_d,
            mask=valid[:, None] & (rope_dims[None, :] < rope_head_size),
            other=0.0,
        )
        tl.store(
            output_ptr
            + rows[:, None] * stride_output_row
            + (latent_rank + rope_dims[None, :]) * stride_output_d,
            tl.where(valid[:, None], rope_values, 0.0),
            mask=row_valid[:, None] & (rope_dims[None, :] < rope_head_size),
        )


@triton.jit
def _remap_oscar_mtp_direct_rows_kernel(
    positions_ptr,
    miss_flags_ptr,
    position_to_miss_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    remapped_ptr,
    num_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    block_size: tl.constexpr,
):
    """Map occurrences to persistent cache slots or compact miss-tail rows."""
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    seq_len = tl.load(seq_lens_ptr)
    hp_row = tl.load(hp_rows_ptr)
    valid = row_valid & (positions >= 0) & (positions < seq_len) & (hp_row >= 0)
    misses = tl.load(miss_flags_ptr + rows, mask=row_valid, other=0).to(tl.int1)
    miss_ids = tl.load(position_to_miss_ptr + positions, mask=misses, other=0)
    cache_slots = positions % cache_capacity
    mapped = tl.where(misses, cache_capacity + miss_ids, cache_slots)
    tl.store(remapped_ptr + rows, tl.where(valid, mapped, -1), mask=row_valid)


@triton.jit
def _remap_oscar_mtp_two_way_rows_kernel(
    positions_ptr,
    cache_tags_ptr,
    miss_flags_ptr,
    position_to_miss_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    remapped_ptr,
    num_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    set_count: tl.constexpr,
    position_mask: tl.constexpr,
    miss_id_mask: tl.constexpr,
    block_size: tl.constexpr,
):
    rows = tl.program_id(0) * block_size + tl.arange(0, block_size)
    row_valid = rows < num_rows
    positions = tl.load(positions_ptr + rows, mask=row_valid, other=-1)
    seq_len = tl.load(seq_lens_ptr)
    hp_row = tl.load(hp_rows_ptr)
    valid = row_valid & (positions >= 0) & (positions < seq_len) & (hp_row >= 0)
    misses = tl.load(miss_flags_ptr + rows, mask=row_valid, other=0).to(tl.int1)
    packed = tl.load(position_to_miss_ptr + positions, mask=misses, other=0)
    miss_ids = packed & miss_id_mask
    set_slots = (positions % set_count) * 2
    raw_tag0 = tl.load(cache_tags_ptr + set_slots, mask=valid, other=-1)
    hit0 = (raw_tag0 >= 0) & ((raw_tag0 & position_mask) == positions)
    hit_slots = tl.where(hit0, set_slots, set_slots + 1)
    mapped = tl.where(misses, cache_capacity + miss_ids, hit_slots)
    tl.store(remapped_ptr + rows, tl.where(valid, mapped, -1), mask=row_valid)


@triton.jit
def _commit_oscar_mtp_temporal_unique_misses_kernel(
    positions_ptr,
    miss_positions_ptr,
    miss_count_ptr,
    miss_values_ptr,
    cache_slot_owners_ptr,
    cache_tags_ptr,
    cache_values_ptr,
    stride_miss_row: tl.constexpr,
    stride_miss_d: tl.constexpr,
    stride_cache_row: tl.constexpr,
    stride_cache_d: tl.constexpr,
    max_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    latent_rank: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    active_rows = tl.load(miss_count_ptr)
    if tl.program_id(0) * block_m < active_rows:
        row_valid = rows < active_rows
        dim_valid = dims < latent_rank
        miss_positions = tl.load(
            miss_positions_ptr + rows,
            mask=row_valid,
            other=-1,
        )
        slots = miss_positions % cache_capacity
        owners = tl.load(cache_slot_owners_ptr + slots, mask=row_valid, other=-1)
        owner_positions = tl.load(
            positions_ptr + owners,
            mask=row_valid & (owners >= 0),
            other=-2,
        )
        commit = row_valid & (owner_positions == miss_positions)
        values = tl.load(
            miss_values_ptr
            + rows[:, None] * stride_miss_row
            + dims[None, :] * stride_miss_d,
            mask=commit[:, None] & dim_valid[None, :],
            other=0.0,
        )
        tl.store(
            cache_values_ptr
            + slots[:, None] * stride_cache_row
            + dims[None, :] * stride_cache_d,
            values,
            mask=commit[:, None] & dim_valid[None, :],
        )
        if tl.program_id(1) == 0:
            tl.store(cache_tags_ptr + slots, miss_positions, mask=commit)


@triton.jit
def _commit_oscar_mtp_temporal_two_way_unique_misses_kernel(
    positions_ptr,
    position_to_miss_ptr,
    miss_positions_ptr,
    miss_count_ptr,
    miss_values_ptr,
    cache_slot_owners_ptr,
    cache_tags_ptr,
    cache_values_ptr,
    stride_miss_row: tl.constexpr,
    stride_miss_d: tl.constexpr,
    stride_cache_row: tl.constexpr,
    stride_cache_d: tl.constexpr,
    max_rows: tl.constexpr,
    set_count: tl.constexpr,
    state_bit: tl.constexpr,
    position_mask: tl.constexpr,
    miss_id_bits: tl.constexpr,
    latent_rank: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    active_rows = tl.load(miss_count_ptr)
    if tl.program_id(0) * block_m < active_rows:
        row_valid = rows < active_rows
        dim_valid = dims < latent_rank
        miss_positions = tl.load(
            miss_positions_ptr + rows,
            mask=row_valid,
            other=-1,
        )
        packed = tl.load(
            position_to_miss_ptr + miss_positions,
            mask=row_valid,
            other=0,
        )
        slots = packed >> miss_id_bits
        owners = tl.load(cache_slot_owners_ptr + slots, mask=row_valid, other=-1)
        owner_positions = tl.load(
            positions_ptr + owners,
            mask=row_valid & (owners >= 0),
            other=-2,
        )
        commit = row_valid & (owner_positions == miss_positions)
        values = tl.load(
            miss_values_ptr
            + rows[:, None] * stride_miss_row
            + dims[None, :] * stride_miss_d,
            mask=commit[:, None] & dim_valid[None, :],
            other=0.0,
        )
        tl.store(
            cache_values_ptr
            + slots[:, None] * stride_cache_row
            + dims[None, :] * stride_cache_d,
            values,
            mask=commit[:, None] & dim_valid[None, :],
        )
        if tl.program_id(1) == 0:
            set_slots = (miss_positions % set_count) * 2
            selected_way1 = (slots & 1) != 0
            tl.store(
                cache_tags_ptr + set_slots,
                miss_positions | state_bit,
                mask=commit & ~selected_way1,
            )
            tl.store(
                cache_tags_ptr + set_slots + 1,
                miss_positions,
                mask=commit & selected_way1,
            )
            raw_tag0 = tl.load(
                cache_tags_ptr + set_slots,
                mask=commit & selected_way1,
                other=-1,
            )
            tl.store(
                cache_tags_ptr + set_slots,
                raw_tag0 & position_mask,
                mask=commit & selected_way1,
            )


@triton.jit
def _commit_oscar_mtp_direct_unique_misses_kernel(
    positions_ptr,
    miss_positions_ptr,
    miss_count_ptr,
    cache_slot_owners_ptr,
    cache_tags_ptr,
    direct_values_ptr,
    stride_direct_row: tl.constexpr,
    stride_direct_d: tl.constexpr,
    max_rows: tl.constexpr,
    cache_capacity: tl.constexpr,
    row_width: tl.constexpr,
    block_m: tl.constexpr,
    block_d: tl.constexpr,
):
    """Commit compact miss-tail rows after attention has consumed old hits."""
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    dims = tl.program_id(1) * block_d + tl.arange(0, block_d)
    active_rows = tl.load(miss_count_ptr)
    if tl.program_id(0) * block_m < active_rows:
        row_valid = rows < active_rows
        dim_valid = dims < row_width
        miss_positions = tl.load(miss_positions_ptr + rows, mask=row_valid, other=-1)
        slots = miss_positions % cache_capacity
        owners = tl.load(cache_slot_owners_ptr + slots, mask=row_valid, other=-1)
        owner_positions = tl.load(
            positions_ptr + owners,
            mask=row_valid & (owners >= 0),
            other=-2,
        )
        commit = row_valid & (owner_positions == miss_positions)
        values = tl.load(
            direct_values_ptr
            + (cache_capacity + rows[:, None]) * stride_direct_row
            + dims[None, :] * stride_direct_d,
            mask=commit[:, None] & dim_valid[None, :],
            other=0.0,
        )
        tl.store(
            direct_values_ptr
            + slots[:, None] * stride_direct_row
            + dims[None, :] * stride_direct_d,
            values,
            mask=commit[:, None] & dim_valid[None, :],
        )
        if tl.program_id(1) == 0:
            tl.store(cache_tags_ptr + slots, miss_positions, mask=commit)


@triton.jit
def _save_and_remap_oscar_topk_kernel(
    indices_ptr,
    saved_indices_ptr,
    numel,
    row_offset,
    num_rows,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    logical_index = tl.load(indices_ptr + offsets, mask=mask, other=-1)
    tl.store(saved_indices_ptr + offsets, logical_index, mask=mask)
    local_index = logical_index - row_offset
    valid = (logical_index >= row_offset) & (local_index < num_rows)
    tl.store(indices_ptr + offsets, tl.where(valid, local_index, -1), mask=mask)


@triton.jit
def _restore_oscar_topk_kernel(
    indices_ptr,
    saved_indices_ptr,
    numel,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < numel
    logical_index = tl.load(saved_indices_ptr + offsets, mask=mask, other=-1)
    tl.store(indices_ptr + offsets, logical_index, mask=mask)


@triton.jit
def _merge_oscar_chunked_attention_kernel(
    output_ptr,
    output_lse_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    num_tokens,
    num_heads: tl.constexpr,
    head_size: tl.constexpr,
    block_d: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, block_d)
    output_offset = (token * num_heads + head) * head_size
    lse_offset = token * num_heads + head
    output_lse = tl.load(output_lse_ptr + lse_offset)
    partial_lse = tl.load(partial_lse_ptr + lse_offset)
    output_valid = output_lse != -float("inf")
    partial_valid = partial_lse != -float("inf")
    max_lse = tl.maximum(
        tl.where(output_valid, output_lse, -float("inf")),
        tl.where(partial_valid, partial_lse, -float("inf")),
    )
    output_exp = tl.where(output_valid, tl.exp(output_lse - max_lse), 0.0)
    partial_exp = tl.where(partial_valid, tl.exp(partial_lse - max_lse), 0.0)
    exp_sum = output_exp + partial_exp
    exp_sum_safe = tl.where(exp_sum > 0.0, exp_sum, 1.0)
    dim_mask = dims < head_size
    output = tl.load(
        output_ptr + output_offset + dims,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    partial_output = tl.load(
        partial_output_ptr + output_offset + dims,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    merged = (output * output_exp + partial_output * partial_exp) / exp_sum_safe
    tl.store(output_ptr + output_offset + dims, merged, mask=dim_mask)
    merged_lse = tl.where(
        exp_sum > 0.0,
        tl.log(exp_sum) + max_lse,
        -float("inf"),
    )
    tl.store(output_lse_ptr + lse_offset, merged_lse)


def can_use_oscar_bf16_materialized_read(
    *,
    capability_major: int,
    num_requests: int,
    num_heads: int,
    latent_rank: int,
    rope_head_size: int,
    group_size: int,
    prefix_tokens: int,
    recent_tokens: int,
    topk: int,
) -> bool:
    """Return whether the frozen C008 target geometry is active."""
    return (
        capability_major == 8
        and num_requests == 1
        and num_heads in (8, 16, 32)
        and latent_rank == 512
        and rope_head_size == 64
        and group_size == 128
        and prefix_tokens == 64
        and recent_tokens == 256
        and topk == 2048
    )


def allocate_oscar_bf16_materialization_workspace(
    reference: torch.Tensor,
    max_rows: int,
    num_heads: int = 8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Allocate the reusable BF16 dense materialization workspace."""
    if max_rows <= 0:
        raise ValueError("max_rows must be positive")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    if reference.ndim != 2 or reference.shape[1] != 2048:
        raise ValueError("C012 workspace requires a [max_tokens, 2048] reference")
    device = reference.device
    max_tokens = reference.shape[0]
    # The BF16 history buffer also backs the int32 top-k save area. Each
    # int32 value needs storage for two BF16 elements.
    history_rows = max(max_rows, (2 * reference.numel() + 511) // 512)
    return (
        torch.empty((history_rows, 512), dtype=torch.bfloat16, device=device),
        torch.empty((max_rows,), dtype=torch.uint8, device=device),
        torch.empty((max_rows, 576), dtype=torch.bfloat16, device=device),
        torch.empty((max_rows,), dtype=torch.int32, device=device),
        torch.empty((max_tokens, num_heads, 512), dtype=torch.bfloat16, device=device),
        torch.empty((max_tokens, num_heads), dtype=torch.float32, device=device),
        torch.empty((max_tokens, num_heads), dtype=torch.float32, device=device),
    )


def prepare_oscar_bf16_materialization_workspace(
    reference: torch.Tensor,
    num_heads: int = 8,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return the single C012 workspace shared by all layers on one device."""
    key = (reference.device.type, reference.device.index, num_heads)
    workspace = _workspace_cache.get(key)
    if workspace is not None:
        return workspace
    if reference.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "C012 materialization workspace must be prepared before CUDA Graph capture"
        )
    workspace = allocate_oscar_bf16_materialization_workspace(
        reference,
        OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
        num_heads,
    )
    _workspace_cache[key] = workspace
    return workspace


def allocate_oscar_mtp_temporal_cache(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one layer's persistent history-only MTP materialization cache."""
    return (
        torch.empty(
            (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512),
            dtype=torch.bfloat16,
            device=reference.device,
        ),
        torch.full(
            (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,),
            -1,
            dtype=torch.int32,
            device=reference.device,
        ),
    )


def allocate_oscar_mtp_direct_attention_cache(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate one layer's cache plus a contiguous compact-miss tail."""
    return (
        torch.empty(
            (
                OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS,
                576,
            ),
            dtype=torch.bfloat16,
            device=reference.device,
        ),
        torch.full(
            (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,),
            -1,
            dtype=torch.int32,
            device=reference.device,
        ),
    )


def allocate_oscar_mtp_temporal_cache_with_direct_storage(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose a contiguous temporal view backed by direct-sized storage."""
    direct_values, cache_tags = allocate_oscar_mtp_direct_attention_cache(reference)
    temporal_values = direct_values.view(-1)[: OSCAR_MTP_TEMPORAL_CACHE_CAPACITY * 512]
    temporal_values = temporal_values.view(OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
    return temporal_values, cache_tags


def allocate_oscar_mtp_temporal_cache_with_split_direct_storage(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match direct-cache bytes without oversizing temporal values storage."""
    direct_value_elements = (
        OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS
    ) * 576
    temporal_value_elements = OSCAR_MTP_TEMPORAL_CACHE_CAPACITY * 512
    padding_bf16_elements = direct_value_elements - temporal_value_elements
    if padding_bf16_elements % 2 != 0:
        raise AssertionError("BF16 padding must be representable as INT32 storage")
    padding_int32_elements = padding_bf16_elements // 2

    temporal_values = torch.empty(
        (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512),
        dtype=torch.bfloat16,
        device=reference.device,
    )
    tag_backing = torch.empty(
        padding_int32_elements + OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,
        dtype=torch.int32,
        device=reference.device,
    )
    cache_tags = tag_backing[padding_int32_elements:]
    cache_tags.fill_(-1)
    return temporal_values, cache_tags


def allocate_oscar_mtp_row576_temporal_cache(
    reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match direct-cache bytes with normal cache rows and a D576 stride."""
    direct_value_elements = (
        OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS
    ) * 576
    cache_value_elements = OSCAR_MTP_TEMPORAL_CACHE_CAPACITY * 576
    padding_bf16_elements = direct_value_elements - cache_value_elements
    if padding_bf16_elements % 2 != 0:
        raise AssertionError("BF16 padding must be representable as INT32 storage")
    padding_int32_elements = padding_bf16_elements // 2

    cache_value_backing = torch.empty(
        (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 576),
        dtype=torch.bfloat16,
        device=reference.device,
    )
    temporal_values = cache_value_backing[:, :512]
    tag_backing = torch.empty(
        padding_int32_elements + OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,
        dtype=torch.int32,
        device=reference.device,
    )
    cache_tags = tag_backing[padding_int32_elements:]
    cache_tags.fill_(-1)
    return temporal_values, cache_tags


def prepare_oscar_mtp_temporal_workspace(
    reference: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return Graph-stable dedup scratch shared by all layers on one device."""
    key = (reference.device.type, reference.device.index)
    workspace = _mtp_temporal_workspace_cache.get(key)
    if workspace is not None:
        return workspace
    if reference.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "OSCAR MTP temporal workspace must be prepared before CUDA Graph capture"
        )
    device = reference.device
    workspace = (
        torch.empty(
            OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(
            OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(
            OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
            dtype=torch.int32,
            device=device,
        ),
        torch.empty(
            OSCAR_MTP_TEMPORAL_MAX_ROWS,
            dtype=torch.uint8,
            device=device,
        ),
        torch.empty(
            OSCAR_MTP_TEMPORAL_MAX_ROWS,
            dtype=torch.int32,
            device=device,
        ),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.empty(
            OSCAR_MTP_TEMPORAL_MAX_ROWS,
            dtype=torch.int32,
            device=device,
        ),
    )
    _mtp_temporal_workspace_cache[key] = workspace
    return workspace


def reset_oscar_mtp_temporal_cache(
    cache: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Invalidate one layer's cache before a new request prefill."""
    cache[1].fill_(-1)


def seed_oscar_mtp_temporal_cache_rows(
    values: torch.Tensor,
    positions: torch.Tensor,
    valid_rows: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Seed stable prefill history rows before they are quantized to INT2."""
    if values.ndim != 2 or values.shape[1] != 512 or values.dtype != torch.bfloat16:
        raise ValueError("OSCAR MTP temporal seed values must be [N, 512] BF16")
    num_rows = values.shape[0]
    if (
        positions.ndim != 1
        or positions.shape[0] != num_rows
        or positions.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("OSCAR MTP temporal seed positions are invalid")
    if (
        valid_rows.ndim != 1
        or valid_rows.shape[0] != num_rows
        or valid_rows.dtype != torch.bool
    ):
        raise ValueError("OSCAR MTP temporal seed mask is invalid")
    cache_values, cache_tags = cache
    if (
        cache_values.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
        or cache_values.dtype != torch.bfloat16
        or cache_tags.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        or cache_tags.dtype != torch.int32
    ):
        raise ValueError("OSCAR MTP temporal seed cache is invalid")
    if num_rows > OSCAR_MTP_TEMPORAL_CACHE_CAPACITY:
        raise ValueError("OSCAR MTP temporal seed rows exceed cache capacity")
    if not (
        values.device
        == positions.device
        == valid_rows.device
        == cache_values.device
        == cache_tags.device
    ):
        raise ValueError("OSCAR MTP temporal seed tensors must share one device")
    if num_rows == 0:
        return
    block_m = 16
    block_d = 128
    _seed_oscar_mtp_temporal_rows_kernel[
        (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_d))
    ](
        values,
        positions,
        valid_rows,
        cache_values,
        cache_tags,
        stride_values_row=values.stride(0),
        stride_values_d=values.stride(1),
        stride_cache_row=cache_values.stride(0),
        stride_cache_d=cache_values.stride(1),
        num_rows=num_rows,
        cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        latent_rank=512,
        block_m=block_m,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )


def seed_oscar_mtp_temporal_cache_recent(
    recent: torch.Tensor,
    positions: torch.Tensor,
    hp_rows: torch.Tensor,
    page_ids: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor],
    *,
    prefix_tokens: int,
) -> None:
    """Seed prefill rows that move from the BF16 recent ring into history."""
    if recent.ndim != 3 or recent.shape[2] != 512 or recent.dtype != torch.bfloat16:
        raise ValueError("OSCAR MTP recent seed values must be [B, N, 512] BF16")
    num_rows = positions.shape[0]
    for name, tensor in (
        ("positions", positions),
        ("hp_rows", hp_rows),
        ("page_ids", page_ids),
    ):
        if (
            tensor.ndim != 1
            or tensor.shape[0] != num_rows
            or tensor.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError(f"OSCAR MTP recent seed {name} is invalid")
    cache_values, cache_tags = cache
    if (
        cache_values.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
        or cache_values.dtype != torch.bfloat16
        or cache_tags.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        or cache_tags.dtype != torch.int32
    ):
        raise ValueError("OSCAR MTP recent seed cache is invalid")
    if prefix_tokens <= 0 or num_rows > OSCAR_MTP_TEMPORAL_CACHE_CAPACITY:
        raise ValueError("OSCAR MTP recent seed geometry is invalid")
    if not (
        recent.device
        == positions.device
        == hp_rows.device
        == page_ids.device
        == cache_values.device
        == cache_tags.device
    ):
        raise ValueError("OSCAR MTP recent seed tensors must share one device")
    if num_rows == 0:
        return
    block_m = 16
    block_d = 128
    _seed_oscar_mtp_temporal_recent_kernel[
        (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_d))
    ](
        recent,
        positions,
        hp_rows,
        page_ids,
        cache_values,
        cache_tags,
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_d=recent.stride(2),
        stride_cache_row=cache_values.stride(0),
        stride_cache_d=cache_values.stride(1),
        num_rows=num_rows,
        prefix_tokens=prefix_tokens,
        recent_capacity=recent.shape[1],
        cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        latent_rank=512,
        block_m=block_m,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )


def materialize_oscar_mla_bf16_rows_temporal(
    *,
    positions: torch.Tensor,
    num_rows: int,
    num_requests: int,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    rope: torch.Tensor,
    rope_block_table: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    history_page_table: torch.Tensor,
    hp_rows: torch.Tensor,
    seq_lens: torch.Tensor,
    inverse_rotation: torch.Tensor,
    history_rotated: torch.Tensor,
    history_mask: torch.Tensor,
    output_kv: torch.Tensor,
    remapped_indices: torch.Tensor,
    temporal_workspace: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    temporal_cache: tuple[torch.Tensor, torch.Tensor],
    recent_tokens: int,
    dual_source_attention: bool = False,
    two_way: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize selected MTP rows with an exact history-only demand cache."""
    if num_requests != 1 or not 0 < num_rows <= OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP temporal materialization requires batch1 rows")
    if two_way and not dual_source_attention:
        raise ValueError("OSCAR MTP temporal two-way requires dual-source attention")
    if (
        positions.ndim != 1
        or positions.shape[0] < num_rows
        or positions.dtype != torch.int32
        or not positions.is_contiguous()
    ):
        raise ValueError("OSCAR MTP temporal positions must be contiguous int32")
    if (
        prefix.shape[1:] != (64, 512)
        or recent.shape[2] != 512
        or not 0 < recent_tokens <= recent.shape[1]
        or rope.shape[2] != 64
        or history_data.shape[2] != 128
        or history_scale.shape[2] != 4
        or inverse_rotation.shape != (512, 512)
    ):
        raise ValueError("OSCAR MTP temporal materialization geometry mismatch")
    if inverse_rotation.dtype != torch.bfloat16 or not inverse_rotation.is_contiguous():
        raise ValueError("OSCAR MTP temporal inverse rotation must be contiguous BF16")
    if history_rotated.shape[0] < 2 * OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP temporal BF16 scratch is too small")
    if history_mask.shape[0] < num_rows or history_mask.dtype != torch.uint8:
        raise ValueError("OSCAR MTP temporal history mask is invalid")
    if output_kv.shape[0] < num_rows or output_kv.shape[1:] != (576,):
        raise ValueError("OSCAR MTP temporal output workspace is invalid")
    if output_kv.dtype != torch.bfloat16:
        raise ValueError("OSCAR MTP temporal output workspace must be BF16")
    if remapped_indices.shape[0] < num_rows or remapped_indices.dtype != torch.int32:
        raise ValueError("OSCAR MTP temporal remap workspace is invalid")

    cache_values, cache_tags = temporal_cache
    if (
        cache_values.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
        or cache_values.dtype != torch.bfloat16
        or cache_tags.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        or cache_tags.dtype != torch.int32
    ):
        raise ValueError("OSCAR MTP temporal layer cache is invalid")
    if dual_source_attention and cache_values.stride() != (512, 1):
        raise ValueError("OSCAR MTP dual-source cache must use D512 storage")
    (
        position_owners,
        position_to_miss,
        cache_slot_owners,
        miss_flags,
        occurrence_to_miss,
        miss_count,
        miss_positions,
    ) = temporal_workspace
    if (
        position_owners.shape != (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,)
        or position_to_miss.shape != (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,)
        or cache_slot_owners.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        or miss_flags.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
        or occurrence_to_miss.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
        or miss_count.shape != (1,)
        or miss_positions.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
    ):
        raise ValueError("OSCAR MTP temporal shared workspace is invalid")
    tensors = (
        positions,
        prefix,
        recent,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        inverse_rotation,
        history_rotated,
        history_mask,
        output_kv,
        remapped_indices,
        cache_values,
        cache_tags,
        *temporal_workspace,
    )
    if any(tensor.device != prefix.device for tensor in tensors):
        raise ValueError("OSCAR MTP temporal tensors must share one device")

    miss_history = history_rotated[:OSCAR_MTP_TEMPORAL_MAX_ROWS]
    miss_values = history_rotated[
        OSCAR_MTP_TEMPORAL_MAX_ROWS : 2 * OSCAR_MTP_TEMPORAL_MAX_ROWS
    ]
    position_owners.fill_(2_147_483_647)
    cache_slot_owners.fill_(-1)
    miss_count.zero_()
    block_size = 256
    if two_way:
        _mark_oscar_mtp_temporal_two_way_miss_owners_kernel[
            (triton.cdiv(num_rows, block_size),)
        ](
            positions,
            cache_tags,
            position_owners,
            position_to_miss,
            cache_slot_owners,
            miss_flags,
            seq_lens,
            num_rows=num_rows,
            set_count=OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT,
            state_bit=OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT,
            position_mask=OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK,
            max_positions=OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
            prefix_tokens=64,
            recent_tokens=recent_tokens,
            block_size=block_size,
            num_warps=4,
            num_stages=1,
        )
        _assign_oscar_mtp_temporal_two_way_unique_misses_kernel[
            (triton.cdiv(num_rows, block_size),)
        ](
            positions,
            position_owners,
            miss_flags,
            miss_count,
            miss_positions,
            position_to_miss,
            num_rows=num_rows,
            miss_id_bits=_OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_BITS,
            block_size=block_size,
            num_warps=4,
            num_stages=1,
        )
    else:
        _mark_oscar_mtp_temporal_miss_owners_kernel[
            (triton.cdiv(num_rows, block_size),)
        ](
            positions,
            cache_tags,
            position_owners,
            cache_slot_owners,
            miss_flags,
            seq_lens,
            num_rows=num_rows,
            cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
            max_positions=OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
            prefix_tokens=64,
            recent_tokens=recent_tokens,
            block_size=block_size,
            num_warps=4,
            num_stages=1,
        )
        _assign_oscar_mtp_temporal_unique_misses_kernel[
            (triton.cdiv(num_rows, block_size),)
        ](
            positions,
            position_owners,
            miss_flags,
            miss_count,
            miss_positions,
            position_to_miss,
            num_rows=num_rows,
            block_size=block_size,
            num_warps=4,
            num_stages=1,
        )
    _gather_oscar_mtp_temporal_unique_misses_kernel[(num_rows,)](
        miss_positions,
        miss_count,
        prefix,
        recent,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        miss_history,
        miss_values,
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_d=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_d=recent.stride(2),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_history_table_page=history_page_table.stride(1),
        stride_history_row=miss_history.stride(0),
        stride_history_d=miss_history.stride(1),
        stride_output_row=miss_values.stride(0),
        stride_output_d=miss_values.stride(1),
        max_rows=num_rows,
        prefix_tokens=64,
        recent_tokens=recent_tokens,
        recent_capacity_tokens=recent.shape[1],
        history_block_size=history_data.shape[1],
        latent_rank=512,
        group_size=128,
        block_d=512,
        num_warps=4,
        num_stages=1,
    )
    block_m = 32
    block_n = 64
    block_k = 32
    _oscar_mtp_temporal_runtime_rows_addmm_kernel[
        (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_n))
    ](
        miss_history,
        inverse_rotation,
        miss_values,
        miss_count,
        stride_lhs_m=miss_history.stride(0),
        stride_lhs_k=miss_history.stride(1),
        stride_rhs_k=inverse_rotation.stride(0),
        stride_rhs_n=inverse_rotation.stride(1),
        stride_output_m=miss_values.stride(0),
        stride_output_n=miss_values.stride(1),
        max_rows=num_rows,
        k_size=512,
        n_size=512,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=8,
        num_stages=2,
    )
    if dual_source_attention:
        if two_way:
            _remap_oscar_mtp_two_way_rows_kernel[(triton.cdiv(num_rows, block_size),)](
                positions,
                cache_tags,
                miss_flags,
                position_to_miss,
                hp_rows,
                seq_lens,
                remapped_indices,
                num_rows=num_rows,
                cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
                set_count=OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT,
                position_mask=OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK,
                miss_id_mask=_OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_MASK,
                block_size=block_size,
                num_warps=4,
                num_stages=1,
            )
        else:
            _remap_oscar_mtp_direct_rows_kernel[(triton.cdiv(num_rows, block_size),)](
                positions,
                miss_flags,
                position_to_miss,
                hp_rows,
                seq_lens,
                remapped_indices,
                num_rows=num_rows,
                cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
                block_size=block_size,
                num_warps=4,
                num_stages=1,
            )
        return (
            miss_values[:num_rows],
            remapped_indices[:num_rows].view(1, 1, num_rows),
        )
    block_d = 64
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_d))
    _finalize_oscar_mtp_temporal_rows_kernel[grid](
        positions,
        miss_flags,
        position_to_miss,
        miss_values,
        cache_values,
        rope,
        rope_block_table,
        hp_rows,
        seq_lens,
        output_kv,
        history_mask,
        remapped_indices,
        stride_miss_row=miss_values.stride(0),
        stride_miss_d=miss_values.stride(1),
        stride_cache_row=cache_values.stride(0),
        stride_cache_d=cache_values.stride(1),
        stride_rope_block=rope.stride(0),
        stride_rope_token=rope.stride(1),
        stride_rope_d=rope.stride(2),
        stride_rope_table_page=rope_block_table.stride(1),
        stride_output_row=output_kv.stride(0),
        stride_output_d=output_kv.stride(1),
        num_rows=num_rows,
        cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        prefix_tokens=64,
        recent_tokens=recent_tokens,
        rope_block_size=rope.shape[1],
        latent_rank=512,
        rope_head_size=64,
        block_m=block_m,
        block_d=block_d,
        num_warps=8,
        num_stages=1,
    )
    _commit_oscar_mtp_temporal_unique_misses_kernel[grid](
        positions,
        miss_positions,
        miss_count,
        miss_values,
        cache_slot_owners,
        cache_tags,
        cache_values,
        stride_miss_row=miss_values.stride(0),
        stride_miss_d=miss_values.stride(1),
        stride_cache_row=cache_values.stride(0),
        stride_cache_d=cache_values.stride(1),
        max_rows=num_rows,
        cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        latent_rank=512,
        block_m=block_m,
        block_d=block_d,
        num_warps=8,
        num_stages=1,
    )
    return (
        output_kv[:num_rows].view(num_rows, 1, 576),
        remapped_indices[:num_rows].view(1, 1, num_rows),
    )


def materialize_oscar_mla_bf16_rows_direct_attention(
    *,
    positions: torch.Tensor,
    num_rows: int,
    num_requests: int,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    rope: torch.Tensor,
    rope_block_table: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    history_page_table: torch.Tensor,
    hp_rows: torch.Tensor,
    seq_lens: torch.Tensor,
    inverse_rotation: torch.Tensor,
    history_rotated: torch.Tensor,
    remapped_indices: torch.Tensor,
    temporal_workspace: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_cache: tuple[torch.Tensor, torch.Tensor],
    recent_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expose persistent cache slots and compact misses directly to attention."""
    if num_requests != 1 or not 0 < num_rows <= OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP direct attention requires batch1 rows")
    if (
        positions.ndim != 1
        or positions.shape[0] < num_rows
        or positions.dtype != torch.int32
        or not positions.is_contiguous()
    ):
        raise ValueError("OSCAR MTP direct positions must be contiguous int32")
    if (
        prefix.shape[1:] != (64, 512)
        or recent.shape[2] != 512
        or not 0 < recent_tokens <= recent.shape[1]
        or rope.shape[2] != 64
        or history_data.shape[2] != 128
        or history_scale.shape[2] != 4
        or inverse_rotation.shape != (512, 512)
    ):
        raise ValueError("OSCAR MTP direct attention geometry mismatch")
    if inverse_rotation.dtype != torch.bfloat16 or not inverse_rotation.is_contiguous():
        raise ValueError("OSCAR MTP direct inverse rotation must be contiguous BF16")
    if history_rotated.shape[0] < OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP direct BF16 scratch is too small")
    if remapped_indices.shape[0] < num_rows or remapped_indices.dtype != torch.int32:
        raise ValueError("OSCAR MTP direct remap workspace is invalid")

    direct_values, cache_tags = direct_cache
    direct_capacity = OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY
    if (
        direct_values.shape != (direct_capacity + OSCAR_MTP_TEMPORAL_MAX_ROWS, 576)
        or direct_values.dtype != torch.bfloat16
        or cache_tags.shape != (direct_capacity,)
        or cache_tags.dtype != torch.int32
    ):
        raise ValueError("OSCAR MTP direct layer cache is invalid")
    (
        position_owners,
        position_to_miss,
        cache_slot_owners,
        miss_flags,
        occurrence_to_miss,
        miss_count,
        miss_positions,
    ) = temporal_workspace
    if (
        position_owners.shape != (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,)
        or position_to_miss.shape != (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,)
        or cache_slot_owners.shape[0] < direct_capacity
        or miss_flags.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
        or occurrence_to_miss.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
        or miss_count.shape != (1,)
        or miss_positions.shape != (OSCAR_MTP_TEMPORAL_MAX_ROWS,)
    ):
        raise ValueError("OSCAR MTP direct shared workspace is invalid")
    tensors = (
        positions,
        prefix,
        recent,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        inverse_rotation,
        history_rotated,
        remapped_indices,
        direct_values,
        cache_tags,
        *temporal_workspace,
    )
    if any(tensor.device != prefix.device for tensor in tensors):
        raise ValueError("OSCAR MTP direct tensors must share one device")

    miss_history = history_rotated[:OSCAR_MTP_TEMPORAL_MAX_ROWS]
    miss_values = direct_values[direct_capacity:, :512]
    position_owners.fill_(2_147_483_647)
    cache_slot_owners[:direct_capacity].fill_(-1)
    miss_count.zero_()
    block_size = 256
    _mark_oscar_mtp_temporal_miss_owners_kernel[(triton.cdiv(num_rows, block_size),)](
        positions,
        cache_tags,
        position_owners,
        cache_slot_owners,
        miss_flags,
        seq_lens,
        num_rows=num_rows,
        cache_capacity=direct_capacity,
        max_positions=OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
        prefix_tokens=64,
        recent_tokens=recent_tokens,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )
    _assign_oscar_mtp_temporal_unique_misses_kernel[
        (triton.cdiv(num_rows, block_size),)
    ](
        positions,
        position_owners,
        miss_flags,
        miss_count,
        miss_positions,
        position_to_miss,
        num_rows=num_rows,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )
    _gather_oscar_mtp_direct_unique_misses_kernel[(num_rows,)](
        miss_positions,
        miss_count,
        prefix,
        recent,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        miss_history,
        direct_values,
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_d=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_d=recent.stride(2),
        stride_rope_block=rope.stride(0),
        stride_rope_token=rope.stride(1),
        stride_rope_d=rope.stride(2),
        stride_rope_table_page=rope_block_table.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_history_table_page=history_page_table.stride(1),
        stride_history_row=miss_history.stride(0),
        stride_history_d=miss_history.stride(1),
        stride_direct_row=direct_values.stride(0),
        stride_direct_d=direct_values.stride(1),
        max_rows=num_rows,
        direct_cache_capacity=direct_capacity,
        prefix_tokens=64,
        recent_tokens=recent_tokens,
        recent_capacity_tokens=recent.shape[1],
        rope_block_size=rope.shape[1],
        rope_head_size=64,
        history_block_size=history_data.shape[1],
        latent_rank=512,
        group_size=128,
        block_d=512,
        num_warps=4,
        num_stages=1,
    )
    block_m = 32
    block_n = 64
    block_k = 32
    _oscar_mtp_temporal_runtime_rows_addmm_kernel[
        (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_n))
    ](
        miss_history,
        inverse_rotation,
        miss_values,
        miss_count,
        stride_lhs_m=miss_history.stride(0),
        stride_lhs_k=miss_history.stride(1),
        stride_rhs_k=inverse_rotation.stride(0),
        stride_rhs_n=inverse_rotation.stride(1),
        stride_output_m=miss_values.stride(0),
        stride_output_n=miss_values.stride(1),
        max_rows=num_rows,
        k_size=512,
        n_size=512,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=8,
        num_stages=2,
    )
    _remap_oscar_mtp_direct_rows_kernel[(triton.cdiv(num_rows, block_size),)](
        positions,
        miss_flags,
        position_to_miss,
        hp_rows,
        seq_lens,
        remapped_indices,
        num_rows=num_rows,
        cache_capacity=direct_capacity,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )
    return (
        direct_values.view(-1, 1, 576),
        remapped_indices[:num_rows].view(1, 1, num_rows),
    )


def commit_oscar_mla_dual_source_attention_misses(
    *,
    positions: torch.Tensor,
    num_rows: int,
    miss_values: torch.Tensor,
    temporal_workspace: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    temporal_cache: tuple[torch.Tensor, torch.Tensor],
    two_way: bool = False,
) -> None:
    """Commit D512 misses after dual-source attention has consumed old hits."""
    cache_values, cache_tags = temporal_cache
    cache_slot_owners = temporal_workspace[2]
    miss_count = temporal_workspace[5]
    miss_positions = temporal_workspace[6]
    if not 0 < num_rows <= OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP dual-source commit row count is invalid")
    if (
        miss_values.shape != (num_rows, 512)
        or miss_values.dtype != torch.bfloat16
        or miss_values.stride(1) != 1
    ):
        raise ValueError("OSCAR MTP dual-source commit misses are invalid")
    if (
        cache_values.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
        or cache_values.dtype != torch.bfloat16
        or cache_values.stride() != (512, 1)
        or cache_tags.shape != (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
        or cache_tags.dtype != torch.int32
    ):
        raise ValueError("OSCAR MTP dual-source commit cache is invalid")
    block_m = 32
    block_d = 64
    grid = (triton.cdiv(num_rows, block_m), triton.cdiv(512, block_d))
    if two_way:
        _commit_oscar_mtp_temporal_two_way_unique_misses_kernel[grid](
            positions,
            temporal_workspace[1],
            miss_positions,
            miss_count,
            miss_values,
            cache_slot_owners,
            cache_tags,
            cache_values,
            stride_miss_row=miss_values.stride(0),
            stride_miss_d=miss_values.stride(1),
            stride_cache_row=cache_values.stride(0),
            stride_cache_d=cache_values.stride(1),
            max_rows=num_rows,
            set_count=OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT,
            state_bit=OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT,
            position_mask=OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK,
            miss_id_bits=_OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_BITS,
            latent_rank=512,
            block_m=block_m,
            block_d=block_d,
            num_warps=8,
            num_stages=1,
        )
    else:
        _commit_oscar_mtp_temporal_unique_misses_kernel[grid](
            positions,
            miss_positions,
            miss_count,
            miss_values,
            cache_slot_owners,
            cache_tags,
            cache_values,
            stride_miss_row=miss_values.stride(0),
            stride_miss_d=miss_values.stride(1),
            stride_cache_row=cache_values.stride(0),
            stride_cache_d=cache_values.stride(1),
            max_rows=num_rows,
            cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
            latent_rank=512,
            block_m=block_m,
            block_d=block_d,
            num_warps=8,
            num_stages=1,
        )


def commit_oscar_mla_direct_attention_misses(
    *,
    positions: torch.Tensor,
    num_rows: int,
    temporal_workspace: tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ],
    direct_cache: tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Commit direct-attention misses after the attention kernel is queued."""
    direct_values, cache_tags = direct_cache
    cache_slot_owners = temporal_workspace[2]
    miss_count = temporal_workspace[5]
    miss_positions = temporal_workspace[6]
    direct_capacity = OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY
    if not 0 < num_rows <= OSCAR_MTP_TEMPORAL_MAX_ROWS:
        raise ValueError("OSCAR MTP direct commit row count is invalid")
    if direct_values.shape != (
        direct_capacity + OSCAR_MTP_TEMPORAL_MAX_ROWS,
        576,
    ) or cache_tags.shape != (direct_capacity,):
        raise ValueError("OSCAR MTP direct commit cache is invalid")
    block_m = 32
    block_d = 64
    _commit_oscar_mtp_direct_unique_misses_kernel[
        (triton.cdiv(num_rows, block_m), triton.cdiv(576, block_d))
    ](
        positions,
        miss_positions,
        miss_count,
        cache_slot_owners,
        cache_tags,
        direct_values,
        stride_direct_row=direct_values.stride(0),
        stride_direct_d=direct_values.stride(1),
        max_rows=num_rows,
        cache_capacity=direct_capacity,
        row_width=576,
        block_m=block_m,
        block_d=block_d,
        num_warps=8,
        num_stages=1,
    )


def save_and_remap_oscar_topk_for_chunk(
    indices: torch.Tensor,
    history_rotated: torch.Tensor,
    row_offset: int,
    num_rows: int,
) -> torch.Tensor:
    """Save logical top-k indices and replace them with chunk-local indices."""
    if indices.dtype != torch.int32 or not indices.is_contiguous():
        raise ValueError("C012 top-k indices must be contiguous int32")
    if history_rotated.dtype != torch.bfloat16 or not history_rotated.is_contiguous():
        raise ValueError("dense history workspace must be contiguous BF16")
    if row_offset < 0 or num_rows <= 0:
        raise ValueError("C012 chunk bounds must be positive")
    saved_indices = history_rotated.view(torch.int32).view(-1)
    if saved_indices.numel() < indices.numel():
        raise ValueError("C012 history workspace cannot hold the logical top-k copy")
    saved_indices = saved_indices[: indices.numel()]
    block_size = 256
    _save_and_remap_oscar_topk_kernel[(triton.cdiv(indices.numel(), block_size),)](
        indices,
        saved_indices,
        indices.numel(),
        row_offset,
        num_rows,
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )
    return saved_indices


def restore_oscar_topk_after_chunk(
    indices: torch.Tensor,
    saved_indices: torch.Tensor,
) -> None:
    """Restore logical top-k indices after one chunk attention call."""
    if indices.dtype != torch.int32 or not indices.is_contiguous():
        raise ValueError("C012 top-k indices must be contiguous int32")
    if saved_indices.dtype != torch.int32 or saved_indices.numel() < indices.numel():
        raise ValueError("C012 saved top-k workspace is invalid")
    block_size = 256
    _restore_oscar_topk_kernel[(triton.cdiv(indices.numel(), block_size),)](
        indices,
        saved_indices,
        indices.numel(),
        block_size=block_size,
        num_warps=4,
        num_stages=1,
    )


def merge_oscar_chunked_attention_states(
    output: torch.Tensor,
    output_lse: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
) -> None:
    """Merge one BF16 partial state into `[tokens, heads]` natural-log LSE."""
    if output.shape != partial_output.shape or output.dtype != torch.bfloat16:
        raise ValueError("C012 partial outputs must have identical BF16 shapes")
    if output.ndim != 3 or output.shape[1] <= 0 or output.shape[2] != 512:
        raise ValueError("C012 partial output must be [tokens, heads, 512]")
    expected_lse_shape = output.shape[:2]
    if (
        output_lse.shape != expected_lse_shape
        or partial_lse.shape != expected_lse_shape
        or output_lse.dtype != torch.float32
        or partial_lse.dtype != torch.float32
    ):
        raise ValueError("C012 LSE buffers must be FP32 [tokens, heads]")
    tensors = (output, output_lse, partial_output, partial_lse)
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("C012 merge buffers must be contiguous")
    if any(tensor.device != output.device for tensor in tensors):
        raise ValueError("C012 merge buffers must share one device")
    _merge_oscar_chunked_attention_kernel[(output.shape[0], output.shape[1])](
        output,
        output_lse,
        partial_output,
        partial_lse,
        output.shape[0],
        num_heads=output.shape[1],
        head_size=512,
        block_d=512,
        num_warps=4,
        num_stages=1,
    )


def restore_oscar_mla_hp_rows(
    *,
    positions: torch.Tensor,
    hp_rows: torch.Tensor,
    page_ids: torch.Tensor,
    page_offsets: torch.Tensor,
    num_rows: int,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    inverse_rotation: torch.Tensor,
    history_rotated: torch.Tensor,
    restored: torch.Tensor,
) -> None:
    """Restore cached prefix/recent rows from canonical block-indexed INT2."""
    max_rows = prefix.shape[0] * (prefix.shape[1] + recent.shape[1])
    if not 0 < num_rows <= max_rows:
        raise ValueError("invalid OSCAR MLA restore row count")
    for name, tensor in (
        ("positions", positions),
        ("hp_rows", hp_rows),
        ("page_ids", page_ids),
        ("page_offsets", page_offsets),
    ):
        if tensor.ndim != 1 or tensor.dtype not in (torch.int32, torch.int64):
            raise ValueError(f"{name} must be a one-dimensional integer tensor")
        if tensor.shape[0] < num_rows:
            raise ValueError(f"{name} is smaller than num_rows")
    latent_rank = prefix.shape[2]
    if history_data.shape[2] * 4 != latent_rank:
        raise ValueError("canonical INT2 rank does not match BF16 windows")
    if inverse_rotation.shape != (latent_rank, latent_rank):
        raise ValueError("inverse rotation does not match the OSCAR latent rank")
    if history_rotated.shape[0] < num_rows or history_rotated.shape[1] != latent_rank:
        raise ValueError("history restore workspace is too small")
    if restored.shape[0] < num_rows or restored.shape[1] < latent_rank:
        raise ValueError("restored BF16 workspace is too small")
    tensors = (
        positions,
        hp_rows,
        page_ids,
        page_offsets,
        history_data,
        history_scale,
        history_zero,
        prefix,
        recent,
        inverse_rotation,
        history_rotated,
        restored,
    )
    if any(tensor.device != prefix.device for tensor in tensors):
        raise ValueError("all OSCAR MLA restore tensors must share one device")

    block_d = triton.next_power_of_2(latent_rank)
    _gather_canonical_restore_kernel[(num_rows,)](
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        history_rotated,
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
        stride_output_row=history_rotated.stride(0),
        stride_output_dim=history_rotated.stride(1),
        latent_rank=latent_rank,
        group_size=latent_rank // history_scale.shape[2],
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    restored_latent = restored[:num_rows, :latent_rank]
    torch.addmm(
        restored_latent,
        history_rotated[:num_rows],
        inverse_rotation,
        beta=0.0,
        alpha=1.0,
        out=restored_latent,
    )
    _scatter_restored_hp_kernel[(num_rows,)](
        restored_latent,
        positions,
        hp_rows,
        prefix,
        recent,
        num_rows,
        stride_restored_row=restored_latent.stride(0),
        stride_restored_dim=restored_latent.stride(1),
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_dim=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_dim=recent.stride(2),
        prefix_tokens=prefix.shape[1],
        recent_tokens=recent.shape[1],
        latent_rank=latent_rank,
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )


def materialize_oscar_mla_bf16_rows(
    *,
    positions: torch.Tensor | None,
    num_rows: int,
    row_offset: int = 0,
    num_requests: int,
    prefix: torch.Tensor,
    recent: torch.Tensor,
    rope: torch.Tensor,
    rope_block_table: torch.Tensor,
    history_data: torch.Tensor,
    history_scale: torch.Tensor,
    history_zero: torch.Tensor,
    history_page_table: torch.Tensor,
    hp_rows: torch.Tensor,
    seq_lens: torch.Tensor,
    inverse_rotation: torch.Tensor,
    history_rotated: torch.Tensor,
    history_mask: torch.Tensor,
    output_kv: torch.Tensor,
    remapped_indices: torch.Tensor,
    recent_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize batch-1 logical or selected OSCAR rows as contiguous BF16 KV."""
    if num_rows <= 0:
        raise ValueError("num_rows must be positive")
    if row_offset < 0 or (positions is not None and row_offset != 0):
        raise ValueError("row_offset is only supported for logical prefill rows")
    if num_requests != 1:
        raise ValueError("C008 materialization requires one active request")
    if positions is not None and (
        positions.ndim != 1
        or positions.shape[0] < num_rows
        or positions.dtype not in (torch.int32, torch.int64)
        or not positions.is_contiguous()
    ):
        raise ValueError("positions must be contiguous int32/int64 with num_rows items")
    if prefix.shape[0] < 1 or recent.shape[0] < 1:
        raise ValueError("C008 materialization requires allocated prefix/recent rows")
    if recent_tokens is None:
        recent_tokens = recent.shape[1]
    if (
        prefix.shape[1:] != (64, 512)
        or recent.shape[2] != 512
        or not 0 < recent_tokens <= recent.shape[1]
    ):
        raise ValueError(
            "C008 prototype requires prefix64/logical recent<=capacity/D512"
        )
    if history_data.shape[2] != 128 or history_scale.shape[2] != 4:
        raise ValueError("C008 prototype requires D512 INT2 group128 history")
    if rope.shape[2] != 64 or inverse_rotation.shape != (512, 512):
        raise ValueError("C008 prototype requires RoPE64 and a 512x512 rotation")
    if inverse_rotation.dtype != torch.bfloat16 or not inverse_rotation.is_contiguous():
        raise ValueError("dense inverse rotation must be contiguous BF16")
    if rope_block_table.shape[0] < 1 or history_page_table.shape[0] < 1:
        raise ValueError("C008 materialization requires allocated page-table rows")
    if hp_rows.shape[0] < 1 or seq_lens.shape[0] < 1:
        raise ValueError("C008 prototype requires one hp row and sequence length")
    if (
        history_rotated.shape[0] < num_rows
        or history_rotated.shape[1] != 512
        or history_rotated.dtype != torch.bfloat16
        or not history_rotated.is_contiguous()
    ):
        raise ValueError("history_rotated workspace must be contiguous BF16 [N, 512]")
    if history_mask.shape[0] < num_rows or history_mask.dtype != torch.uint8:
        raise ValueError("history_mask workspace is too small or has wrong dtype")
    if output_kv.shape[0] < num_rows or output_kv.shape[1:] != (576,):
        raise ValueError("output_kv workspace is too small or has wrong shape")
    if output_kv.dtype != torch.bfloat16:
        raise ValueError("output_kv workspace must be BF16")
    if remapped_indices.shape[0] < num_rows or remapped_indices.dtype != torch.int32:
        raise ValueError("remapped_indices workspace is too small or has wrong dtype")
    tensors: tuple[torch.Tensor, ...] = (
        prefix,
        recent,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        inverse_rotation,
        history_rotated,
        history_mask,
        output_kv,
        remapped_indices,
    )
    if positions is not None:
        tensors += (positions,)
    if any(t.device != prefix.device for t in tensors):
        raise ValueError("all materialization tensors must share one device")

    positions_arg = seq_lens if positions is None else positions
    block_d = 512
    _gather_oscar_mla_rows_kernel[(num_rows, triton.cdiv(512, block_d))](
        positions_arg,
        prefix,
        recent,
        rope,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        history_page_table,
        hp_rows,
        seq_lens,
        history_rotated,
        history_mask,
        output_kv,
        remapped_indices,
        num_rows,
        row_offset,
        stride_prefix_row=prefix.stride(0),
        stride_prefix_token=prefix.stride(1),
        stride_prefix_d=prefix.stride(2),
        stride_recent_row=recent.stride(0),
        stride_recent_token=recent.stride(1),
        stride_recent_d=recent.stride(2),
        stride_rope_block=rope.stride(0),
        stride_rope_token=rope.stride(1),
        stride_rope_d=rope.stride(2),
        stride_rope_table_b=rope_block_table.stride(0),
        stride_rope_table_page=rope_block_table.stride(1),
        stride_data_page=history_data.stride(0),
        stride_data_token=history_data.stride(1),
        stride_data_byte=history_data.stride(2),
        stride_scale_page=history_scale.stride(0),
        stride_scale_token=history_scale.stride(1),
        stride_scale_group=history_scale.stride(2),
        stride_zero_page=history_zero.stride(0),
        stride_zero_token=history_zero.stride(1),
        stride_zero_group=history_zero.stride(2),
        stride_history_table_b=history_page_table.stride(0),
        stride_history_table_page=history_page_table.stride(1),
        stride_history_row=history_rotated.stride(0),
        stride_history_d=history_rotated.stride(1),
        stride_output_row=output_kv.stride(0),
        stride_output_d=output_kv.stride(1),
        prefix_tokens=64,
        recent_tokens=recent_tokens,
        recent_capacity_tokens=recent.shape[1],
        rope_block_size=rope.shape[1],
        rope_head_size=64,
        history_block_size=history_data.shape[1],
        latent_rank=512,
        group_size=128,
        block_d=block_d,
        use_positions=positions is not None,
        num_warps=4,
        num_stages=1,
    )
    dense_output = output_kv[:num_rows, :512]
    torch.addmm(
        dense_output,
        history_rotated[:num_rows],
        inverse_rotation,
        beta=1.0,
        alpha=1.0,
        out=dense_output,
    )
    return (
        output_kv[:num_rows].view(num_rows, 1, 576),
        remapped_indices[:num_rows].view(1, 1, num_rows),
    )
