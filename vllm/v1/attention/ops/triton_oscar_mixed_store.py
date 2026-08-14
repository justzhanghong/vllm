# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton writes and demotion for OSCAR's BF16 prefix/recent arena."""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.oscar_cache_contract import (
    validate_oscar_separated_arenas,
)
from vllm.v1.attention.ops.triton_oscar_store import _quantize_pack_int2_vec


@triton.jit
def _store_hp_kernel(
    Key_ptr,
    Value_ptr,
    Prefix_ptr,
    Recent_ptr,
    Token_req_ptr,
    Query_start_ptr,
    Seq_lens_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    num_tokens,
    stride_key_token: tl.constexpr,
    stride_key_head: tl.constexpr,
    stride_key_dim: tl.constexpr,
    stride_value_token: tl.constexpr,
    stride_value_head: tl.constexpr,
    stride_value_dim: tl.constexpr,
    stride_prefix_slot: tl.constexpr,
    stride_prefix_head: tl.constexpr,
    stride_prefix_kv: tl.constexpr,
    stride_recent_slot: tl.constexpr,
    stride_recent_head: tl.constexpr,
    stride_recent_kv: tl.constexpr,
    stride_prefix_pages_req: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // NUM_HEADS
    head_idx = pid % NUM_HEADS
    if token_idx >= num_tokens:
        return

    req_idx = tl.load(Token_req_ptr + token_idx)
    query_start = tl.load(Query_start_ptr + req_idx)
    query_end = tl.load(Query_start_ptr + req_idx + 1)
    final_seq_len = tl.load(Seq_lens_ptr + req_idx)
    hp_row = tl.load(HP_rows_ptr + req_idx)
    if hp_row < 0:
        return
    pos = final_seq_len - (query_end - query_start) + token_idx - query_start
    recent_start = tl.maximum(PREFIX_TOKENS, final_seq_len - RECENT_TOKENS)
    is_prefix = pos < PREFIX_TOKENS
    is_recent = pos >= recent_start
    if not (is_prefix or is_recent):
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    key_base = (
        token_idx.to(tl.int64) * stride_key_token
        + head_idx.to(tl.int64) * stride_key_head
    )
    value_base = (
        token_idx.to(tl.int64) * stride_value_token
        + head_idx.to(tl.int64) * stride_value_head
    )
    key = tl.load(Key_ptr + key_base + d_offs * stride_key_dim, mask=d_mask).to(
        tl.bfloat16
    )
    value = tl.load(Value_ptr + value_base + d_offs * stride_value_dim, mask=d_mask).to(
        tl.bfloat16
    )

    if USE_PREFIX_PAGE_TABLE:
        prefix_page = tl.load(
            Prefix_pages_ptr
            + req_idx * stride_prefix_pages_req
            + pos // PREFIX_BLOCK_SIZE,
            mask=is_prefix,
            other=0,
        )
        prefix_idx = prefix_page * PREFIX_BLOCK_SIZE + pos % PREFIX_BLOCK_SIZE
    else:
        prefix_idx = hp_row * PREFIX_TOKENS + pos
    prefix_base = (
        prefix_idx.to(tl.int64) * stride_prefix_slot
        + head_idx.to(tl.int64) * stride_prefix_head
    )
    tl.store(
        Prefix_ptr + prefix_base + d_offs,
        key,
        mask=d_mask & is_prefix,
    )
    tl.store(
        Prefix_ptr + prefix_base + stride_prefix_kv + d_offs,
        value,
        mask=d_mask & is_prefix,
    )

    recent_idx = (pos - PREFIX_TOKENS) % RECENT_CAPACITY
    recent_base = (hp_row * RECENT_CAPACITY + recent_idx).to(
        tl.int64
    ) * stride_recent_slot + head_idx.to(tl.int64) * stride_recent_head
    tl.store(
        Recent_ptr + recent_base + d_offs,
        key,
        mask=d_mask & is_recent & ~is_prefix,
    )
    tl.store(
        Recent_ptr + recent_base + stride_recent_kv + d_offs,
        value,
        mask=d_mask & is_recent & ~is_prefix,
    )


@triton.jit
def _fused_qk_rotation_hp_store_kernel(
    Query_ptr,
    Key_ptr,
    Value_ptr,
    K_rotation_ptr,
    Q_rotation_ptr,
    Q_rot_ptr,
    Prefix_ptr,
    Recent_ptr,
    Token_req_ptr,
    Query_start_ptr,
    Seq_lens_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    num_tokens,
    stride_query_token: tl.constexpr,
    stride_query_head: tl.constexpr,
    stride_key_token: tl.constexpr,
    stride_key_head: tl.constexpr,
    stride_value_token: tl.constexpr,
    stride_value_head: tl.constexpr,
    stride_k_rotation_row: tl.constexpr,
    stride_q_rotation_row: tl.constexpr,
    stride_q_rot_token: tl.constexpr,
    stride_q_rot_head: tl.constexpr,
    stride_prefix_slot: tl.constexpr,
    stride_prefix_head: tl.constexpr,
    stride_prefix_kv: tl.constexpr,
    stride_recent_slot: tl.constexpr,
    stride_recent_head: tl.constexpr,
    stride_recent_kv: tl.constexpr,
    stride_prefix_pages_req: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    PREFIX_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    column_tile = tl.program_id(1)
    if token_idx >= num_tokens:
        return

    input_offs = tl.arange(0, 128)
    output_offs = column_tile * 16 + tl.arange(0, 16)
    query_heads = tl.arange(0, 32)
    kv_heads = tl.arange(0, 16)
    kv_head_mask = kv_heads < 8

    query_base = token_idx.to(tl.int64) * stride_query_token
    query = tl.load(
        Query_ptr
        + query_base
        + query_heads[:, None].to(tl.int64) * stride_query_head
        + input_offs[None, :]
    ).to(tl.bfloat16)
    q_rotation = tl.load(
        Q_rotation_ptr
        + input_offs[:, None].to(tl.int64) * stride_q_rotation_row
        + output_offs[None, :]
    ).to(tl.bfloat16)
    query_rot = tl.dot(query, q_rotation)
    query_rot_base = token_idx.to(tl.int64) * stride_q_rot_token
    tl.store(
        Q_rot_ptr
        + query_rot_base
        + query_heads[:, None].to(tl.int64) * stride_q_rot_head
        + output_offs[None, :],
        query_rot.to(tl.bfloat16),
    )

    key_base = token_idx.to(tl.int64) * stride_key_token
    key = tl.load(
        Key_ptr
        + key_base
        + kv_heads[:, None].to(tl.int64) * stride_key_head
        + input_offs[None, :],
        mask=kv_head_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    k_rotation = tl.load(
        K_rotation_ptr
        + input_offs[:, None].to(tl.int64) * stride_k_rotation_row
        + output_offs[None, :]
    ).to(tl.float32)
    key_rot = tl.dot(key, k_rotation, input_precision="ieee")
    value_base = token_idx.to(tl.int64) * stride_value_token
    value = tl.load(
        Value_ptr
        + value_base
        + kv_heads[:, None].to(tl.int64) * stride_value_head
        + output_offs[None, :],
        mask=kv_head_mask[:, None],
        other=0.0,
    ).to(tl.bfloat16)

    req_idx = tl.load(Token_req_ptr + token_idx)
    query_start = tl.load(Query_start_ptr + req_idx)
    query_end = tl.load(Query_start_ptr + req_idx + 1)
    final_seq_len = tl.load(Seq_lens_ptr + req_idx)
    hp_row = tl.load(HP_rows_ptr + req_idx)
    if hp_row < 0:
        return
    pos = final_seq_len - (query_end - query_start) + token_idx - query_start
    recent_start = tl.maximum(PREFIX_TOKENS, final_seq_len - RECENT_TOKENS)
    is_prefix = pos < PREFIX_TOKENS
    is_recent = pos >= recent_start
    if not (is_prefix or is_recent):
        return

    prefix_page = tl.load(
        Prefix_pages_ptr + req_idx * stride_prefix_pages_req + pos // PREFIX_BLOCK_SIZE,
        mask=is_prefix,
        other=0,
    )
    prefix_idx = prefix_page * PREFIX_BLOCK_SIZE + pos % PREFIX_BLOCK_SIZE
    prefix_base = (
        prefix_idx.to(tl.int64) * stride_prefix_slot
        + kv_heads[:, None].to(tl.int64) * stride_prefix_head
        + output_offs[None, :]
    )
    prefix_mask = kv_head_mask[:, None] & is_prefix
    tl.store(Prefix_ptr + prefix_base, key_rot.to(tl.bfloat16), mask=prefix_mask)
    tl.store(
        Prefix_ptr + prefix_base + stride_prefix_kv,
        value,
        mask=prefix_mask,
    )

    recent_idx = (pos - PREFIX_TOKENS) % RECENT_CAPACITY
    recent_base = (
        (hp_row * RECENT_CAPACITY + recent_idx).to(tl.int64) * stride_recent_slot
        + kv_heads[:, None].to(tl.int64) * stride_recent_head
        + output_offs[None, :]
    )
    recent_mask = kv_head_mask[:, None] & is_recent & ~is_prefix
    tl.store(Recent_ptr + recent_base, key_rot.to(tl.bfloat16), mask=recent_mask)
    tl.store(
        Recent_ptr + recent_base + stride_recent_kv,
        value,
        mask=recent_mask,
    )


@triton.jit
def _demote_hp_kernel(
    Recent_ptr,
    K_data_ptr,
    V_data_ptr,
    K_meta_ptr,
    V_meta_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    HP_rows_ptr,
    Shared_hit_lens_ptr,
    Query_start_ptr,
    stride_recent_slot: tl.constexpr,
    stride_recent_head: tl.constexpr,
    stride_recent_kv: tl.constexpr,
    stride_data_block: tl.constexpr,
    stride_data_pos: tl.constexpr,
    stride_data_head: tl.constexpr,
    stride_meta_block: tl.constexpr,
    stride_meta_pos: tl.constexpr,
    stride_meta_head: tl.constexpr,
    stride_bt_b: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,
):
    req_idx = tl.program_id(0)
    demote_idx = tl.program_id(1)
    head_idx = tl.program_id(2)
    final_seq_len = tl.load(Seq_lens_ptr + req_idx)
    if SINGLE_TOKEN:
        query_len = 1
    else:
        query_len = tl.load(Query_start_ptr + req_idx + 1) - tl.load(
            Query_start_ptr + req_idx
        )
    cached_len = final_seq_len - query_len
    shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
    old_recent_start = tl.maximum(
        tl.maximum(PREFIX_TOKENS, shared_hit_len), cached_len - RECENT_TOKENS
    )
    new_recent_start = tl.maximum(PREFIX_TOKENS, final_seq_len - RECENT_TOKENS)
    demote_end = tl.minimum(cached_len, new_recent_start)
    demote_position = old_recent_start + demote_idx
    if demote_position >= demote_end:
        return

    page_idx = demote_position // BLOCK_SIZE
    page_off = demote_position % BLOCK_SIZE
    block = tl.load(Block_table_ptr + req_idx * stride_bt_b + page_idx).to(tl.int64)
    slot = block * BLOCK_SIZE + page_off
    hp_row = tl.load(HP_rows_ptr + req_idx)
    recent_idx = (demote_position - PREFIX_TOKENS) % RECENT_CAPACITY
    recent_offset = hp_row * RECENT_CAPACITY + recent_idx
    hp_base = (
        recent_offset * stride_recent_slot + head_idx.to(tl.int64) * stride_recent_head
    )
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    key = tl.load(Recent_ptr + hp_base + d_offs, mask=d_mask).to(tl.float32)
    value = tl.load(Recent_ptr + hp_base + stride_recent_kv + d_offs, mask=d_mask).to(
        tl.float32
    )

    block = (slot // BLOCK_SIZE).to(tl.int64)
    offset = (slot % BLOCK_SIZE).to(tl.int64)
    data_dst = (
        block * stride_data_block
        + offset * stride_data_pos
        + head_idx.to(tl.int64) * stride_data_head
    )
    meta_dst = (
        block * stride_meta_block
        + offset * stride_meta_pos
        + head_idx.to(tl.int64) * stride_meta_head
    )
    _quantize_pack_int2_vec(
        key,
        K_data_ptr,
        K_meta_ptr,
        data_dst,
        meta_dst,
        d_offs,
        d_mask,
        D=HEAD_DIM,
        LEVELS=KEY_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=K_CLIP_INDEX,
    )
    _quantize_pack_int2_vec(
        value,
        V_data_ptr,
        V_meta_ptr,
        data_dst,
        meta_dst,
        d_offs,
        d_mask,
        D=HEAD_DIM,
        LEVELS=VALUE_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=V_CLIP_INDEX,
    )


def _clip_index(ratio: float, head_dim: int) -> int:
    if ratio <= 0.0:
        return -1
    return min(int(ratio * head_dim), head_dim - 1)


def is_oscar_fused_qk_rotation_hp_store_eligible(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_rotation: torch.Tensor,
    q_rotation: torch.Tensor,
    prefix_cache: torch.Tensor,
    recent_cache: torch.Tensor,
    *,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    prefix_page_ids: torch.Tensor,
    prefix_block_size: int,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int,
) -> bool:
    """Fail closed unless the C110 pure-decode specialization is exact."""
    tensors = (
        query,
        key,
        value,
        k_rotation,
        q_rotation,
        prefix_cache,
        recent_cache,
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        hp_row_ids,
        prefix_page_ids,
    )
    integer_metadata = (
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        hp_row_ids,
        prefix_page_ids,
    )
    return bool(
        query.shape == (1, 32, 128)
        and key.shape == (1, 8, 128)
        and value.shape == key.shape
        and query.dtype == key.dtype == value.dtype == torch.bfloat16
        and query.stride()[1:] == (128, 1)
        and key.stride()[1:] == (128, 1)
        and value.stride()[1:] == (128, 1)
        and k_rotation.shape == (128, 128)
        and k_rotation.dtype == torch.float32
        and k_rotation.is_contiguous()
        and q_rotation.shape == (128, 128)
        and q_rotation.dtype == torch.bfloat16
        and q_rotation.is_contiguous()
        and prefix_cache.shape[1:] == (8, 2, 128)
        and recent_cache.shape == (recent_capacity, 8, 2, 128)
        and prefix_cache.dtype == recent_cache.dtype == torch.bfloat16
        and prefix_cache.is_contiguous()
        and recent_cache.is_contiguous()
        and token_to_req_indices.shape == (1,)
        and query_start_loc.shape == (2,)
        and seq_lens.shape == (1,)
        and hp_row_ids.shape == (1,)
        and prefix_page_ids.shape == (1, 4)
        and all(tensor.dtype == torch.int32 for tensor in integer_metadata)
        and all(tensor.device == query.device for tensor in tensors)
        and prefix_block_size == 16
        and prefix_tokens == 64
        and recent_tokens == 256
        and recent_capacity == 272
        and prefix_cache.shape[0] >= prefix_tokens
    )


def oscar_fused_qk_rotation_hp_store(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_rotation: torch.Tensor,
    q_rotation: torch.Tensor,
    prefix_cache: torch.Tensor,
    recent_cache: torch.Tensor,
    *,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    prefix_page_ids: torch.Tensor,
    prefix_block_size: int,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int,
) -> torch.Tensor:
    if not is_oscar_fused_qk_rotation_hp_store_eligible(
        query,
        key,
        value,
        k_rotation,
        q_rotation,
        prefix_cache,
        recent_cache,
        token_to_req_indices=token_to_req_indices,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        hp_row_ids=hp_row_ids,
        prefix_page_ids=prefix_page_ids,
        prefix_block_size=prefix_block_size,
        prefix_tokens=prefix_tokens,
        recent_tokens=recent_tokens,
        recent_capacity=recent_capacity,
    ):
        raise ValueError("unsupported OSCAR fused Q/K rotation HP-store layout")
    q_rot = torch.empty_like(query)
    num_tokens = query.shape[0]
    _fused_qk_rotation_hp_store_kernel[(num_tokens, 8)](
        query,
        key,
        value,
        k_rotation,
        q_rotation,
        q_rot,
        prefix_cache,
        recent_cache,
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        hp_row_ids,
        prefix_page_ids,
        num_tokens,
        stride_query_token=query.stride(0),
        stride_query_head=query.stride(1),
        stride_key_token=key.stride(0),
        stride_key_head=key.stride(1),
        stride_value_token=value.stride(0),
        stride_value_head=value.stride(1),
        stride_k_rotation_row=k_rotation.stride(0),
        stride_q_rotation_row=q_rotation.stride(0),
        stride_q_rot_token=q_rot.stride(0),
        stride_q_rot_head=q_rot.stride(1),
        stride_prefix_slot=prefix_cache.stride(0),
        stride_prefix_head=prefix_cache.stride(1),
        stride_prefix_kv=prefix_cache.stride(2),
        stride_recent_slot=recent_cache.stride(0),
        stride_recent_head=recent_cache.stride(1),
        stride_recent_kv=recent_cache.stride(2),
        stride_prefix_pages_req=prefix_page_ids.stride(0),
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        RECENT_CAPACITY=recent_capacity,
        PREFIX_BLOCK_SIZE=prefix_block_size,
        num_warps=4,
        num_stages=1,
    )
    return q_rot


def oscar_store_hp(
    key_rot: torch.Tensor,
    value_rot: torch.Tensor,
    prefix_cache: torch.Tensor,
    recent_cache: torch.Tensor,
    *,
    token_to_req_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    prefix_page_ids: torch.Tensor | None = None,
    prefix_block_size: int = 16,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int | None = None,
) -> None:
    num_tokens, num_heads, head_dim = key_rot.shape
    if num_tokens == 0:
        return
    block_d = triton.next_power_of_2(head_dim)
    use_prefix_page_table = prefix_page_ids is not None
    recent_capacity = recent_capacity or recent_tokens
    if prefix_page_ids is None:
        prefix_page_ids = hp_row_ids
    else:
        if prefix_page_ids.ndim != 2 or prefix_page_ids.shape[1] <= 0:
            raise ValueError("OSCAR prefix page table must be a non-empty 2D tensor")
        if prefix_page_ids.shape[1] * prefix_block_size != prefix_tokens:
            raise ValueError(
                "OSCAR prefix page table width must cover the prefix window"
            )
    _store_hp_kernel[(num_tokens * num_heads,)](
        key_rot,
        value_rot,
        prefix_cache,
        recent_cache,
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        hp_row_ids,
        prefix_page_ids,
        num_tokens,
        stride_key_token=key_rot.stride(0),
        stride_key_head=key_rot.stride(1),
        stride_key_dim=key_rot.stride(2),
        stride_value_token=value_rot.stride(0),
        stride_value_head=value_rot.stride(1),
        stride_value_dim=value_rot.stride(2),
        stride_prefix_slot=prefix_cache.stride(0),
        stride_prefix_head=prefix_cache.stride(1),
        stride_prefix_kv=prefix_cache.stride(2),
        stride_recent_slot=recent_cache.stride(0),
        stride_recent_head=recent_cache.stride(1),
        stride_recent_kv=recent_cache.stride(2),
        stride_prefix_pages_req=(
            prefix_page_ids.stride(0) if use_prefix_page_table else 0
        ),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        RECENT_CAPACITY=recent_capacity,
        PREFIX_BLOCK_SIZE=prefix_block_size,
        USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )


def oscar_demote_hp(
    recent_cache: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    hp_row_ids: torch.Tensor,
    *,
    shared_hit_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = 1,
    prefix_tokens: int,
    recent_tokens: int,
    recent_capacity: int | None = None,
    key_levels: int,
    value_levels: int,
    data_bytes: int,
    k_clip_ratio: float,
    v_clip_ratio: float,
) -> None:
    validate_oscar_separated_arenas(
        k_data,
        v_data,
        k_meta,
        v_meta,
        data_bytes=data_bytes,
    )
    if max_query_len <= 0:
        return
    recent_capacity = recent_capacity or recent_tokens
    num_heads = k_data.shape[2]
    head_dim = data_bytes * 4
    block_d = triton.next_power_of_2(head_dim)
    block_pack = triton.next_power_of_2(data_bytes)
    num_reqs = seq_lens.shape[0]
    single_token = query_start_loc is None
    if query_start_loc is None:
        query_start_loc = seq_lens
    if shared_hit_tokens is None:
        shared_hit_tokens = torch.zeros_like(hp_row_ids)
    elif shared_hit_tokens.ndim != 1:
        raise ValueError("OSCAR shared hit lengths must be a 1D tensor")
    _demote_hp_kernel[(num_reqs, max_query_len, num_heads)](
        recent_cache,
        k_data,
        v_data,
        k_meta,
        v_meta,
        block_table,
        seq_lens,
        hp_row_ids,
        shared_hit_tokens,
        query_start_loc,
        stride_recent_slot=recent_cache.stride(0),
        stride_recent_head=recent_cache.stride(1),
        stride_recent_kv=recent_cache.stride(2),
        stride_data_block=k_data.stride(0),
        stride_data_pos=k_data.stride(1),
        stride_data_head=k_data.stride(2),
        stride_meta_block=k_meta.stride(0),
        stride_meta_pos=k_meta.stride(1),
        stride_meta_head=k_meta.stride(2),
        stride_bt_b=block_table.stride(0),
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=k_data.shape[1],
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        DATA_BYTES=data_bytes,
        K_CLIP_INDEX=_clip_index(k_clip_ratio, head_dim),
        V_CLIP_INDEX=_clip_index(v_clip_ratio, head_dim),
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        RECENT_CAPACITY=recent_capacity,
        SINGLE_TOKEN=single_token,
        BLOCK_D=block_d,
        BLOCK_PACK=block_pack,
        num_warps=4,
        num_stages=1,
    )
