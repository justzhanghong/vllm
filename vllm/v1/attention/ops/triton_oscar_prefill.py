# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiled fused OSCAR attention over cached mixed BF16/INT2 K/V."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _oscar_materialize_prefill_kv_kernel(
    Current_k_ptr,
    Current_v_ptr,
    KV_cache_ptr,
    KV_meta_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    Block_table_ptr,
    Cached_lens_ptr,
    Query_start_ptr,
    Seq_start_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Shared_hit_lens_ptr,
    K_out_ptr,
    V_out_ptr,
    stride_current_k_token,
    stride_current_k_head,
    stride_current_k_dim,
    stride_current_v_token,
    stride_current_v_head,
    stride_current_v_dim,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_prefix_slot,
    stride_prefix_head,
    stride_prefix_kv,
    stride_recent_slot,
    stride_recent_head,
    stride_recent_kv,
    stride_bt_req,
    stride_prefix_pages_req,
    stride_out_token,
    stride_out_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    req_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    token_offs = tl.program_id(2) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    seq_start = tl.load(Seq_start_ptr + req_idx)
    seq_end = tl.load(Seq_start_ptr + req_idx + 1)
    seq_len = seq_end - seq_start
    token_mask = token_offs < seq_len
    cached_len = tl.load(Cached_lens_ptr + req_idx)
    is_cached = token_mask & (token_offs < cached_len)
    d_offs = tl.arange(0, BLOCK_D)
    dim_mask = d_offs < HEAD_DIM

    page_idx = token_offs // BLOCK_SIZE
    page_off = token_offs % BLOCK_SIZE
    block_num = tl.load(
        Block_table_ptr + req_idx * stride_bt_req + page_idx,
        mask=is_cached,
        other=0,
    ).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + page_off.to(tl.int64) * stride_cache_pos
        + tl.cast(head_idx, tl.int64) * stride_cache_head
    )
    shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
    recent_start = tl.maximum(
        tl.maximum(PREFIX_TOKENS, shared_hit_len), cached_len - RECENT_TOKENS
    )
    is_prefix = is_cached & (token_offs < PREFIX_TOKENS)
    is_recent = is_cached & (token_offs >= recent_start)
    quant_mask = is_cached & ~(is_prefix | is_recent)

    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2
    k_byte = tl.load(
        KV_cache_ptr + slot_base[:, None] + byte_idx[None, :],
        mask=quant_mask[:, None] & dim_mask[None, :],
        other=0,
    ).to(tl.int32)
    q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)
    k_meta = slot_base + KEY_DATA_BYTES
    k_scale = tl.load(KV_meta_ptr + k_meta // 2, mask=quant_mask, other=0.0).to(
        tl.float32
    )
    k_zero = tl.load(KV_meta_ptr + k_meta // 2 + 1, mask=quant_mask, other=0.0).to(
        tl.float32
    )
    keys = (q_k - k_zero[:, None]) * k_scale[:, None]

    v_base = slot_base + KEY_PACKED
    v_byte = tl.load(
        KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
        mask=quant_mask[:, None] & dim_mask[None, :],
        other=0,
    ).to(tl.int32)
    q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)
    v_meta = v_base + VALUE_DATA_BYTES
    v_scale = tl.load(KV_meta_ptr + v_meta // 2, mask=quant_mask, other=0.0).to(
        tl.float32
    )
    v_zero = tl.load(KV_meta_ptr + v_meta // 2 + 1, mask=quant_mask, other=0.0).to(
        tl.float32
    )
    values = (q_v - v_zero[:, None]) * v_scale[:, None]

    prefix_page = tl.load(
        Prefix_pages_ptr + req_idx * stride_prefix_pages_req + page_idx,
        mask=is_prefix,
        other=0,
    )
    prefix_idx = prefix_page * BLOCK_SIZE + page_off
    prefix_base = (
        prefix_idx.to(tl.int64) * stride_prefix_slot
        + tl.cast(head_idx, tl.int64) * stride_prefix_head
    )
    prefix_keys = tl.load(
        Prefix_cache_ptr + prefix_base[:, None] + d_offs[None, :],
        mask=is_prefix[:, None] & dim_mask[None, :],
        other=0.0,
    )
    prefix_values = tl.load(
        Prefix_cache_ptr + prefix_base[:, None] + stride_prefix_kv + d_offs[None, :],
        mask=is_prefix[:, None] & dim_mask[None, :],
        other=0.0,
    )
    hp_row = tl.load(HP_rows_ptr + req_idx)
    recent_idx = hp_row * RECENT_TOKENS + (token_offs - PREFIX_TOKENS) % RECENT_TOKENS
    recent_base = (
        recent_idx.to(tl.int64) * stride_recent_slot
        + tl.cast(head_idx, tl.int64) * stride_recent_head
    )
    recent_keys = tl.load(
        Recent_cache_ptr + recent_base[:, None] + d_offs[None, :],
        mask=is_recent[:, None] & dim_mask[None, :],
        other=0.0,
    )
    recent_values = tl.load(
        Recent_cache_ptr + recent_base[:, None] + stride_recent_kv + d_offs[None, :],
        mask=is_recent[:, None] & dim_mask[None, :],
        other=0.0,
    )
    keys = tl.where(is_prefix[:, None], prefix_keys, keys)
    keys = tl.where(is_recent[:, None], recent_keys, keys)
    values = tl.where(is_prefix[:, None], prefix_values, values)
    values = tl.where(is_recent[:, None], recent_values, values)

    current_idx = tl.load(Query_start_ptr + req_idx) + token_offs - cached_len
    is_current = token_mask & ~is_cached
    current_k_base = (
        current_idx.to(tl.int64) * stride_current_k_token
        + tl.cast(head_idx, tl.int64) * stride_current_k_head
    )
    current_v_base = (
        current_idx.to(tl.int64) * stride_current_v_token
        + tl.cast(head_idx, tl.int64) * stride_current_v_head
    )
    current_keys = tl.load(
        Current_k_ptr
        + current_k_base[:, None]
        + d_offs[None, :] * stride_current_k_dim,
        mask=is_current[:, None] & dim_mask[None, :],
        other=0.0,
    )
    current_values = tl.load(
        Current_v_ptr
        + current_v_base[:, None]
        + d_offs[None, :] * stride_current_v_dim,
        mask=is_current[:, None] & dim_mask[None, :],
        other=0.0,
    )
    keys = tl.where(is_current[:, None], current_keys, keys)
    values = tl.where(is_current[:, None], current_values, values)

    output_token = seq_start + token_offs
    output_base = (
        output_token.to(tl.int64) * stride_out_token
        + tl.cast(head_idx, tl.int64) * stride_out_head
    )
    output_mask = token_mask[:, None] & dim_mask[None, :]
    tl.store(K_out_ptr + output_base[:, None] + d_offs[None, :], keys, mask=output_mask)
    tl.store(
        V_out_ptr + output_base[:, None] + d_offs[None, :],
        values,
        mask=output_mask,
    )


def oscar_materialize_prefill_kv(
    current_key: torch.Tensor,
    current_value: torch.Tensor,
    kv_cache: torch.Tensor,
    prefix_cache: torch.Tensor,
    recent_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_start_loc: torch.Tensor,
    hp_row_ids: torch.Tensor,
    prefix_page_ids: torch.Tensor,
    shared_hit_tokens: torch.Tensor,
    output_key: torch.Tensor,
    output_value: torch.Tensor,
    *,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
    prefix_tokens: int,
    recent_tokens: int,
    max_seq_len: int,
) -> None:
    num_reqs = cached_lens.shape[0]
    head_dim = current_key.shape[-1]
    block_tokens = 8
    grid = (
        num_reqs,
        current_key.shape[1],
        triton.cdiv(max_seq_len, block_tokens),
    )
    _oscar_materialize_prefill_kv_kernel[grid](
        current_key,
        current_value,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        prefix_cache,
        recent_cache,
        block_table,
        cached_lens,
        query_start_loc,
        seq_start_loc,
        hp_row_ids,
        prefix_page_ids,
        shared_hit_tokens,
        output_key,
        output_value,
        current_key.stride(0),
        current_key.stride(1),
        current_key.stride(2),
        current_value.stride(0),
        current_value.stride(1),
        current_value.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        prefix_cache.stride(0),
        prefix_cache.stride(1),
        prefix_cache.stride(2),
        recent_cache.stride(0),
        recent_cache.stride(1),
        recent_cache.stride(2),
        block_table.stride(0),
        prefix_page_ids.stride(0),
        output_key.stride(0),
        output_key.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=kv_cache.shape[1],
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        BLOCK_TOKENS=block_tokens,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
        num_stages=1,
    )


@triton.jit
def _advance_cached_prefill_attention(
    q,
    keys,
    values,
    m_mask,
    n_mask,
    m_i,
    l_i,
    acc,
    ATTN_SCALE: tl.constexpr,
):
    scores = tl.dot(q, tl.trans(keys.to(tl.bfloat16))) * ATTN_SCALE
    scores = tl.where(m_mask[:, None] & n_mask[None, :], scores, -float("inf"))
    m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
    alpha = tl.exp(m_i - m_ij)
    p = tl.exp(scores - m_ij[:, None])
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), values.to(tl.bfloat16))
    l_i = l_i * alpha + tl.sum(p, axis=1)
    return m_ij, l_i, acc


@triton.jit
def _oscar_cached_prefill_kernel(
    Q_ptr,
    Current_k_ptr,
    Current_v_ptr,
    KV_cache_ptr,
    KV_meta_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Shared_hit_lens_ptr,
    Block_table_ptr,
    Cached_lens_ptr,
    Query_start_ptr,
    Out_ptr,
    LSE_ptr,
    stride_qn,
    stride_qh,
    stride_current_k_token,
    stride_current_k_head,
    stride_current_k_dim,
    stride_current_v_token,
    stride_current_v_head,
    stride_current_v_dim,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_prefix_slot,
    stride_prefix_head,
    stride_prefix_kv,
    stride_recent_slot,
    stride_recent_head,
    stride_recent_kv,
    stride_prefix_pages_req,
    stride_bt_req,
    stride_on,
    stride_oh,
    stride_lseh,
    stride_lsen,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    QUERY_HEADS_PER_PROGRAM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAX_QUERY_LEN: tl.constexpr,
):
    req_idx = tl.program_id(0)
    query_head_group = tl.program_id(1)
    query_block = tl.program_id(2)
    first_query_head = query_head_group * QUERY_HEADS_PER_PROGRAM
    kv_head = first_query_head // KV_GROUP_SIZE

    query_start = tl.load(Query_start_ptr + req_idx)
    query_end = tl.load(Query_start_ptr + req_idx + 1)
    hm_offs = tl.arange(0, BLOCK_M * QUERY_HEADS_PER_PROGRAM)
    query_heads = first_query_head + hm_offs // BLOCK_M
    m_offs = query_start + query_block * BLOCK_M + hm_offs % BLOCK_M
    m_mask = (m_offs < query_end) & (query_heads < NUM_QUERY_HEADS)
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    q = tl.load(
        Q_ptr
        + m_offs[:, None] * stride_qn
        + query_heads[:, None] * stride_qh
        + d_offs[None, :],
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)

    cached_len = tl.load(Cached_lens_ptr + req_idx)
    shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
    recent_start = tl.maximum(
        tl.maximum(PREFIX_TOKENS, shared_hit_len), cached_len - RECENT_TOKENS
    )
    hp_row = tl.load(HP_rows_ptr + req_idx)
    bt_base = req_idx * stride_bt_req

    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2
    m_i = tl.full([BLOCK_M * QUERY_HEADS_PER_PROGRAM], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M * QUERY_HEADS_PER_PROGRAM], tl.float32)
    acc = tl.zeros([BLOCK_M * QUERY_HEADS_PER_PROGRAM, BLOCK_D], tl.float32)

    prefix_end = tl.minimum(cached_len, PREFIX_TOKENS)
    for start_n in range(0, prefix_end, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < prefix_end
        prefix_page = tl.load(
            Prefix_pages_ptr + req_idx * stride_prefix_pages_req + n_offs // BLOCK_SIZE,
            mask=n_mask,
            other=0,
        )
        prefix_idx = prefix_page * BLOCK_SIZE + n_offs % BLOCK_SIZE
        prefix_base = (
            prefix_idx.to(tl.int64) * stride_prefix_slot
            + tl.cast(kv_head, tl.int64) * stride_prefix_head
        )
        keys = tl.load(
            Prefix_cache_ptr + prefix_base[:, None] + d_offs[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            Prefix_cache_ptr
            + prefix_base[:, None]
            + stride_prefix_kv
            + d_offs[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.float32)
        m_i, l_i, acc = _advance_cached_prefill_attention(
            q, keys, values, m_mask, n_mask, m_i, l_i, acc, ATTN_SCALE
        )

    for start_n in range(PREFIX_TOKENS, recent_start, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = (n_offs < recent_start) & (n_offs < cached_len)
        page_idx = n_offs // BLOCK_SIZE
        page_off = n_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=n_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)
        k_meta = slot_bases + KEY_DATA_BYTES
        k_scale = tl.load(KV_meta_ptr + k_meta // 2, mask=n_mask, other=0.0).to(
            tl.float32
        )
        k_zero = tl.load(KV_meta_ptr + k_meta // 2 + 1, mask=n_mask, other=0.0).to(
            tl.float32
        )
        keys = (q_k - k_zero[:, None]) * k_scale[:, None]

        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)
        v_meta = v_base + VALUE_DATA_BYTES
        v_scale = tl.load(KV_meta_ptr + v_meta // 2, mask=n_mask, other=0.0).to(
            tl.float32
        )
        v_zero = tl.load(KV_meta_ptr + v_meta // 2 + 1, mask=n_mask, other=0.0).to(
            tl.float32
        )
        values = (q_v - v_zero[:, None]) * v_scale[:, None]
        m_i, l_i, acc = _advance_cached_prefill_attention(
            q, keys, values, m_mask, n_mask, m_i, l_i, acc, ATTN_SCALE
        )

    for start_n in range(recent_start, cached_len, BLOCK_N):
        n_offs = start_n + tl.arange(0, BLOCK_N)
        n_mask = n_offs < cached_len
        recent_idx = hp_row * RECENT_TOKENS + (n_offs - PREFIX_TOKENS) % RECENT_TOKENS
        recent_base = (
            recent_idx.to(tl.int64) * stride_recent_slot
            + tl.cast(kv_head, tl.int64) * stride_recent_head
        )
        keys = tl.load(
            Recent_cache_ptr + recent_base[:, None] + d_offs[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.float32)
        values = tl.load(
            Recent_cache_ptr
            + recent_base[:, None]
            + stride_recent_kv
            + d_offs[None, :],
            mask=n_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.float32)
        m_i, l_i, acc = _advance_cached_prefill_attention(
            q, keys, values, m_mask, n_mask, m_i, l_i, acc, ATTN_SCALE
        )

    query_len = query_end - query_start
    query_positions = m_offs - query_start
    for start_n in range(0, MAX_QUERY_LEN, BLOCK_N):
        current_offs = start_n + tl.arange(0, BLOCK_N)
        current_mask = current_offs < query_len
        current_indices = query_start + current_offs
        current_k_base = (
            current_indices[:, None] * stride_current_k_token
            + tl.cast(kv_head, tl.int64) * stride_current_k_head
            + d_offs[None, :] * stride_current_k_dim
        )
        current_v_base = (
            current_indices[:, None] * stride_current_v_token
            + tl.cast(kv_head, tl.int64) * stride_current_v_head
            + d_offs[None, :] * stride_current_v_dim
        )
        keys = tl.load(
            Current_k_ptr + current_k_base,
            mask=current_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        values = tl.load(
            Current_v_ptr + current_v_base,
            mask=current_mask[:, None] & d_mask[None, :],
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(keys.to(tl.bfloat16))) * ATTN_SCALE
        causal_mask = current_offs[None, :] <= query_positions[:, None]
        scores = tl.where(
            m_mask[:, None] & current_mask[None, :] & causal_mask,
            scores,
            -float("inf"),
        )
        m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(scores - m_ij[:, None])
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), values.to(tl.bfloat16))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_ij

    safe_l = tl.where(l_i > 0.0, l_i, 1.0)
    output = acc / safe_l[:, None]
    tl.store(
        Out_ptr
        + m_offs[:, None] * stride_on
        + query_heads[:, None] * stride_oh
        + d_offs[None, :],
        output,
        mask=m_mask[:, None] & d_mask[None, :],
    )
    tl.store(
        LSE_ptr + query_heads * stride_lseh + m_offs * stride_lsen,
        m_i + tl.log(safe_l),
        mask=m_mask,
    )


def oscar_cached_prefill_attention(
    q_rot: torch.Tensor,
    kv_cache: torch.Tensor,
    prefix_cache: torch.Tensor,
    recent_cache: torch.Tensor,
    block_table: torch.Tensor,
    cached_lens: torch.Tensor,
    query_start_loc: torch.Tensor,
    hp_row_ids: torch.Tensor,
    prefix_page_ids: torch.Tensor,
    shared_hit_tokens: torch.Tensor,
    scale: float,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
    prefix_tokens: int,
    recent_tokens: int,
    max_query_len: int,
    v_rotation_t: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    current_key: torch.Tensor | None = None,
    current_value: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, num_query_heads, head_dim = q_rot.shape
    num_reqs = cached_lens.shape[0]
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    if prefix_page_ids.ndim != 2:
        raise ValueError("OSCAR prefix page table must be a 2D tensor")
    if prefix_page_ids.shape[1] * block_size != prefix_tokens:
        raise ValueError("OSCAR prefix page table width must cover the prefix window")
    if shared_hit_tokens.shape != cached_lens.shape:
        raise ValueError("OSCAR shared hit lengths must match cached sequence lengths")
    if (current_key is None) != (current_value is None):
        raise ValueError("OSCAR full prefill requires both current K and V")

    q_rot = (
        q_rot.contiguous() if current_key is not None else q_rot.contiguous().float()
    )
    if v_rotation_t is not None and output_dtype not in (None, q_rot.dtype):
        raise ValueError("OSCAR rotated cached-prefill output must remain FP32")
    if current_key is not None:
        assert current_value is not None
        if v_rotation_t is not None:
            raise ValueError("OSCAR full prefill requires absorbed V rotation")
        expected_shape = (num_tokens, num_kv_heads, head_dim)
        if current_key.shape != expected_shape or current_value.shape != expected_shape:
            raise ValueError("OSCAR current K/V shape does not match the query batch")
        current_max_query_len = max_query_len
    else:
        current_key = q_rot
        current_value = q_rot
        current_max_query_len = 0
    output = torch.empty_like(q_rot, dtype=output_dtype)
    lse = torch.empty(
        num_query_heads, num_tokens, dtype=torch.float32, device=q_rot.device
    )
    block_m = 32
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    kv_group_size = num_query_heads // num_kv_heads
    query_heads_per_program = 2 if kv_group_size % 2 == 0 else 1
    grid = (
        num_reqs,
        triton.cdiv(num_query_heads, query_heads_per_program),
        triton.cdiv(max_query_len, block_m),
    )
    _oscar_cached_prefill_kernel[grid](
        q_rot,
        current_key,
        current_value,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        prefix_cache,
        recent_cache,
        hp_row_ids,
        prefix_page_ids,
        shared_hit_tokens,
        block_table,
        cached_lens,
        query_start_loc,
        output,
        lse,
        q_rot.stride(0),
        q_rot.stride(1),
        current_key.stride(0),
        current_key.stride(1),
        current_key.stride(2),
        current_value.stride(0),
        current_value.stride(1),
        current_value.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        prefix_cache.stride(0),
        prefix_cache.stride(1),
        prefix_cache.stride(2),
        recent_cache.stride(0),
        recent_cache.stride(1),
        recent_cache.stride(2),
        prefix_page_ids.stride(0),
        block_table.stride(0),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        lse.stride(1),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        KV_GROUP_SIZE=kv_group_size,
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        ATTN_SCALE=scale,
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        QUERY_HEADS_PER_PROGRAM=query_heads_per_program,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_D=block_d,
        MAX_QUERY_LEN=current_max_query_len,
        num_warps=4 if query_heads_per_program == 2 else 2,
        num_stages=2,
    )
    if v_rotation_t is not None:
        rotated_output = torch.empty_like(output)
        torch.mm(
            output.view(-1, head_dim),
            v_rotation_t,
            out=rotated_output.view(-1, head_dim),
        )
        output = rotated_output
    return output, lse
