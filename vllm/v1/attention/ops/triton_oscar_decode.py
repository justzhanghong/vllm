# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused OSCAR INT2 decode attention.

Decode path: split-KV tiled scoring + value accumulation (stage1) followed
by a log-sum-exp reduction across splits (stage2, reused from
``triton_decode_attention``).

Keys and values are stored as asymmetric INT2 (4 indices per byte) with one
BF16 ``(scale, zero_point)`` pair per vector. The query passed in is already
rotated by ``R_k`` so that scores against the rotated stored keys equal the
true ``Q K^T``; the value-side ``R_v^T`` inverse is applied by the caller to
the returned output, which lives in rotated-V space.
"""

import math
import sys

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.oscar_cache_contract import (
    has_linear_oscar_arena_layout,
    validate_oscar_separated_arenas,
)
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

_GROUPED_H4_PREINDEXED_QUANT_SPLITS = 12
_GROUPED_H4_HP_SPLITS = 8
_GROUPED_H4_MIXED_TOTAL_SPLITS = (
    _GROUPED_H4_PREINDEXED_QUANT_SPLITS + _GROUPED_H4_HP_SPLITS
)


@triton.jit
def _materialize_oscar_slot_ids_kernel(
    Block_table_ptr,
    Seq_lens_ptr,
    Physical_slot_ids_ptr,
    stride_bt_b,
    stride_slots_b,
    BLOCK_SIZE: tl.constexpr,
    MAX_SEQ_LEN: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    offsets = tl.program_id(1) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    seq_len = tl.load(Seq_lens_ptr + req_idx)
    mask = (offsets < seq_len) & (offsets < MAX_SEQ_LEN)
    physical_blocks = tl.load(
        Block_table_ptr + req_idx * stride_bt_b + offsets // BLOCK_SIZE,
        mask=mask,
        other=0,
    ).to(tl.int64)
    page_offsets = offsets % BLOCK_SIZE
    physical_slots = physical_blocks * BLOCK_SIZE + page_offsets.to(tl.int64)
    tl.store(
        Physical_slot_ids_ptr + req_idx * stride_slots_b + offsets,
        physical_slots,
        mask=mask,
    )


def materialize_oscar_slot_ids(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    physical_slot_ids: torch.Tensor,
    block_size: int,
) -> None:
    """Expand one KV group's decode block table into its stable slot buffer."""
    if block_table.ndim != 2 or physical_slot_ids.ndim != 2:
        raise ValueError("OSCAR block table and physical slot IDs must be 2D")
    if seq_lens.ndim != 1 or seq_lens.shape[0] != block_table.shape[0]:
        raise ValueError("OSCAR sequence lengths must match the block table")
    if physical_slot_ids.shape[0] < block_table.shape[0]:
        raise ValueError("OSCAR physical slot buffer has too few request rows")
    if physical_slot_ids.dtype != torch.int64:
        raise ValueError("OSCAR physical slot IDs must use int64")
    if block_size <= 0:
        raise ValueError("OSCAR block size must be positive")
    max_seq_len = physical_slot_ids.shape[1]
    if max_seq_len <= 0:
        return
    block_tokens = 256
    grid = (block_table.shape[0], triton.cdiv(max_seq_len, block_tokens))
    _materialize_oscar_slot_ids_kernel[grid](
        block_table,
        seq_lens,
        physical_slot_ids,
        block_table.stride(0),
        physical_slot_ids.stride(0),
        BLOCK_SIZE=block_size,
        MAX_SEQ_LEN=max_seq_len,
        BLOCK_TOKENS=block_tokens,
        num_warps=4,
    )


@triton.jit
def _oscar_decode_stage1(
    Q_rot_ptr,  # [B, Hq, D] fp32 — query already rotated by R_k
    K_data_ptr,
    V_data_ptr,
    K_meta_ptr,
    V_meta_ptr,
    Prefix_cache_ptr,  # [prefix_slots, Hk, 2, D] bf16
    Recent_cache_ptr,  # [recent_slots, Hk, 2, D] bf16
    HP_rows_ptr,  # [B] int32
    Prefix_pages_ptr,  # [B, prefix_pages] int32
    Query_to_req_ptr,  # [B] int32 when multiple queries share a request
    Shared_hit_lens_ptr,  # [requests] int32 initial locally shared hit length
    Recent_extra_ptr,  # [requests] int32 pending BF16 tokens beyond logical recent
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] fp32
    stride_qb,
    stride_qh,
    stride_data_block,
    stride_data_pos,
    stride_data_head,
    stride_meta_block,
    stride_meta_pos,
    stride_meta_head,
    stride_prefix_slot,
    stride_prefix_head,
    stride_prefix_kv,
    stride_recent_slot,
    stride_recent_head,
    stride_recent_kv,
    stride_prefix_pages_req,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,  # D // 4
    VALUE_DATA_BYTES: tl.constexpr,  # D // 4
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    sid = tl.program_id(2)

    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    kv_head = hid // KV_GROUP_SIZE
    seq_len = tl.load(Seq_lens_ptr + bid)
    hp_row = 0
    if MIXED_KV:
        hp_row = tl.load(HP_rows_ptr + req_idx)
        recent_extra = tl.load(Recent_extra_ptr + req_idx)
    else:
        recent_extra = 0

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # INT2 unpack index vectors (loop-invariant): 4 indices per byte.
    if HEAD_DIM == 128:
        byte_idx = d_offs % KEY_DATA_BYTES
        bit_shift = (d_offs // KEY_DATA_BYTES) * 2
    else:
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2

    q_base = bid * stride_qb + hid * stride_qh
    q_rot = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(tl.float32)

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = req_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        data_bases = (
            block_nums * stride_data_block
            + page_off.to(tl.int64) * stride_data_pos
            + tl.cast(kv_head, tl.int64) * stride_data_head
        )
        meta_bases = (
            block_nums * stride_meta_block
            + page_off.to(tl.int64) * stride_meta_pos
            + tl.cast(kv_head, tl.int64) * stride_meta_head
        )

        # ---- dequant K (INT2) and score ----
        if MIXED_KV:
            shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
            recent_start = tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len),
                seq_len - RECENT_TOKENS - recent_extra,
            )
            is_hp = kv_mask & ((kv_offs < PREFIX_TOKENS) | (kv_offs >= recent_start))
        else:
            is_hp = tl.zeros([BLOCK_KV], dtype=tl.int1)

        quant_mask = kv_mask & ~is_hp
        k_byte = tl.load(
            K_data_ptr + data_bases[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)

        k_scale = tl.load(K_meta_ptr + meta_bases, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        k_zero = tl.load(K_meta_ptr + meta_bases + 1, mask=quant_mask, other=0.0).to(
            tl.float32
        )

        keys = (q_k - k_zero[:, None]) * k_scale[:, None]
        if MIXED_KV:
            if USE_PREFIX_PAGE_TABLE:
                prefix_page = tl.load(
                    Prefix_pages_ptr
                    + req_idx * stride_prefix_pages_req
                    + kv_offs // BLOCK_SIZE,
                    mask=kv_mask & (kv_offs < PREFIX_TOKENS),
                    other=0,
                )
                prefix_idx = prefix_page * BLOCK_SIZE + kv_offs % BLOCK_SIZE
            else:
                prefix_idx = hp_row * PREFIX_TOKENS + kv_offs
            prefix_base = (
                prefix_idx.to(tl.int64) * stride_prefix_slot
                + tl.cast(kv_head, tl.int64) * stride_prefix_head
            )
            prefix_keys = tl.load(
                Prefix_cache_ptr + prefix_base[:, None] + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_idx = (
                hp_row * RECENT_CAPACITY
                + (kv_offs - PREFIX_TOKENS) % RECENT_CAPACITY
            )
            recent_base = (
                recent_idx.to(tl.int64) * stride_recent_slot
                + tl.cast(kv_head, tl.int64) * stride_recent_head
            )
            recent_keys = tl.load(
                Recent_cache_ptr + recent_base[:, None] + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_keys = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None], prefix_keys, recent_keys
            )
            keys = tl.where(is_hp[:, None], hp_keys, keys)
        score_q = q_rot.to(tl.bfloat16)
        score_keys = keys.to(tl.bfloat16)
        scores = (
            tl.sum(
                tl.where(
                    d_mask[None, :], score_q[None, :] * score_keys, 0.0
                ),
                axis=1,
            )
            * ATTN_SCALE
        )
        scores = tl.where(kv_mask, scores, -float("inf"))

        n_e_max = tl.maximum(tl.max(scores, 0), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max)

        # ---- dequant V (INT2) ----
        v_byte = tl.load(
            V_data_ptr + data_bases[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)

        v_scale = tl.load(V_meta_ptr + meta_bases, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        v_zero = tl.load(V_meta_ptr + meta_bases + 1, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        values = (q_v - v_zero[:, None]) * v_scale[:, None]
        if MIXED_KV:
            prefix_values = tl.load(
                Prefix_cache_ptr
                + prefix_base[:, None]
                + stride_prefix_kv
                + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_values = tl.load(
                Recent_cache_ptr
                + recent_base[:, None]
                + stride_recent_kv
                + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_values = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None],
                prefix_values,
                recent_values,
            )
            values = tl.where(is_hp[:, None], hp_values, values)

        acc = acc * re_scale + tl.sum(p[:, None] * values, 0)
        l_prev = l_prev * re_scale + tl.sum(p, 0)
        m_prev = n_e_max

    out_base = bid * stride_mid_b + hid * stride_mid_h + sid * stride_mid_s
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(Mid_o_ptr + out_base + d_offs, acc / safe_l, mask=d_mask)
    tl.store(Mid_o_ptr + out_base + HEAD_DIM, m_prev + tl.log(safe_l))


@triton.jit
def _oscar_decode_stage1_grouped_h4(
    Q_rot_ptr,
    KV_cache_ptr,
    KV_meta_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Query_to_req_ptr,
    Shared_hit_lens_ptr,
    Recent_extra_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
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
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)
    head0 = kv_head * KV_GROUP_SIZE

    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    seq_len = tl.load(Seq_lens_ptr + bid)
    hp_row = 0
    if MIXED_KV:
        hp_row = tl.load(HP_rows_ptr + req_idx)
        recent_extra = tl.load(Recent_extra_ptr + req_idx)
    else:
        recent_extra = 0

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    if HEAD_DIM == 128:
        byte_idx = d_offs % KEY_DATA_BYTES
        bit_shift = (d_offs // KEY_DATA_BYTES) * 2
    else:
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2

    q_base = bid * stride_qb + head0 * stride_qh
    q0 = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(
        tl.bfloat16
    )
    q1 = tl.load(
        Q_rot_ptr + q_base + stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)
    q2 = tl.load(
        Q_rot_ptr + q_base + 2 * stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)
    q3 = tl.load(
        Q_rot_ptr + q_base + 3 * stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)

    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = req_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        if MIXED_KV:
            shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
            recent_start = tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len),
                seq_len - RECENT_TOKENS - recent_extra,
            )
            is_hp = kv_mask & ((kv_offs < PREFIX_TOKENS) | (kv_offs >= recent_start))
        else:
            is_hp = tl.zeros([BLOCK_KV], dtype=tl.int1)

        quant_mask = kv_mask & ~is_hp
        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)
        k_meta = slot_bases + KEY_DATA_BYTES
        k_scale = tl.load(KV_meta_ptr + k_meta // 2, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        k_zero = tl.load(
            KV_meta_ptr + k_meta // 2 + 1, mask=quant_mask, other=0.0
        ).to(tl.float32)
        keys = (q_k - k_zero[:, None]) * k_scale[:, None]

        if MIXED_KV:
            if USE_PREFIX_PAGE_TABLE:
                prefix_page = tl.load(
                    Prefix_pages_ptr
                    + req_idx * stride_prefix_pages_req
                    + kv_offs // BLOCK_SIZE,
                    mask=kv_mask & (kv_offs < PREFIX_TOKENS),
                    other=0,
                )
                prefix_idx = prefix_page * BLOCK_SIZE + kv_offs % BLOCK_SIZE
            else:
                prefix_idx = hp_row * PREFIX_TOKENS + kv_offs
            prefix_base = (
                prefix_idx.to(tl.int64) * stride_prefix_slot
                + tl.cast(kv_head, tl.int64) * stride_prefix_head
            )
            prefix_keys = tl.load(
                Prefix_cache_ptr + prefix_base[:, None] + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_idx = (
                hp_row * RECENT_CAPACITY
                + (kv_offs - PREFIX_TOKENS) % RECENT_CAPACITY
            )
            recent_base = (
                recent_idx.to(tl.int64) * stride_recent_slot
                + tl.cast(kv_head, tl.int64) * stride_recent_head
            )
            recent_keys = tl.load(
                Recent_cache_ptr + recent_base[:, None] + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_keys = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None], prefix_keys, recent_keys
            )
            keys = tl.where(is_hp[:, None], hp_keys, keys)

        score_keys = keys.to(tl.bfloat16)
        score_mask = d_mask[None, :]
        scores0 = tl.sum(tl.where(score_mask, q0[None, :] * score_keys, 0.0), 1)
        scores1 = tl.sum(tl.where(score_mask, q1[None, :] * score_keys, 0.0), 1)
        scores2 = tl.sum(tl.where(score_mask, q2[None, :] * score_keys, 0.0), 1)
        scores3 = tl.sum(tl.where(score_mask, q3[None, :] * score_keys, 0.0), 1)
        scores0 = tl.where(kv_mask, scores0 * ATTN_SCALE, -float("inf"))
        scores1 = tl.where(kv_mask, scores1 * ATTN_SCALE, -float("inf"))
        scores2 = tl.where(kv_mask, scores2 * ATTN_SCALE, -float("inf"))
        scores3 = tl.where(kv_mask, scores3 * ATTN_SCALE, -float("inf"))

        n0 = tl.maximum(tl.max(scores0, 0), m0)
        n1 = tl.maximum(tl.max(scores1, 0), m1)
        n2 = tl.maximum(tl.max(scores2, 0), m2)
        n3 = tl.maximum(tl.max(scores3, 0), m3)
        r0 = tl.exp(m0 - n0)
        r1 = tl.exp(m1 - n1)
        r2 = tl.exp(m2 - n2)
        r3 = tl.exp(m3 - n3)
        p0 = tl.exp(scores0 - n0)
        p1 = tl.exp(scores1 - n1)
        p2 = tl.exp(scores2 - n2)
        p3 = tl.exp(scores3 - n3)

        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)
        v_meta = v_base + VALUE_DATA_BYTES
        v_scale = tl.load(KV_meta_ptr + v_meta // 2, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        v_zero = tl.load(
            KV_meta_ptr + v_meta // 2 + 1, mask=quant_mask, other=0.0
        ).to(tl.float32)
        values = (q_v - v_zero[:, None]) * v_scale[:, None]

        if MIXED_KV:
            prefix_values = tl.load(
                Prefix_cache_ptr
                + prefix_base[:, None]
                + stride_prefix_kv
                + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_values = tl.load(
                Recent_cache_ptr
                + recent_base[:, None]
                + stride_recent_kv
                + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_values = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None], prefix_values, recent_values
            )
            values = tl.where(is_hp[:, None], hp_values, values)

        acc0 = acc0 * r0 + tl.sum(p0[:, None] * values, 0)
        acc1 = acc1 * r1 + tl.sum(p1[:, None] * values, 0)
        acc2 = acc2 * r2 + tl.sum(p2[:, None] * values, 0)
        acc3 = acc3 * r3 + tl.sum(p3[:, None] * values, 0)
        l0 = l0 * r0 + tl.sum(p0, 0)
        l1 = l1 * r1 + tl.sum(p1, 0)
        l2 = l2 * r2 + tl.sum(p2, 0)
        l3 = l3 * r3 + tl.sum(p3, 0)
        m0 = n0
        m1 = n1
        m2 = n2
        m3 = n3

    split_base = bid * stride_mid_b + sid * stride_mid_s
    safe_l0 = tl.where(l0 > 0.0, l0, 1.0)
    safe_l1 = tl.where(l1 > 0.0, l1, 1.0)
    safe_l2 = tl.where(l2 > 0.0, l2, 1.0)
    safe_l3 = tl.where(l3 > 0.0, l3, 1.0)
    out0 = split_base + head0 * stride_mid_h
    out1 = out0 + stride_mid_h
    out2 = out0 + 2 * stride_mid_h
    out3 = out0 + 3 * stride_mid_h
    tl.store(Mid_o_ptr + out0 + d_offs, acc0 / safe_l0, mask=d_mask)
    tl.store(Mid_o_ptr + out1 + d_offs, acc1 / safe_l1, mask=d_mask)
    tl.store(Mid_o_ptr + out2 + d_offs, acc2 / safe_l2, mask=d_mask)
    tl.store(Mid_o_ptr + out3 + d_offs, acc3 / safe_l3, mask=d_mask)
    tl.store(Mid_o_ptr + out0 + HEAD_DIM, m0 + tl.log(safe_l0))
    tl.store(Mid_o_ptr + out1 + HEAD_DIM, m1 + tl.log(safe_l1))
    tl.store(Mid_o_ptr + out2 + HEAD_DIM, m2 + tl.log(safe_l2))
    tl.store(Mid_o_ptr + out3 + HEAD_DIM, m3 + tl.log(safe_l3))


@triton.jit
def _oscar_decode_stage1_grouped_h4_qk(
    Q_rot_ptr,
    KV_cache_ptr,
    KV_meta_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Query_to_req_ptr,
    Shared_hit_lens_ptr,
    Recent_extra_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Score_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_prefix_slot,
    stride_prefix_head,
    stride_recent_slot,
    stride_recent_head,
    stride_prefix_pages_req,
    stride_bt_b,
    stride_score_b,
    stride_score_h,
    stride_score_t,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Grouped-H4 K dequant and QK score stage with FP32 global output."""
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)
    head0 = kv_head * KV_GROUP_SIZE

    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    seq_len = tl.load(Seq_lens_ptr + bid)
    hp_row = 0
    if MIXED_KV:
        hp_row = tl.load(HP_rows_ptr + req_idx)
        recent_extra = tl.load(Recent_extra_ptr + req_idx)
    else:
        recent_extra = 0

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    if HEAD_DIM == 128:
        byte_idx = d_offs % KEY_DATA_BYTES
        bit_shift = (d_offs // KEY_DATA_BYTES) * 2
    else:
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2

    q_base = bid * stride_qb + head0 * stride_qh
    q0 = tl.load(Q_rot_ptr + q_base + d_offs, mask=d_mask, other=0.0).to(
        tl.bfloat16
    )
    q1 = tl.load(
        Q_rot_ptr + q_base + stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)
    q2 = tl.load(
        Q_rot_ptr + q_base + 2 * stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)
    q3 = tl.load(
        Q_rot_ptr + q_base + 3 * stride_qh + d_offs, mask=d_mask, other=0.0
    ).to(tl.bfloat16)
    bt_base = req_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        if MIXED_KV:
            shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
            recent_start = tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len),
                seq_len - RECENT_TOKENS - recent_extra,
            )
            is_hp = kv_mask & ((kv_offs < PREFIX_TOKENS) | (kv_offs >= recent_start))
        else:
            is_hp = tl.zeros([BLOCK_KV], dtype=tl.int1)

        quant_mask = kv_mask & ~is_hp
        k_byte = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_k = ((k_byte >> bit_shift[None, :]) & (KEY_LEVELS - 1)).to(tl.float32)
        k_meta = slot_bases + KEY_DATA_BYTES
        k_scale = tl.load(KV_meta_ptr + k_meta // 2, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        k_zero = tl.load(
            KV_meta_ptr + k_meta // 2 + 1, mask=quant_mask, other=0.0
        ).to(tl.float32)
        keys = (q_k - k_zero[:, None]) * k_scale[:, None]

        if MIXED_KV:
            if USE_PREFIX_PAGE_TABLE:
                prefix_page = tl.load(
                    Prefix_pages_ptr
                    + req_idx * stride_prefix_pages_req
                    + kv_offs // BLOCK_SIZE,
                    mask=kv_mask & (kv_offs < PREFIX_TOKENS),
                    other=0,
                )
                prefix_idx = prefix_page * BLOCK_SIZE + kv_offs % BLOCK_SIZE
            else:
                prefix_idx = hp_row * PREFIX_TOKENS + kv_offs
            prefix_base = (
                prefix_idx.to(tl.int64) * stride_prefix_slot
                + tl.cast(kv_head, tl.int64) * stride_prefix_head
            )
            prefix_keys = tl.load(
                Prefix_cache_ptr + prefix_base[:, None] + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_idx = (
                hp_row * RECENT_CAPACITY
                + (kv_offs - PREFIX_TOKENS) % RECENT_CAPACITY
            )
            recent_base = (
                recent_idx.to(tl.int64) * stride_recent_slot
                + tl.cast(kv_head, tl.int64) * stride_recent_head
            )
            recent_keys = tl.load(
                Recent_cache_ptr + recent_base[:, None] + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_keys = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None], prefix_keys, recent_keys
            )
            keys = tl.where(is_hp[:, None], hp_keys, keys)

        score_keys = keys.to(tl.bfloat16)
        q_head = tl.arange(0, 4)[:, None]
        q_heads = tl.where(q_head == 0, q0[None, :], q1[None, :])
        q_heads = tl.where(q_head == 2, q2[None, :], q_heads)
        q_heads = tl.where(q_head == 3, q3[None, :], q_heads)

        q_quarters = tl.reshape(q_heads, (4, 4, 32))
        q_quarters = tl.permute(q_quarters, (0, 2, 1))
        q_quarters = tl.reshape(q_quarters, (4, 32, 2, 2))
        q_even, q_odd = tl.split(q_quarters)
        q_d0, q_d2 = tl.split(q_even)
        q_d1, q_d3 = tl.split(q_odd)

        key_quarters = tl.reshape(score_keys, (BLOCK_KV, 4, 32))
        key_quarters = tl.permute(key_quarters, (0, 2, 1))
        key_quarters = tl.reshape(key_quarters, (BLOCK_KV, 32, 2, 2))
        key_even, key_odd = tl.split(key_quarters)
        key_d0, key_d2 = tl.split(key_even)
        key_d1, key_d3 = tl.split(key_odd)

        score_heads = tl.dot(q_d0, tl.trans(key_d0))
        score_heads = score_heads + tl.dot(q_d1, tl.trans(key_d1))
        score_heads = score_heads + tl.dot(q_d2, tl.trans(key_d2))
        score_heads = score_heads + tl.dot(q_d3, tl.trans(key_d3))
        score_pairs = tl.reshape(tl.trans(score_heads), (BLOCK_KV, 2, 2))
        score_even, score_odd = tl.split(score_pairs)
        scores0, scores2 = tl.split(score_even)
        scores1, scores3 = tl.split(score_odd)
        scores0 = tl.where(kv_mask, scores0 * ATTN_SCALE, -float("inf"))
        scores1 = tl.where(kv_mask, scores1 * ATTN_SCALE, -float("inf"))
        scores2 = tl.where(kv_mask, scores2 * ATTN_SCALE, -float("inf"))
        scores3 = tl.where(kv_mask, scores3 * ATTN_SCALE, -float("inf"))

        score_base = bid * stride_score_b + head0 * stride_score_h
        tl.store(
            Score_ptr + score_base + kv_offs * stride_score_t,
            scores0,
            mask=kv_mask,
        )
        tl.store(
            Score_ptr + score_base + stride_score_h + kv_offs * stride_score_t,
            scores1,
            mask=kv_mask,
        )
        tl.store(
            Score_ptr + score_base + 2 * stride_score_h + kv_offs * stride_score_t,
            scores2,
            mask=kv_mask,
        )
        tl.store(
            Score_ptr + score_base + 3 * stride_score_h + kv_offs * stride_score_t,
            scores3,
            mask=kv_mask,
        )


@triton.jit
def _oscar_decode_stage1_grouped_h4_v(
    KV_cache_ptr,
    KV_meta_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Query_to_req_ptr,
    Shared_hit_lens_ptr,
    Recent_extra_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Score_ptr,
    Mid_o_ptr,
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
    stride_bt_b,
    stride_score_b,
    stride_score_h,
    stride_score_t,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    KEY_PACKED: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_KV: tl.constexpr,
):
    """Grouped-H4 online softmax and V stage reading FP32 global scores."""
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)
    head0 = kv_head * KV_GROUP_SIZE

    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    seq_len = tl.load(Seq_lens_ptr + bid)
    hp_row = 0
    if MIXED_KV:
        hp_row = tl.load(HP_rows_ptr + req_idx)
        recent_extra = tl.load(Recent_extra_ptr + req_idx)
    else:
        recent_extra = 0

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
    if HEAD_DIM == 128:
        byte_idx = d_offs % VALUE_DATA_BYTES
        bit_shift = (d_offs // VALUE_DATA_BYTES) * 2
    else:
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2
    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    acc0 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_D], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_D], dtype=tl.float32)
    bt_base = req_idx * stride_bt_b

    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )
        if MIXED_KV:
            shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
            recent_start = tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len),
                seq_len - RECENT_TOKENS - recent_extra,
            )
            is_hp = kv_mask & ((kv_offs < PREFIX_TOKENS) | (kv_offs >= recent_start))
        else:
            is_hp = tl.zeros([BLOCK_KV], dtype=tl.int1)

        score_base = bid * stride_score_b + head0 * stride_score_h
        scores0 = tl.load(
            Score_ptr + score_base + kv_offs * stride_score_t, mask=kv_mask, other=0.0
        )
        scores1 = tl.load(
            Score_ptr
            + score_base
            + stride_score_h
            + kv_offs * stride_score_t,
            mask=kv_mask,
            other=0.0,
        )
        scores2 = tl.load(
            Score_ptr
            + score_base
            + 2 * stride_score_h
            + kv_offs * stride_score_t,
            mask=kv_mask,
            other=0.0,
        )
        scores3 = tl.load(
            Score_ptr
            + score_base
            + 3 * stride_score_h
            + kv_offs * stride_score_t,
            mask=kv_mask,
            other=0.0,
        )
        scores0 = tl.where(kv_mask, scores0, -float("inf"))
        scores1 = tl.where(kv_mask, scores1, -float("inf"))
        scores2 = tl.where(kv_mask, scores2, -float("inf"))
        scores3 = tl.where(kv_mask, scores3, -float("inf"))

        n0 = tl.maximum(tl.max(scores0, 0), m0)
        n1 = tl.maximum(tl.max(scores1, 0), m1)
        n2 = tl.maximum(tl.max(scores2, 0), m2)
        n3 = tl.maximum(tl.max(scores3, 0), m3)
        r0 = tl.exp(m0 - n0)
        r1 = tl.exp(m1 - n1)
        r2 = tl.exp(m2 - n2)
        r3 = tl.exp(m3 - n3)
        p0 = tl.exp(scores0 - n0)
        p1 = tl.exp(scores1 - n1)
        p2 = tl.exp(scores2 - n2)
        p3 = tl.exp(scores3 - n3)

        quant_mask = kv_mask & ~is_hp
        v_base = slot_bases + KEY_PACKED
        v_byte = tl.load(
            KV_cache_ptr + v_base[:, None] + byte_idx[None, :],
            mask=quant_mask[:, None] & d_mask[None, :],
            other=0,
        ).to(tl.int32)
        q_v = ((v_byte >> bit_shift[None, :]) & (VALUE_LEVELS - 1)).to(tl.float32)
        v_meta = v_base + VALUE_DATA_BYTES
        v_scale = tl.load(KV_meta_ptr + v_meta // 2, mask=quant_mask, other=0.0).to(
            tl.float32
        )
        v_zero = tl.load(
            KV_meta_ptr + v_meta // 2 + 1, mask=quant_mask, other=0.0
        ).to(tl.float32)
        values = (q_v - v_zero[:, None]) * v_scale[:, None]

        if MIXED_KV:
            if USE_PREFIX_PAGE_TABLE:
                prefix_page = tl.load(
                    Prefix_pages_ptr
                    + req_idx * stride_prefix_pages_req
                    + kv_offs // BLOCK_SIZE,
                    mask=kv_mask & (kv_offs < PREFIX_TOKENS),
                    other=0,
                )
                prefix_idx = prefix_page * BLOCK_SIZE + kv_offs % BLOCK_SIZE
            else:
                prefix_idx = hp_row * PREFIX_TOKENS + kv_offs
            prefix_base = (
                prefix_idx.to(tl.int64) * stride_prefix_slot
                + tl.cast(kv_head, tl.int64) * stride_prefix_head
            )
            recent_idx = (
                hp_row * RECENT_CAPACITY
                + (kv_offs - PREFIX_TOKENS) % RECENT_CAPACITY
            )
            recent_base = (
                recent_idx.to(tl.int64) * stride_recent_slot
                + tl.cast(kv_head, tl.int64) * stride_recent_head
            )
            prefix_values = tl.load(
                Prefix_cache_ptr
                + prefix_base[:, None]
                + stride_prefix_kv
                + d_offs[None, :],
                mask=(kv_mask & (kv_offs < PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            recent_values = tl.load(
                Recent_cache_ptr
                + recent_base[:, None]
                + stride_recent_kv
                + d_offs[None, :],
                mask=(is_hp & (kv_offs >= PREFIX_TOKENS))[:, None]
                & d_mask[None, :],
                other=0,
            ).to(tl.float32)
            hp_values = tl.where(
                (kv_offs < PREFIX_TOKENS)[:, None], prefix_values, recent_values
            )
            values = tl.where(is_hp[:, None], hp_values, values)

        p01 = tl.join(p0, p1)
        p23 = tl.join(p2, p3)
        probs = tl.join(p01, p23)
        probs = tl.permute(probs, (2, 1, 0))
        probs = tl.reshape(probs, (4, BLOCK_KV))
        value_dot = tl.dot(probs.to(tl.bfloat16), values.to(tl.bfloat16))
        value_dot = tl.permute(value_dot, (1, 0))
        value_dot = tl.reshape(value_dot, (BLOCK_D, 2, 2))
        value_dot_even, value_dot_odd = tl.split(value_dot)
        dot0, dot2 = tl.split(value_dot_even)
        dot1, dot3 = tl.split(value_dot_odd)
        acc0 = acc0 * r0 + dot0
        acc1 = acc1 * r1 + dot1
        acc2 = acc2 * r2 + dot2
        acc3 = acc3 * r3 + dot3
        l0 = l0 * r0 + tl.sum(p0, 0)
        l1 = l1 * r1 + tl.sum(p1, 0)
        l2 = l2 * r2 + tl.sum(p2, 0)
        l3 = l3 * r3 + tl.sum(p3, 0)
        m0 = n0
        m1 = n1
        m2 = n2
        m3 = n3

    split_base = bid * stride_mid_b + sid * stride_mid_s
    safe_l0 = tl.where(l0 > 0.0, l0, 1.0)
    safe_l1 = tl.where(l1 > 0.0, l1, 1.0)
    safe_l2 = tl.where(l2 > 0.0, l2, 1.0)
    safe_l3 = tl.where(l3 > 0.0, l3, 1.0)
    out0 = split_base + head0 * stride_mid_h
    out1 = out0 + stride_mid_h
    out2 = out0 + 2 * stride_mid_h
    out3 = out0 + 3 * stride_mid_h
    tl.store(Mid_o_ptr + out0 + d_offs, acc0 / safe_l0, mask=d_mask)
    tl.store(Mid_o_ptr + out1 + d_offs, acc1 / safe_l1, mask=d_mask)
    tl.store(Mid_o_ptr + out2 + d_offs, acc2 / safe_l2, mask=d_mask)
    tl.store(Mid_o_ptr + out3 + d_offs, acc3 / safe_l3, mask=d_mask)
    tl.store(Mid_o_ptr + out0 + HEAD_DIM, m0 + tl.log(safe_l0))
    tl.store(Mid_o_ptr + out1 + HEAD_DIM, m1 + tl.log(safe_l1))
    tl.store(Mid_o_ptr + out2 + HEAD_DIM, m2 + tl.log(safe_l2))
    tl.store(Mid_o_ptr + out3 + HEAD_DIM, m3 + tl.log(safe_l3))


@triton.jit
def _oscar_decode_quant_stage1_grouped_h4(
    Q_rot_ptr,
    K_data_ptr,
    V_data_ptr,
    K_meta_pair_ptr,
    V_meta_pair_ptr,
    Query_to_req_ptr,
    Shared_hit_lens_ptr,
    Recent_extra_ptr,
    Physical_slot_ids_ptr,
    Seq_lens_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_data_pos,
    stride_data_head,
    stride_meta_pair_pos,
    stride_meta_pair_head,
    stride_slots_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,
    NUM_QUANT_SPLITS: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Tensor-core INT2 stage for D128/Hq:Hk=4 using quarter packing."""
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)
    head0 = kv_head * 4
    heads = head0 + tl.arange(0, 4)
    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    seq_len = tl.load(Seq_lens_ptr + bid)

    quant_start = 0
    quant_end = seq_len
    if MIXED_KV:
        shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
        recent_extra = tl.load(Recent_extra_ptr + req_idx)
        quant_start = tl.minimum(PREFIX_TOKENS, seq_len)
        quant_end = tl.minimum(
            seq_len,
            tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len),
                seq_len - RECENT_TOKENS - recent_extra,
            ),
        )
    quant_len = tl.maximum(quant_end - quant_start, 0)
    split_len = tl.cdiv(quant_len, NUM_QUANT_SPLITS)
    split_start = quant_start + split_len * sid
    split_end = tl.minimum(split_start + split_len, quant_end)

    offs_quarter = tl.arange(0, 32)
    q_base = bid * stride_qb + heads[:, None] * stride_qh
    q0 = tl.load(Q_rot_ptr + q_base + offs_quarter[None, :]).to(tl.bfloat16)
    q1 = tl.load(Q_rot_ptr + q_base + (offs_quarter + 32)[None, :]).to(
        tl.bfloat16
    )
    q2 = tl.load(Q_rot_ptr + q_base + (offs_quarter + 64)[None, :]).to(
        tl.bfloat16
    )
    q3 = tl.load(Q_rot_ptr + q_base + (offs_quarter + 96)[None, :]).to(
        tl.bfloat16
    )
    q01 = tl.join(q0, q1)
    q23 = tl.join(q2, q3)
    query = tl.join(q01, q23)
    query = tl.reshape(query, (4, 32, 4))
    query = tl.permute(query, (0, 2, 1))
    query = tl.reshape(query, (4, 128))
    n_range = tl.arange(0, BLOCK_N)
    e_max = tl.zeros([4], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([4], dtype=tl.float32)
    acc0 = tl.zeros([4, 32], dtype=tl.float32)
    acc1 = tl.zeros([4, 32], dtype=tl.float32)
    acc2 = tl.zeros([4, 32], dtype=tl.float32)
    acc3 = tl.zeros([4, 32], dtype=tl.float32)

    if split_end > split_start:
        for start_n in range(split_start, split_end, BLOCK_N):
            kv_offs = start_n + n_range
            kv_mask = kv_offs < split_end
            physical_slots = tl.load(
                Physical_slot_ids_ptr + req_idx * stride_slots_b + kv_offs,
                mask=kv_mask,
                other=0,
            ).to(tl.int64)
            data_bases = (
                physical_slots * stride_data_pos
                + tl.cast(kv_head, tl.int64) * stride_data_head
            )
            meta_bases = (
                physical_slots * stride_meta_pair_pos
                + tl.cast(kv_head, tl.int64) * stride_meta_pair_head
            )

            k_packed = tl.load(
                K_data_ptr + data_bases[None, :] + offs_quarter[:, None],
                mask=kv_mask[None, :],
                other=0,
            )
            k_meta_pair = tl.load(
                K_meta_pair_ptr + meta_bases,
                mask=kv_mask,
                other=0x00003F80,
            )
            v_packed = tl.load(
                V_data_ptr + data_bases[:, None] + offs_quarter[None, :],
                mask=kv_mask[:, None],
                other=0,
            )
            v_meta_pair = tl.load(
                V_meta_pair_ptr + meta_bases,
                mask=kv_mask,
                other=0x00003F80,
            )
            k_meta_bits = k_meta_pair.to(tl.uint32, bitcast=True)
            k_scale = (k_meta_bits & 0xFFFF).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )
            k_zero = ((k_meta_bits >> 16) & 0xFFFF).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )
            v_meta_bits = v_meta_pair.to(tl.uint32, bitcast=True)
            v_scale = (v_meta_bits & 0xFFFF).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )
            v_zero = ((v_meta_bits >> 16) & 0xFFFF).to(tl.uint16).to(
                tl.bfloat16, bitcast=True
            )
            k0 = (
                (k_packed & (KEY_LEVELS - 1)).to(tl.bfloat16) - k_zero[None, :]
            ) * k_scale[None, :]
            k1 = (
                ((k_packed >> 2) & (KEY_LEVELS - 1)).to(tl.bfloat16) - k_zero[None, :]
            ) * k_scale[None, :]
            k2 = (
                ((k_packed >> 4) & (KEY_LEVELS - 1)).to(tl.bfloat16) - k_zero[None, :]
            ) * k_scale[None, :]
            k3 = (
                ((k_packed >> 6) & (KEY_LEVELS - 1)).to(tl.bfloat16) - k_zero[None, :]
            ) * k_scale[None, :]
            k01 = tl.join(k0, k1)
            k23 = tl.join(k2, k3)
            keys = tl.join(k01, k23)
            keys = tl.reshape(keys, (32, BLOCK_N, 4))
            keys = tl.permute(keys, (2, 0, 1))
            keys = tl.reshape(keys, (128, BLOCK_N))
            scores = tl.dot(query, keys) * ATTN_SCALE
            scores = tl.where(kv_mask[None, :], scores, -float("inf"))

            v0 = (
                (v_packed & (VALUE_LEVELS - 1)).to(tl.bfloat16) - v_zero[:, None]
            ) * v_scale[:, None]
            v1 = (
                ((v_packed >> 2) & (VALUE_LEVELS - 1)).to(tl.bfloat16) - v_zero[:, None]
            ) * v_scale[:, None]
            v2 = (
                ((v_packed >> 4) & (VALUE_LEVELS - 1)).to(tl.bfloat16) - v_zero[:, None]
            ) * v_scale[:, None]
            v3 = (
                ((v_packed >> 6) & (VALUE_LEVELS - 1)).to(tl.bfloat16) - v_zero[:, None]
            ) * v_scale[:, None]

            next_max = tl.maximum(tl.max(scores, 1), e_max)
            rescale = tl.exp(e_max - next_max)
            probs = tl.exp(scores - next_max[:, None])
            acc0 = acc0 * rescale[:, None] + tl.dot(probs.to(tl.bfloat16), v0)
            acc1 = acc1 * rescale[:, None] + tl.dot(probs.to(tl.bfloat16), v1)
            acc2 = acc2 * rescale[:, None] + tl.dot(probs.to(tl.bfloat16), v2)
            acc3 = acc3 * rescale[:, None] + tl.dot(probs.to(tl.bfloat16), v3)
            e_sum = e_sum * rescale + tl.sum(probs, 1)
            e_max = next_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    base = bid * stride_mid_b + heads[:, None] * stride_mid_h + sid * stride_mid_s
    tl.store(Mid_o_ptr + base + offs_quarter[None, :], acc0 / safe_sum[:, None])
    tl.store(
        Mid_o_ptr + base + (offs_quarter + 32)[None, :],
        acc1 / safe_sum[:, None],
    )
    tl.store(
        Mid_o_ptr + base + (offs_quarter + 64)[None, :],
        acc2 / safe_sum[:, None],
    )
    tl.store(
        Mid_o_ptr + base + (offs_quarter + 96)[None, :],
        acc3 / safe_sum[:, None],
    )
    lse = tl.where(e_sum > 0.0, e_max + tl.log(safe_sum), -float("inf"))
    lse_base = bid * stride_mid_b + heads * stride_mid_h + sid * stride_mid_s
    tl.store(Mid_o_ptr + lse_base + HEAD_DIM, lse)


@triton.jit
def _oscar_decode_hp_stage1(
    Q_rot_ptr,
    Prefix_cache_ptr,
    Recent_cache_ptr,
    HP_rows_ptr,
    Prefix_pages_ptr,
    Query_to_req_ptr,
    Shared_hit_lens_ptr,
    Recent_extra_ptr,
    Seq_lens_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_prefix_slot,
    stride_prefix_head,
    stride_prefix_kv,
    stride_recent_slot,
    stride_recent_head,
    stride_recent_kv,
    stride_prefix_pages_req,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
    RECENT_CAPACITY: tl.constexpr,
    NUM_HP_SPLITS: tl.constexpr,
    HP_PARTIAL_START: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Compute grouped-H4 BF16 HP partials with tensor-core dots."""
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)
    head0 = kv_head * KV_GROUP_SIZE
    heads = head0 + tl.arange(0, BLOCK_H)
    head_mask = heads < head0 + KV_GROUP_SIZE
    req_idx = tl.load(Query_to_req_ptr + bid) if MAPPED_QUERIES else bid
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    partial_idx = HP_PARTIAL_START + sid
    out_base = (
        bid * stride_mid_b + heads[:, None] * stride_mid_h + partial_idx * stride_mid_s
    )

    if not MIXED_KV:
        tl.store(
            Mid_o_ptr + out_base + d_offs[None, :],
            0.0,
            mask=head_mask[:, None] & d_mask[None, :],
        )
        lse_base = (
            bid * stride_mid_b
            + heads * stride_mid_h
            + partial_idx * stride_mid_s
        )
        tl.store(
            Mid_o_ptr + lse_base + HEAD_DIM,
            -float("inf"),
            mask=head_mask,
        )
        return

    seq_len = tl.load(Seq_lens_ptr + bid)
    hp_row = tl.load(HP_rows_ptr + req_idx)
    shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
    recent_extra = tl.load(Recent_extra_ptr + req_idx)
    prefix_len = tl.minimum(PREFIX_TOKENS, seq_len)
    recent_start = tl.minimum(
        seq_len,
        tl.maximum(
            tl.maximum(PREFIX_TOKENS, shared_hit_len),
            seq_len - RECENT_TOKENS - recent_extra,
        ),
    )
    hp_len = prefix_len + seq_len - recent_start
    split_len = tl.cdiv(tl.cdiv(hp_len, NUM_HP_SPLITS), BLOCK_N) * BLOCK_N
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, hp_len)
    q_base = bid * stride_qb + heads[:, None] * stride_qh
    query = tl.load(
        Q_rot_ptr + q_base + d_offs[None, :],
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    kv_range = tl.arange(0, BLOCK_N)
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) - float("inf")
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], dtype=tl.float32)

    for start_n in range(split_start, split_end, BLOCK_N):
        hp_offs = start_n + kv_range
        kv_mask = hp_offs < split_end
        is_prefix = hp_offs < prefix_len
        logical_pos = tl.where(is_prefix, hp_offs, recent_start + hp_offs - prefix_len)
        if USE_PREFIX_PAGE_TABLE:
            prefix_page = tl.load(
                Prefix_pages_ptr
                + req_idx * stride_prefix_pages_req
                + logical_pos // BLOCK_SIZE,
                mask=kv_mask & is_prefix,
                other=0,
            )
            prefix_idx = prefix_page * BLOCK_SIZE + logical_pos % BLOCK_SIZE
        else:
            prefix_idx = hp_row * PREFIX_TOKENS + logical_pos
        prefix_base = (
            prefix_idx.to(tl.int64) * stride_prefix_slot
            + tl.cast(kv_head, tl.int64) * stride_prefix_head
        )
        prefix_keys = tl.load(
            Prefix_cache_ptr + prefix_base[None, :] + d_offs[:, None],
            mask=(kv_mask & is_prefix)[None, :] & d_mask[:, None],
            other=0.0,
        )
        recent_idx = (
            hp_row * RECENT_CAPACITY
            + (logical_pos - PREFIX_TOKENS) % RECENT_CAPACITY
        )
        recent_base = (
            recent_idx.to(tl.int64) * stride_recent_slot
            + tl.cast(kv_head, tl.int64) * stride_recent_head
        )
        recent_keys = tl.load(
            Recent_cache_ptr + recent_base[None, :] + d_offs[:, None],
            mask=(kv_mask & ~is_prefix)[None, :] & d_mask[:, None],
            other=0.0,
        )
        keys = tl.where(is_prefix[None, :], prefix_keys, recent_keys).to(tl.bfloat16)
        scores = tl.dot(query, keys) * ATTN_SCALE
        scores = tl.where(
            head_mask[:, None] & kv_mask[None, :],
            scores,
            -float("inf"),
        )

        prefix_values = tl.load(
            Prefix_cache_ptr
            + prefix_base[:, None]
            + stride_prefix_kv
            + d_offs[None, :],
            mask=(kv_mask & is_prefix)[:, None] & d_mask[None, :],
            other=0.0,
        )
        recent_values = tl.load(
            Recent_cache_ptr
            + recent_base[:, None]
            + stride_recent_kv
            + d_offs[None, :],
            mask=(kv_mask & ~is_prefix)[:, None] & d_mask[None, :],
            other=0.0,
        )
        values = tl.where(is_prefix[:, None], prefix_values, recent_values).to(
            tl.bfloat16
        )
        next_max = tl.maximum(tl.max(scores, 1), e_max)
        rescale = tl.exp(e_max - next_max)
        probs = tl.exp(scores - next_max[:, None])
        acc = acc * rescale[:, None] + tl.dot(probs.to(tl.bfloat16), values)
        e_sum = e_sum * rescale + tl.sum(probs, 1)
        e_max = next_max

    safe_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    tl.store(
        Mid_o_ptr + out_base + d_offs[None, :],
        acc / safe_sum[:, None],
        mask=head_mask[:, None] & d_mask[None, :],
    )
    lse = tl.where(e_sum > 0.0, e_max + tl.log(safe_sum), -float("inf"))
    lse_base = bid * stride_mid_b + heads * stride_mid_h + partial_idx * stride_mid_s
    tl.store(Mid_o_ptr + lse_base + HEAD_DIM, lse, mask=head_mask)


@triton.jit
def _oscar_finite_lse_stage2(
    Mid_O,
    Output_ptr,
    LSE_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """Merge OSCAR partials by finite LSE, independent of token geometry."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    mid_base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh

    for partial_idx in range(0, NUM_PARTIALS):
        partial_base = mid_base + partial_idx * stride_mid_os
        tlogic = tl.load(Mid_O + partial_base + Lv)
        is_finite = (tlogic > -float("inf")) & (tlogic < float("inf"))
        if is_finite:
            partial = tl.load(
                Mid_O + partial_base + offs_d,
                mask=mask_d,
                other=0.0,
            )
            next_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp2((e_max - next_max) * 1.4426950408889634)
            partial_scale = tl.exp2((tlogic - next_max) * 1.4426950408889634)
            acc = acc * old_scale + partial * partial_scale
            e_sum = e_sum * old_scale + partial_scale
            e_max = next_max

    safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out = tl.where(e_sum > 0.0, acc / safe_e_sum, 0.0)
    tl.store(
        Output_ptr + cur_batch * stride_obs + cur_head * stride_oh + offs_d,
        out,
        mask=mask_d,
    )
    lse_out = tl.where(
        e_sum > 0.0, e_max + tl.log(safe_e_sum), -float("inf")
    )
    tl.store(LSE_ptr + cur_batch * stride_lse_bs + cur_head, lse_out)


@triton.jit
def _oscar_finite_lse_inverse_v_bf16_stage2(
    Mid_O,
    Rotation_t_ptr,
    Output_ptr,
    LSE_ptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_rotation_row,
    stride_rotation_col,
    stride_obs,
    stride_oh,
    stride_lse_bs,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
):
    """Merge finite-LSE partials and inverse-rotate into BF16 output."""
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)
    mid_base = cur_batch * stride_mid_ob + cur_head * stride_mid_oh

    for partial_idx in range(0, NUM_PARTIALS):
        partial_base = mid_base + partial_idx * stride_mid_os
        tlogic = tl.load(Mid_O + partial_base + Lv)
        is_finite = (tlogic > -float("inf")) & (tlogic < float("inf"))
        if is_finite:
            partial = tl.load(
                Mid_O + partial_base + offs_d,
                mask=mask_d,
                other=0.0,
            )
            next_max = tl.maximum(tlogic, e_max)
            old_scale = tl.exp2((e_max - next_max) * 1.4426950408889634)
            partial_scale = tl.exp2((tlogic - next_max) * 1.4426950408889634)
            acc = acc * old_scale + partial * partial_scale
            e_sum = e_sum * old_scale + partial_scale
            e_max = next_max

    safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)
    out = tl.where(e_sum > 0.0, acc / safe_e_sum, 0.0)
    row = out.to(tl.bfloat16)
    rotation = tl.load(
        Rotation_t_ptr
        + offs_d[:, None] * stride_rotation_row
        + offs_d[None, :] * stride_rotation_col,
        mask=mask_d[:, None] & mask_d[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    result = tl.sum(row[:, None].to(tl.float32) * rotation, axis=0)
    tl.store(
        Output_ptr
        + cur_batch * stride_obs
        + cur_head * stride_oh
        + offs_d,
        result,
        mask=mask_d,
    )
    lse_out = tl.where(
        e_sum > 0.0, e_max + tl.log(safe_e_sum), -float("inf")
    )
    tl.store(LSE_ptr + cur_batch * stride_lse_bs + cur_head, lse_out)


@triton.jit
def _oscar_full_dequant_kv(
    K_data_ptr,
    V_data_ptr,
    K_meta_ptr,
    V_meta_ptr,
    Block_table_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space K
    V_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space V
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_data_block,
    stride_data_pos,
    stride_data_head,
    stride_meta_block,
    stride_meta_pos,
    stride_meta_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    VALUE_DATA_BYTES: tl.constexpr,
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Dequant cached INT2 K/V to fp16 (still in rotated space)."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    data_base = (
        block_num * stride_data_block
        + tl.cast(page_off, tl.int64) * stride_data_pos
        + tl.cast(hid, tl.int64) * stride_data_head
    )
    meta_base = (
        block_num * stride_meta_block
        + tl.cast(page_off, tl.int64) * stride_meta_pos
        + tl.cast(hid, tl.int64) * stride_meta_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    if HEAD_DIM == 128:
        byte_idx = d_offs % KEY_DATA_BYTES
        bit_shift = (d_offs // KEY_DATA_BYTES) * 2
    else:
        byte_idx = d_offs // 4
        bit_shift = (d_offs % 4) * 2

    # K
    k_byte = tl.load(K_data_ptr + data_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_k = ((k_byte >> bit_shift) & (KEY_LEVELS - 1)).to(tl.float32)
    ksc = tl.load(K_meta_ptr + meta_base).to(tl.float32)
    kzr = tl.load(K_meta_ptr + meta_base + 1).to(tl.float32)
    k_recon = (q_k - kzr) * ksc
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # V
    v_byte = tl.load(V_data_ptr + data_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_v = ((v_byte >> bit_shift) & (VALUE_LEVELS - 1)).to(tl.float32)
    vsc = tl.load(V_meta_ptr + meta_base).to(tl.float32)
    vzr = tl.load(V_meta_ptr + meta_base + 1).to(tl.float32)
    v_recon = (q_v - vzr) * vsc
    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_recon.to(tl.float16), mask=d_mask)


_layout_cache: dict = {}
_grouped_h4_score_workspace_cache: dict[tuple, torch.Tensor] = {}


def _get_grouped_h4_score_workspace(
    q_rot: torch.Tensor, block_table: torch.Tensor, block_size: int
) -> torch.Tensor:
    """Reuse one pre-capture FP32 score workspace across sequential layers."""
    batch_size, num_query_heads, _ = q_rot.shape
    token_capacity = block_table.shape[1] * block_size
    key = (
        q_rot.device.type,
        q_rot.device.index,
        batch_size,
        num_query_heads,
        token_capacity,
    )
    workspace = _grouped_h4_score_workspace_cache.get(key)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "OSCAR Grouped-H4 score workspace must be allocated before "
                "CUDA Graph capture"
            )
        workspace = torch.empty(
            batch_size,
            num_query_heads,
            token_capacity,
            dtype=torch.float32,
            device=q_rot.device,
        )
        _grouped_h4_score_workspace_cache[key] = workspace
    return workspace


def _use_grouped_h4_stage1(
    num_query_heads: int, num_kv_heads: int, head_dim: int
) -> bool:
    return (
        head_dim == 128
        and num_kv_heads > 0
        and num_query_heads == 4 * num_kv_heads
    )


def _has_linear_physical_slot_layout(
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
) -> bool:
    metadata_pair_views_supported = sys.byteorder == "little" and all(
        tensor.dtype == torch.bfloat16
        and tensor.ndim == 4
        and tensor.shape[-1] == 2
        and tensor.stride(-1) == 1
        and tensor.storage_offset() % 2 == 0
        and tensor.data_ptr() % 4 == 0
        and all(stride % 2 == 0 for stride in tensor.stride()[:-1])
        for tensor in (k_meta, v_meta)
    )
    return bool(
        metadata_pair_views_supported
        and has_linear_oscar_arena_layout(k_data, v_data, k_meta, v_meta)
    )


def is_oscar_grouped_h4_eligible(
    q_rot: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
    physical_slot_ids: torch.Tensor | None,
) -> bool:
    """Return whether decode can use the BF16 Grouped-H4 path."""
    Hq, D = q_rot.shape[1], q_rot.shape[2]
    Hk = k_data.shape[2]
    return (
        _use_grouped_h4_stage1(Hq, Hk, D)
        and physical_slot_ids is not None
        and _has_linear_physical_slot_layout(k_data, v_data, k_meta, v_meta)
        and q_rot.dtype == torch.bfloat16
        and q_rot.is_contiguous()
    )


def _grouped_h4_partial_counts(
    num_quant_splits: int, mixed_kv: bool
) -> tuple[int, int]:
    """Return BF16 HP and unified (quant + BF16 HP) partial counts."""
    num_hp_splits = _GROUPED_H4_HP_SPLITS if mixed_kv else 1
    return num_hp_splits, num_quant_splits + num_hp_splits


def _use_bf16_inverse_v_output_alias(
    head_dim: int,
    v_rotation_t: torch.Tensor | None,
    output_buf: torch.Tensor | None,
    batch_size: int,
    num_query_heads: int,
) -> bool:
    return bool(
        head_dim == 128
        and v_rotation_t is not None
        and v_rotation_t.dtype == torch.bfloat16
        and tuple(v_rotation_t.shape) == (head_dim, head_dim)
        and v_rotation_t.is_contiguous()
        and output_buf is not None
        and output_buf.dtype == torch.bfloat16
        and tuple(output_buf.shape) == (batch_size, num_query_heads, head_dim)
        and output_buf.is_contiguous()
        and v_rotation_t.device == output_buf.device
    )


def _use_fused_inverse_v_bf16_output(
    grouped_h4: bool,
    head_dim: int,
    num_total_splits: int,
    v_rotation_t: torch.Tensor | None,
    output_buf: torch.Tensor | None,
    mid_o_buf: torch.Tensor | None,
    lse_buf: torch.Tensor | None,
    batch_size: int,
    num_query_heads: int,
) -> bool:
    return bool(
        grouped_h4
        and num_total_splits == _GROUPED_H4_MIXED_TOTAL_SPLITS
        and _use_bf16_inverse_v_output_alias(
            head_dim,
            v_rotation_t,
            output_buf,
            batch_size,
            num_query_heads,
        )
        and mid_o_buf is not None
        and mid_o_buf.dtype == torch.float32
        and tuple(mid_o_buf.shape)
        == (batch_size, num_query_heads, num_total_splits, head_dim + 1)
        and mid_o_buf.is_contiguous()
        and lse_buf is not None
        and lse_buf.dtype == torch.float32
        and tuple(lse_buf.shape) == (batch_size, num_query_heads)
        and lse_buf.is_contiguous()
        and mid_o_buf.device == output_buf.device
        and lse_buf.device == output_buf.device
    )


@triton.jit
def _inverse_v_rotation_kernel(
    Input_ptr,
    Rotation_t_ptr,
    Output_ptr,
    stride_input_b,
    stride_input_h,
    stride_rotation_row,
    stride_rotation_col,
    stride_output_b,
    stride_output_h,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    bid = tl.program_id(0)
    hid = tl.program_id(1)
    d = tl.arange(0, BLOCK_D)
    mask = d < HEAD_DIM
    row = tl.load(
        Input_ptr + bid * stride_input_b + hid * stride_input_h + d,
        mask=mask,
        other=0.0,
    ).to(tl.bfloat16)
    rotation = tl.load(
        Rotation_t_ptr
        + d[:, None] * stride_rotation_row
        + d[None, :] * stride_rotation_col,
        mask=mask[:, None] & mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    result = tl.sum(row[:, None].to(tl.float32) * rotation, axis=0)
    tl.store(
        Output_ptr + bid * stride_output_b + hid * stride_output_h + d,
        result,
        mask=mask,
    )


def oscar_decode_attention(
    q_rot: torch.Tensor,  # [B, Hq, D] — query already rotated by R_k
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    max_num_kv_splits: int = 16,
    prefix_cache: torch.Tensor | None = None,
    recent_cache: torch.Tensor | None = None,
    hp_row_ids: torch.Tensor | None = None,
    prefix_page_ids: torch.Tensor | None = None,
    prefix_tokens: int | None = None,
    recent_tokens: int | None = None,
    recent_capacity: int | None = None,
    recent_extra: torch.Tensor | None = None,
    v_rotation_t: torch.Tensor | None = None,
    query_to_req_indices: torch.Tensor | None = None,
    shared_hit_tokens: torch.Tensor | None = None,
    physical_slot_ids: torch.Tensor | None = None,
    return_lse: bool = False,
    use_grouped_h4: bool | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused OSCAR decode, optionally including the Triton V inverse."""
    validate_oscar_separated_arenas(
        k_data,
        v_data,
        k_meta,
        v_meta,
        data_bytes=key_data_bytes,
    )
    if key_packed_size != key_data_bytes + 4:
        raise ValueError("OSCAR separated decode requires one BF16 K meta pair")
    B, Hq, D = q_rot.shape
    Hk = k_data.shape[2]
    block_size = k_data.shape[1]
    kv_group_size = Hq // Hk
    grouped_h4_supported = _use_grouped_h4_stage1(Hq, Hk, D)
    if use_grouped_h4 is True and not grouped_h4_supported:
        raise ValueError("Grouped-H4 OSCAR decode requires Hq/Hk=4 and head_dim=128")
    if use_grouped_h4 is True and physical_slot_ids is None:
        raise ValueError("Grouped-H4 OSCAR decode requires preindexed slot IDs")
    grouped_h4 = (
        grouped_h4_supported and physical_slot_ids is not None
        if use_grouped_h4 is None
        else use_grouped_h4
    )
    if grouped_h4 and not _has_linear_physical_slot_layout(
        k_data, v_data, k_meta, v_meta
    ):
        if use_grouped_h4 is True:
            raise ValueError(
                "Grouped-H4 preindexed slots require a linear paged cache"
            )
        grouped_h4 = False
    device = q_rot.device
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = (
        _GROUPED_H4_PREINDEXED_QUANT_SPLITS if grouped_h4 else max_num_kv_splits
    )
    mixed_kv = prefix_cache is not None
    NUM_HP_SPLITS, NUM_TOTAL_SPLITS = (
        _grouped_h4_partial_counts(NUM_KV_SPLITS, mixed_kv)
        if grouped_h4
        else (0, NUM_KV_SPLITS)
    )
    mapped_queries = query_to_req_indices is not None
    use_prefix_page_table = prefix_page_ids is not None
    if mixed_kv and (
        recent_cache is None
        or hp_row_ids is None
        or prefix_tokens is None
        or recent_tokens is None
    ):
        raise ValueError(
            "Both BF16 caches and their window sizes are required for mixed decode"
        )
    if prefix_cache is None:
        prefix_cache = k_meta
    if recent_cache is None:
        recent_cache = k_meta
    if hp_row_ids is None:
        hp_row_ids = seq_lens
    prefix_tokens = prefix_tokens or 0
    recent_tokens = recent_tokens or 1
    recent_capacity = recent_capacity or recent_tokens
    if prefix_page_ids is None:
        prefix_page_ids = hp_row_ids
    elif prefix_page_ids.ndim != 2:
        raise ValueError("OSCAR prefix page table must be a 2D tensor")
    elif prefix_page_ids.shape[1] * block_size != prefix_tokens:
        raise ValueError("OSCAR prefix page table width must cover the prefix window")
    if query_to_req_indices is None:
        query_to_req_indices = seq_lens
    if shared_hit_tokens is None:
        shared_hit_tokens = torch.zeros_like(hp_row_ids)
    elif shared_hit_tokens.ndim != 1:
        raise ValueError("OSCAR shared hit lengths must be a 1D tensor")
    if recent_extra is None:
        recent_extra = torch.zeros_like(hp_row_ids)
    elif recent_extra.ndim != 1:
        raise ValueError("OSCAR recent extra lengths must be a 1D tensor")
    if grouped_h4:
        assert physical_slot_ids is not None
        if physical_slot_ids.ndim != 2:
            raise ValueError("OSCAR physical slot IDs must be a 2D tensor")
        if physical_slot_ids.dtype != torch.int64:
            raise ValueError("OSCAR physical slot IDs must use int64")
        if physical_slot_ids.device != k_data.device:
            raise ValueError("OSCAR physical slot IDs must share the KV cache device")
        if physical_slot_ids.shape[0] < block_table.shape[0]:
            raise ValueError("OSCAR physical slot IDs have too few request rows")
        if physical_slot_ids.shape[1] < block_table.shape[1] * block_size:
            raise ValueError(
                "OSCAR physical slot buffer does not cover the block table"
            )

    if grouped_h4 and q_rot.dtype == torch.bfloat16:
        q_rot = q_rot.contiguous()
    else:
        q_rot = q_rot.contiguous().float()

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_TOTAL_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_TOTAL_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_TOTAL_SPLITS, D + 1, dtype=torch.float32, device=device
        )

    if grouped_h4:
        k_meta_pairs = k_meta.view(torch.int32).squeeze(-1)
        v_meta_pairs = v_meta.view(torch.int32).squeeze(-1)
        _oscar_decode_quant_stage1_grouped_h4[(B, Hk, NUM_KV_SPLITS)](
            q_rot,
            k_data,
            v_data,
            k_meta_pairs,
            v_meta_pairs,
            query_to_req_indices,
            shared_hit_tokens,
            recent_extra,
            physical_slot_ids,
            seq_lens,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            k_data.stride(1),
            k_data.stride(2),
            k_meta_pairs.stride(1),
            k_meta_pairs.stride(2),
            physical_slot_ids.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            HEAD_DIM=D,
            NUM_QUANT_SPLITS=NUM_KV_SPLITS,
            KEY_DATA_BYTES=key_data_bytes,
            VALUE_DATA_BYTES=value_data_bytes,
            KEY_LEVELS=key_levels,
            VALUE_LEVELS=value_levels,
            ATTN_SCALE=scale,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            BLOCK_N=128,
            num_warps=4,
            num_stages=3,
        )
        _oscar_decode_hp_stage1[(B, Hk, NUM_HP_SPLITS)](
            q_rot,
            prefix_cache,
            recent_cache,
            hp_row_ids,
            prefix_page_ids,
            query_to_req_indices,
            shared_hit_tokens,
            recent_extra,
            seq_lens,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            prefix_cache.stride(0),
            prefix_cache.stride(1),
            prefix_cache.stride(2),
            recent_cache.stride(0),
            recent_cache.stride(1),
            recent_cache.stride(2),
            prefix_page_ids.stride(0) if use_prefix_page_table else 0,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            KV_GROUP_SIZE=kv_group_size,
            ATTN_SCALE=scale,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            RECENT_CAPACITY=recent_capacity,
            NUM_HP_SPLITS=NUM_HP_SPLITS,
            HP_PARTIAL_START=NUM_KV_SPLITS,
            BLOCK_D=BLOCK_D,
            BLOCK_N=32,
            BLOCK_H=16,
            num_warps=4,
            num_stages=2,
        )
    else:
        grid = (B, Hq, NUM_KV_SPLITS)
        _oscar_decode_stage1[grid](
            q_rot,
            k_data,
            v_data,
            k_meta,
            v_meta,
            prefix_cache,
            recent_cache,
            hp_row_ids,
            prefix_page_ids,
            query_to_req_indices,
            shared_hit_tokens,
            recent_extra,
            block_table,
            seq_lens,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            k_data.stride(0),
            k_data.stride(1),
            k_data.stride(2),
            k_meta.stride(0),
            k_meta.stride(1),
            k_meta.stride(2),
            prefix_cache.stride(0),
            prefix_cache.stride(1),
            prefix_cache.stride(2),
            recent_cache.stride(0),
            recent_cache.stride(1),
            recent_cache.stride(2),
            prefix_page_ids.stride(0) if use_prefix_page_table else 0,
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            KEY_DATA_BYTES=key_data_bytes,
            VALUE_DATA_BYTES=value_data_bytes,
            KEY_LEVELS=key_levels,
            VALUE_LEVELS=value_levels,
            ATTN_SCALE=scale,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            RECENT_CAPACITY=recent_capacity,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=64,
            num_warps=2,
            num_stages=2,
        )

    candidate_output = None
    if (
        output_buf is not None
        and output_buf.ndim == 3
        and output_buf.shape[0] >= B
        and output_buf.shape[1] >= Hq
        and output_buf.shape[2] >= D
    ):
        candidate_output = (
            output_buf
            if tuple(output_buf.shape) == (B, Hq, D)
            else output_buf[:B, :Hq, :D]
        )
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
    use_fused_inverse_v = _use_fused_inverse_v_bf16_output(
        grouped_h4,
        D,
        NUM_TOTAL_SPLITS,
        v_rotation_t,
        candidate_output,
        mid_o,
        lse,
        B,
        Hq,
    )
    if use_fused_inverse_v:
        assert candidate_output is not None
        output = candidate_output
    else:
        output = torch.empty(B, Hq, D, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
    if use_fused_inverse_v:
        _oscar_finite_lse_inverse_v_bf16_stage2[grid2](
            mid_o,
            v_rotation_t,
            output,
            lse,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            v_rotation_t.stride(0),
            v_rotation_t.stride(1),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_PARTIALS=NUM_TOTAL_SPLITS,
            BLOCK_DV=BLOCK_D,
            Lv=D,
            num_warps=4,
            num_stages=2,
        )
    elif grouped_h4:
        _oscar_finite_lse_stage2[grid2](
            mid_o,
            output,
            lse,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_PARTIALS=NUM_TOTAL_SPLITS,
            BLOCK_DV=BLOCK_D,
            Lv=D,
            num_warps=4,
            num_stages=2,
        )
    else:
        _fwd_kernel_stage2[grid2](
            mid_o,
            output,
            lse,
            seq_lens,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            output.stride(0),
            output.stride(1),
            lse.stride(0),
            NUM_KV_SPLITS=NUM_TOTAL_SPLITS,
            BLOCK_DV=BLOCK_D,
            Lv=D,
            OUTPUT_FP16=0,
            num_warps=4,
            num_stages=2,
        )
    if v_rotation_t is not None and not use_fused_inverse_v:
        rotated_output = torch.empty_like(output)
        _inverse_v_rotation_kernel[(B, Hq)](
            output,
            v_rotation_t,
            rotated_output,
            output.stride(0),
            output.stride(1),
            v_rotation_t.stride(0),
            v_rotation_t.stride(1),
            rotated_output.stride(0),
            rotated_output.stride(1),
            HEAD_DIM=D,
            BLOCK_D=BLOCK_D,
            num_warps=4,
            num_stages=1,
        )
        output = rotated_output
    if return_lse:
        return output, lse
    return output


def oscar_full_dequant_kv(
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
    block_table: torch.Tensor,
    cached_len: int,
    num_kv_heads: int,
    head_dim: int,
    key_levels: int,
    value_levels: int,
    key_data_bytes: int,
    key_packed_size: int,
    value_data_bytes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequant the first ``cached_len`` cached tokens to fp16 (rotated space).

    Returns ``(k, v)`` each ``[cached_len, Hk, D]``.
    """
    validate_oscar_separated_arenas(
        k_data,
        v_data,
        k_meta,
        v_meta,
        data_bytes=key_data_bytes,
    )
    if key_packed_size != key_data_bytes + 4:
        raise ValueError("OSCAR separated dequant requires one BF16 K meta pair")
    device = k_data.device
    block_size = k_data.shape[1]
    alloc_len = math.ceil(cached_len / block_size) * block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    k_buf = torch.empty(
        1, num_kv_heads, alloc_len, head_dim, dtype=torch.float16, device=device
    )
    v_buf = torch.empty_like(k_buf)

    grid = (alloc_len, num_kv_heads)
    _oscar_full_dequant_kv[grid](
        k_data,
        v_data,
        k_meta,
        v_meta,
        block_table,
        k_buf,
        v_buf,
        k_buf.stride(0),
        k_buf.stride(1),
        k_buf.stride(2),
        v_buf.stride(0),
        v_buf.stride(1),
        v_buf.stride(2),
        k_data.stride(0),
        k_data.stride(1),
        k_data.stride(2),
        k_meta.stride(0),
        k_meta.stride(1),
        k_meta.stride(2),
        block_table.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        KEY_DATA_BYTES=key_data_bytes,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    k = k_buf[0, :, :cached_len, :].transpose(0, 1)  # [cached_len, Hk, D]
    v = v_buf[0, :, :cached_len, :].transpose(0, 1)
    return k, v
