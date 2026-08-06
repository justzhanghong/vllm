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

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2


@triton.jit
def _oscar_decode_stage1(
    Q_rot_ptr,  # [B, Hq, D] fp32 — query already rotated by R_k
    KV_cache_ptr,  # [num_blocks, block_size, Hk, slot_size] uint8
    KV_meta_ptr,  # BF16 view of the same cache storage
    Prefix_cache_ptr,  # [prefix_slots, Hk, 2, D] bf16
    Recent_cache_ptr,  # [recent_slots, Hk, 2, D] bf16
    HP_rows_ptr,  # [B] int32
    Prefix_pages_ptr,  # [B, prefix_pages] int32
    Query_to_req_ptr,  # [B] int32 when multiple queries share a request
    Shared_hit_lens_ptr,  # [requests] int32 initial locally shared hit length
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] fp32
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
    KEY_DATA_BYTES: tl.constexpr,  # D // 4
    KEY_PACKED: tl.constexpr,  # key region size incl. meta
    VALUE_DATA_BYTES: tl.constexpr,  # D // 4
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
    MIXED_KV: tl.constexpr,
    MAPPED_QUERIES: tl.constexpr,
    USE_PREFIX_PAGE_TABLE: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
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

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # INT2 unpack index vectors (loop-invariant): 4 indices per byte.
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
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ---- dequant K (INT2) and score ----
        if MIXED_KV:
            shared_hit_len = tl.load(Shared_hit_lens_ptr + req_idx)
            recent_start = tl.maximum(
                tl.maximum(PREFIX_TOKENS, shared_hit_len), seq_len - RECENT_TOKENS
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
        k_zero = tl.load(KV_meta_ptr + k_meta // 2 + 1, mask=quant_mask, other=0.0).to(
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
                hp_row * RECENT_TOKENS + (kv_offs - PREFIX_TOKENS) % RECENT_TOKENS
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
        v_zero = tl.load(KV_meta_ptr + v_meta // 2 + 1, mask=quant_mask, other=0.0).to(
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

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
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
                tl.maximum(PREFIX_TOKENS, shared_hit_len), seq_len - RECENT_TOKENS
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
                hp_row * RECENT_TOKENS + (kv_offs - PREFIX_TOKENS) % RECENT_TOKENS
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

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
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
                tl.maximum(PREFIX_TOKENS, shared_hit_len), seq_len - RECENT_TOKENS
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
                hp_row * RECENT_TOKENS + (kv_offs - PREFIX_TOKENS) % RECENT_TOKENS
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

    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)
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
                tl.maximum(PREFIX_TOKENS, shared_hit_len), seq_len - RECENT_TOKENS
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
                hp_row * RECENT_TOKENS + (kv_offs - PREFIX_TOKENS) % RECENT_TOKENS
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
def _oscar_full_dequant_kv(
    KV_cache_ptr,
    KV_meta_ptr,
    Block_table_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space K
    V_out_ptr,  # [B, Hk, max_seq, D] fp16 — rotated-space V
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KEY_DATA_BYTES: tl.constexpr,
    KEY_PACKED: tl.constexpr,
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
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    byte_idx = d_offs // 4
    bit_shift = (d_offs % 4) * 2

    # K
    k_byte = tl.load(KV_cache_ptr + slot_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_k = ((k_byte >> bit_shift) & (KEY_LEVELS - 1)).to(tl.float32)
    k_meta = slot_base + KEY_DATA_BYTES
    ksc = tl.load(KV_meta_ptr + k_meta // 2).to(tl.float32)
    kzr = tl.load(KV_meta_ptr + k_meta // 2 + 1).to(tl.float32)
    k_recon = (q_k - kzr) * ksc
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # V
    v_base = slot_base + KEY_PACKED
    v_byte = tl.load(KV_cache_ptr + v_base + byte_idx, mask=d_mask, other=0).to(
        tl.int32
    )
    q_v = ((v_byte >> bit_shift) & (VALUE_LEVELS - 1)).to(tl.float32)
    v_meta = v_base + VALUE_DATA_BYTES
    vsc = tl.load(KV_meta_ptr + v_meta // 2).to(tl.float32)
    vzr = tl.load(KV_meta_ptr + v_meta // 2 + 1).to(tl.float32)
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
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
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
    v_rotation_t: torch.Tensor | None = None,
    query_to_req_indices: torch.Tensor | None = None,
    shared_hit_tokens: torch.Tensor | None = None,
    return_lse: bool = False,
    use_grouped_h4: bool | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Fused OSCAR decode, optionally including the Triton V inverse."""
    B, Hq, D = q_rot.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    grouped_h4_supported = _use_grouped_h4_stage1(Hq, Hk, D)
    if use_grouped_h4 is True and not grouped_h4_supported:
        raise ValueError("Grouped-H4 OSCAR decode requires Hq/Hk=4 and head_dim=128")
    grouped_h4 = grouped_h4_supported if use_grouped_h4 is None else use_grouped_h4
    device = q_rot.device
    BLOCK_D = triton.next_power_of_2(D)
    NUM_KV_SPLITS = max_num_kv_splits
    mixed_kv = prefix_cache is not None
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
        prefix_cache = kv_cache
    if recent_cache is None:
        recent_cache = kv_cache
    if hp_row_ids is None:
        hp_row_ids = seq_lens
    prefix_tokens = prefix_tokens or 0
    recent_tokens = recent_tokens or 1
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

    q_rot = q_rot.contiguous().float()

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B, Hq, NUM_KV_SPLITS, D + 1, dtype=torch.float32, device=device
        )

    if grouped_h4:
        score_workspace = _get_grouped_h4_score_workspace(
            q_rot, block_table, block_size
        )
        grid = (B, Hk, NUM_KV_SPLITS)
        _oscar_decode_stage1_grouped_h4_qk[grid](
            q_rot,
            kv_cache,
            kv_cache.view(torch.bfloat16),
            prefix_cache,
            recent_cache,
            hp_row_ids,
            prefix_page_ids,
            query_to_req_indices,
            shared_hit_tokens,
            block_table,
            seq_lens,
            score_workspace,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            prefix_cache.stride(0),
            prefix_cache.stride(1),
            recent_cache.stride(0),
            recent_cache.stride(1),
            prefix_page_ids.stride(0) if use_prefix_page_table else 0,
            block_table.stride(0),
            score_workspace.stride(0),
            score_workspace.stride(1),
            score_workspace.stride(2),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            KEY_DATA_BYTES=key_data_bytes,
            KEY_LEVELS=key_levels,
            ATTN_SCALE=scale,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=64,
            num_warps=2,
            num_stages=2,
        )
        _oscar_decode_stage1_grouped_h4_v[grid](
            kv_cache,
            kv_cache.view(torch.bfloat16),
            prefix_cache,
            recent_cache,
            hp_row_ids,
            prefix_page_ids,
            query_to_req_indices,
            shared_hit_tokens,
            block_table,
            seq_lens,
            score_workspace,
            mid_o,
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            prefix_cache.stride(0),
            prefix_cache.stride(1),
            prefix_cache.stride(2),
            recent_cache.stride(0),
            recent_cache.stride(1),
            recent_cache.stride(2),
            prefix_page_ids.stride(0) if use_prefix_page_table else 0,
            block_table.stride(0),
            score_workspace.stride(0),
            score_workspace.stride(1),
            score_workspace.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            KEY_PACKED=key_packed_size,
            VALUE_DATA_BYTES=value_data_bytes,
            VALUE_LEVELS=value_levels,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=64,
            num_warps=2,
            num_stages=2,
        )
    else:
        grid = (B, Hq, NUM_KV_SPLITS)
        _oscar_decode_stage1[grid](
            q_rot,
            kv_cache,
            kv_cache.view(torch.bfloat16),
            prefix_cache,
            recent_cache,
            hp_row_ids,
            prefix_page_ids,
            query_to_req_indices,
            shared_hit_tokens,
            block_table,
            seq_lens,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
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
            KEY_PACKED=key_packed_size,
            VALUE_DATA_BYTES=value_data_bytes,
            KEY_LEVELS=key_levels,
            VALUE_LEVELS=value_levels,
            ATTN_SCALE=scale,
            MIXED_KV=mixed_kv,
            MAPPED_QUERIES=mapped_queries,
            USE_PREFIX_PAGE_TABLE=use_prefix_page_table,
            PREFIX_TOKENS=prefix_tokens,
            RECENT_TOKENS=recent_tokens,
            BLOCK_D=BLOCK_D,
            BLOCK_KV=64,
            num_warps=2,
            num_stages=2,
        )

    out_dtype = torch.float32
    if output_buf is not None and output_buf.shape[0] >= B:
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)

    grid2 = (B, Hq)
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
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=BLOCK_D,
        Lv=D,
        num_warps=4,
        num_stages=2,
    )
    if v_rotation_t is not None:
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
    kv_cache: torch.Tensor,
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
    device = kv_cache.device
    block_size = kv_cache.shape[1]
    alloc_len = math.ceil(cached_len / block_size) * block_size
    BLOCK_D = triton.next_power_of_2(head_dim)
    k_buf = torch.empty(
        1, num_kv_heads, alloc_len, head_dim, dtype=torch.float16, device=device
    )
    v_buf = torch.empty_like(k_buf)

    grid = (alloc_len, num_kv_heads)
    _oscar_full_dequant_kv[grid](
        kv_cache,
        kv_cache.view(torch.bfloat16),
        block_table,
        k_buf,
        v_buf,
        k_buf.stride(0),
        k_buf.stride(1),
        k_buf.stride(2),
        v_buf.stride(0),
        v_buf.stride(1),
        v_buf.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        block_table.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_SIZE=block_size,
        NUM_KV_HEADS=num_kv_heads,
        KEY_DATA_BYTES=key_data_bytes,
        KEY_PACKED=key_packed_size,
        VALUE_DATA_BYTES=value_data_bytes,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )
    k = k_buf[0, :, :cached_len, :].transpose(0, 1)  # [cached_len, Hk, D]
    v = v_buf[0, :, :cached_len, :].transpose(0, 1)
    return k, v
