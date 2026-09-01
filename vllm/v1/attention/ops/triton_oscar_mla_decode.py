# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse mixed-tier OSCAR MLA decode kernels for A800/SM80."""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_oscar_mla_store import (
    _require_cuda_tensor,
    _validate_history_tensors,
    oscar_mla_rotate,
)


@triton.jit
def _mixed_sparse_decode_stage1(
    query_ptr,
    query_rotated_ptr,
    query_rope_ptr,
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
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
    mid_bf16_ptr,
    mid_history_ptr,
    mid_lse_ptr,
    stride_query_b: tl.constexpr,
    stride_query_h: tl.constexpr,
    stride_query_d: tl.constexpr,
    stride_query_rotated_b: tl.constexpr,
    stride_query_rotated_h: tl.constexpr,
    stride_query_rotated_d: tl.constexpr,
    stride_query_rope_b: tl.constexpr,
    stride_query_rope_h: tl.constexpr,
    stride_query_rope_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_block_table_b: tl.constexpr,
    stride_rope_block_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_split: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_split: tl.constexpr,
    num_splits: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    packed_group_bytes: tl.constexpr,
    attention_scale: tl.constexpr,
    num_requests: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
    block_r: tl.constexpr,
):
    query_row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)
    request = tl.load(
        query_request_indices_ptr + query_row * stride_query_request,
    )
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    query = tl.load(
        query_ptr
        + query_row * stride_query_b
        + head * stride_query_h
        + dims * stride_query_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    query_rotated = tl.load(
        query_rotated_ptr
        + query_row * stride_query_rotated_b
        + head * stride_query_rotated_h
        + dims * stride_query_rotated_d,
        mask=dim_mask,
        other=0.0,
    ).to(tl.float32)
    rope_dims = tl.arange(0, block_r)
    rope_dim_mask = rope_dims < rope_head_size
    query_rope = tl.load(
        query_rope_ptr
        + query_row * stride_query_rope_b
        + head * stride_query_rope_h
        + rope_dims * stride_query_rope_d,
        mask=rope_dim_mask,
        other=0.0,
    ).to(tl.float32)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    split_len = tl.cdiv(topk, num_splits)
    split_start = split * split_len
    split_end = tl.minimum(split_start + split_len, topk)
    token_offsets = tl.arange(0, block_t)
    m_prev = -float("inf")
    l_prev = 0.0
    bf16_acc = tl.zeros((block_d,), dtype=tl.float32)
    history_acc = tl.zeros((block_d,), dtype=tl.float32)

    for tile_start in range(split_start, split_end, block_t):
        selected_offsets = tile_start + token_offsets
        selected_mask = selected_offsets < split_end
        tokens = tl.load(
            selected_tokens_ptr
            + query_row * stride_selected_b
            + selected_offsets * stride_selected_k,
            mask=selected_mask,
            other=-1,
        )
        valid = (
            selected_mask
            & request_valid
            & (query_position >= 0)
            & (tokens >= 0)
            & (tokens < causal_seq_len)
            & (hp_row >= 0)
        )
        is_prefix = valid & (tokens < prefix_tokens)
        is_recent = valid & (tokens >= recent_start)
        is_history = valid & ~is_prefix & ~is_recent

        prefix_base = hp_row * stride_prefix_row + tokens * stride_prefix_token
        prefix_values = tl.load(
            prefix_ptr + prefix_base[:, None] + dims[None, :] * stride_prefix_d,
            mask=is_prefix[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        recent_indices = (tokens - prefix_tokens) % recent_capacity_tokens
        recent_base = hp_row * stride_recent_row + recent_indices * stride_recent_token
        recent_values = tl.load(
            recent_ptr + recent_base[:, None] + dims[None, :] * stride_recent_d,
            mask=is_recent[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        bf16_values = tl.where(
            is_prefix[:, None],
            prefix_values,
            recent_values,
        )

        history_indices = tokens - prefix_tokens
        logical_pages = history_indices // history_block_size
        page_offsets = history_indices % history_block_size
        physical_pages = tl.load(
            history_page_table_ptr
            + safe_request * stride_page_table_b
            + logical_pages * stride_page_table_page,
            mask=is_history,
            other=0,
        )
        byte_offsets = dims // 4
        shifts = (dims % 4) * 2
        data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
        packed = tl.load(
            history_data_ptr
            + data_base[:, None]
            + byte_offsets[None, :] * stride_data_byte,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> shifts[None, :]) & 0x3).to(tl.float32)
        groups = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_pages[:, None] * stride_scale_page
            + page_offsets[:, None] * stride_scale_token
            + groups[None, :] * stride_scale_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_pages[:, None] * stride_zero_page
            + page_offsets[:, None] * stride_zero_token
            + groups[None, :] * stride_zero_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        history_values = (quantized - zero) * scale

        rope_logical_pages = tokens // rope_block_size
        rope_page_offsets = tokens % rope_block_size
        rope_physical_pages = tl.load(
            rope_block_table_ptr
            + safe_request * stride_rope_block_table_b
            + rope_logical_pages * stride_rope_block_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + rope_physical_pages[:, None] * stride_rope_block
            + rope_page_offsets[:, None] * stride_rope_token
            + rope_dims[None, :] * stride_rope_d,
            mask=valid[:, None] & rope_dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        rope_scores = tl.sum(
            rope_values * query_rope[None, :],
            axis=1,
        )

        bf16_scores = tl.sum(
            bf16_values * query[None, :],
            axis=1,
        )
        history_scores = tl.sum(
            history_values * query_rotated[None, :],
            axis=1,
        )
        scores = (
            tl.where(
                is_history,
                history_scores,
                bf16_scores,
            )
            + rope_scores
        )
        scores *= attention_scale
        scores = tl.where(valid, scores, -float("inf"))
        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            m_new = tl.maximum(tl.max(scores, axis=0), m_prev)
            previous_scale = tl.exp(m_prev - m_new)
            probabilities = tl.exp(scores - m_new)
            probabilities = tl.where(valid, probabilities, 0.0)
            bf16_acc = bf16_acc * previous_scale + tl.sum(
                probabilities[:, None]
                * tl.where(is_history[:, None], 0.0, bf16_values),
                axis=0,
            )
            history_acc = history_acc * previous_scale + tl.sum(
                probabilities[:, None]
                * tl.where(is_history[:, None], history_values, 0.0),
                axis=0,
            )
            l_prev = l_prev * previous_scale + tl.sum(probabilities, axis=0)
            m_prev = m_new

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    mid_base = query_row * stride_mid_b + head * stride_mid_h + split * stride_mid_split
    tl.store(
        mid_bf16_ptr + mid_base + dims * stride_mid_d,
        bf16_acc / safe_l,
        mask=dim_mask,
    )
    tl.store(
        mid_history_ptr + mid_base + dims * stride_mid_d,
        history_acc / safe_l,
        mask=dim_mask,
    )
    local_lse = tl.where(
        l_prev > 0.0,
        m_prev + tl.log(safe_l),
        -float("inf"),
    )
    tl.store(
        mid_lse_ptr
        + query_row * stride_lse_b
        + head * stride_lse_h
        + split * stride_lse_split,
        local_lse,
    )


@triton.jit
def _mixed_sparse_decode_grouped_h4_qk(
    query_ptr,
    query_rotated_ptr,
    query_rope_ptr,
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
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
    score_ptr,
    stride_query_b: tl.constexpr,
    stride_query_h: tl.constexpr,
    stride_query_d: tl.constexpr,
    stride_query_rotated_b: tl.constexpr,
    stride_query_rotated_h: tl.constexpr,
    stride_query_rotated_d: tl.constexpr,
    stride_query_rope_b: tl.constexpr,
    stride_query_rope_h: tl.constexpr,
    stride_query_rope_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_block_table_b: tl.constexpr,
    stride_rope_block_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_score_b: tl.constexpr,
    stride_score_h: tl.constexpr,
    stride_score_k: tl.constexpr,
    num_splits: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    attention_scale: tl.constexpr,
    num_requests: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
    block_r: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    split = tl.program_id(2)
    head0 = head_group * 4
    request = tl.load(
        query_request_indices_ptr + query_row * stride_query_request,
    )
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    rope_dims = tl.arange(0, block_r)
    rope_dim_mask = rope_dims < rope_head_size
    split_len = tl.cdiv(topk, num_splits)
    split_start = split * split_len
    split_end = tl.minimum(split_start + split_len, topk)
    token_offsets = tl.arange(0, block_t)

    for tile_start in range(split_start, split_end, block_t):
        selected_offsets = tile_start + token_offsets
        selected_mask = selected_offsets < split_end
        tokens = tl.load(
            selected_tokens_ptr
            + query_row * stride_selected_b
            + selected_offsets * stride_selected_k,
            mask=selected_mask,
            other=-1,
        )
        valid = (
            selected_mask
            & request_valid
            & (query_position >= 0)
            & (tokens >= 0)
            & (tokens < causal_seq_len)
            & (hp_row >= 0)
        )
        is_prefix = valid & (tokens < prefix_tokens)
        is_recent = valid & (tokens >= recent_start)
        is_history = valid & ~is_prefix & ~is_recent

        prefix_base = hp_row * stride_prefix_row + tokens * stride_prefix_token
        prefix_values = tl.load(
            prefix_ptr + prefix_base[:, None] + dims[None, :] * stride_prefix_d,
            mask=is_prefix[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        recent_indices = (tokens - prefix_tokens) % recent_capacity_tokens
        recent_base = hp_row * stride_recent_row + recent_indices * stride_recent_token
        recent_values = tl.load(
            recent_ptr + recent_base[:, None] + dims[None, :] * stride_recent_d,
            mask=is_recent[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        bf16_values = tl.where(is_prefix[:, None], prefix_values, recent_values)

        history_indices = tokens - prefix_tokens
        logical_pages = history_indices // history_block_size
        page_offsets = history_indices % history_block_size
        physical_pages = tl.load(
            history_page_table_ptr
            + safe_request * stride_page_table_b
            + logical_pages * stride_page_table_page,
            mask=is_history,
            other=0,
        )
        byte_offsets = dims // 4
        shifts = (dims % 4) * 2
        data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
        packed = tl.load(
            history_data_ptr
            + data_base[:, None]
            + byte_offsets[None, :] * stride_data_byte,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> shifts[None, :]) & 0x3).to(tl.float32)
        groups = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_pages[:, None] * stride_scale_page
            + page_offsets[:, None] * stride_scale_token
            + groups[None, :] * stride_scale_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_pages[:, None] * stride_zero_page
            + page_offsets[:, None] * stride_zero_token
            + groups[None, :] * stride_zero_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        history_values = (quantized - zero) * scale

        rope_logical_pages = tokens // rope_block_size
        rope_page_offsets = tokens % rope_block_size
        rope_physical_pages = tl.load(
            rope_block_table_ptr
            + safe_request * stride_rope_block_table_b
            + rope_logical_pages * stride_rope_block_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + rope_physical_pages[:, None] * stride_rope_block
            + rope_page_offsets[:, None] * stride_rope_token
            + rope_dims[None, :] * stride_rope_d,
            mask=valid[:, None] & rope_dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Keep each head's query live only until its FP32 score is stored.
        for head_offset in tl.static_range(0, 4):
            head = head0 + head_offset
            query = tl.load(
                query_ptr
                + query_row * stride_query_b
                + head * stride_query_h
                + dims * stride_query_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            query_rotated = tl.load(
                query_rotated_ptr
                + query_row * stride_query_rotated_b
                + head * stride_query_rotated_h
                + dims * stride_query_rotated_d,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            query_rope = tl.load(
                query_rope_ptr
                + query_row * stride_query_rope_b
                + head * stride_query_rope_h
                + rope_dims * stride_query_rope_d,
                mask=rope_dim_mask,
                other=0.0,
            ).to(tl.float32)
            rope_scores = tl.sum(rope_values * query_rope[None, :], axis=1)
            bf16_scores = tl.sum(bf16_values * query[None, :], axis=1)
            history_scores = tl.sum(
                history_values * query_rotated[None, :],
                axis=1,
            )
            scores = tl.where(is_history, history_scores, bf16_scores) + rope_scores
            scores *= attention_scale
            scores = tl.where(valid, scores, -float("inf"))
            tl.store(
                score_ptr
                + query_row * stride_score_b
                + head * stride_score_h
                + selected_offsets * stride_score_k,
                scores,
                mask=selected_mask,
            )


@triton.jit
def _mixed_sparse_decode_grouped_h4_v(
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
    prefix_ptr,
    recent_ptr,
    history_data_ptr,
    history_scale_ptr,
    history_zero_ptr,
    history_page_table_ptr,
    hp_rows_ptr,
    seq_lens_ptr,
    score_ptr,
    mid_bf16_ptr,
    mid_history_ptr,
    mid_lse_ptr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
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
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_score_b: tl.constexpr,
    stride_score_h: tl.constexpr,
    stride_score_k: tl.constexpr,
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_split: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_split: tl.constexpr,
    num_splits: tl.constexpr,
    num_d_shards: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    num_requests: tl.constexpr,
    block_t: tl.constexpr,
    block_v: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    split_and_shard = tl.program_id(2)
    split = split_and_shard // num_d_shards
    d_shard = split_and_shard % num_d_shards
    head0 = head_group * 4
    request = tl.load(
        query_request_indices_ptr + query_row * stride_query_request,
    )
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    dims = d_shard * block_v + tl.arange(0, block_v)
    dim_mask = dims < latent_rank
    split_len = tl.cdiv(topk, num_splits)
    split_start = split * split_len
    split_end = tl.minimum(split_start + split_len, topk)
    token_offsets = tl.arange(0, block_t)
    m0 = -float("inf")
    m1 = -float("inf")
    m2 = -float("inf")
    m3 = -float("inf")
    l0 = 0.0
    l1 = 0.0
    l2 = 0.0
    l3 = 0.0
    bf16_acc0 = tl.zeros((block_v,), dtype=tl.float32)
    bf16_acc1 = tl.zeros((block_v,), dtype=tl.float32)
    bf16_acc2 = tl.zeros((block_v,), dtype=tl.float32)
    bf16_acc3 = tl.zeros((block_v,), dtype=tl.float32)
    history_acc0 = tl.zeros((block_v,), dtype=tl.float32)
    history_acc1 = tl.zeros((block_v,), dtype=tl.float32)
    history_acc2 = tl.zeros((block_v,), dtype=tl.float32)
    history_acc3 = tl.zeros((block_v,), dtype=tl.float32)

    for tile_start in range(split_start, split_end, block_t):
        selected_offsets = tile_start + token_offsets
        selected_mask = selected_offsets < split_end
        tokens = tl.load(
            selected_tokens_ptr
            + query_row * stride_selected_b
            + selected_offsets * stride_selected_k,
            mask=selected_mask,
            other=-1,
        )
        valid = (
            selected_mask
            & request_valid
            & (query_position >= 0)
            & (tokens >= 0)
            & (tokens < causal_seq_len)
            & (hp_row >= 0)
        )
        is_prefix = valid & (tokens < prefix_tokens)
        is_recent = valid & (tokens >= recent_start)
        is_history = valid & ~is_prefix & ~is_recent

        prefix_base = hp_row * stride_prefix_row + tokens * stride_prefix_token
        prefix_values = tl.load(
            prefix_ptr + prefix_base[:, None] + dims[None, :] * stride_prefix_d,
            mask=is_prefix[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        recent_indices = (tokens - prefix_tokens) % recent_capacity_tokens
        recent_base = hp_row * stride_recent_row + recent_indices * stride_recent_token
        recent_values = tl.load(
            recent_ptr + recent_base[:, None] + dims[None, :] * stride_recent_d,
            mask=is_recent[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        bf16_values = tl.where(is_prefix[:, None], prefix_values, recent_values)

        history_indices = tokens - prefix_tokens
        logical_pages = history_indices // history_block_size
        page_offsets = history_indices % history_block_size
        physical_pages = tl.load(
            history_page_table_ptr
            + safe_request * stride_page_table_b
            + logical_pages * stride_page_table_page,
            mask=is_history,
            other=0,
        )
        byte_offsets = dims // 4
        shifts = (dims % 4) * 2
        data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
        packed = tl.load(
            history_data_ptr
            + data_base[:, None]
            + byte_offsets[None, :] * stride_data_byte,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> shifts[None, :]) & 0x3).to(tl.float32)
        groups = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_pages[:, None] * stride_scale_page
            + page_offsets[:, None] * stride_scale_token
            + groups[None, :] * stride_scale_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_pages[:, None] * stride_zero_page
            + page_offsets[:, None] * stride_zero_token
            + groups[None, :] * stride_zero_group,
            mask=is_history[:, None] & dim_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        history_values = (quantized - zero) * scale

        score_base = query_row * stride_score_b + head0 * stride_score_h
        scores0 = tl.load(
            score_ptr + score_base + selected_offsets * stride_score_k,
            mask=selected_mask,
            other=-float("inf"),
        )
        scores1 = tl.load(
            score_ptr + score_base + stride_score_h + selected_offsets * stride_score_k,
            mask=selected_mask,
            other=-float("inf"),
        )
        scores2 = tl.load(
            score_ptr
            + score_base
            + 2 * stride_score_h
            + selected_offsets * stride_score_k,
            mask=selected_mask,
            other=-float("inf"),
        )
        scores3 = tl.load(
            score_ptr
            + score_base
            + 3 * stride_score_h
            + selected_offsets * stride_score_k,
            mask=selected_mask,
            other=-float("inf"),
        )
        scores0 = tl.where(valid, scores0, -float("inf"))
        scores1 = tl.where(valid, scores1, -float("inf"))
        scores2 = tl.where(valid, scores2, -float("inf"))
        scores3 = tl.where(valid, scores3, -float("inf"))
        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            n0 = tl.maximum(tl.max(scores0, axis=0), m0)
            n1 = tl.maximum(tl.max(scores1, axis=0), m1)
            n2 = tl.maximum(tl.max(scores2, axis=0), m2)
            n3 = tl.maximum(tl.max(scores3, axis=0), m3)
            r0 = tl.exp(m0 - n0)
            r1 = tl.exp(m1 - n1)
            r2 = tl.exp(m2 - n2)
            r3 = tl.exp(m3 - n3)
            p0 = tl.where(valid, tl.exp(scores0 - n0), 0.0)
            p1 = tl.where(valid, tl.exp(scores1 - n1), 0.0)
            p2 = tl.where(valid, tl.exp(scores2 - n2), 0.0)
            p3 = tl.where(valid, tl.exp(scores3 - n3), 0.0)
            bf16_only = tl.where(is_history[:, None], 0.0, bf16_values)
            history_only = tl.where(is_history[:, None], history_values, 0.0)
            p01 = tl.join(p0, p1)
            p23 = tl.join(p2, p3)
            probabilities = tl.join(p01, p23)
            probabilities = tl.permute(probabilities, (2, 1, 0))
            probabilities = tl.reshape(probabilities, (4, block_t))

            bf16_dot = tl.dot(
                probabilities.to(tl.bfloat16),
                bf16_only.to(tl.bfloat16),
            )
            bf16_dot = tl.permute(bf16_dot, (1, 0))
            bf16_dot = tl.reshape(bf16_dot, (block_v, 2, 2))
            bf16_dot_even, bf16_dot_odd = tl.split(bf16_dot)
            bf16_dot0, bf16_dot2 = tl.split(bf16_dot_even)
            bf16_dot1, bf16_dot3 = tl.split(bf16_dot_odd)

            history_dot = tl.dot(
                probabilities.to(tl.bfloat16),
                history_only.to(tl.bfloat16),
            )
            history_dot = tl.permute(history_dot, (1, 0))
            history_dot = tl.reshape(history_dot, (block_v, 2, 2))
            history_dot_even, history_dot_odd = tl.split(history_dot)
            history_dot0, history_dot2 = tl.split(history_dot_even)
            history_dot1, history_dot3 = tl.split(history_dot_odd)

            bf16_acc0 = bf16_acc0 * r0 + bf16_dot0
            bf16_acc1 = bf16_acc1 * r1 + bf16_dot1
            bf16_acc2 = bf16_acc2 * r2 + bf16_dot2
            bf16_acc3 = bf16_acc3 * r3 + bf16_dot3
            history_acc0 = history_acc0 * r0 + history_dot0
            history_acc1 = history_acc1 * r1 + history_dot1
            history_acc2 = history_acc2 * r2 + history_dot2
            history_acc3 = history_acc3 * r3 + history_dot3
            l0 = l0 * r0 + tl.sum(p0, axis=0)
            l1 = l1 * r1 + tl.sum(p1, axis=0)
            l2 = l2 * r2 + tl.sum(p2, axis=0)
            l3 = l3 * r3 + tl.sum(p3, axis=0)
            m0 = n0
            m1 = n1
            m2 = n2
            m3 = n3

    safe_l0 = tl.where(l0 > 0.0, l0, 1.0)
    safe_l1 = tl.where(l1 > 0.0, l1, 1.0)
    safe_l2 = tl.where(l2 > 0.0, l2, 1.0)
    safe_l3 = tl.where(l3 > 0.0, l3, 1.0)
    split_base = query_row * stride_mid_b + split * stride_mid_split
    out0 = split_base + head0 * stride_mid_h
    out1 = out0 + stride_mid_h
    out2 = out0 + 2 * stride_mid_h
    out3 = out0 + 3 * stride_mid_h
    tl.store(
        mid_bf16_ptr + out0 + dims * stride_mid_d,
        bf16_acc0 / safe_l0,
        mask=dim_mask,
    )
    tl.store(
        mid_bf16_ptr + out1 + dims * stride_mid_d,
        bf16_acc1 / safe_l1,
        mask=dim_mask,
    )
    tl.store(
        mid_bf16_ptr + out2 + dims * stride_mid_d,
        bf16_acc2 / safe_l2,
        mask=dim_mask,
    )
    tl.store(
        mid_bf16_ptr + out3 + dims * stride_mid_d,
        bf16_acc3 / safe_l3,
        mask=dim_mask,
    )
    tl.store(
        mid_history_ptr + out0 + dims * stride_mid_d,
        history_acc0 / safe_l0,
        mask=dim_mask,
    )
    tl.store(
        mid_history_ptr + out1 + dims * stride_mid_d,
        history_acc1 / safe_l1,
        mask=dim_mask,
    )
    tl.store(
        mid_history_ptr + out2 + dims * stride_mid_d,
        history_acc2 / safe_l2,
        mask=dim_mask,
    )
    tl.store(
        mid_history_ptr + out3 + dims * stride_mid_d,
        history_acc3 / safe_l3,
        mask=dim_mask,
    )
    tl.store(
        mid_lse_ptr
        + query_row * stride_lse_b
        + head0 * stride_lse_h
        + split * stride_lse_split,
        tl.where(l0 > 0.0, m0 + tl.log(safe_l0), -float("inf")),
        mask=d_shard == 0,
    )
    tl.store(
        mid_lse_ptr
        + query_row * stride_lse_b
        + (head0 + 1) * stride_lse_h
        + split * stride_lse_split,
        tl.where(l1 > 0.0, m1 + tl.log(safe_l1), -float("inf")),
        mask=d_shard == 0,
    )
    tl.store(
        mid_lse_ptr
        + query_row * stride_lse_b
        + (head0 + 2) * stride_lse_h
        + split * stride_lse_split,
        tl.where(l2 > 0.0, m2 + tl.log(safe_l2), -float("inf")),
        mask=d_shard == 0,
    )
    tl.store(
        mid_lse_ptr
        + query_row * stride_lse_b
        + (head0 + 3) * stride_lse_h
        + split * stride_lse_split,
        tl.where(l3 > 0.0, m3 + tl.log(safe_l3), -float("inf")),
        mask=d_shard == 0,
    )


@triton.jit
def _mixed_sparse_prefill_stage1(
    query_ptr,
    query_rotated_ptr,
    query_rope_ptr,
    selected_tokens_ptr,
    query_request_indices_ptr,
    query_positions_ptr,
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
    mid_bf16_ptr,
    mid_history_ptr,
    mid_lse_ptr,
    stride_query_b: tl.constexpr,
    stride_query_h: tl.constexpr,
    stride_query_d: tl.constexpr,
    stride_query_rotated_b: tl.constexpr,
    stride_query_rotated_h: tl.constexpr,
    stride_query_rotated_d: tl.constexpr,
    stride_query_rope_b: tl.constexpr,
    stride_query_rope_h: tl.constexpr,
    stride_query_rope_d: tl.constexpr,
    stride_selected_b: tl.constexpr,
    stride_selected_k: tl.constexpr,
    stride_query_request: tl.constexpr,
    stride_query_position: tl.constexpr,
    stride_prefix_row: tl.constexpr,
    stride_prefix_token: tl.constexpr,
    stride_prefix_d: tl.constexpr,
    stride_recent_row: tl.constexpr,
    stride_recent_token: tl.constexpr,
    stride_recent_d: tl.constexpr,
    stride_rope_block: tl.constexpr,
    stride_rope_token: tl.constexpr,
    stride_rope_d: tl.constexpr,
    stride_rope_block_table_b: tl.constexpr,
    stride_rope_block_table_page: tl.constexpr,
    stride_data_page: tl.constexpr,
    stride_data_token: tl.constexpr,
    stride_data_byte: tl.constexpr,
    stride_scale_page: tl.constexpr,
    stride_scale_token: tl.constexpr,
    stride_scale_group: tl.constexpr,
    stride_zero_page: tl.constexpr,
    stride_zero_token: tl.constexpr,
    stride_zero_group: tl.constexpr,
    stride_page_table_b: tl.constexpr,
    stride_page_table_page: tl.constexpr,
    stride_hp_rows: tl.constexpr,
    stride_seq_lens: tl.constexpr,
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_split: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_split: tl.constexpr,
    topk: tl.constexpr,
    prefix_tokens: tl.constexpr,
    recent_tokens: tl.constexpr,
    recent_capacity_tokens: tl.constexpr,
    rope_block_size: tl.constexpr,
    rope_head_size: tl.constexpr,
    history_block_size: tl.constexpr,
    latent_rank: tl.constexpr,
    group_size: tl.constexpr,
    attention_scale: tl.constexpr,
    num_requests: tl.constexpr,
    num_heads: tl.constexpr,
    block_h: tl.constexpr,
    block_t: tl.constexpr,
    block_d: tl.constexpr,
    block_r: tl.constexpr,
):
    query_row = tl.program_id(0)
    head_group = tl.program_id(1)
    heads = head_group * block_h + tl.arange(0, block_h)
    head_mask = heads < num_heads
    request = tl.load(
        query_request_indices_ptr + query_row * stride_query_request,
    )
    request_valid = (request >= 0) & (request < num_requests)
    safe_request = tl.where(request_valid, request, 0)
    query_position = tl.load(query_positions_ptr + query_row * stride_query_position)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    query = tl.load(
        query_ptr
        + query_row * stride_query_b
        + heads[:, None] * stride_query_h
        + dims[None, :] * stride_query_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    query_rotated = tl.load(
        query_rotated_ptr
        + query_row * stride_query_rotated_b
        + heads[:, None] * stride_query_rotated_h
        + dims[None, :] * stride_query_rotated_d,
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    rope_dims = tl.arange(0, block_r)
    rope_dim_mask = rope_dims < rope_head_size
    query_rope = tl.load(
        query_rope_ptr
        + query_row * stride_query_rope_b
        + heads[:, None] * stride_query_rope_h
        + rope_dims[None, :] * stride_query_rope_d,
        mask=head_mask[:, None] & rope_dim_mask[None, :],
        other=0.0,
    ).to(tl.bfloat16)
    hp_row = tl.load(hp_rows_ptr + safe_request * stride_hp_rows)
    seq_len = tl.load(seq_lens_ptr + safe_request * stride_seq_lens)
    causal_seq_len = tl.minimum(seq_len, query_position + 1)
    effective_topk = tl.minimum(topk, causal_seq_len)
    recent_start = tl.maximum(prefix_tokens, seq_len - recent_tokens)

    token_offsets = tl.arange(0, block_t)
    m_prev = tl.full((block_h,), -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros((block_h,), dtype=tl.float32)
    bf16_acc = tl.zeros((block_h, block_d), dtype=tl.float32)
    history_acc = tl.zeros((block_h, block_d), dtype=tl.float32)

    for tile_start in tl.range(0, effective_topk, block_t):
        selected_offsets = tile_start + token_offsets
        selected_mask = selected_offsets < topk
        tokens = tl.load(
            selected_tokens_ptr
            + query_row * stride_selected_b
            + selected_offsets * stride_selected_k,
            mask=selected_mask,
            other=-1,
        )
        valid = (
            selected_mask
            & request_valid
            & (query_position >= 0)
            & (tokens >= 0)
            & (tokens < causal_seq_len)
            & (hp_row >= 0)
        )
        is_prefix = valid & (tokens < prefix_tokens)
        is_recent = valid & (tokens >= recent_start)
        is_bf16 = is_prefix | is_recent
        is_history = valid & ~is_prefix & ~is_recent
        has_bf16 = tl.sum(is_bf16.to(tl.int32), axis=0) > 0

        prefix_base = hp_row * stride_prefix_row + tokens * stride_prefix_token
        prefix_values = tl.load(
            prefix_ptr + prefix_base[None, :] + dims[:, None] * stride_prefix_d,
            mask=dim_mask[:, None] & is_prefix[None, :],
            other=0.0,
        )
        recent_indices = (tokens - prefix_tokens) % recent_capacity_tokens
        recent_base = hp_row * stride_recent_row + recent_indices * stride_recent_token
        recent_values = tl.load(
            recent_ptr + recent_base[None, :] + dims[:, None] * stride_recent_d,
            mask=dim_mask[:, None] & is_recent[None, :],
            other=0.0,
        )
        bf16_values = tl.where(
            is_prefix[None, :],
            prefix_values,
            recent_values,
        ).to(tl.bfloat16)

        history_indices = tokens - prefix_tokens
        logical_pages = history_indices // history_block_size
        page_offsets = history_indices % history_block_size
        physical_pages = tl.load(
            history_page_table_ptr
            + safe_request * stride_page_table_b
            + logical_pages * stride_page_table_page,
            mask=is_history,
            other=0,
        )
        byte_offsets = dims // 4
        shifts = (dims % 4) * 2
        data_base = physical_pages * stride_data_page + page_offsets * stride_data_token
        packed = tl.load(
            history_data_ptr
            + data_base[None, :]
            + byte_offsets[:, None] * stride_data_byte,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0,
        ).to(tl.int32)
        quantized = ((packed >> shifts[:, None]) & 0x3).to(tl.float32)
        groups = dims // group_size
        scale = tl.load(
            history_scale_ptr
            + physical_pages[None, :] * stride_scale_page
            + page_offsets[None, :] * stride_scale_token
            + groups[:, None] * stride_scale_group,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0.0,
        ).to(tl.float32)
        zero = tl.load(
            history_zero_ptr
            + physical_pages[None, :] * stride_zero_page
            + page_offsets[None, :] * stride_zero_token
            + groups[:, None] * stride_zero_group,
            mask=dim_mask[:, None] & is_history[None, :],
            other=0.0,
        ).to(tl.float32)
        history_values = (quantized - zero) * scale

        rope_logical_pages = tokens // rope_block_size
        rope_page_offsets = tokens % rope_block_size
        rope_physical_pages = tl.load(
            rope_block_table_ptr
            + safe_request * stride_rope_block_table_b
            + rope_logical_pages * stride_rope_block_table_page,
            mask=valid,
            other=0,
        )
        rope_values = tl.load(
            rope_ptr
            + rope_physical_pages[None, :] * stride_rope_block
            + rope_page_offsets[None, :] * stride_rope_token
            + rope_dims[:, None] * stride_rope_d,
            mask=rope_dim_mask[:, None] & valid[None, :],
            other=0.0,
        ).to(tl.bfloat16)

        bf16_scores = tl.zeros((block_h, block_t), dtype=tl.float32)
        if has_bf16:
            bf16_scores = tl.dot(query, bf16_values)
        history_scores = tl.dot(
            query_rotated,
            history_values,
            input_precision="tf32",
        )
        rope_scores = tl.dot(query_rope, rope_values)
        scores = (
            tl.where(
                is_history[None, :],
                history_scores,
                bf16_scores,
            )
            + rope_scores
        )
        scores *= attention_scale
        score_mask = head_mask[:, None] & valid[None, :]
        scores = tl.where(score_mask, scores, -float("inf"))
        if tl.sum(valid.to(tl.int32), axis=0) > 0:
            m_new = tl.maximum(tl.max(scores, axis=1), m_prev)
            previous_scale = tl.exp(m_prev - m_new)
            probabilities = tl.exp(scores - m_new[:, None])
            probabilities = tl.where(score_mask, probabilities, 0.0)
            bf16_contribution = tl.zeros((block_h, block_d), dtype=tl.float32)
            if has_bf16:
                bf16_contribution = tl.dot(
                    probabilities,
                    tl.trans(bf16_values.to(tl.float32)),
                    input_precision="tf32",
                )
            bf16_acc = bf16_acc * previous_scale[:, None] + bf16_contribution
            history_acc = history_acc * previous_scale[:, None] + tl.dot(
                probabilities,
                tl.trans(history_values),
                input_precision="tf32",
            )
            l_prev = l_prev * previous_scale + tl.sum(probabilities, axis=1)
            m_prev = m_new

    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    mid_base = (
        query_row * stride_mid_b
        + heads[:, None] * stride_mid_h
        + dims[None, :] * stride_mid_d
    )
    output_mask = head_mask[:, None] & dim_mask[None, :]
    tl.store(
        mid_bf16_ptr + mid_base,
        bf16_acc / safe_l[:, None],
        mask=output_mask,
    )
    tl.store(
        mid_history_ptr + mid_base,
        history_acc / safe_l[:, None],
        mask=output_mask,
    )
    local_lse = tl.where(
        l_prev > 0.0,
        m_prev + tl.log(safe_l),
        -float("inf"),
    )
    tl.store(
        mid_lse_ptr + query_row * stride_lse_b + heads * stride_lse_h,
        local_lse,
        mask=head_mask,
    )


@triton.jit
def _merge_mixed_splits_kernel(
    mid_bf16_ptr,
    mid_history_ptr,
    mid_lse_ptr,
    bf16_output_ptr,
    history_output_ptr,
    output_lse_ptr,
    stride_mid_b: tl.constexpr,
    stride_mid_h: tl.constexpr,
    stride_mid_split: tl.constexpr,
    stride_mid_d: tl.constexpr,
    stride_lse_b: tl.constexpr,
    stride_lse_h: tl.constexpr,
    stride_lse_split: tl.constexpr,
    stride_output_b: tl.constexpr,
    stride_output_h: tl.constexpr,
    stride_output_d: tl.constexpr,
    stride_output_lse_b: tl.constexpr,
    stride_output_lse_h: tl.constexpr,
    num_splits: tl.constexpr,
    latent_rank: tl.constexpr,
    block_s: tl.constexpr,
    block_d: tl.constexpr,
):
    request = tl.program_id(0)
    head = tl.program_id(1)
    splits = tl.arange(0, block_s)
    split_mask = splits < num_splits
    lse = tl.load(
        mid_lse_ptr
        + request * stride_lse_b
        + head * stride_lse_h
        + splits * stride_lse_split,
        mask=split_mask,
        other=-float("inf"),
    ).to(tl.float32)
    maximum = tl.max(lse, axis=0)
    finite = maximum > -float("inf")
    weights = tl.where(split_mask, tl.exp(lse - maximum), 0.0)
    weights = tl.where(finite, weights, 0.0)
    denominator = tl.sum(weights, axis=0)
    weights = weights / tl.where(denominator > 0.0, denominator, 1.0)

    dims = tl.arange(0, block_d)
    dim_mask = dims < latent_rank
    mid_base = request * stride_mid_b + head * stride_mid_h
    bf16 = tl.load(
        mid_bf16_ptr
        + mid_base
        + splits[:, None] * stride_mid_split
        + dims[None, :] * stride_mid_d,
        mask=split_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    history = tl.load(
        mid_history_ptr
        + mid_base
        + splits[:, None] * stride_mid_split
        + dims[None, :] * stride_mid_d,
        mask=split_mask[:, None] & dim_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    bf16_merged = tl.sum(weights[:, None] * bf16, axis=0)
    history_merged = tl.sum(weights[:, None] * history, axis=0)
    output_base = request * stride_output_b + head * stride_output_h
    tl.store(
        bf16_output_ptr + output_base + dims * stride_output_d,
        bf16_merged,
        mask=dim_mask,
    )
    tl.store(
        history_output_ptr + output_base + dims * stride_output_d,
        history_merged,
        mask=dim_mask,
    )
    global_lse = tl.where(
        denominator > 0.0,
        maximum + tl.log(denominator),
        -float("inf"),
    )
    tl.store(
        output_lse_ptr + request * stride_output_lse_b + head * stride_output_lse_h,
        global_lse,
    )


@triton.jit
def _add_outputs_kernel(
    left_ptr,
    right_ptr,
    output_ptr,
    num_rows,
    latent_rank: tl.constexpr,
    stride_left_row: tl.constexpr,
    stride_left_d: tl.constexpr,
    stride_right_row: tl.constexpr,
    stride_right_d: tl.constexpr,
    stride_output_row: tl.constexpr,
    stride_output_d: tl.constexpr,
    block_d: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    dims = tl.arange(0, block_d)
    mask = dims < latent_rank
    left = tl.load(
        left_ptr + row * stride_left_row + dims * stride_left_d,
        mask=mask,
    ).to(tl.float32)
    right = tl.load(
        right_ptr + row * stride_right_row + dims * stride_right_d,
        mask=mask,
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + dims * stride_output_d,
        left + right,
        mask=mask,
    )


def _validate_attention_inputs(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    selected_tokens: torch.Tensor,
    query_request_indices: torch.Tensor,
    query_positions: torch.Tensor,
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
    rotation: torch.Tensor,
) -> tuple[int, int, int, int]:
    _require_cuda_tensor(
        query,
        name="query",
        ndim=3,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    num_queries, num_heads, latent_rank = query.shape
    if num_queries <= 0 or num_heads <= 0 or latent_rank <= 0:
        raise ValueError("query dimensions must all be positive")
    _require_cuda_tensor(
        query_rope,
        name="query_rope",
        ndim=3,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    if query_rope.shape[:2] != (num_queries, num_heads):
        raise ValueError("query_rope batch and head dimensions must match query")
    if query_rope.shape[2] <= 0:
        raise ValueError("query_rope head size must be positive")
    _require_cuda_tensor(
        selected_tokens,
        name="selected_tokens",
        ndim=2,
        dtype=(torch.int32, torch.int64),
    )
    if selected_tokens.shape[0] != num_queries or selected_tokens.shape[1] <= 0:
        raise ValueError("selected tokens must be non-empty with one row per query")
    for name, tensor in (
        ("query_request_indices", query_request_indices),
        ("query_positions", query_positions),
    ):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_queries:
            raise ValueError(f"{name} must have one entry per query")
    for name, tensor in (("prefix", prefix), ("recent", recent)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=3,
            dtype=torch.bfloat16,
        )
        if tensor.shape[2] != latent_rank:
            raise ValueError(f"{name} latent rank must match query")
    if prefix.shape[0] != recent.shape[0]:
        raise ValueError("prefix/recent row capacities must match")
    if prefix.shape[1] <= 0 or recent.shape[1] <= 0:
        raise ValueError("prefix/recent windows must be positive")
    _require_cuda_tensor(
        rope,
        name="rope",
        ndim=3,
        dtype=torch.bfloat16,
    )
    if rope.shape[0] <= 0 or rope.shape[1] <= 0:
        raise ValueError("RoPE cache must contain at least one non-empty block")
    if rope.shape[2] != query_rope.shape[2]:
        raise ValueError("RoPE cache head size must match query_rope")
    num_groups, group_size, _, history_rank = _validate_history_tensors(
        history_data,
        history_scale,
        history_zero,
    )
    if history_rank != latent_rank:
        raise ValueError("history latent rank must match query")
    _require_cuda_tensor(
        history_page_table,
        name="history_page_table",
        ndim=2,
        dtype=(torch.int32, torch.int64),
    )
    num_requests = history_page_table.shape[0]
    if num_requests <= 0:
        raise ValueError("history page table must contain at least one request")
    _require_cuda_tensor(
        rope_block_table,
        name="rope_block_table",
        ndim=2,
        dtype=(torch.int32, torch.int64),
    )
    if rope_block_table.shape[0] != num_requests:
        raise ValueError("RoPE block table must have one row per request")
    for name, tensor in (("hp_rows", hp_rows), ("seq_lens", seq_lens)):
        _require_cuda_tensor(
            tensor,
            name=name,
            ndim=1,
            dtype=(torch.int32, torch.int64),
        )
        if tensor.shape[0] != num_requests:
            raise ValueError(f"{name} must have one entry per request")
    _require_cuda_tensor(
        rotation,
        name="rotation",
        ndim=2,
        dtype=(torch.bfloat16, torch.float16, torch.float32),
    )
    if rotation.shape != (latent_rank, latent_rank):
        raise ValueError("rotation shape must match query latent rank")
    device = query.device
    tensors = (
        query_rope,
        selected_tokens,
        query_request_indices,
        query_positions,
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
        rotation,
    )
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("all OSCAR MLA attention tensors must share one CUDA device")
    return num_queries, num_heads, latent_rank, group_size


def _prefill_head_block_size(num_heads: int) -> int:
    if num_heads <= 0:
        raise ValueError("num_heads must be positive")
    # The production-proven SM80 kernel groups eight heads. Larger groups can
    # exceed shared memory or fault for PP8/TP2's 32 local heads, so tile them.
    return 8


_grouped_h4_score_workspace_cache: dict[tuple[str, int | None], torch.Tensor] = {}


def prepare_grouped_h4_score_workspace(reference: torch.Tensor) -> torch.Tensor:
    key = (reference.device.type, reference.device.index)
    workspace = _grouped_h4_score_workspace_cache.get(key)
    if workspace is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "OSCAR Grouped-H4 score workspace must be allocated "
                "before CUDA Graph capture"
            )
        workspace = torch.empty(
            (1, 8, 2048),
            dtype=torch.float32,
            device=reference.device,
        )
        _grouped_h4_score_workspace_cache[key] = workspace
    return workspace


def _use_grouped_h4_decode_qkv_split(
    *,
    enabled: bool,
    num_queries: int,
    num_requests: int,
    num_heads: int,
    latent_rank: int,
    rope_head_size: int,
    topk: int,
    num_splits: int,
    prefix_tokens: int,
    recent_tokens: int,
) -> bool:
    return (
        enabled
        and num_queries == 1
        and num_requests == 1
        and num_heads == 8
        and latent_rank == 512
        and rope_head_size == 64
        and topk == 2048
        and num_splits == 16
        and prefix_tokens == 64
        and recent_tokens == 256
    )


def _oscar_mla_sparse_attention(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    selected_tokens: torch.Tensor,
    query_request_indices: torch.Tensor,
    query_positions: torch.Tensor,
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
    rotation: torch.Tensor,
    *,
    inverse_rotation: torch.Tensor | None = None,
    attention_scale: float | None = None,
    num_splits: int = 16,
    mid_bf16: torch.Tensor | None = None,
    mid_history: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    score_workspace: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    recent_tokens: int | None = None,
    group_prefill_heads: bool = False,
    group_decode_h4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend directly to causal DSA-selected tokens across all latent pools."""
    num_queries, num_heads, latent_rank, group_size = _validate_attention_inputs(
        query,
        query_rope,
        selected_tokens,
        query_request_indices,
        query_positions,
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
        rotation,
    )
    if inverse_rotation is None:
        inverse_rotation = rotation.T
    if inverse_rotation.shape != (latent_rank, latent_rank):
        raise ValueError("inverse rotation shape must match query latent rank")
    if inverse_rotation.device != query.device:
        raise ValueError("inverse rotation must share the query CUDA device")
    if num_splits <= 0 or num_splits > 32:
        raise ValueError("num_splits must be in [1, 32]")
    if recent_tokens is None:
        recent_tokens = recent.shape[1]
    if not 0 < recent_tokens <= recent.shape[1]:
        raise ValueError("logical recent window must fit the physical recent pool")
    topk = selected_tokens.shape[1]
    num_splits = min(num_splits, topk)
    attention_scale = (
        (latent_rank + query_rope.shape[2]) ** -0.5
        if attention_scale is None
        else attention_scale
    )
    if not math.isfinite(attention_scale) or attention_scale <= 0:
        raise ValueError("attention_scale must be finite and positive")

    flat_query = query.reshape(num_queries * num_heads, latent_rank)
    query_rotated = oscar_mla_rotate(flat_query, rotation).view(
        num_queries,
        num_heads,
        latent_rank,
    )
    mid_shape = (num_queries, num_heads, num_splits, latent_rank)
    for name, tensor in (
        ("mid_bf16", mid_bf16),
        ("mid_history", mid_history),
    ):
        if tensor is not None and (
            tensor.shape != mid_shape
            or tensor.dtype != torch.float32
            or tensor.device != query.device
        ):
            raise ValueError(f"{name} has incompatible shape, dtype, or device")
    if mid_bf16 is None:
        mid_bf16 = torch.empty(
            mid_shape,
            dtype=torch.float32,
            device=query.device,
        )
    if mid_history is None:
        mid_history = torch.empty_like(mid_bf16)
    if mid_lse is None:
        mid_lse = torch.empty(
            (num_queries, num_heads, num_splits),
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        mid_lse.shape != (num_queries, num_heads, num_splits)
        or mid_lse.dtype != torch.float32
        or mid_lse.device != query.device
    ):
        raise ValueError("mid_lse has incompatible shape, dtype, or device")

    block_d = triton.next_power_of_2(latent_rank)
    block_t = 16
    packed_group_bytes = group_size // 4
    use_grouped_h4_decode = _use_grouped_h4_decode_qkv_split(
        enabled=group_decode_h4,
        num_queries=num_queries,
        num_requests=seq_lens.shape[0],
        num_heads=num_heads,
        latent_rank=latent_rank,
        rope_head_size=query_rope.shape[2],
        topk=topk,
        num_splits=num_splits,
        prefix_tokens=prefix.shape[1],
        recent_tokens=recent_tokens,
    )
    if use_grouped_h4_decode:
        score_shape = (num_queries, num_heads, topk)
        if score_workspace is None:
            score_workspace = prepare_grouped_h4_score_workspace(query)
        if (
            score_workspace.shape != score_shape
            or score_workspace.dtype != torch.float32
            or score_workspace.device != query.device
            or not score_workspace.is_contiguous()
        ):
            raise ValueError("score_workspace has incompatible shape, dtype, or device")

        grouped_grid = (num_queries, num_heads // 4, num_splits)
        _mixed_sparse_decode_grouped_h4_qk[grouped_grid](
            query,
            query_rotated,
            query_rope,
            selected_tokens,
            query_request_indices,
            query_positions,
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
            score_workspace,
            stride_query_b=query.stride(0),
            stride_query_h=query.stride(1),
            stride_query_d=query.stride(2),
            stride_query_rotated_b=query_rotated.stride(0),
            stride_query_rotated_h=query_rotated.stride(1),
            stride_query_rotated_d=query_rotated.stride(2),
            stride_query_rope_b=query_rope.stride(0),
            stride_query_rope_h=query_rope.stride(1),
            stride_query_rope_d=query_rope.stride(2),
            stride_selected_b=selected_tokens.stride(0),
            stride_selected_k=selected_tokens.stride(1),
            stride_query_request=query_request_indices.stride(0),
            stride_query_position=query_positions.stride(0),
            stride_prefix_row=prefix.stride(0),
            stride_prefix_token=prefix.stride(1),
            stride_prefix_d=prefix.stride(2),
            stride_recent_row=recent.stride(0),
            stride_recent_token=recent.stride(1),
            stride_recent_d=recent.stride(2),
            stride_rope_block=rope.stride(0),
            stride_rope_token=rope.stride(1),
            stride_rope_d=rope.stride(2),
            stride_rope_block_table_b=rope_block_table.stride(0),
            stride_rope_block_table_page=rope_block_table.stride(1),
            stride_data_page=history_data.stride(0),
            stride_data_token=history_data.stride(1),
            stride_data_byte=history_data.stride(2),
            stride_scale_page=history_scale.stride(0),
            stride_scale_token=history_scale.stride(1),
            stride_scale_group=history_scale.stride(2),
            stride_zero_page=history_zero.stride(0),
            stride_zero_token=history_zero.stride(1),
            stride_zero_group=history_zero.stride(2),
            stride_page_table_b=history_page_table.stride(0),
            stride_page_table_page=history_page_table.stride(1),
            stride_hp_rows=hp_rows.stride(0),
            stride_seq_lens=seq_lens.stride(0),
            stride_score_b=score_workspace.stride(0),
            stride_score_h=score_workspace.stride(1),
            stride_score_k=score_workspace.stride(2),
            num_splits=num_splits,
            topk=topk,
            prefix_tokens=prefix.shape[1],
            recent_tokens=recent_tokens,
            recent_capacity_tokens=recent.shape[1],
            rope_block_size=rope.shape[1],
            rope_head_size=rope.shape[2],
            history_block_size=history_data.shape[1],
            latent_rank=latent_rank,
            group_size=group_size,
            attention_scale=attention_scale,
            num_requests=seq_lens.shape[0],
            block_t=block_t,
            block_d=block_d,
            block_r=triton.next_power_of_2(rope.shape[2]),
            num_warps=4,
            num_stages=1,
        )
        num_d_shards = latent_rank // 64
        grouped_v_grid = (
            num_queries,
            num_heads // 4,
            num_splits * num_d_shards,
        )
        _mixed_sparse_decode_grouped_h4_v[grouped_v_grid](
            selected_tokens,
            query_request_indices,
            query_positions,
            prefix,
            recent,
            history_data,
            history_scale,
            history_zero,
            history_page_table,
            hp_rows,
            seq_lens,
            score_workspace,
            mid_bf16,
            mid_history,
            mid_lse,
            stride_selected_b=selected_tokens.stride(0),
            stride_selected_k=selected_tokens.stride(1),
            stride_query_request=query_request_indices.stride(0),
            stride_query_position=query_positions.stride(0),
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
            stride_page_table_b=history_page_table.stride(0),
            stride_page_table_page=history_page_table.stride(1),
            stride_hp_rows=hp_rows.stride(0),
            stride_seq_lens=seq_lens.stride(0),
            stride_score_b=score_workspace.stride(0),
            stride_score_h=score_workspace.stride(1),
            stride_score_k=score_workspace.stride(2),
            stride_mid_b=mid_bf16.stride(0),
            stride_mid_h=mid_bf16.stride(1),
            stride_mid_split=mid_bf16.stride(2),
            stride_mid_d=mid_bf16.stride(3),
            stride_lse_b=mid_lse.stride(0),
            stride_lse_h=mid_lse.stride(1),
            stride_lse_split=mid_lse.stride(2),
            num_splits=num_splits,
            num_d_shards=num_d_shards,
            topk=topk,
            prefix_tokens=prefix.shape[1],
            recent_tokens=recent_tokens,
            recent_capacity_tokens=recent.shape[1],
            history_block_size=history_data.shape[1],
            latent_rank=latent_rank,
            group_size=group_size,
            num_requests=seq_lens.shape[0],
            block_t=block_t,
            block_v=64,
            num_warps=4,
            num_stages=1,
        )

    stage1 = _mixed_sparse_decode_stage1
    stage1_grid: tuple[int, ...] = (num_queries, num_heads, num_splits)
    stage1_extra: dict[str, int] = {
        "num_splits": num_splits,
        "packed_group_bytes": packed_group_bytes,
    }
    use_grouped_prefill = group_prefill_heads and num_splits == 1
    if use_grouped_prefill:
        block_h = _prefill_head_block_size(num_heads)
        stage1 = _mixed_sparse_prefill_stage1
        stage1_grid = (num_queries, triton.cdiv(num_heads, block_h))
        stage1_extra = {
            "num_heads": num_heads,
            "block_h": block_h,
        }
    if not use_grouped_h4_decode:
        stage1[stage1_grid](
            query,
            query_rotated,
            query_rope,
            selected_tokens,
            query_request_indices,
            query_positions,
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
            mid_bf16,
            mid_history,
            mid_lse,
            stride_query_b=query.stride(0),
            stride_query_h=query.stride(1),
            stride_query_d=query.stride(2),
            stride_query_rotated_b=query_rotated.stride(0),
            stride_query_rotated_h=query_rotated.stride(1),
            stride_query_rotated_d=query_rotated.stride(2),
            stride_query_rope_b=query_rope.stride(0),
            stride_query_rope_h=query_rope.stride(1),
            stride_query_rope_d=query_rope.stride(2),
            stride_selected_b=selected_tokens.stride(0),
            stride_selected_k=selected_tokens.stride(1),
            stride_query_request=query_request_indices.stride(0),
            stride_query_position=query_positions.stride(0),
            stride_prefix_row=prefix.stride(0),
            stride_prefix_token=prefix.stride(1),
            stride_prefix_d=prefix.stride(2),
            stride_recent_row=recent.stride(0),
            stride_recent_token=recent.stride(1),
            stride_recent_d=recent.stride(2),
            stride_rope_block=rope.stride(0),
            stride_rope_token=rope.stride(1),
            stride_rope_d=rope.stride(2),
            stride_rope_block_table_b=rope_block_table.stride(0),
            stride_rope_block_table_page=rope_block_table.stride(1),
            stride_data_page=history_data.stride(0),
            stride_data_token=history_data.stride(1),
            stride_data_byte=history_data.stride(2),
            stride_scale_page=history_scale.stride(0),
            stride_scale_token=history_scale.stride(1),
            stride_scale_group=history_scale.stride(2),
            stride_zero_page=history_zero.stride(0),
            stride_zero_token=history_zero.stride(1),
            stride_zero_group=history_zero.stride(2),
            stride_page_table_b=history_page_table.stride(0),
            stride_page_table_page=history_page_table.stride(1),
            stride_hp_rows=hp_rows.stride(0),
            stride_seq_lens=seq_lens.stride(0),
            stride_mid_b=mid_bf16.stride(0),
            stride_mid_h=mid_bf16.stride(1),
            stride_mid_split=mid_bf16.stride(2),
            stride_mid_d=mid_bf16.stride(3),
            stride_lse_b=mid_lse.stride(0),
            stride_lse_h=mid_lse.stride(1),
            stride_lse_split=mid_lse.stride(2),
            topk=topk,
            prefix_tokens=prefix.shape[1],
            recent_tokens=recent_tokens,
            recent_capacity_tokens=recent.shape[1],
            rope_block_size=rope.shape[1],
            rope_head_size=rope.shape[2],
            history_block_size=history_data.shape[1],
            latent_rank=latent_rank,
            group_size=group_size,
            attention_scale=attention_scale,
            num_requests=seq_lens.shape[0],
            block_t=block_t,
            block_d=block_d,
            block_r=triton.next_power_of_2(rope.shape[2]),
            **stage1_extra,
            num_warps=8 if use_grouped_prefill else 4,
            num_stages=1,
        )

    merged_shape = (num_queries, num_heads, latent_rank)
    bf16_merged = torch.empty(
        merged_shape,
        dtype=torch.float32,
        device=query.device,
    )
    history_merged = torch.empty_like(bf16_merged)
    if output_lse is None:
        output_lse = torch.empty(
            (num_queries, num_heads),
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        output_lse.shape != (num_queries, num_heads)
        or output_lse.dtype != torch.float32
        or output_lse.device != query.device
    ):
        raise ValueError("output_lse has incompatible shape, dtype, or device")
    _merge_mixed_splits_kernel[(num_queries, num_heads)](
        mid_bf16,
        mid_history,
        mid_lse,
        bf16_merged,
        history_merged,
        output_lse,
        stride_mid_b=mid_bf16.stride(0),
        stride_mid_h=mid_bf16.stride(1),
        stride_mid_split=mid_bf16.stride(2),
        stride_mid_d=mid_bf16.stride(3),
        stride_lse_b=mid_lse.stride(0),
        stride_lse_h=mid_lse.stride(1),
        stride_lse_split=mid_lse.stride(2),
        stride_output_b=bf16_merged.stride(0),
        stride_output_h=bf16_merged.stride(1),
        stride_output_d=bf16_merged.stride(2),
        stride_output_lse_b=output_lse.stride(0),
        stride_output_lse_h=output_lse.stride(1),
        num_splits=num_splits,
        latent_rank=latent_rank,
        block_s=triton.next_power_of_2(num_splits),
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )

    flat_history = history_merged.view(num_queries * num_heads, latent_rank)
    history_original = oscar_mla_rotate(
        flat_history,
        inverse_rotation,
    )
    if output is None:
        output = torch.empty(
            merged_shape,
            dtype=torch.float32,
            device=query.device,
        )
    elif (
        output.shape != merged_shape
        or output.dtype != torch.float32
        or output.device != query.device
    ):
        raise ValueError("output has incompatible shape, dtype, or device")
    flat_bf16 = bf16_merged.view(num_queries * num_heads, latent_rank)
    flat_output = output.view(num_queries * num_heads, latent_rank)
    _add_outputs_kernel[(num_queries * num_heads,)](
        flat_bf16,
        history_original,
        flat_output,
        num_queries * num_heads,
        latent_rank=latent_rank,
        stride_left_row=flat_bf16.stride(0),
        stride_left_d=flat_bf16.stride(1),
        stride_right_row=history_original.stride(0),
        stride_right_d=history_original.stride(1),
        stride_output_row=flat_output.stride(0),
        stride_output_d=flat_output.stride(1),
        block_d=block_d,
        num_warps=4,
        num_stages=1,
    )
    return output, output_lse


def oscar_mla_sparse_decode(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    selected_tokens: torch.Tensor,
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
    rotation: torch.Tensor,
    *,
    inverse_rotation: torch.Tensor | None = None,
    attention_scale: float | None = None,
    num_splits: int = 16,
    mid_bf16: torch.Tensor | None = None,
    mid_history: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    score_workspace: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    recent_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend one decode query per request to DSA-selected cache tokens."""
    if query.shape[0] != seq_lens.shape[0]:
        raise ValueError("decode requires exactly one query per request")
    query_request_indices = torch.arange(
        query.shape[0],
        dtype=torch.int32,
        device=query.device,
    )
    query_positions = seq_lens - 1
    return _oscar_mla_sparse_attention(
        query,
        query_rope,
        selected_tokens,
        query_request_indices,
        query_positions,
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
        rotation,
        inverse_rotation=inverse_rotation,
        attention_scale=attention_scale,
        num_splits=num_splits,
        mid_bf16=mid_bf16,
        mid_history=mid_history,
        mid_lse=mid_lse,
        score_workspace=score_workspace,
        output=output,
        output_lse=output_lse,
        recent_tokens=recent_tokens,
        group_decode_h4=True,
    )


def oscar_mla_sparse_prefill(
    query: torch.Tensor,
    query_rope: torch.Tensor,
    selected_tokens: torch.Tensor,
    query_request_indices: torch.Tensor,
    query_positions: torch.Tensor,
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
    rotation: torch.Tensor,
    *,
    inverse_rotation: torch.Tensor | None = None,
    attention_scale: float | None = None,
    num_splits: int = 16,
    mid_bf16: torch.Tensor | None = None,
    mid_history: torch.Tensor | None = None,
    mid_lse: torch.Tensor | None = None,
    score_workspace: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
    output_lse: torch.Tensor | None = None,
    recent_tokens: int | None = None,
    group_prefill_heads: bool = True,
    group_decode_h4: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend multi-token prefill queries with request mapping and causality."""
    return _oscar_mla_sparse_attention(
        query,
        query_rope,
        selected_tokens,
        query_request_indices,
        query_positions,
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
        rotation,
        inverse_rotation=inverse_rotation,
        attention_scale=attention_scale,
        num_splits=num_splits,
        mid_bf16=mid_bf16,
        mid_history=mid_history,
        mid_lse=mid_lse,
        score_workspace=score_workspace,
        output=output,
        output_lse=output_lse,
        recent_tokens=recent_tokens,
        group_prefill_heads=group_prefill_heads,
        group_decode_h4=group_decode_h4,
    )
