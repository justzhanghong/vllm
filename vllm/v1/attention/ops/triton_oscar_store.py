# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton store kernel for OSCAR INT2 KV cache.

The dense rotation (``K @ R_k``, ``V @ R_v``) is done externally as a GEMM.
This kernel takes rotated K/V and, per (token, head):

  1. applies OSCAR percentile clipping;
  2. computes the per-vector asymmetric INT2 quantizer;
  3. packs four 2-bit indices per byte;
  4. scatters the packed bytes plus a BF16 ``(scale, zero_point)`` pair into the
     combined KV cache slot ``[key_packed | value_packed]``.

Single quantization group per vector (``group_size >= head_dim``); this is
the ``head_dim <= 128`` regime that the OSCAR presets target.
"""

import torch

from vllm.triton_utils import tl, triton


def _int2_byte_index_and_shift(dim: int, head_dim: int) -> tuple[int, int]:
    """Return the physical byte and bit shift for one logical INT2 element."""
    if head_dim == 128:
        quarter = head_dim // 4
        return dim % quarter, (dim // quarter) * 2
    return dim // 4, (dim % 4) * 2


@triton.jit
def _quantize_pack_int2_vec(
    vec,
    KV_cache_ptr,  # flattened uint8 cache
    KV_meta_ptr,  # BF16 view of the same cache storage
    region_base,  # byte offset of this region within the slot (0 or key_packed)
    slot_base,  # byte offset of this slot+head in the cache
    d_offs,  # tl.arange(0, BLOCK_D)
    d_mask,  # d_offs < D
    D: tl.constexpr,
    LEVELS: tl.constexpr,  # 2 ** quant_bits (== 4 for INT2)
    DATA_BYTES: tl.constexpr,  # ceil(D * bits / 8) == D // 4 for INT2
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,  # next_pow2(DATA_BYTES)
    CLIP_INDEX: tl.constexpr,
):
    """Clip, asymmetric INT2 quantize, pack, and store one vector."""
    if CLIP_INDEX >= 0:
        sorted_abs = tl.sort(tl.abs(vec))
        threshold = tl.sum(tl.where(d_offs == CLIP_INDEX, sorted_abs, 0.0), axis=0)
        vec = tl.minimum(tl.maximum(vec, -threshold), threshold)

    vmin = tl.min(tl.where(d_mask, vec, float("inf")), axis=0)
    vmax = tl.max(tl.where(d_mask, vec, -float("inf")), axis=0)
    scale = tl.maximum(vmax - vmin, 1e-8) / (LEVELS - 1)

    zero = -vmin / scale

    # Match OSCAR SGLang: q = clamp(round(x / scale + zero_point), 0, levels-1).
    q = tl.minimum(tl.maximum((vec / scale + zero + 0.5).to(tl.int32), 0), LEVELS - 1)

    shifts = tl.arange(0, 4) * 2
    if D == 128:
        q_grp = tl.reshape(q, [4, BLOCK_D // 4])
        packed = tl.sum((q_grp & 0x3) << shifts[:, None], axis=0).to(tl.uint8)
    else:
        q_grp = tl.reshape(q, [BLOCK_D // 4, 4])
        packed = tl.sum((q_grp & 0x3) << shifts[None, :], axis=1).to(tl.uint8)
    pack_offs = tl.arange(0, BLOCK_PACK)
    pack_mask = pack_offs < DATA_BYTES
    tl.store(
        KV_cache_ptr + slot_base + region_base + pack_offs,
        packed,
        mask=pack_mask,
    )

    # Store BF16 scale and zero point right after the packed data.
    meta = region_base + DATA_BYTES
    meta_offset = (slot_base + meta) // 2
    tl.store(KV_meta_ptr + meta_offset, scale.to(tl.bfloat16))
    tl.store(KV_meta_ptr + meta_offset + 1, zero.to(tl.bfloat16))


@triton.jit
def _store_int2_vec(
    Src_ptr,  # [N, H, D] — rotated K or V
    KV_cache_ptr,  # flattened uint8 cache
    KV_meta_ptr,  # BF16 view of the same cache storage
    base,  # token/head offset into Src_ptr
    stride_dim: tl.constexpr,
    region_base,  # byte offset of this region within the slot (0 or key_packed)
    slot_base,  # byte offset of this slot+head in the cache
    d_offs,
    d_mask,
    D: tl.constexpr,
    LEVELS: tl.constexpr,
    DATA_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_PACK: tl.constexpr,
    CLIP_INDEX: tl.constexpr,
):
    vec = tl.load(Src_ptr + base + d_offs * stride_dim, mask=d_mask, other=0.0).to(
        tl.float32
    )
    _quantize_pack_int2_vec(
        vec,
        KV_cache_ptr,
        KV_meta_ptr,
        region_base,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        LEVELS=LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=CLIP_INDEX,
    )


@triton.jit
def _oscar_store_kernel(
    Key_ptr,  # [N, H, D] — rotated+clipped keys
    Value_ptr,  # [N, H, D] — rotated+clipped values
    KV_cache_ptr,  # flattened uint8
    KV_meta_ptr,  # BF16 view of the same cache storage
    Slot_mapping_ptr,  # [N] int
    Token_req_ptr,
    Query_start_ptr,
    Seq_lens_ptr,
    stride_key_token: tl.constexpr,
    stride_key_head: tl.constexpr,
    stride_key_dim: tl.constexpr,
    stride_value_token: tl.constexpr,
    stride_value_head: tl.constexpr,
    stride_value_dim: tl.constexpr,
    stride_cache_block: tl.constexpr,
    stride_cache_pos: tl.constexpr,
    stride_cache_head: tl.constexpr,
    D: tl.constexpr,
    H: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    KEY_PACKED: tl.constexpr,  # bytes of the key region (incl. its meta)
    KEY_LEVELS: tl.constexpr,
    VALUE_LEVELS: tl.constexpr,
    K_CLIP_INDEX: tl.constexpr,
    V_CLIP_INDEX: tl.constexpr,
    DATA_BYTES: tl.constexpr,  # == D // 4 for INT2 (shared by K and V)
    BLOCK_PACK: tl.constexpr,
    MIXED_LAYOUT: tl.constexpr,
    PREFIX_TOKENS: tl.constexpr,
    RECENT_TOKENS: tl.constexpr,
):
    pid = tl.program_id(0)
    token_idx = pid // H
    head_idx = pid % H

    if MIXED_LAYOUT:
        req_idx = tl.load(Token_req_ptr + token_idx)
        query_start = tl.load(Query_start_ptr + req_idx)
        query_end = tl.load(Query_start_ptr + req_idx + 1)
        seq_len = tl.load(Seq_lens_ptr + req_idx)
        pos = seq_len - (query_end - query_start) + token_idx - query_start
        recent_start = tl.maximum(PREFIX_TOKENS, seq_len - RECENT_TOKENS)
        if pos < PREFIX_TOKENS or pos >= recent_start:
            return

    slot = tl.load(Slot_mapping_ptr + token_idx)
    if slot < 0:
        return
    blk = (slot // BLOCK_SIZE).to(tl.int64)
    off = (slot % BLOCK_SIZE).to(tl.int64)
    slot_base = (
        blk * stride_cache_block
        + off * stride_cache_pos
        + tl.cast(head_idx, tl.int64) * stride_cache_head
    )

    key_base = (
        token_idx.to(tl.int64) * stride_key_token
        + head_idx.to(tl.int64) * stride_key_head
    )
    value_base = (
        token_idx.to(tl.int64) * stride_value_token
        + head_idx.to(tl.int64) * stride_value_head
    )
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < D

    # Key region at offset 0, value region at offset KEY_PACKED.
    _store_int2_vec(
        Key_ptr,
        KV_cache_ptr,
        KV_meta_ptr,
        key_base,
        stride_key_dim,
        0,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        LEVELS=KEY_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=K_CLIP_INDEX,
    )
    _store_int2_vec(
        Value_ptr,
        KV_cache_ptr,
        KV_meta_ptr,
        value_base,
        stride_value_dim,
        KEY_PACKED,
        slot_base,
        d_offs,
        d_mask,
        D=D,
        LEVELS=VALUE_LEVELS,
        DATA_BYTES=DATA_BYTES,
        BLOCK_D=BLOCK_D,
        BLOCK_PACK=BLOCK_PACK,
        CLIP_INDEX=V_CLIP_INDEX,
    )


def oscar_store(
    key_rot: torch.Tensor,  # [N, H, D] fp32/fp16 — rotated (+clipped) keys
    value_rot: torch.Tensor,  # [N, H, D] — rotated (+clipped) values
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, slot_size] uint8
    slot_mapping: torch.Tensor,  # [N]
    key_levels: int,
    value_levels: int,
    key_packed_size: int,
    data_bytes: int,
    k_clip_ratio: float = 0.0,
    v_clip_ratio: float = 0.0,
    token_to_req_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    seq_lens: torch.Tensor | None = None,
    prefix_tokens: int = 0,
    recent_tokens: int = 0,
) -> None:
    """Quantize rotated K/V to INT2 and scatter into the combined cache."""
    N, H, D = key_rot.shape
    if N == 0:
        return
    NH = N * H
    block_size = kv_cache.shape[1]
    BLOCK_D = triton.next_power_of_2(D)
    BLOCK_PACK = triton.next_power_of_2(data_bytes)
    mixed_layout = token_to_req_indices is not None
    if mixed_layout and (query_start_loc is None or seq_lens is None):
        raise ValueError("OSCAR mixed store requires complete request metadata")
    if token_to_req_indices is None:
        token_to_req_indices = slot_mapping
    if query_start_loc is None:
        query_start_loc = slot_mapping
    if seq_lens is None:
        seq_lens = slot_mapping

    def clip_index(ratio: float) -> int:
        if ratio <= 0.0:
            return -1
        return min(int(ratio * D), D - 1)

    grid = (NH,)
    _oscar_store_kernel[grid](
        key_rot,
        value_rot,
        kv_cache,
        kv_cache.view(torch.bfloat16),
        slot_mapping,
        token_to_req_indices,
        query_start_loc,
        seq_lens,
        stride_key_token=key_rot.stride(0),
        stride_key_head=key_rot.stride(1),
        stride_key_dim=key_rot.stride(2),
        stride_value_token=value_rot.stride(0),
        stride_value_head=value_rot.stride(1),
        stride_value_dim=value_rot.stride(2),
        stride_cache_block=kv_cache.stride(0),
        stride_cache_pos=kv_cache.stride(1),
        stride_cache_head=kv_cache.stride(2),
        D=D,
        H=H,
        BLOCK_SIZE=block_size,
        BLOCK_D=BLOCK_D,
        KEY_PACKED=key_packed_size,
        KEY_LEVELS=key_levels,
        VALUE_LEVELS=value_levels,
        K_CLIP_INDEX=clip_index(k_clip_ratio),
        V_CLIP_INDEX=clip_index(v_clip_ratio),
        DATA_BYTES=data_bytes,
        BLOCK_PACK=BLOCK_PACK,
        MIXED_LAYOUT=mixed_layout,
        PREFIX_TOKENS=prefix_tokens,
        RECENT_TOKENS=recent_tokens,
        num_warps=4,
        num_stages=1,
    )
