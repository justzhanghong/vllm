# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for OSCAR INT2 KV-cache quantization.

Validates:
  * config slot geometry,
  * the store + dequant round-trip (INT2 pack/unpack consistency vs a pure
    PyTorch reference quantizer),
  * the fused INT2 decode-attention kernel vs a reference attention computed
    on the dequantized cache.

Run: ``pytest tests/quantization/test_oscar.py``.
"""

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="OSCAR kernels require CUDA/Triton."
)


def _ref_int2_quant_dequant(x: torch.Tensor, levels: int) -> torch.Tensor:
    """Per-vector asymmetric uniform quant-dequant, matching the kernel."""
    vmin = x.amin(dim=-1, keepdim=True)
    vmax = x.amax(dim=-1, keepdim=True)
    scale = (vmax - vmin).clamp_min(1e-8) / (levels - 1)
    zero = -vmin / scale
    q = torch.clamp((x / scale + zero + 0.5).to(torch.int32), 0, levels - 1)
    stored_scale = scale.to(torch.bfloat16).float()
    stored_zero = zero.to(torch.bfloat16).float()
    return (q.float() - stored_zero) * stored_scale


def test_config_geometry():
    c = OscarConfig.from_cache_dtype("oscar_int2", 128)
    assert c.key_levels == 4 and c.value_levels == 4
    assert c.key_data_bytes == 32 and c.value_data_bytes == 32
    assert c.meta_bytes == 4
    assert c.key_packed_size == 36 and c.value_packed_size == 36
    assert c.slot_size_aligned == 72


def test_store_metadata_is_bf16():
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", 128)
    assert cfg.meta_bytes == 4
    cache = _make_cache(1, 16, 1, cfg.slot_size_aligned, device)
    key = torch.linspace(-1.0, 1.0, 128, device=device).view(1, 1, 128)
    value = torch.linspace(-2.0, 2.0, 128, device=device).view(1, 1, 128)
    oscar_store(
        key,
        value,
        cache,
        torch.zeros(1, dtype=torch.int32, device=device),
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
    )

    cache_bf16 = cache.view(torch.bfloat16)
    key_meta = cfg.key_data_bytes // 2
    value_meta = (cfg.key_packed_size + cfg.value_data_bytes) // 2
    expected_key_scale = ((key.max() - key.min()) / 3).to(torch.bfloat16)
    expected_key_zero = (-key.min() / ((key.max() - key.min()) / 3)).to(torch.bfloat16)
    expected_value_scale = ((value.max() - value.min()) / 3).to(torch.bfloat16)
    expected_value_zero = (-value.min() / ((value.max() - value.min()) / 3)).to(
        torch.bfloat16
    )
    torch.testing.assert_close(cache_bf16[0, 0, 0, key_meta], expected_key_scale)
    torch.testing.assert_close(cache_bf16[0, 0, 0, key_meta + 1], expected_key_zero)
    torch.testing.assert_close(cache_bf16[0, 0, 0, value_meta], expected_value_scale)
    torch.testing.assert_close(cache_bf16[0, 0, 0, value_meta + 1], expected_value_zero)


@pytest.mark.parametrize("head_dim", [64, 128])
def test_store_int2_physical_layout(head_dim):
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    codes = torch.arange(head_dim, device=device, dtype=torch.float32) % 4
    key = codes.view(1, 1, head_dim)
    cache = _make_cache(1, 16, 1, cfg.slot_size_aligned, device)
    oscar_store(
        key,
        key,
        cache,
        torch.zeros(1, dtype=torch.int32, device=device),
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
    )

    scale = ((key.max() - key.min()) / 3).to(torch.bfloat16).float()
    zero = (-key.min() / scale).to(torch.bfloat16).float()
    quant = torch.clamp((key / scale + zero + 0.5).int(), 0, 3).view(-1)
    if head_dim == 128:
        expected = (
            quant[:32]
            | (quant[32:64] << 2)
            | (quant[64:96] << 4)
            | (quant[96:128] << 6)
        ).to(torch.uint8)
    else:
        expected = (
            (quant.view(-1, 4) * torch.tensor([1, 4, 16, 64], device=device))
            .sum(1)
            .to(torch.uint8)
        )
    torch.testing.assert_close(cache[0, 0, 0, : cfg.key_data_bytes], expected)


def test_mixed_kernels_reject_wrong_prefix_page_width():
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_decode_attention
    from vllm.v1.attention.ops.triton_oscar_mixed_store import oscar_store_hp

    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", 128)
    cache, prefix_cache, recent_cache = _make_mixed_cache(1, 16, 1, cfg, device)
    bad_prefix_pages = torch.arange(3, dtype=torch.int32, device=device).view(1, 3)
    hp_rows = torch.zeros(1, dtype=torch.int32, device=device)
    seq_lens = torch.ones(1, dtype=torch.int32, device=device)

    with pytest.raises(ValueError, match="width must cover"):
        oscar_store_hp(
            torch.zeros(1, 1, 128, dtype=torch.float32, device=device),
            torch.zeros(1, 1, 128, dtype=torch.float32, device=device),
            prefix_cache,
            recent_cache,
            token_to_req_indices=torch.zeros(1, dtype=torch.int32, device=device),
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
            seq_lens=seq_lens,
            hp_row_ids=hp_rows,
            prefix_page_ids=bad_prefix_pages,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )

    with pytest.raises(ValueError, match="width must cover"):
        oscar_decode_attention(
            torch.zeros(1, 1, 128, dtype=torch.float32, device=device),
            cache,
            torch.zeros(1, 1, dtype=torch.int32, device=device),
            seq_lens,
            128**-0.5,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_data_bytes=cfg.key_data_bytes,
            key_packed_size=cfg.key_packed_size,
            value_data_bytes=cfg.value_data_bytes,
            prefix_cache=prefix_cache,
            recent_cache=recent_cache,
            hp_row_ids=hp_rows,
            prefix_page_ids=bad_prefix_pages,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )


def _make_cache(num_blocks, block_size, num_kv_heads, slot_size, device):
    return torch.zeros(
        num_blocks,
        block_size,
        num_kv_heads,
        slot_size,
        dtype=torch.uint8,
        device=device,
    )


def _make_mixed_cache(num_blocks, block_size, num_kv_heads, cfg, device, num_hp_rows=1):
    quant = _make_cache(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg.slot_size_aligned,
        device,
    )
    hp_tail = (num_kv_heads, 2, cfg.head_dim)
    prefix = torch.zeros(
        num_hp_rows * cfg.prefix_tokens,
        *hp_tail,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.zeros(
        num_hp_rows * cfg.recent_tokens,
        *hp_tail,
        dtype=torch.bfloat16,
        device=device,
    )
    return quant, prefix, recent


@pytest.mark.parametrize("head_dim", [64, 128])
def test_store_dequant_roundtrip(head_dim):
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_full_dequant_kv
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(0)
    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    N, Hk = 40, 4
    block_size, num_blocks = 16, 8

    key = torch.randn(N, Hk, head_dim + 3, device=device)[..., :head_dim]
    value = torch.randn(N, Hk, head_dim + 5, device=device)[..., :head_dim]
    assert not key.is_contiguous() and not value.is_contiguous()
    assert key.stride() != value.stride()
    cache = _make_cache(num_blocks, block_size, Hk, cfg.slot_size_aligned, device)
    # Contiguous slots 0..N-1.
    slot_mapping = torch.arange(N, device=device, dtype=torch.int32)

    oscar_store(
        key,
        value,
        cache,
        slot_mapping,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
    )

    # Single contiguous block table covering all tokens.
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(1, -1)
    k_deq, v_deq = oscar_full_dequant_kv(
        cache,
        block_table,
        N,
        Hk,
        head_dim,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )

    k_ref = _ref_int2_quant_dequant(key, cfg.key_levels)
    v_ref = _ref_int2_quant_dequant(value, cfg.value_levels)

    # fp16 storage of the dequantized result -> compare in fp16 tolerance.
    torch.testing.assert_close(k_deq.float(), k_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(v_deq.float(), v_ref, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("head_dim,Hq,Hk", [(128, 8, 2), (64, 4, 4)])
def test_decode_attention_matches_reference(head_dim, Hq, Hk):
    from vllm.v1.attention.ops.triton_oscar_decode import (
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(1)
    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", head_dim)
    B = 2
    seq_len = 48
    block_size, num_blocks = 16, B * 8
    scale = head_dim**-0.5

    # Per-batch contiguous storage.
    key = torch.randn(B, seq_len, Hk, head_dim, device=device)
    value = torch.randn(B, seq_len, Hk, head_dim, device=device)
    cache = _make_cache(num_blocks, block_size, Hk, cfg.slot_size_aligned, device)

    blocks_per_req = num_blocks // B
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).view(
        B, blocks_per_req
    )

    # Store all tokens for both requests.
    for b in range(B):
        base = b * blocks_per_req * block_size
        slot = torch.arange(base, base + seq_len, device=device, dtype=torch.int32)
        oscar_store(
            key[b],
            value[b],
            cache,
            slot,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
        )

    q = torch.randn(B, Hq, head_dim, device=device)
    v_rotation_t = torch.linalg.qr(
        torch.randn(head_dim, head_dim, device=device, dtype=torch.float32)
    ).Q
    seq_lens = torch.full((B,), seq_len, device=device, dtype=torch.int32)

    out = oscar_decode_attention(
        q,
        cache,
        block_table,
        seq_lens,
        scale,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
        v_rotation_t=v_rotation_t,
    )
    if head_dim == 128:
        legacy_out = oscar_decode_attention(
            q,
            cache,
            block_table,
            seq_lens,
            scale,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_data_bytes=cfg.key_data_bytes,
            key_packed_size=cfg.key_packed_size,
            value_data_bytes=cfg.value_data_bytes,
            max_num_kv_splits=4,
            v_rotation_t=v_rotation_t,
            use_grouped_h4=False,
        )
        torch.testing.assert_close(out, legacy_out, atol=6e-3, rtol=6e-3)

    # Reference: attention over the *dequantized* cache (what the kernel reads).
    ref = torch.empty(B, Hq, head_dim, device=device)
    g = Hq // Hk
    for b in range(B):
        k_deq, v_deq = oscar_full_dequant_kv(
            cache,
            block_table[b : b + 1],
            seq_len,
            Hk,
            head_dim,
            cfg.key_levels,
            cfg.value_levels,
            cfg.key_data_bytes,
            cfg.key_packed_size,
            cfg.value_data_bytes,
        )
        for h in range(Hq):
            kh = k_deq[:, h // g, :].float()
            vh = v_deq[:, h // g, :].float()
            s = (q[b, h].float() @ kh.t()) * scale
            p = torch.softmax(s, dim=-1)
            ref[b, h] = (p @ vh) @ v_rotation_t

    torch.testing.assert_close(out.float(), ref, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize("mixed_kv", [False, True])
def test_preindexed_grouped_h4_decode_matches_legacy_and_reference(mixed_kv):
    from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
    from vllm.v1.attention.ops.triton_oscar_decode import (
        materialize_oscar_slot_ids,
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_mixed_store import oscar_store_hp
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(17)
    device = "cuda"
    cfg = OscarConfig.from_cache_dtype("oscar_int2", 128)
    Hq, Hk, D = 32, 8, 128
    block_size = 16
    seq_len = 384 if mixed_kv else 48
    num_blocks = (seq_len + block_size - 1) // block_size
    cache = _make_cache(
        num_blocks,
        block_size,
        Hk,
        cfg.slot_size_aligned,
        device,
    )
    block_table = torch.roll(
        torch.arange(num_blocks, device=device, dtype=torch.int32), shifts=1
    )[None, :]
    assert not torch.equal(
        block_table,
        torch.arange(num_blocks, device=device, dtype=torch.int32)[None, :],
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.int32)
    slots = (
        block_table[0, positions // block_size] * block_size
        + positions % block_size
    )
    key = torch.randn(seq_len, Hk, D, device=device)
    value = torch.randn(seq_len, Hk, D, device=device)

    prefix_cache = recent_cache = hp_row_ids = None
    if mixed_kv:
        _, prefix_cache, recent_cache = _make_mixed_cache(
            num_blocks, block_size, Hk, cfg, device
        )
        part = partition_tokens(
            seq_len,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )
        oscar_store(
            key[part.history.start : part.history.stop],
            value[part.history.start : part.history.stop],
            cache,
            slots[part.history.start : part.history.stop],
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
            k_clip_ratio=cfg.k_clip_ratio,
            v_clip_ratio=cfg.v_clip_ratio,
        )
        hp_row_ids = torch.zeros(1, dtype=torch.int32, device=device)
        oscar_store_hp(
            key,
            value,
            prefix_cache,
            recent_cache,
            token_to_req_indices=torch.zeros(
                seq_len, dtype=torch.int32, device=device
            ),
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            hp_row_ids=hp_row_ids,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )
    else:
        oscar_store(
            key,
            value,
            cache,
            slots,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
        )

    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    physical_slot_ids = torch.full(
        (1, num_blocks * block_size),
        -1,
        dtype=torch.int64,
        device=device,
    )
    materialize_oscar_slot_ids(
        block_table,
        seq_lens,
        physical_slot_ids,
        block_size,
    )
    torch.testing.assert_close(
        physical_slot_ids[0, :seq_len], slots.to(torch.int64), atol=0, rtol=0
    )

    query = torch.randn(1, Hq, D, device=device)
    decode_kwargs = {
        "key_levels": cfg.key_levels,
        "value_levels": cfg.value_levels,
        "key_data_bytes": cfg.key_data_bytes,
        "key_packed_size": cfg.key_packed_size,
        "value_data_bytes": cfg.value_data_bytes,
        "max_num_kv_splits": 4,
        "prefix_cache": prefix_cache,
        "recent_cache": recent_cache,
        "hp_row_ids": hp_row_ids,
        "prefix_tokens": cfg.prefix_tokens if mixed_kv else None,
        "recent_tokens": cfg.recent_tokens if mixed_kv else None,
    }
    fast = oscar_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        D**-0.5,
        physical_slot_ids=physical_slot_ids,
        use_grouped_h4=True,
        **decode_kwargs,
    )
    legacy = oscar_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        D**-0.5,
        use_grouped_h4=False,
        **decode_kwargs,
    )
    torch.testing.assert_close(fast, legacy, atol=6e-3, rtol=6e-3)

    key_ref, value_ref = oscar_full_dequant_kv(
        cache,
        block_table,
        seq_len,
        Hk,
        D,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )
    if mixed_kv:
        for token_range in (part.prefix, part.recent):
            start, stop = token_range.start, token_range.stop
            key_ref[start:stop] = key[start:stop].bfloat16().to(key_ref.dtype)
            value_ref[start:stop] = value[start:stop].bfloat16().to(
                value_ref.dtype
            )
    reference = torch.empty_like(fast)
    for head_idx in range(Hq):
        kv_head = head_idx // (Hq // Hk)
        scores = (query[0, head_idx].float() @ key_ref[:, kv_head].float().T) * (
            D**-0.5
        )
        reference[0, head_idx] = torch.softmax(scores, dim=-1) @ value_ref[
            :, kv_head
        ].float()
    torch.testing.assert_close(
        fast.float(), reference.float(), atol=6e-3, rtol=6e-3
    )

    auto_missing_slots = oscar_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        D**-0.5,
        **decode_kwargs,
    )
    torch.testing.assert_close(auto_missing_slots, legacy, atol=0, rtol=0)
    with pytest.raises(ValueError, match="preindexed slot IDs"):
        oscar_decode_attention(
            query,
            cache,
            block_table,
            seq_lens,
            D**-0.5,
            use_grouped_h4=True,
            **decode_kwargs,
        )

    padded_cache = torch.empty(
        num_blocks,
        block_size * 2,
        Hk,
        cfg.slot_size_aligned,
        dtype=torch.uint8,
        device=device,
    )
    nonlinear_cache = padded_cache[:, ::2]
    nonlinear_cache.copy_(cache)
    nonlinear_auto = oscar_decode_attention(
        query,
        nonlinear_cache,
        block_table,
        seq_lens,
        D**-0.5,
        physical_slot_ids=physical_slot_ids,
        **decode_kwargs,
    )
    nonlinear_legacy = oscar_decode_attention(
        query,
        nonlinear_cache,
        block_table,
        seq_lens,
        D**-0.5,
        use_grouped_h4=False,
        **decode_kwargs,
    )
    torch.testing.assert_close(nonlinear_auto, nonlinear_legacy, atol=0, rtol=0)
    with pytest.raises(ValueError, match="linear paged cache"):
        oscar_decode_attention(
            query,
            nonlinear_cache,
            block_table,
            seq_lens,
            D**-0.5,
            physical_slot_ids=physical_slot_ids,
            use_grouped_h4=True,
            **decode_kwargs,
        )


def test_mixed_store_demote_and_decode_matches_reference():
    from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
    from vllm.v1.attention.ops.triton_oscar_decode import (
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_mixed_store import (
        oscar_demote_hp,
        oscar_store_hp,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(2)
    device = "cuda"
    cfg = OscarConfig()
    Hq, Hk, D = 8, 2, 128
    block_size, num_blocks = 16, 32
    seq_len = 384
    cache, prefix_cache, recent_cache = _make_mixed_cache(
        num_blocks, block_size, Hk, cfg, device
    )
    # Use a non-identity logical-to-physical mapping for INT2 slots. The
    # single-request BF16 arena intentionally stays in fixed physical pages.
    block_table = torch.randperm(num_blocks, device=device, dtype=torch.int64).to(
        torch.int32
    )[None, :]
    positions = torch.arange(seq_len + 1, device=device, dtype=torch.int32)
    slots = (
        block_table[0, positions // block_size] * block_size + positions % block_size
    )
    key = torch.randn(seq_len + 1, Hk, D + 3, device=device)[..., :D]
    value = torch.randn(seq_len + 1, Hk, D + 5, device=device)[..., :D]
    assert not key.is_contiguous() and not value.is_contiguous()
    assert key.stride() != value.stride()

    part = partition_tokens(
        seq_len,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    oscar_store(
        key[part.history.start : part.history.stop],
        value[part.history.start : part.history.stop],
        cache,
        slots[part.history.start : part.history.stop],
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
    )
    oscar_store_hp(
        key[:seq_len],
        value[:seq_len],
        prefix_cache,
        recent_cache,
        token_to_req_indices=torch.zeros(seq_len, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        hp_row_ids=torch.tensor([0], dtype=torch.int32, device=device),
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    # Decode one token: position 128 leaves recent and is demoted to INT2.
    final_len = seq_len + 1
    seq_lens = torch.tensor([final_len], dtype=torch.int32, device=device)
    hp_row_ids = torch.tensor([0], dtype=torch.int32, device=device)
    oscar_demote_hp(
        recent_cache,
        cache,
        block_table,
        seq_lens,
        hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
    )
    oscar_store_hp(
        key[seq_len : seq_len + 1],
        value[seq_len : seq_len + 1],
        prefix_cache,
        recent_cache,
        token_to_req_indices=torch.zeros(1, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seq_lens=seq_lens,
        hp_row_ids=hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    q = torch.randn(1, Hq, D, device=device)
    out = oscar_decode_attention(
        q,
        cache,
        block_table,
        seq_lens,
        D**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
        prefix_cache=prefix_cache,
        recent_cache=recent_cache,
        hp_row_ids=hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    legacy_out = oscar_decode_attention(
        q,
        cache,
        block_table,
        seq_lens,
        D**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
        prefix_cache=prefix_cache,
        recent_cache=recent_cache,
        hp_row_ids=hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        use_grouped_h4=False,
    )
    torch.testing.assert_close(out, legacy_out, atol=6e-3, rtol=6e-3)

    k_ref, v_ref = oscar_full_dequant_kv(
        cache,
        block_table,
        final_len,
        Hk,
        D,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )
    final_part = partition_tokens(
        final_len,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    for positions in (final_part.prefix, final_part.recent):
        start, stop = positions.start, positions.stop
        k_ref[start:stop] = key[start:stop].bfloat16().to(k_ref.dtype)
        v_ref[start:stop] = value[start:stop].bfloat16().to(v_ref.dtype)

    ref = torch.empty_like(out)
    for h in range(Hq):
        kh = k_ref[:, h // (Hq // Hk)].float()
        vh = v_ref[:, h // (Hq // Hk)].float()
        weights = torch.softmax((q[0, h].float() @ kh.T) * D**-0.5, dim=-1)
        ref[0, h] = weights @ vh
    torch.testing.assert_close(out.float(), ref.float(), atol=6e-3, rtol=6e-3)


def test_batched_mixed_cache_uses_scheduler_bf16_ownership():
    from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
    from vllm.v1.attention.ops.triton_oscar_decode import (
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_mixed_store import (
        oscar_demote_hp,
        oscar_store_hp,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(3)
    device = "cuda"
    cfg = OscarConfig()
    batch_size, num_query_heads, num_kv_heads, head_dim = 2, 8, 2, 128
    block_size, seq_len = 16, 384
    blocks_per_req = (seq_len + 1 + block_size - 1) // block_size
    num_blocks = batch_size * blocks_per_req
    cache, prefix_cache, recent_cache = _make_mixed_cache(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg,
        device,
        num_hp_rows=batch_size,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).view(
        batch_size, blocks_per_req
    )
    positions = torch.arange(seq_len + 1, dtype=torch.int32, device=device)
    slots = (
        block_table[:, positions // block_size] * block_size + positions % block_size
    )
    key = torch.randn(batch_size, seq_len + 1, num_kv_heads, head_dim, device=device)
    value = torch.randn_like(key)
    hp_row_ids = torch.tensor([1, 0], dtype=torch.int32, device=device)
    prefix_page_ids = torch.arange(8, dtype=torch.int32, device=device).view(2, 4)

    initial_query_start = torch.tensor(
        [0, seq_len, 2 * seq_len], dtype=torch.int32, device=device
    )
    initial_token_reqs = torch.arange(
        batch_size, dtype=torch.int32, device=device
    ).repeat_interleave(seq_len)
    initial_seq_lens = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    initial_key = key[:, :seq_len].reshape(-1, num_kv_heads, head_dim)
    initial_value = value[:, :seq_len].reshape(-1, num_kv_heads, head_dim)
    oscar_store(
        initial_key,
        initial_value,
        cache,
        slots[:, :seq_len].reshape(-1),
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
        token_to_req_indices=initial_token_reqs,
        query_start_loc=initial_query_start,
        seq_lens=initial_seq_lens,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    oscar_store_hp(
        initial_key,
        initial_value,
        prefix_cache,
        recent_cache,
        token_to_req_indices=initial_token_reqs,
        query_start_loc=initial_query_start,
        seq_lens=initial_seq_lens,
        hp_row_ids=hp_row_ids,
        prefix_page_ids=prefix_page_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    final_seq_lens = initial_seq_lens + 1
    decode_query_start = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    decode_token_reqs = torch.arange(batch_size, dtype=torch.int32, device=device)
    oscar_demote_hp(
        recent_cache,
        cache,
        block_table,
        final_seq_lens,
        hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
    )
    new_key = key[:, seq_len]
    new_value = value[:, seq_len]
    oscar_store_hp(
        new_key,
        new_value,
        prefix_cache,
        recent_cache,
        token_to_req_indices=decode_token_reqs,
        query_start_loc=decode_query_start,
        seq_lens=final_seq_lens,
        hp_row_ids=hp_row_ids,
        prefix_page_ids=prefix_page_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    query = torch.randn(batch_size, num_query_heads, head_dim, device=device)
    output = oscar_decode_attention(
        query,
        cache,
        block_table,
        final_seq_lens,
        head_dim**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
        prefix_cache=prefix_cache,
        recent_cache=recent_cache,
        hp_row_ids=hp_row_ids,
        prefix_page_ids=prefix_page_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    reference = torch.empty_like(output)
    final_partition = partition_tokens(
        seq_len + 1,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    for batch_idx in range(batch_size):
        key_ref, value_ref = oscar_full_dequant_kv(
            cache,
            block_table[batch_idx : batch_idx + 1],
            seq_len + 1,
            num_kv_heads,
            head_dim,
            cfg.key_levels,
            cfg.value_levels,
            cfg.key_data_bytes,
            cfg.key_packed_size,
            cfg.value_data_bytes,
        )
        for token_range in (final_partition.prefix, final_partition.recent):
            start, stop = token_range.start, token_range.stop
            key_ref[start:stop] = (
                key[batch_idx, start:stop].bfloat16().to(key_ref.dtype)
            )
            value_ref[start:stop] = (
                value[batch_idx, start:stop].bfloat16().to(value_ref.dtype)
            )
        for head_idx in range(num_query_heads):
            kv_head = head_idx // (num_query_heads // num_kv_heads)
            scores = (
                query[batch_idx, head_idx].float() @ key_ref[:, kv_head].float().T
            ) * head_dim**-0.5
            reference[batch_idx, head_idx] = (
                torch.softmax(scores, dim=-1) @ value_ref[:, kv_head].float()
            )

    torch.testing.assert_close(output.float(), reference.float(), atol=6e-3, rtol=6e-3)

    query_to_req = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device=device)
    mapped_query = torch.randn(
        len(query_to_req), num_query_heads, head_dim, device=device
    )
    query_seq_lens = final_seq_lens[query_to_req.long()]
    mapped_output, mapped_lse = oscar_decode_attention(
        mapped_query,
        cache,
        block_table,
        query_seq_lens,
        head_dim**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=4,
        prefix_cache=prefix_cache,
        recent_cache=recent_cache,
        hp_row_ids=hp_row_ids,
        prefix_page_ids=prefix_page_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        query_to_req_indices=query_to_req,
        return_lse=True,
    )
    reference_outputs = []
    reference_lses = []
    for query_idx, request_idx in enumerate(query_to_req.tolist()):
        one_output, one_lse = oscar_decode_attention(
            mapped_query[query_idx : query_idx + 1],
            cache,
            block_table[request_idx : request_idx + 1],
            final_seq_lens[request_idx : request_idx + 1],
            head_dim**-0.5,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_data_bytes=cfg.key_data_bytes,
            key_packed_size=cfg.key_packed_size,
            value_data_bytes=cfg.value_data_bytes,
            max_num_kv_splits=4,
            prefix_cache=prefix_cache,
            recent_cache=recent_cache,
            hp_row_ids=hp_row_ids[request_idx : request_idx + 1],
            prefix_page_ids=prefix_page_ids[request_idx : request_idx + 1],
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
            return_lse=True,
        )
        reference_outputs.append(one_output)
        reference_lses.append(one_lse)
    torch.testing.assert_close(
        mapped_output,
        torch.cat(reference_outputs),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        mapped_lse,
        torch.cat(reference_lses),
        atol=0,
        rtol=0,
    )


def test_prefix_hit_does_not_read_unowned_recent_row():
    from vllm.v1.attention.ops.triton_oscar_decode import (
        oscar_decode_attention,
        oscar_full_dequant_kv,
    )
    from vllm.v1.attention.ops.triton_oscar_mixed_store import (
        oscar_demote_hp,
        oscar_store_hp,
    )
    from vllm.v1.attention.ops.triton_oscar_prefill import (
        oscar_cached_prefill_attention,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(31)
    device = "cuda"
    cfg = OscarConfig()
    num_query_heads, num_kv_heads, head_dim = 8, 2, 128
    block_size, shared_len = 16, 384
    num_blocks = shared_len // block_size
    cache, prefix_cache, recent_cache = _make_mixed_cache(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg,
        device,
        num_hp_rows=2,
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    slots = torch.arange(shared_len, dtype=torch.int32, device=device)
    key = torch.randn(
        shared_len, num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    value = torch.randn_like(key)
    query_start_loc = torch.tensor([0, shared_len], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([shared_len], dtype=torch.int32, device=device)
    token_to_req = torch.zeros(shared_len, dtype=torch.int32, device=device)
    source_hp_row = torch.tensor([0], dtype=torch.int32, device=device)
    shared_prefix_pages = torch.arange(
        cfg.prefix_tokens // block_size, dtype=torch.int32, device=device
    ).unsqueeze(0)

    oscar_store(
        key.float(),
        value.float(),
        cache,
        slots,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
        token_to_req_indices=token_to_req,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    oscar_store_hp(
        key.float(),
        value.float(),
        prefix_cache,
        recent_cache,
        token_to_req_indices=token_to_req,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        hp_row_ids=source_hp_row,
        prefix_page_ids=shared_prefix_pages,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    # A prefix-hit request owns row 1, but has not written any recent tokens yet.
    recent_cache[cfg.recent_tokens :].fill_(1000)
    query = torch.zeros(
        1, num_query_heads, head_dim, dtype=torch.float32, device=device
    )
    output = oscar_decode_attention(
        query,
        cache,
        block_table,
        seq_lens,
        head_dim**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        max_num_kv_splits=1,
        prefix_cache=prefix_cache,
        recent_cache=recent_cache,
        hp_row_ids=torch.tensor([1], dtype=torch.int32, device=device),
        prefix_page_ids=shared_prefix_pages,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        query_to_req_indices=torch.tensor([0], dtype=torch.int32, device=device),
        shared_hit_tokens=seq_lens,
    )

    key_ref, value_ref = oscar_full_dequant_kv(
        cache,
        block_table,
        shared_len,
        num_kv_heads,
        head_dim,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )
    key_ref[: cfg.prefix_tokens] = key[: cfg.prefix_tokens].float()
    value_ref[: cfg.prefix_tokens] = value[: cfg.prefix_tokens].float()
    reference = value_ref.mean(dim=0).repeat_interleave(
        num_query_heads // num_kv_heads, dim=0
    )
    torch.testing.assert_close(
        output[0].float(), reference.float(), atol=6e-3, rtol=6e-3
    )
    v_rotation_t = torch.linalg.qr(
        torch.randn(head_dim, head_dim, dtype=torch.float32, device=device)
    ).Q.contiguous()
    tiled_output, tiled_lse = oscar_cached_prefill_attention(
        query,
        cache,
        prefix_cache,
        recent_cache,
        block_table,
        seq_lens,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        shared_prefix_pages,
        seq_lens,
        scale=head_dim**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        max_query_len=1,
        v_rotation_t=v_rotation_t,
    )
    rotated_reference = torch.matmul(reference.float(), v_rotation_t)
    torch.testing.assert_close(
        tiled_output[0].float(), rotated_reference, atol=1.5e-2, rtol=1.5e-2
    )
    assert tiled_lse.shape == (num_query_heads, 1)
    assert torch.isfinite(tiled_lse).all()

    absorbed_output, absorbed_lse = oscar_cached_prefill_attention(
        query,
        cache,
        prefix_cache,
        recent_cache,
        block_table,
        seq_lens,
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        shared_prefix_pages,
        seq_lens,
        scale=head_dim**-0.5,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_data_bytes=cfg.key_data_bytes,
        key_packed_size=cfg.key_packed_size,
        value_data_bytes=cfg.value_data_bytes,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        max_query_len=1,
        output_dtype=torch.bfloat16,
    )
    assert absorbed_output.dtype == torch.bfloat16
    torch.testing.assert_close(
        absorbed_output[0].float(), reference.float(), atol=1.5e-2, rtol=1.5e-2
    )
    torch.testing.assert_close(absorbed_lse, tiled_lse, atol=0, rtol=0)

    quant_before = cache.clone()
    oscar_demote_hp(
        recent_cache,
        cache,
        block_table,
        torch.tensor([shared_len + 17], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        shared_hit_tokens=seq_lens,
        query_start_loc=torch.tensor([0, 17], dtype=torch.int32, device=device),
        max_query_len=17,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
    )
    torch.testing.assert_close(cache, quant_before, atol=0, rtol=0)


def test_chunked_prefill_demotes_every_evicted_recent_token():
    from vllm.v1.attention.ops.triton_oscar_mixed_store import (
        oscar_demote_hp,
        oscar_store_hp,
    )

    torch.manual_seed(4)
    device = "cuda"
    cfg = OscarConfig()
    num_kv_heads, head_dim = 2, 128
    block_size, cached_len, query_len = 16, 384, 17
    final_len = cached_len + query_len
    num_blocks = (final_len + block_size - 1) // block_size
    cache, prefix_cache, recent_cache = _make_mixed_cache(
        num_blocks, block_size, num_kv_heads, cfg, device
    )
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    key = torch.randn(cached_len, num_kv_heads, head_dim, device=device)
    value = torch.randn_like(key)
    oscar_store_hp(
        key,
        value,
        prefix_cache,
        recent_cache,
        token_to_req_indices=torch.zeros(cached_len, dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, cached_len], dtype=torch.int32, device=device),
        seq_lens=torch.tensor([cached_len], dtype=torch.int32, device=device),
        hp_row_ids=torch.tensor([0], dtype=torch.int32, device=device),
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    oscar_demote_hp(
        recent_cache,
        cache,
        block_table,
        torch.tensor([final_len], dtype=torch.int32, device=device),
        torch.tensor([0], dtype=torch.int32, device=device),
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        max_query_len=query_len,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=0.0,
        v_clip_ratio=0.0,
    )

    single_token_cache = torch.zeros_like(cache)
    for step in range(1, query_len + 1):
        oscar_demote_hp(
            recent_cache,
            single_token_cache,
            block_table,
            torch.tensor([cached_len + step], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
            k_clip_ratio=0.0,
            v_clip_ratio=0.0,
        )

    torch.testing.assert_close(cache, single_token_cache, atol=0, rtol=0)
    demote_start = cached_len - cfg.recent_tokens
    demote_end = demote_start + query_len
    for position in range(demote_start, demote_end):
        assert cache[position // block_size, position % block_size].any()
    assert not cache[demote_end // block_size, demote_end % block_size].any()


@pytest.mark.parametrize("chunk_size", [128, 256, 257, 2048, 8192])
def test_chunked_and_non_chunked_final_tiers_are_byte_exact(chunk_size):
    from vllm.v1.attention.ops.triton_oscar_mixed_store import (
        oscar_demote_hp,
        oscar_store_hp,
    )
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store

    torch.manual_seed(6)
    device = "cuda"
    cfg = OscarConfig()
    num_kv_heads, head_dim = 2, 128
    block_size, seq_len = 16, 8449
    num_blocks = (seq_len + block_size - 1) // block_size
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    slots = (
        block_table[0, positions // block_size] * block_size + positions % block_size
    )
    key = torch.randn(
        seq_len,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    value = torch.randn_like(key)
    hp_row_ids = torch.tensor([0], dtype=torch.int32, device=device)

    reference_cache, reference_prefix, reference_recent = _make_mixed_cache(
        num_blocks, block_size, num_kv_heads, cfg, device
    )
    reference_qsl = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    reference_seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    reference_token_reqs = torch.zeros(seq_len, dtype=torch.int32, device=device)
    oscar_store(
        key.float(),
        value.float(),
        reference_cache,
        slots,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
        token_to_req_indices=reference_token_reqs,
        query_start_loc=reference_qsl,
        seq_lens=reference_seq_lens,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    oscar_store_hp(
        key.float(),
        value.float(),
        reference_prefix,
        reference_recent,
        token_to_req_indices=reference_token_reqs,
        query_start_loc=reference_qsl,
        seq_lens=reference_seq_lens,
        hp_row_ids=hp_row_ids,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    chunked_cache, chunked_prefix, chunked_recent = _make_mixed_cache(
        num_blocks, block_size, num_kv_heads, cfg, device
    )
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        current_len = end - start
        current_qsl = torch.tensor([0, current_len], dtype=torch.int32, device=device)
        current_seq_lens = torch.tensor([end], dtype=torch.int32, device=device)
        current_token_reqs = torch.zeros(current_len, dtype=torch.int32, device=device)
        oscar_store(
            key[start:end].float(),
            value[start:end].float(),
            chunked_cache,
            slots[start:end],
            key_levels=cfg.key_levels,
            value_levels=cfg.value_levels,
            key_packed_size=cfg.key_packed_size,
            data_bytes=cfg.key_data_bytes,
            k_clip_ratio=cfg.k_clip_ratio,
            v_clip_ratio=cfg.v_clip_ratio,
            token_to_req_indices=current_token_reqs,
            query_start_loc=current_qsl,
            seq_lens=current_seq_lens,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )
        if start:
            oscar_demote_hp(
                chunked_recent,
                chunked_cache,
                block_table,
                current_seq_lens,
                hp_row_ids,
                query_start_loc=current_qsl,
                max_query_len=current_len,
                prefix_tokens=cfg.prefix_tokens,
                recent_tokens=cfg.recent_tokens,
                key_levels=cfg.key_levels,
                value_levels=cfg.value_levels,
                key_packed_size=cfg.key_packed_size,
                data_bytes=cfg.key_data_bytes,
                k_clip_ratio=cfg.k_clip_ratio,
                v_clip_ratio=cfg.v_clip_ratio,
            )
        oscar_store_hp(
            key[start:end].float(),
            value[start:end].float(),
            chunked_prefix,
            chunked_recent,
            token_to_req_indices=current_token_reqs,
            query_start_loc=current_qsl,
            seq_lens=current_seq_lens,
            hp_row_ids=hp_row_ids,
            prefix_tokens=cfg.prefix_tokens,
            recent_tokens=cfg.recent_tokens,
        )

    torch.testing.assert_close(chunked_cache, reference_cache, atol=0, rtol=0)
    torch.testing.assert_close(chunked_prefix, reference_prefix, atol=0, rtol=0)
    torch.testing.assert_close(chunked_recent, reference_recent, atol=0, rtol=0)


@pytest.mark.parametrize("absorb_v_rotation", [False, True])
def test_chunked_prefill_merges_cached_and_current_attention_before_store(
    absorb_v_rotation,
    monkeypatch,
):
    from types import SimpleNamespace

    from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
    from vllm.v1.attention.backends import oscar_attn as oscar_attn_module
    from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
    from vllm.v1.attention.backends.oscar_attn import (
        OscarAttentionImpl,
        OscarMetadata,
    )
    from vllm.v1.attention.ops.triton_oscar_decode import oscar_full_dequant_kv
    from vllm.v1.attention.ops.triton_oscar_mixed_store import oscar_store_hp
    from vllm.v1.attention.ops.triton_oscar_store import oscar_store
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        reset_workspace_manager,
    )

    torch.manual_seed(5)
    device = "cuda"
    cfg = OscarConfig(absorb_v_rotation=absorb_v_rotation)
    num_query_heads, num_kv_heads, head_dim = 8, 2, 128
    block_size, cached_len = 16, 384
    query_lens = [17, 13]
    final_seq_lens = [cached_len + query_lens[0], query_lens[1]]
    max_blocks = (final_seq_lens[0] + block_size - 1) // block_size
    num_blocks = max_blocks + 1
    cache, prefix_cache, recent_cache = _make_mixed_cache(
        num_blocks,
        block_size,
        num_kv_heads,
        cfg,
        device,
        num_hp_rows=2,
    )
    block_table = torch.empty(2, max_blocks, dtype=torch.int32, device=device)
    block_table[0] = torch.arange(max_blocks, dtype=torch.int32, device=device)
    block_table[1].fill_(max_blocks)

    old_key = torch.randn(
        cached_len,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    old_value = torch.randn_like(old_key)
    key_rotation = torch.linalg.qr(
        torch.randn(head_dim, head_dim, dtype=torch.float32, device=device)
    ).Q.contiguous()
    value_rotation = torch.linalg.qr(
        torch.randn(head_dim, head_dim, dtype=torch.float32, device=device)
    ).Q.contiguous()
    old_positions = torch.arange(cached_len, dtype=torch.int32, device=device)
    old_slots = (
        block_table[0, old_positions // block_size] * block_size
        + old_positions % block_size
    )
    old_qsl = torch.tensor([0, cached_len], dtype=torch.int32, device=device)
    old_seq_lens = torch.tensor([cached_len], dtype=torch.int32, device=device)
    old_token_reqs = torch.zeros(cached_len, dtype=torch.int32, device=device)
    oscar_store(
        torch.matmul(old_key.float(), key_rotation),
        torch.matmul(old_value.float(), value_rotation),
        cache,
        old_slots,
        key_levels=cfg.key_levels,
        value_levels=cfg.value_levels,
        key_packed_size=cfg.key_packed_size,
        data_bytes=cfg.key_data_bytes,
        k_clip_ratio=cfg.k_clip_ratio,
        v_clip_ratio=cfg.v_clip_ratio,
        token_to_req_indices=old_token_reqs,
        query_start_loc=old_qsl,
        seq_lens=old_seq_lens,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    oscar_store_hp(
        torch.matmul(old_key.float(), key_rotation),
        torch.matmul(old_value.float(), value_rotation),
        prefix_cache,
        recent_cache,
        token_to_req_indices=old_token_reqs,
        query_start_loc=old_qsl,
        seq_lens=old_seq_lens,
        hp_row_ids=torch.tensor([0], dtype=torch.int32, device=device),
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )

    total_queries = sum(query_lens)
    query = torch.randn(
        total_queries,
        num_query_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    key = torch.randn(
        total_queries,
        num_kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    original_value = torch.randn_like(key)
    if absorb_v_rotation:
        rotated_value = torch.matmul(original_value.float(), value_rotation).to(
            original_value.dtype
        )
        qkv_width = (num_query_heads + 2 * num_kv_heads) * head_dim
        fused_qkv = torch.empty(
            total_queries,
            qkv_width,
            dtype=original_value.dtype,
            device=device,
        )
        value = fused_qkv[:, -num_kv_heads * head_dim :].view_as(original_value)
        value.copy_(rotated_value)
        assert not value.is_contiguous()
        assert value.stride(0) != key.stride(0)
    else:
        value = original_value
    query_start_loc = torch.tensor(
        [0, query_lens[0], total_queries], dtype=torch.int32, device=device
    )
    token_to_req = torch.tensor(
        [0] * query_lens[0] + [1] * query_lens[1],
        dtype=torch.int32,
        device=device,
    )
    current_positions = torch.cat(
        (
            torch.arange(
                cached_len,
                final_seq_lens[0],
                dtype=torch.int32,
                device=device,
            ),
            torch.arange(query_lens[1], dtype=torch.int32, device=device),
        )
    )
    slot_mapping = torch.empty(total_queries, dtype=torch.int64, device=device)
    first_positions = current_positions[: query_lens[0]]
    slot_mapping[: query_lens[0]] = (
        block_table[0, first_positions // block_size] * block_size
        + first_positions % block_size
    )
    second_positions = current_positions[query_lens[0] :]
    slot_mapping[query_lens[0] :] = (
        block_table[1, second_positions // block_size] * block_size
        + second_positions % block_size
    )

    cached_key, cached_value = oscar_full_dequant_kv(
        cache,
        block_table[0:1],
        cached_len,
        num_kv_heads,
        head_dim,
        cfg.key_levels,
        cfg.value_levels,
        cfg.key_data_bytes,
        cfg.key_packed_size,
        cfg.value_data_bytes,
    )
    cached_key = torch.matmul(cached_key.float(), key_rotation.T)
    cached_value = torch.matmul(cached_value.float(), value_rotation.T)
    old_partition = partition_tokens(
        cached_len,
        prefix_tokens=cfg.prefix_tokens,
        recent_tokens=cfg.recent_tokens,
    )
    for token_range in (old_partition.prefix, old_partition.recent):
        start, stop = token_range.start, token_range.stop
        cached_key[start:stop] = old_key[start:stop].float()
        cached_value[start:stop] = old_value[start:stop].float()

    reference = torch.empty_like(query, dtype=torch.float32)
    for request_idx, query_len in enumerate(query_lens):
        q_start = int(query_start_loc[request_idx])
        q_end = int(query_start_loc[request_idx + 1])
        request_cached_len = cached_len if request_idx == 0 else 0
        current_key = key[q_start:q_end].float()
        current_value = original_value[q_start:q_end].float()
        request_key = (
            torch.cat((cached_key, current_key)) if request_cached_len else current_key
        )
        request_value = (
            torch.cat((cached_value, current_value))
            if request_cached_len
            else current_value
        )
        for query_idx in range(query_len):
            visible = request_cached_len + query_idx + 1
            for head_idx in range(num_query_heads):
                kv_head = head_idx // (num_query_heads // num_kv_heads)
                scores = (
                    query[q_start + query_idx, head_idx].float()
                    @ request_key[:visible, kv_head].T
                ) * head_dim**-0.5
                reference[q_start + query_idx, head_idx] = (
                    torch.softmax(scores, dim=-1) @ request_value[:visible, kv_head]
                )

    impl = object.__new__(OscarAttentionImpl)
    impl.num_heads = num_query_heads
    impl.num_kv_heads = num_kv_heads
    impl.head_size = head_dim
    impl.scale = head_dim**-0.5
    impl.cfg = cfg
    impl.fa_version = get_flash_attn_version(head_size=head_dim)
    impl.max_num_kv_splits = 4
    layer = SimpleNamespace(
        _oscar_ready=True,
        _oscar_Rk=key_rotation,
        _oscar_Rv=value_rotation,
        _oscar_RvT=value_rotation.T.contiguous(),
        _oscar_Rk_fast=key_rotation.to(query.dtype),
        _oscar_RvT_fast=value_rotation.T.contiguous().to(query.dtype),
        oscar_v_rotation_absorbed=absorb_v_rotation,
    )
    metadata = OscarMetadata(
        seq_lens=torch.tensor(final_seq_lens, dtype=torch.int32, device=device),
        slot_mapping=slot_mapping,
        block_table=block_table,
        query_start_loc=query_start_loc,
        hp_row_ids=torch.tensor([0, 1], dtype=torch.int32, device=device),
        prefix_page_ids=torch.arange(8, dtype=torch.int32, device=device).view(2, 4),
        shared_hit_tokens=torch.zeros(2, dtype=torch.int32, device=device),
        token_to_req_indices=token_to_req,
        seq_start_loc=torch.tensor(
            [0, final_seq_lens[0], sum(final_seq_lens)],
            dtype=torch.int32,
            device=device,
        ),
        cached_lens=torch.tensor([cached_len, 0], dtype=torch.int32, device=device),
        num_actual_tokens=total_queries,
        max_query_len=max(query_lens),
        max_seq_len=max(final_seq_lens),
        is_prefill=True,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens_cpu=torch.tensor(final_seq_lens, dtype=torch.int32),
    )
    output = torch.empty_like(query)
    fused_output = torch.empty_like(query)
    total_materialized_tokens = sum(final_seq_lens)
    original_cached_prefill = oscar_attn_module.oscar_cached_prefill_attention

    def check_cached_prefill_q_dtype(q_rot, *args, **kwargs):
        expected_dtype = query.dtype if absorb_v_rotation else torch.float32
        assert q_rot.dtype == expected_dtype
        return original_cached_prefill(q_rot, *args, **kwargs)

    monkeypatch.setattr(
        oscar_attn_module,
        "oscar_cached_prefill_attention",
        check_cached_prefill_q_dtype,
    )
    reset_workspace_manager()
    init_workspace_manager(torch.device(device))
    try:
        impl._materialize_max_tokens = total_materialized_tokens - 1
        assert impl._materialize_tokens(metadata) == 0
        original_flash_attn = impl._flash_attn_varlen
        if absorb_v_rotation:

            def fail_suffix_flash_attn(*args, **kwargs):
                raise AssertionError(
                    "absorbed cached prefill must fuse current attention"
                )

            impl._flash_attn_varlen = fail_suffix_flash_attn
        impl.forward(
            layer,
            query,
            key,
            value,
            [cache.clone(), prefix_cache.clone(), recent_cache.clone()],
            metadata,
            output=fused_output,
        )
        impl._flash_attn_varlen = original_flash_attn
        impl._materialize_max_tokens = total_materialized_tokens
        assert impl._materialize_tokens(metadata) == total_materialized_tokens
        impl.forward(
            layer,
            query,
            key,
            value,
            [cache, prefix_cache, recent_cache],
            metadata,
            output=output,
        )
    finally:
        reset_workspace_manager()

    torch.testing.assert_close(
        fused_output.float(), reference, atol=1.5e-2, rtol=1.5e-2
    )
    torch.testing.assert_close(output.float(), reference, atol=1.5e-2, rtol=1.5e-2)
    for position in range(cached_len - cfg.recent_tokens, 145):
        block = block_table[0, position // block_size]
        assert cache[block, position % block_size].any()
