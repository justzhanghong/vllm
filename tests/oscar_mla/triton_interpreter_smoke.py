# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-interpreter smoke for OSCAR MLA Triton kernels.

This module runs in a fresh process with ``TRITON_INTERPRET=1``. It validates
Triton semantics without creating a CUDA context; SM80 compilation and A800
launch remain separate acceptance gates.
"""

import torch

from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    mixed_latent_attention_with_lse,
)
from vllm.v1.attention.ops import triton_oscar_mla_decode as decode
from vllm.v1.attention.ops import triton_oscar_mla_materialize as materialize
from vllm.v1.attention.ops import triton_oscar_mla_store as store

store._require_cuda_tensor = lambda *args, **kwargs: None
decode._require_cuda_tensor = lambda *args, **kwargs: None

torch.manual_seed(7)
latent_rank = 512
sequence_length = 5
latent = torch.randn(sequence_length, latent_rank, dtype=torch.bfloat16)
query = torch.randn(1, 1, latent_rank, dtype=torch.bfloat16)
rope_values = torch.randn(sequence_length, 64, dtype=torch.bfloat16)
query_rope = torch.randn(1, 1, 64, dtype=torch.bfloat16)
rope_cache = torch.zeros(1, 16, 64, dtype=torch.bfloat16)
rope_block_table = torch.zeros(1, 1, dtype=torch.int32)
rotation = torch.eye(latent_rank, dtype=torch.bfloat16).T
assert not rotation.is_contiguous()
prefix = torch.zeros(1, 2, latent_rank, dtype=torch.bfloat16)
recent = torch.zeros(1, 2, latent_rank, dtype=torch.bfloat16)
positions = torch.arange(sequence_length, dtype=torch.int32)
zero_index = torch.zeros(1, dtype=torch.int32)

store.oscar_mla_store_bf16(
    latent,
    prefix,
    recent,
    positions,
    torch.full_like(positions, sequence_length),
    torch.zeros_like(positions),
)
store.oscar_mla_store_rope(
    rope_values,
    rope_cache,
    torch.arange(sequence_length, dtype=torch.int32),
)
torch.testing.assert_close(prefix[0], latent[:2])
torch.testing.assert_close(recent[0, 0], latent[4])
torch.testing.assert_close(recent[0, 1], latent[3])
torch.testing.assert_close(rope_cache[0, :sequence_length], rope_values)

history_data = torch.zeros(
    1,
    16,
    latent_rank // 4,
    dtype=torch.uint8,
)
history_scale = torch.zeros(
    1,
    16,
    latent_rank // 128,
    dtype=torch.float32,
)
history_zero = torch.zeros_like(history_scale)
store.oscar_mla_rotate_quantize_store(
    latent[2:3],
    rotation,
    history_data,
    history_scale,
    history_zero,
    zero_index,
    zero_index,
    clip_ratio=0.96,
)
history = store.oscar_mla_dequantize_history(
    history_data,
    history_scale,
    history_zero,
    zero_index,
    zero_index,
)
output, lse = decode.oscar_mla_sparse_decode(
    query,
    query_rope,
    torch.arange(sequence_length, dtype=torch.int32).unsqueeze(0),
    prefix,
    recent,
    rope_cache,
    rope_block_table,
    history_data,
    history_scale,
    history_zero,
    torch.zeros(1, 1, dtype=torch.int32),
    zero_index,
    torch.tensor([sequence_length], dtype=torch.int32),
    rotation,
    num_splits=2,
)
expected, expected_lse = mixed_latent_attention_with_lse(
    query.float(),
    prefix_latent=latent[:2].float(),
    recent_latent=latent[3:].float(),
    history_rotated=history,
    rotation=rotation.float(),
    query_rope=query_rope.float(),
    prefix_rope=rope_values[:2].float(),
    history_rope=rope_values[2:3].float(),
    recent_rope=rope_values[3:].float(),
)
torch.testing.assert_close(output, expected, atol=1e-5, rtol=1e-5)
torch.testing.assert_close(lse, expected_lse, atol=1e-5, rtol=1e-5)
assert bool(output.isfinite().all())
assert bool(lse.isfinite().all())

prefill_output, prefill_lse = decode.oscar_mla_sparse_prefill(
    query.repeat(2, 1, 1),
    query_rope.repeat(2, 1, 1),
    torch.arange(sequence_length, dtype=torch.int32).repeat(2, 1),
    torch.zeros(2, dtype=torch.int32),
    torch.tensor([2, 4], dtype=torch.int32),
    prefix,
    recent,
    rope_cache,
    rope_block_table,
    history_data,
    history_scale,
    history_zero,
    torch.zeros(1, 1, dtype=torch.int32),
    zero_index,
    torch.tensor([sequence_length], dtype=torch.int32),
    rotation,
    num_splits=2,
)
prefill_expected_first, prefill_expected_lse_first = mixed_latent_attention_with_lse(
    query.float(),
    prefix_latent=latent[:2].float(),
    recent_latent=latent[:0].float(),
    history_rotated=history,
    rotation=rotation.float(),
    query_rope=query_rope.float(),
    prefix_rope=rope_values[:2].float(),
    history_rope=rope_values[2:3].float(),
    recent_rope=rope_values[:0].float(),
)
prefill_expected = torch.cat((prefill_expected_first, expected), dim=0)
prefill_expected_lse = torch.cat((prefill_expected_lse_first, expected_lse), dim=0)
torch.testing.assert_close(
    prefill_output,
    prefill_expected,
    atol=1e-5,
    rtol=1e-5,
)
torch.testing.assert_close(
    prefill_lse,
    prefill_expected_lse,
    atol=1e-5,
    rtol=1e-5,
)
assert bool(prefill_output.isfinite().all())
assert bool(prefill_lse.isfinite().all())

request_count = 2
batched_latent = torch.randn(
    request_count,
    sequence_length,
    latent_rank,
    dtype=torch.bfloat16,
)
batched_query = torch.randn(
    request_count,
    1,
    latent_rank,
    dtype=torch.bfloat16,
)
batched_rope_values = torch.randn(
    request_count,
    sequence_length,
    64,
    dtype=torch.bfloat16,
)
batched_query_rope = torch.randn(
    request_count,
    1,
    64,
    dtype=torch.bfloat16,
)
batched_prefix = torch.zeros(
    request_count,
    2,
    latent_rank,
    dtype=torch.bfloat16,
)
batched_recent = torch.zeros_like(batched_prefix)
batched_positions = torch.arange(sequence_length, dtype=torch.int32).repeat(
    request_count
)
batched_hp_rows = torch.arange(request_count, dtype=torch.int32).repeat_interleave(
    sequence_length
)
store.oscar_mla_store_bf16(
    batched_latent.flatten(0, 1),
    batched_prefix,
    batched_recent,
    batched_positions,
    torch.full_like(batched_positions, sequence_length),
    batched_hp_rows,
)

batched_rope_cache = torch.zeros(
    request_count,
    16,
    64,
    dtype=torch.bfloat16,
)
batched_rope_slots = (
    torch.arange(request_count, dtype=torch.int32).repeat_interleave(sequence_length)
    * 16
    + batched_positions
)
store.oscar_mla_store_rope(
    batched_rope_values.flatten(0, 1),
    batched_rope_cache,
    batched_rope_slots,
)
batched_rope_block_table = torch.arange(
    request_count,
    dtype=torch.int32,
).unsqueeze(1)

batched_history_data = torch.zeros(
    request_count,
    16,
    latent_rank // 4,
    dtype=torch.uint8,
)
batched_history_scale = torch.zeros(
    request_count,
    16,
    latent_rank // 128,
    dtype=torch.float32,
)
batched_history_zero = torch.zeros_like(batched_history_scale)
batched_page_ids = torch.arange(request_count, dtype=torch.int32)
store.oscar_mla_rotate_quantize_store(
    batched_latent[:, 2],
    rotation,
    batched_history_data,
    batched_history_scale,
    batched_history_zero,
    batched_page_ids,
    torch.zeros(request_count, dtype=torch.int32),
    clip_ratio=0.96,
)
batched_history = store.oscar_mla_dequantize_history(
    batched_history_data,
    batched_history_scale,
    batched_history_zero,
    batched_page_ids,
    torch.zeros(request_count, dtype=torch.int32),
)
batched_output, batched_lse = decode.oscar_mla_sparse_decode(
    batched_query,
    batched_query_rope,
    torch.cat(
        (
            torch.arange(sequence_length, dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
        )
    ).repeat(request_count, 1),
    batched_prefix,
    batched_recent,
    batched_rope_cache,
    batched_rope_block_table,
    batched_history_data,
    batched_history_scale,
    batched_history_zero,
    batched_page_ids.unsqueeze(1),
    torch.arange(request_count, dtype=torch.int32),
    torch.full((request_count,), sequence_length, dtype=torch.int32),
    rotation,
    num_splits=2,
)
batched_expected_rows = []
batched_expected_lse_rows = []
for request_index in range(request_count):
    expected_row, expected_lse_row = mixed_latent_attention_with_lse(
        batched_query[request_index : request_index + 1].float(),
        prefix_latent=batched_latent[request_index, :2].float(),
        recent_latent=batched_latent[request_index, 3:].float(),
        history_rotated=batched_history[request_index : request_index + 1],
        rotation=rotation.float(),
        query_rope=batched_query_rope[request_index : request_index + 1].float(),
        prefix_rope=batched_rope_values[request_index, :2].float(),
        history_rope=batched_rope_values[request_index, 2:3].float(),
        recent_rope=batched_rope_values[request_index, 3:].float(),
    )
    batched_expected_rows.append(expected_row)
    batched_expected_lse_rows.append(expected_lse_row)
batched_expected = torch.cat(batched_expected_rows)
batched_expected_lse = torch.cat(batched_expected_lse_rows)
torch.testing.assert_close(
    batched_output,
    batched_expected,
    atol=1e-5,
    rtol=1e-5,
)
torch.testing.assert_close(
    batched_lse,
    batched_expected_lse,
    atol=1e-5,
    rtol=1e-5,
)
assert bool(batched_output.isfinite().all())
assert bool(batched_lse.isfinite().all())

mtp5_sequence_length = 326
mtp5_logical_recent_tokens = 256
mtp5_physical_recent_tokens = mtp5_logical_recent_tokens + 5
mtp5_latent = torch.randn(
    mtp5_sequence_length,
    latent_rank,
    dtype=torch.bfloat16,
)
mtp5_rope_values = torch.randn(
    mtp5_sequence_length,
    64,
    dtype=torch.bfloat16,
)
mtp5_prefix = torch.zeros(1, 64, latent_rank, dtype=torch.bfloat16)
mtp5_recent = torch.zeros(
    1,
    mtp5_physical_recent_tokens,
    latent_rank,
    dtype=torch.bfloat16,
)
mtp5_positions = torch.arange(mtp5_sequence_length, dtype=torch.int32)
store.oscar_mla_store_bf16(
    mtp5_latent,
    mtp5_prefix,
    mtp5_recent,
    mtp5_positions,
    torch.full_like(mtp5_positions, mtp5_sequence_length),
    torch.zeros_like(mtp5_positions),
    recent_tokens=mtp5_logical_recent_tokens,
)
torch.testing.assert_close(mtp5_prefix[0, 63], mtp5_latent[63])
torch.testing.assert_close(mtp5_recent[0, 6], mtp5_latent[70])
torch.testing.assert_close(mtp5_recent[0, 255], mtp5_latent[319])
torch.testing.assert_close(mtp5_recent[0, 256], mtp5_latent[320])
torch.testing.assert_close(mtp5_recent[0, 260], mtp5_latent[324])
torch.testing.assert_close(mtp5_recent[0, 0], mtp5_latent[325])

mtp5_history_data = torch.zeros(1, 16, latent_rank // 4, dtype=torch.uint8)
mtp5_history_scale = torch.zeros(
    1,
    16,
    latent_rank // 128,
    dtype=torch.float32,
)
mtp5_history_zero = torch.zeros_like(mtp5_history_scale)
mtp5_history_positions = torch.arange(64, 70, dtype=torch.int32)
store.oscar_mla_rotate_quantize_store(
    mtp5_latent[mtp5_history_positions.long()],
    rotation,
    mtp5_history_data,
    mtp5_history_scale,
    mtp5_history_zero,
    torch.zeros(6, dtype=torch.int32),
    mtp5_history_positions - 64,
    clip_ratio=0.96,
)
mtp5_history = store.oscar_mla_dequantize_history(
    mtp5_history_data,
    mtp5_history_scale,
    mtp5_history_zero,
    torch.zeros(6, dtype=torch.int32),
    mtp5_history_positions - 64,
)
mtp5_rope_pages = (mtp5_sequence_length + 15) // 16
mtp5_rope = torch.zeros(
    mtp5_rope_pages,
    16,
    64,
    dtype=torch.bfloat16,
)
store.oscar_mla_store_rope(
    mtp5_rope_values,
    mtp5_rope,
    mtp5_positions,
)
mtp5_selected_positions = torch.tensor(
    [63, 64, 69, 70, 319, 320, 324, 325],
    dtype=torch.int32,
)
mtp5_num_rows = mtp5_selected_positions.numel()
mtp5_materialized, _ = materialize.materialize_oscar_mla_bf16_rows(
    positions=mtp5_selected_positions,
    num_rows=mtp5_num_rows,
    num_requests=1,
    prefix=mtp5_prefix,
    recent=mtp5_recent,
    rope=mtp5_rope,
    rope_block_table=torch.arange(mtp5_rope_pages, dtype=torch.int32).unsqueeze(0),
    history_data=mtp5_history_data,
    history_scale=mtp5_history_scale,
    history_zero=mtp5_history_zero,
    history_page_table=torch.zeros(1, 1, dtype=torch.int32),
    hp_rows=torch.zeros(1, dtype=torch.int32),
    seq_lens=torch.tensor([mtp5_sequence_length], dtype=torch.int32),
    inverse_rotation=torch.eye(latent_rank, dtype=torch.bfloat16).contiguous(),
    history_rotated=torch.zeros(
        mtp5_num_rows,
        latent_rank,
        dtype=torch.bfloat16,
    ),
    history_mask=torch.zeros(mtp5_num_rows, dtype=torch.uint8),
    output_kv=torch.zeros(
        mtp5_num_rows,
        latent_rank + 64,
        dtype=torch.bfloat16,
    ),
    remapped_indices=torch.zeros(mtp5_num_rows, dtype=torch.int32),
    recent_tokens=mtp5_logical_recent_tokens,
)
mtp5_expected_latent = mtp5_latent[mtp5_selected_positions.long()].clone()
mtp5_expected_latent[1] = mtp5_history[0].bfloat16()
mtp5_expected_latent[2] = mtp5_history[5].bfloat16()
torch.testing.assert_close(
    mtp5_materialized[:, 0, :latent_rank],
    mtp5_expected_latent,
    atol=1e-2,
    rtol=1e-2,
)
torch.testing.assert_close(
    mtp5_materialized[:, 0, latent_rank:],
    mtp5_rope_values[mtp5_selected_positions.long()],
)
mtp5_materialize_error = (
    (mtp5_materialized[:, 0, :latent_rank] - mtp5_expected_latent).abs().max().item()
)
multi_request_error = (batched_output - batched_expected).abs().max().item()
multi_request_lse_error = (batched_lse - batched_expected_lse).abs().max().item()
print(
    "interpreter_smoke",
    f"latent_rank={latent_rank}",
    f"groups={history_scale.shape[-1]}",
    f"max_error={(output - expected).abs().max().item()}",
    f"lse_max_error={(lse - expected_lse).abs().max().item()}",
    f"prefill_max_error={(prefill_output - prefill_expected).abs().max().item()}",
    f"prefill_lse_max_error={(prefill_lse - prefill_expected_lse).abs().max().item()}",
    f"multi_request_max_error={multi_request_error}",
    f"multi_request_lse_max_error={multi_request_lse_error}",
    f"mtp5_materialize_max_error={mtp5_materialize_error}",
)
