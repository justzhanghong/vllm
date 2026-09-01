# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import inspect
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import vllm.v1.attention.ops.triton_oscar_mla_decode as triton_oscar_mla_decode
import vllm.v1.attention.ops.triton_sparse_mla_kernel as triton_sparse_mla_kernel
from vllm.model_executor.layers.quantization.oscar_mla.reference import (
    mixed_latent_attention_with_lse,
)
from vllm.v1.attention.ops import triton_oscar_mla_materialize
from vllm.v1.attention.ops.triton_oscar_mla_decode import (
    _mixed_sparse_decode_grouped_h4_qk,
    _mixed_sparse_decode_grouped_h4_v,
    _mixed_sparse_prefill_stage1,
    _prefill_head_block_size,
    _use_grouped_h4_decode_qkv_split,
    oscar_mla_sparse_decode,
    oscar_mla_sparse_prefill,
    prepare_grouped_h4_score_workspace,
)
from vllm.v1.attention.ops.triton_oscar_mla_materialize import (
    OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
    OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,
    OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
    OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
    OSCAR_MTP_TEMPORAL_MAX_ROWS,
    _gather_oscar_mla_rows_kernel,
    allocate_oscar_bf16_materialization_workspace,
    allocate_oscar_mtp_direct_attention_cache,
    allocate_oscar_mtp_row576_temporal_cache,
    allocate_oscar_mtp_temporal_cache,
    allocate_oscar_mtp_temporal_cache_with_direct_storage,
    allocate_oscar_mtp_temporal_cache_with_split_direct_storage,
    can_use_oscar_bf16_materialized_read,
    commit_oscar_mla_direct_attention_misses,
    commit_oscar_mla_dual_source_attention_misses,
    materialize_oscar_mla_bf16_rows,
    materialize_oscar_mla_bf16_rows_direct_attention,
    materialize_oscar_mla_bf16_rows_temporal,
    prepare_oscar_mtp_temporal_workspace,
    reset_oscar_mtp_temporal_cache,
    restore_oscar_mla_hp_rows,
    seed_oscar_mtp_temporal_cache_recent,
    seed_oscar_mtp_temporal_cache_rows,
)
from vllm.v1.attention.ops.triton_oscar_mla_store import (
    oscar_mla_dequantize_history,
    oscar_mla_rotate_quantize_store,
    oscar_mla_store_bf16,
)
from vllm.v1.attention.ops.triton_sparse_mla_kernel import (
    triton_sparse_mla_attention,
    triton_sparse_mla_attention_dual_source,
)

RUN_CUDA_TESTS = os.environ.get("VLLM_OSCAR_RUN_CUDA_TESTS") == "1"
requires_cuda = pytest.mark.skipif(
    not RUN_CUDA_TESTS or not torch.cuda.is_available(),
    reason="set VLLM_OSCAR_RUN_CUDA_TESTS=1 on an authorized idle GPU",
)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("capability_major", 9),
        ("num_requests", 2),
        ("num_heads", 4),
        ("latent_rank", 128),
        ("rope_head_size", 32),
        ("group_size", 64),
        ("prefix_tokens", 32),
        ("recent_tokens", 128),
        ("topk", 1024),
    ],
)
def test_bf16_materialized_read_guard_is_exact(field: str, value: int) -> None:
    target = {
        "capability_major": 8,
        "num_requests": 1,
        "num_heads": 8,
        "latent_rank": 512,
        "rope_head_size": 64,
        "group_size": 128,
        "prefix_tokens": 64,
        "recent_tokens": 256,
        "topk": 2048,
    }
    assert can_use_oscar_bf16_materialized_read(**target)
    target[field] = value
    assert not can_use_oscar_bf16_materialized_read(**target)


def test_bf16_materialized_read_supports_tp2_local_heads() -> None:
    target = {
        "capability_major": 8,
        "num_requests": 1,
        "num_heads": 32,
        "latent_rank": 512,
        "rope_head_size": 64,
        "group_size": 128,
        "prefix_tokens": 64,
        "recent_tokens": 256,
        "topk": 2048,
    }
    assert can_use_oscar_bf16_materialized_read(**target)


def test_bf16_materialized_read_production_route_is_chunked_and_shared() -> None:
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    init_source = inspect.getsource(TritonMLASparseImpl.__init__)
    forward_source = inspect.getsource(TritonMLASparseImpl.forward_mqa)
    assert "prepare_oscar_bf16_materialization_workspace" in init_source
    assert "can_use_oscar_bf16_materialized_read" in forward_source
    assert forward_source.count("or oscar.max_seq_len is None") == 1
    assert "<= OSCAR_BF16_MATERIALIZATION_MAX_ROWS" in forward_source
    assert "attn_metadata.max_seq_len >= topk_width" not in forward_source
    assert "use_selected_incremental_materialization" in forward_source
    assert "attn_metadata.full_topk_start <= 0" in forward_source
    assert "num_actual_toks * topk_width" in forward_source
    assert "if use_selected_materialization else None" in forward_source
    assert "remapped_indices.view(num_actual_toks, 1, topk_width)" in forward_source
    assert "for row_offset in range(" in forward_source
    assert "row_offset=row_offset" in forward_source
    assert "save_and_remap_oscar_topk_for_chunk(" in forward_source
    assert "restore_oscar_topk_after_chunk(selected, saved_indices)" in forward_source
    assert "merge_oscar_chunked_attention_states(" in forward_source
    assert forward_source.index("restore_oscar_topk_after_chunk") < (
        forward_source.index("merge_oscar_chunked_attention_states")
    )
    assert "triton_sparse_mla_attention(" in forward_source
    assert "assume_valid_indices=False" in forward_source
    assert (
        forward_source.count("inverse_rotation=layer._oscar_inverse_rotation_bf16") == 4
    )
    assert "use_mtp_temporal_materialization" in forward_source
    assert "materialize_oscar_mla_bf16_rows_temporal(" in forward_source
    assert "allocate_oscar_mtp_temporal_cache_with_split_direct_storage" in init_source
    assert "allocate_oscar_mtp_row576_temporal_cache" in init_source
    assert "use_mtp_direct_cache_attention" in forward_source
    assert "materialize_oscar_mla_bf16_rows_direct_attention(" in forward_source
    assert "commit_oscar_mla_direct_attention_misses(" in forward_source
    assert "use_mtp_dual_source_attention" in forward_source
    assert "dual_source_attention=(" in forward_source
    assert "triton_sparse_mla_attention_dual_source(" in forward_source
    assert "commit_oscar_mla_dual_source_attention_misses(" in forward_source
    query_positions_assignment = "self._oscar_query_positions("
    assert forward_source.count(query_positions_assignment) == 1
    assert forward_source.index("self.oscar_read_calls += 1") < forward_source.index(
        query_positions_assignment
    )
    assert forward_source.index(query_positions_assignment) < forward_source.index(
        "return oscar_mla_sparse_prefill("
    )
    assert OSCAR_BF16_MATERIALIZATION_MAX_ROWS == 32769


def test_mtp_dual_source_attention_uses_tp8_head_geometry() -> None:
    source = inspect.getsource(
        triton_sparse_mla_kernel.triton_sparse_mla_attention_dual_source
    )

    assert triton_sparse_mla_kernel._DUAL_SOURCE_BLOCK_H == 8
    assert triton_sparse_mla_kernel._DUAL_SOURCE_NUM_WARPS == 8
    assert "BLOCK_H=_DUAL_SOURCE_BLOCK_H" in source
    assert "num_warps=_DUAL_SOURCE_NUM_WARPS" in source


def test_mtp_dual_source_attention_has_dedicated_kv_reuse_switch() -> None:
    module_source = inspect.getsource(triton_sparse_mla_kernel)
    source = inspect.getsource(
        triton_sparse_mla_kernel.triton_sparse_mla_attention_dual_source
    )

    assert "_DUAL_SOURCE_REUSE_K_AS_V = _env_flag(" in module_source
    assert '"VLLM_OSCAR_MTP_DUAL_SOURCE_REUSE_K_AS_V"' in module_source
    assert "REUSE_K_AS_V=_DUAL_SOURCE_REUSE_K_AS_V" in source
    assert "REUSE_K_AS_V=_REUSE_K_AS_V," not in source


def test_bf16_materialization_workspace_is_bounded() -> None:
    reference = torch.empty((2, 2048), dtype=torch.int32)
    workspace = allocate_oscar_bf16_materialization_workspace(reference, max_rows=3)

    assert [tensor.shape for tensor in workspace] == [
        (16, 512),
        (3,),
        (3, 576),
        (3,),
        (2, 8, 512),
        (2, 8),
        (2, 8),
    ]
    assert [tensor.dtype for tensor in workspace] == [
        torch.bfloat16,
        torch.uint8,
        torch.bfloat16,
        torch.int32,
        torch.bfloat16,
        torch.float32,
        torch.float32,
    ]

    tp2_workspace = allocate_oscar_bf16_materialization_workspace(
        reference, max_rows=3, num_heads=32
    )
    assert tp2_workspace[4].shape == (2, 32, 512)
    assert tp2_workspace[5].shape == (2, 32)
    assert tp2_workspace[6].shape == (2, 32)
    assert all(tensor.is_contiguous() for tensor in workspace)


def test_mtp_temporal_cache_is_per_layer_and_workspace_is_shared() -> None:
    reference = torch.empty((192, 2048), dtype=torch.int32)
    first_cache = allocate_oscar_mtp_temporal_cache(reference)
    second_cache = allocate_oscar_mtp_temporal_cache(reference)
    first_workspace = prepare_oscar_mtp_temporal_workspace(reference)
    second_workspace = prepare_oscar_mtp_temporal_workspace(reference)

    assert first_cache[0].shape == (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
    assert first_cache[0].dtype == torch.bfloat16
    assert first_cache[1].shape == (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,)
    assert first_cache[1].dtype == torch.int32
    assert first_cache[0].data_ptr() != second_cache[0].data_ptr()
    assert first_cache[1].data_ptr() != second_cache[1].data_ptr()
    assert first_workspace is second_workspace
    assert [tensor.shape for tensor in first_workspace] == [
        (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,),
        (OSCAR_MTP_TEMPORAL_MAX_POSITIONS,),
        (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,),
        (OSCAR_MTP_TEMPORAL_MAX_ROWS,),
        (OSCAR_MTP_TEMPORAL_MAX_ROWS,),
        (1,),
        (OSCAR_MTP_TEMPORAL_MAX_ROWS,),
    ]
    first_cache[1].fill_(7)
    reset_oscar_mtp_temporal_cache(first_cache)
    assert torch.equal(first_cache[1], torch.full_like(first_cache[1], -1))


def test_mtp_temporal_two_way_is_explicit_same_capacity_route() -> None:
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    assert triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT == 4096
    assert triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT == 1 << 30
    assert (
        triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK
        == (1 << 30) - 1
    )
    assert OSCAR_MTP_TEMPORAL_CACHE_CAPACITY == 2 * 4096
    materialize_parameters = inspect.signature(
        materialize_oscar_mla_bf16_rows_temporal
    ).parameters
    commit_parameters = inspect.signature(
        commit_oscar_mla_dual_source_attention_misses
    ).parameters
    assert materialize_parameters["two_way"].default is False
    assert commit_parameters["two_way"].default is False
    init_source = inspect.getsource(TritonMLASparseImpl.__init__)
    forward_source = inspect.getsource(TritonMLASparseImpl.forward_mqa)
    assert "_OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED" in init_source
    assert forward_source.count("two_way=_OSCAR_MTP_TEMPORAL_TWO_WAY_ENABLED") == 2


def test_mtp_direct_attention_cache_has_contiguous_miss_tail() -> None:
    reference = torch.empty((192, 2048), dtype=torch.int32)
    values, tags = allocate_oscar_mtp_direct_attention_cache(reference)

    assert values.shape == (
        OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS,
        576,
    )
    assert values.dtype == torch.bfloat16
    assert values.is_contiguous()
    assert tags.shape == (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,)
    assert tags.dtype == torch.int32
    assert torch.equal(tags, torch.full_like(tags, -1))


def test_mtp_temporal_cache_can_use_direct_sized_storage() -> None:
    reference = torch.empty((192, 2048), dtype=torch.int32)
    values, tags = allocate_oscar_mtp_temporal_cache_with_direct_storage(reference)

    assert values.shape == (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
    assert values.dtype == torch.bfloat16
    assert values.stride() == (512, 1)
    assert values.is_contiguous()
    assert (
        values.untyped_storage().nbytes()
        == (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS)
        * 576
        * values.element_size()
    )
    assert tags.shape == (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,)
    assert torch.equal(tags, torch.full_like(tags, -1))


def test_mtp_temporal_cache_can_split_direct_sized_storage() -> None:
    reference = torch.empty((192, 2048), dtype=torch.int32)
    values, tags = allocate_oscar_mtp_temporal_cache_with_split_direct_storage(
        reference
    )

    direct_value_elements = (
        OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS
    ) * 576
    temporal_value_elements = OSCAR_MTP_TEMPORAL_CACHE_CAPACITY * 512
    padding_int32_elements = (direct_value_elements - temporal_value_elements) // 2
    direct_total_bytes = (
        direct_value_elements * torch.empty((), dtype=torch.bfloat16).element_size()
        + OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY
        * torch.empty((), dtype=torch.int32).element_size()
    )

    assert values.shape == (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
    assert values.dtype == torch.bfloat16
    assert values.stride() == (512, 1)
    assert values.is_contiguous()
    assert values.untyped_storage().nbytes() == (
        temporal_value_elements * values.element_size()
    )
    assert tags.shape == (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,)
    assert tags.dtype == torch.int32
    assert tags.storage_offset() == padding_int32_elements
    assert torch.equal(tags, torch.full_like(tags, -1))
    assert (
        values.untyped_storage().nbytes() + tags.untyped_storage().nbytes()
        == direct_total_bytes
    )


def test_mtp_temporal_cache_can_use_row576_direct_sized_storage() -> None:
    reference = torch.empty((192, 2048), dtype=torch.int32)
    values, tags = allocate_oscar_mtp_row576_temporal_cache(reference)

    direct_value_elements = (
        OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY + OSCAR_MTP_TEMPORAL_MAX_ROWS
    ) * 576
    cache_value_elements = OSCAR_MTP_TEMPORAL_CACHE_CAPACITY * 576
    padding_int32_elements = (direct_value_elements - cache_value_elements) // 2
    direct_total_bytes = (
        direct_value_elements * torch.empty((), dtype=torch.bfloat16).element_size()
        + OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY
        * torch.empty((), dtype=torch.int32).element_size()
    )

    assert values.shape == (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY, 512)
    assert values.dtype == torch.bfloat16
    assert values.stride() == (576, 1)
    assert not values.is_contiguous()
    assert values.untyped_storage().nbytes() == (
        cache_value_elements * values.element_size()
    )
    assert tags.shape == (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,)
    assert tags.dtype == torch.int32
    assert tags.storage_offset() == padding_int32_elements
    assert torch.equal(tags, torch.full_like(tags, -1))
    assert (
        values.untyped_storage().nbytes() + tags.untyped_storage().nbytes()
        == direct_total_bytes
    )


@requires_cuda
def test_mtp_temporal_prefill_seed_kernels_preserve_exact_bf16_rows() -> None:
    device = torch.device("cuda:0")
    reference = torch.empty((6, 2048), dtype=torch.int32, device=device)
    cache = allocate_oscar_mtp_temporal_cache(reference)
    reset_oscar_mtp_temporal_cache(cache)

    row_values = torch.arange(6 * 512, dtype=torch.float32, device=device).view(6, 512)
    row_values = (row_values / 32).to(torch.bfloat16)
    row_positions = torch.tensor([64, 65, 66, 67, 68, 69], device=device)
    row_valid = torch.tensor([True, False, True, True, False, True], device=device)
    seed_oscar_mtp_temporal_cache_rows(
        row_values,
        row_positions,
        row_valid,
        cache,
    )

    recent = torch.arange(2 * 261 * 512, dtype=torch.float32, device=device)
    recent = (recent.view(2, 261, 512) / 64).to(torch.bfloat16)
    recent_positions = torch.tensor([320, 321, 322, 323], device=device)
    recent_hp_rows = torch.tensor([1, 0, -1, 1], device=device)
    recent_page_ids = torch.tensor([7, 8, 9, -1], device=device)
    seed_oscar_mtp_temporal_cache_recent(
        recent,
        recent_positions,
        recent_hp_rows,
        recent_page_ids,
        cache,
        prefix_tokens=64,
    )
    torch.accelerator.synchronize()

    cache_values, cache_tags = cache
    for row in (0, 2, 3, 5):
        position = int(row_positions[row].item())
        assert int(cache_tags[position % OSCAR_MTP_TEMPORAL_CACHE_CAPACITY]) == position
        assert torch.equal(
            cache_values[position % OSCAR_MTP_TEMPORAL_CACHE_CAPACITY],
            row_values[row],
        )
    for row in (0, 1):
        position = int(recent_positions[row].item())
        hp_row = int(recent_hp_rows[row].item())
        recent_slot = (position - 64) % recent.shape[1]
        assert int(cache_tags[position % OSCAR_MTP_TEMPORAL_CACHE_CAPACITY]) == position
        assert torch.equal(
            cache_values[position % OSCAR_MTP_TEMPORAL_CACHE_CAPACITY],
            recent[hp_row, recent_slot],
        )
    assert int(cache_tags[65]) == -1
    assert int(cache_tags[68]) == -1
    assert int(cache_tags[322]) == -1
    assert int(cache_tags[323]) == -1


@requires_cuda
def test_mtp_temporal_two_way_round_robin_collision_lifecycle() -> None:
    device = torch.device("cuda:0")
    set_count = triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_SET_COUNT
    state_bit = triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_STATE_BIT
    position_mask = (
        triton_oscar_mla_materialize.OSCAR_MTP_TEMPORAL_TWO_WAY_POSITION_MASK
    )
    miss_id_bits = triton_oscar_mla_materialize._OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_BITS
    miss_id_mask = triton_oscar_mla_materialize._OSCAR_MTP_TEMPORAL_TWO_WAY_MISS_ID_MASK
    max_rows = OSCAR_MTP_TEMPORAL_MAX_ROWS
    cache_values = torch.zeros(
        OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        512,
        dtype=torch.bfloat16,
        device=device,
    )
    cache_tags = torch.full(
        (OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,),
        -1,
        dtype=torch.int32,
        device=device,
    )
    position_owners = torch.empty(
        OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
        dtype=torch.int32,
        device=device,
    )
    position_to_miss = torch.empty_like(position_owners)
    cache_slot_owners = torch.empty(
        OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
        dtype=torch.int32,
        device=device,
    )
    miss_flags = torch.empty(max_rows, dtype=torch.uint8, device=device)
    miss_count = torch.zeros(1, dtype=torch.int32, device=device)
    miss_positions = torch.empty(max_rows, dtype=torch.int32, device=device)
    remapped = torch.empty(max_rows, dtype=torch.int32, device=device)
    hp_rows = torch.zeros(1, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([9000], dtype=torch.int32, device=device)

    def run_step(step_positions: list[int]) -> tuple[int, torch.Tensor]:
        positions = torch.tensor(step_positions, dtype=torch.int32, device=device)
        num_rows = positions.numel()
        position_owners.fill_(2_147_483_647)
        cache_slot_owners.fill_(-1)
        miss_count.zero_()
        triton_oscar_mla_materialize._mark_oscar_mtp_temporal_two_way_miss_owners_kernel[
            (1,)
        ](
            positions,
            cache_tags,
            position_owners,
            position_to_miss,
            cache_slot_owners,
            miss_flags,
            seq_lens,
            num_rows=num_rows,
            set_count=set_count,
            state_bit=state_bit,
            position_mask=position_mask,
            max_positions=OSCAR_MTP_TEMPORAL_MAX_POSITIONS,
            prefix_tokens=64,
            recent_tokens=256,
            block_size=256,
            num_warps=4,
            num_stages=1,
        )
        triton_oscar_mla_materialize._assign_oscar_mtp_temporal_two_way_unique_misses_kernel[
            (1,)
        ](
            positions,
            position_owners,
            miss_flags,
            miss_count,
            miss_positions,
            position_to_miss,
            num_rows=num_rows,
            miss_id_bits=miss_id_bits,
            block_size=256,
            num_warps=4,
            num_stages=1,
        )
        triton_oscar_mla_materialize._remap_oscar_mtp_two_way_rows_kernel[(1,)](
            positions,
            cache_tags,
            miss_flags,
            position_to_miss,
            hp_rows,
            seq_lens,
            remapped,
            num_rows=num_rows,
            cache_capacity=OSCAR_MTP_TEMPORAL_CACHE_CAPACITY,
            set_count=set_count,
            position_mask=position_mask,
            miss_id_mask=miss_id_mask,
            block_size=256,
            num_warps=4,
            num_stages=1,
        )
        torch.accelerator.synchronize()
        active_misses = int(miss_count.item())
        miss_values = torch.empty(
            max(active_misses, 1),
            512,
            dtype=torch.bfloat16,
            device=device,
        )
        for miss_id, position in enumerate(
            miss_positions[:active_misses].cpu().tolist()
        ):
            miss_values[miss_id].fill_(position // set_count + 1)
        triton_oscar_mla_materialize._commit_oscar_mtp_temporal_two_way_unique_misses_kernel[
            (1, 8)
        ](
            positions,
            position_to_miss,
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
            set_count=set_count,
            state_bit=state_bit,
            position_mask=position_mask,
            miss_id_bits=miss_id_bits,
            latent_rank=512,
            block_m=32,
            block_d=64,
            num_warps=8,
            num_stages=1,
        )
        torch.accelerator.synchronize()
        return active_misses, remapped[:num_rows].cpu()

    way0 = (64 % set_count) * 2
    first_misses, _ = run_step([64, 4160])
    assert first_misses == 2
    assert int(cache_tags[way0].item()) == 4160 | state_bit
    assert torch.equal(cache_values[way0], torch.full_like(cache_values[way0], 2))

    second_misses, _ = run_step([64])
    assert second_misses == 1
    assert int(cache_tags[way0].item()) == 4160
    assert int(cache_tags[way0 + 1].item()) == 64
    assert torch.equal(
        cache_values[way0 + 1],
        torch.full_like(cache_values[way0 + 1], 1),
    )

    hit_misses, hit_remapped = run_step([4160, 64])
    assert hit_misses == 0
    assert torch.equal(hit_remapped, torch.tensor([way0, way0 + 1]))

    replacement_misses, _ = run_step([8256])
    assert replacement_misses == 1
    assert int(cache_tags[way0].item()) == 8256 | state_bit
    assert int(cache_tags[way0 + 1].item()) == 64
    assert torch.equal(cache_values[way0], torch.full_like(cache_values[way0], 3))

    reset_oscar_mtp_temporal_cache((cache_values, cache_tags))
    assert torch.equal(cache_tags, torch.full_like(cache_tags, -1))


def test_sparse_mla_lse_out_preserves_existing_positional_signature() -> None:
    parameters = list(inspect.signature(triton_sparse_mla_attention).parameters)
    assert parameters[-1] == "lse_out"


@pytest.mark.parametrize(
    ("is_cuda", "is_capturing", "expected", "expected_capture_queries"),
    [(True, True, True, 1), (True, False, False, 1), (False, True, False, 0)],
)
def test_sparse_mla_cuda_graph_capture_bypasses_autotune(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda: bool,
    is_capturing: bool,
    expected: bool,
    expected_capture_queries: int,
) -> None:
    capture_queries = 0

    def current_stream_is_capturing() -> bool:
        nonlocal capture_queries
        capture_queries += 1
        return is_capturing

    monkeypatch.setattr(
        triton_sparse_mla_kernel,
        "_FINAL_STATIC_BY_TOKENS_ENABLED",
        False,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        current_stream_is_capturing,
    )

    tensor = cast(torch.Tensor, SimpleNamespace(is_cuda=is_cuda))
    assert triton_sparse_mla_kernel._use_static_final_kernel(tensor) is expected
    assert capture_queries == expected_capture_queries


def test_bf16_materializer_separates_active_requests_from_cache_capacity() -> None:
    materializer_source = inspect.getsource(materialize_oscar_mla_bf16_rows)
    assert "if num_requests != 1:" in materializer_source
    assert "if prefix.shape[0] < 1 or recent.shape[0] < 1:" in materializer_source
    assert "prefix.shape[0] != 1" not in materializer_source
    assert "if rope_block_table.shape[0] < 1" in materializer_source
    assert "rope_block_table.shape[0] != 1" not in materializer_source


def test_bf16_materializer_uses_gather_zero_fill_with_dense_addmm() -> None:
    kernel = getattr(
        _gather_oscar_mla_rows_kernel,
        "fn",
        _gather_oscar_mla_rows_kernel,
    )
    kernel_source = inspect.getsource(kernel)
    materializer_source = inspect.getsource(materialize_oscar_mla_bf16_rows)
    assert "tl.where(is_prefix | is_recent, bf16_values, 0.0)" in kernel_source
    assert "tl.where(is_history, (quantized - zero) * scale, 0.0)" in kernel_source
    assert "dense_output = output_kv[:num_rows, :512]" in materializer_source
    assert "torch.addmm(" in materializer_source
    assert "beta=1.0" in materializer_source
    assert "out=dense_output" in materializer_source


@pytest.mark.parametrize(
    ("num_heads", "expected"),
    [(1, 8), (8, 8), (9, 8), (16, 8), (17, 8), (32, 8)],
)
def test_prefill_head_block_size(num_heads: int, expected: int) -> None:
    assert _prefill_head_block_size(num_heads) == expected


def test_grouped_prefill_uses_causal_runtime_loop_bound() -> None:
    kernel = getattr(_mixed_sparse_prefill_stage1, "fn", _mixed_sparse_prefill_stage1)
    source = inspect.getsource(kernel)
    assert "effective_topk = tl.minimum(topk, causal_seq_len)" in source
    assert "tl.range(0, effective_topk, block_t)" in source


def test_grouped_prefill_tiles_large_local_head_counts() -> None:
    source = inspect.getsource(triton_oscar_mla_decode._oscar_mla_sparse_attention)
    assert "group_prefill_heads and num_splits == 1" in source
    assert "triton.cdiv(num_heads, block_h)" in source
    assert "num_warps=8 if use_grouped_prefill else 4" in source


def test_grouped_prefill_skips_zero_bf16_tile_dots() -> None:
    prefill_kernel = getattr(
        _mixed_sparse_prefill_stage1,
        "fn",
        _mixed_sparse_prefill_stage1,
    )
    decode_kernel = getattr(
        triton_oscar_mla_decode._mixed_sparse_decode_stage1,
        "fn",
        triton_oscar_mla_decode._mixed_sparse_decode_stage1,
    )
    source = inspect.getsource(prefill_kernel)
    decode_source = inspect.getsource(decode_kernel)
    assert "has_bf16 = tl.sum(is_bf16.to(tl.int32), axis=0) > 0" in source
    assert source.count("if has_bf16:") == 2
    assert "bf16_acc = bf16_acc * previous_scale[:, None]" in source
    assert "has_bf16" not in decode_source


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("enabled", False),
        ("num_queries", 2),
        ("num_requests", 2),
        ("num_heads", 4),
        ("latent_rank", 128),
        ("rope_head_size", 32),
        ("topk", 1024),
        ("num_splits", 8),
        ("prefix_tokens", 32),
        ("recent_tokens", 128),
    ],
)
def test_grouped_h4_decode_guard_is_exact(field: str, value: object) -> None:
    target: dict[str, object] = {
        "enabled": True,
        "num_queries": 1,
        "num_requests": 1,
        "num_heads": 8,
        "latent_rank": 512,
        "rope_head_size": 64,
        "topk": 2048,
        "num_splits": 16,
        "prefix_tokens": 64,
        "recent_tokens": 256,
    }
    assert _use_grouped_h4_decode_qkv_split(**target)
    target[field] = value
    assert not _use_grouped_h4_decode_qkv_split(**target)


def test_grouped_h4_qkv_split_uses_independent_pool_dots_and_shards_v() -> None:
    qk_kernel = getattr(
        _mixed_sparse_decode_grouped_h4_qk,
        "fn",
        _mixed_sparse_decode_grouped_h4_qk,
    )
    v_kernel = getattr(
        _mixed_sparse_decode_grouped_h4_v,
        "fn",
        _mixed_sparse_decode_grouped_h4_v,
    )
    qk_source = inspect.getsource(qk_kernel)
    v_source = inspect.getsource(v_kernel)
    attention_source = inspect.getsource(
        triton_oscar_mla_decode._oscar_mla_sparse_attention
    )

    assert "tl.static_range(0, 4)" in qk_source
    assert "for head_offset" in qk_source
    assert "tl.dot" not in qk_source
    assert v_source.count("tl.dot(") == 2
    assert "bf16_dot = tl.dot(" in v_source
    assert "history_dot = tl.dot(" in v_source
    assert "probabilities.to(tl.bfloat16)" in v_source
    assert "bf16_only.to(tl.bfloat16)" in v_source
    assert "history_only.to(tl.bfloat16)" in v_source
    assert v_source.count("dtype=tl.float32") == 8
    for pool in ("bf16", "history"):
        for head in range(4):
            assert (
                f"{pool}_acc{head} = {pool}_acc{head} * r{head} + {pool}_dot{head}"
            ) in v_source
            assert f"p{head}[:, None] * {pool}_only" not in v_source
    assert "tl.exp2" not in qk_source + v_source
    assert qk_source.count("tl.sum(") >= 3
    assert v_source.count("tl.sum(") == 5
    assert v_source.count("mask=d_shard == 0") == 4
    assert "split = split_and_shard // num_d_shards" in v_source
    assert "d_shard = split_and_shard % num_d_shards" in v_source
    assert "num_splits * num_d_shards" in attention_source
    assert attention_source.index("_mixed_sparse_decode_grouped_h4_qk") < (
        attention_source.index("_mixed_sparse_decode_grouped_h4_v")
    )
    assert "if not use_grouped_h4_decode:" in attention_source


def test_grouped_h4_score_workspace_is_device_unique_and_contiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = torch.empty(1)
    key = (reference.device.type, reference.device.index)
    triton_oscar_mla_decode._grouped_h4_score_workspace_cache.pop(key, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first = prepare_grouped_h4_score_workspace(reference)
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: pytest.fail("cache hit must not query capture state"),
    )
    second = prepare_grouped_h4_score_workspace(reference)

    assert first is second
    assert first.data_ptr() == second.data_ptr()
    assert first.shape == (1, 8, 2048)
    assert first.dtype == torch.float32
    assert first.is_contiguous()
    assert first.numel() * first.element_size() == 65536


def test_grouped_h4_score_workspace_capture_miss_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = torch.empty(1, device="meta")
    key = (reference.device.type, reference.device.index)
    triton_oscar_mla_decode._grouped_h4_score_workspace_cache.pop(key, None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(RuntimeError, match="before CUDA Graph capture"):
        prepare_grouped_h4_score_workspace(reference)
    assert key not in triton_oscar_mla_decode._grouped_h4_score_workspace_cache


def test_grouped_h4_workspace_is_oscar_only() -> None:
    from vllm.v1.attention.backends.mla.triton_mla_sparse import (
        TritonMLASparseImpl,
    )

    source = inspect.getsource(TritonMLASparseImpl.__init__)
    dtype_guard = 'self.kv_cache_dtype == "oscar_mla_int2"'
    assert dtype_guard in source
    assert source.index(dtype_guard) < source.index(
        "prepare_grouped_h4_score_workspace"
    )


@pytest.mark.parametrize("num_heads", [0, -1])
def test_prefill_head_block_size_rejects_non_positive(num_heads: int) -> None:
    with pytest.raises(ValueError, match="num_heads must be positive"):
        _prefill_head_block_size(num_heads)


def test_grouped_routes_are_forwarded_by_decode_and_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    sentinel = (torch.empty(0), torch.empty(0))

    def record_attention(*args: object, **kwargs: object) -> tuple[torch.Tensor, ...]:
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        triton_oscar_mla_decode,
        "_oscar_mla_sparse_attention",
        record_attention,
    )
    query = torch.empty((1, 1, 1))
    dummy = torch.empty(1)
    seq_lens = torch.ones(1, dtype=torch.int32)

    triton_oscar_mla_decode.oscar_mla_sparse_decode(
        query,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        seq_lens,
        dummy,
        num_splits=16,
    )
    triton_oscar_mla_decode.oscar_mla_sparse_prefill(
        query,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        seq_lens,
        dummy,
        num_splits=1,
    )

    assert calls[0]["num_splits"] == 16
    assert "group_prefill_heads" not in calls[0]
    assert calls[0]["group_decode_h4"] is True
    assert calls[1]["num_splits"] == 1
    assert calls[1]["group_prefill_heads"] is True
    assert calls[1]["group_decode_h4"] is False


def test_triton_interpreter_smoke() -> None:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "PYTHONPATH": str(Path(__file__).parents[2]),
            "TRANSFORMERS_OFFLINE": "1",
            "TRITON_INTERPRET": "1",
        }
    )
    script = Path(__file__).with_name("triton_interpreter_smoke.py")
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "latent_rank=512" in completed.stdout
    assert "max_error=" in completed.stdout
    assert "lse_max_error=" in completed.stdout
    assert "prefill_max_error=" in completed.stdout
    assert "prefill_lse_max_error=" in completed.stdout
    assert "multi_request_max_error=" in completed.stdout
    assert "multi_request_lse_max_error=" in completed.stdout
    assert "mtp5_materialize_max_error=" in completed.stdout


def _rotation(dim: int, *, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(41)
    matrix = torch.randn(dim, dim, generator=generator, device=device)
    q, _ = torch.linalg.qr(matrix.float())
    return q.to(torch.bfloat16)


def _assert_oracle(
    output: torch.Tensor,
    lse: torch.Tensor,
    expected: torch.Tensor,
    expected_lse: torch.Tensor,
    *,
    label: str,
) -> None:
    output_error = (output - expected).abs()
    lse_error = (lse - expected_lse).abs()
    print(
        f"{label} "
        f"output_max_error={output_error.max().item()} "
        f"output_mean_error={output_error.mean().item()} "
        f"lse_max_error={lse_error.max().item()} "
        f"lse_mean_error={lse_error.mean().item()}"
    )
    torch.testing.assert_close(output, expected, atol=0.5, rtol=0.03)
    torch.testing.assert_close(lse, expected_lse, atol=0.05, rtol=0.01)


def _pack_rope_cache(
    values: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks = (values.shape[0] + block_size - 1) // block_size
    cache = torch.zeros(
        num_blocks,
        block_size,
        values.shape[1],
        dtype=torch.bfloat16,
        device=values.device,
    )
    cache.view(-1, values.shape[1])[: values.shape[0]].copy_(values)
    block_table = torch.arange(
        num_blocks,
        dtype=torch.int32,
        device=values.device,
    ).unsqueeze(0)
    return cache, block_table


@requires_cuda
def test_restore_cached_bf16_windows_from_canonical_int2() -> None:
    device = torch.device("cuda")
    dim = 512
    block_size = 16
    final_seq_len = 337
    cache_hit_len = 320
    rotation = _rotation(dim, device=device)
    latent = torch.randn(
        final_seq_len,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    num_pages = (final_seq_len + block_size - 1) // block_size
    data = torch.zeros(
        num_pages,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.zeros(
        num_pages,
        block_size,
        4,
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros_like(scale)
    all_positions = torch.arange(final_seq_len, dtype=torch.int32, device=device)
    oscar_mla_rotate_quantize_store(
        latent,
        rotation,
        data,
        scale,
        zero,
        all_positions // block_size,
        all_positions % block_size,
        clip_ratio=0.96,
    )

    restore_position_list = list(range(64)) + list(range(81, cache_hit_len))
    positions = torch.tensor(restore_position_list, dtype=torch.int32, device=device)
    hp_rows = torch.zeros_like(positions)
    prefix = torch.full(
        (1, 64, dim),
        torch.nan,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.full(
        (1, 259, dim),
        torch.nan,
        dtype=torch.bfloat16,
        device=device,
    )
    history_rotated = torch.empty(
        len(restore_position_list),
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    restored = torch.empty(
        len(restore_position_list),
        576,
        dtype=torch.bfloat16,
        device=device,
    )
    page_ids = positions // block_size
    page_offsets = positions % block_size

    restore_oscar_mla_hp_rows(
        positions=positions,
        hp_rows=hp_rows,
        page_ids=page_ids,
        page_offsets=page_offsets,
        num_rows=positions.numel(),
        history_data=data,
        history_scale=scale,
        history_zero=zero,
        prefix=prefix,
        recent=recent,
        inverse_rotation=rotation.T.contiguous(),
        history_rotated=history_rotated,
        restored=restored,
    )

    dequantized = oscar_mla_dequantize_history(
        data,
        scale,
        zero,
        page_ids,
        page_offsets,
    ).to(torch.bfloat16)
    expected = dequantized @ rotation.T.contiguous()
    torch.testing.assert_close(prefix[0], expected[:64], atol=0, rtol=0)
    recent_indices = (positions[64:] - 64) % recent.shape[1]
    torch.testing.assert_close(
        recent[0, recent_indices.long()],
        expected[64:],
        atol=0,
        rtol=0,
    )
    assert torch.isnan(recent[0, :17]).all()


@requires_cuda
@pytest.mark.parametrize("seq_len", [63, 64, 319, 320, 321])
@pytest.mark.parametrize("num_heads", [1, 4, 8])
def test_sparse_decode_matches_three_pool_oracle(
    seq_len: int,
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    dim = 512
    group_size = 128
    prefix_tokens = 64
    recent_tokens = 256
    block_size = 16
    generator = torch.Generator(device=device).manual_seed(53 + seq_len + num_heads)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query = torch.randn(
        1,
        num_heads,
        dim,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    query_rope = torch.randn(
        1,
        num_heads,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, block_size)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(
        1,
        prefix_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.zeros(
        1,
        recent_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    final_lens = torch.full_like(positions, seq_len)
    hp_rows_for_tokens = torch.zeros_like(positions)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        final_lens,
        hp_rows_for_tokens,
    )

    history_start = prefix_tokens
    history_end = max(prefix_tokens, seq_len - recent_tokens)
    history_len = history_end - history_start
    history_pages = max(1, (history_len + block_size - 1) // block_size)
    data = torch.zeros(
        history_pages,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    scale = torch.zeros(
        history_pages,
        block_size,
        dim // group_size,
        dtype=torch.float32,
        device=device,
    )
    zero = torch.zeros_like(scale)
    if history_len:
        history_latent = latent[history_start:history_end]
        history_indices = torch.arange(
            history_len,
            dtype=torch.int32,
            device=device,
        )
        page_ids = history_indices // block_size
        page_offsets = history_indices % block_size
        oscar_mla_rotate_quantize_store(
            history_latent,
            rotation,
            data,
            scale,
            zero,
            page_ids,
            page_offsets,
            clip_ratio=0.96,
        )
        history_rotated = oscar_mla_dequantize_history(
            data,
            scale,
            zero,
            page_ids,
            page_offsets,
        )
    else:
        history_rotated = torch.empty(
            0,
            dim,
            dtype=torch.float32,
            device=device,
        )

    if num_heads == 8:
        selected = torch.full((1, 2048), -1, dtype=torch.int32, device=device)
        selected[0, :seq_len] = torch.arange(
            seq_len,
            dtype=torch.int32,
            device=device,
        )
        num_splits = 16
    else:
        selected = torch.arange(seq_len, dtype=torch.int32, device=device).unsqueeze(0)
        num_splits = 4
    history_page_table = torch.arange(
        history_pages,
        dtype=torch.int32,
        device=device,
    ).unsqueeze(0)
    hp_rows = torch.zeros(1, dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        data,
        scale,
        zero,
        history_page_table,
        hp_rows,
        seq_lens,
        rotation,
        num_splits=num_splits,
    )

    prefix_end = min(seq_len, prefix_tokens)
    recent_start = max(prefix_end, seq_len - recent_tokens)
    expected, expected_lse = mixed_latent_attention_with_lse(
        query.float(),
        prefix_latent=latent[:prefix_end].float(),
        recent_latent=latent[recent_start:].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
        query_rope=query_rope.float(),
        prefix_rope=rope_values[:prefix_end].float(),
        history_rope=rope_values[prefix_end:recent_start].float(),
        recent_rope=rope_values[recent_start:].float(),
    )
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label=f"decode_heads={num_heads}_seq={seq_len}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


@requires_cuda
def test_sparse_decode_respects_selected_token_ids() -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    generator = torch.Generator(device=device).manual_seed(71)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        1,
        1,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        1,
        1,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, 16)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(1, 64, dim, dtype=torch.bfloat16, device=device)
    recent = torch.zeros(1, 256, dim, dtype=torch.bfloat16, device=device)
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        torch.zeros_like(positions),
    )
    data = torch.zeros(1, 16, dim // 4, dtype=torch.uint8, device=device)
    scale = torch.zeros(1, 16, 4, dtype=torch.float32, device=device)
    zero = torch.zeros_like(scale)
    oscar_mla_rotate_quantize_store(
        latent[64:65],
        rotation,
        data,
        scale,
        zero,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        clip_ratio=1.0,
    )
    history_rotated = oscar_mla_dequantize_history(
        data,
        scale,
        zero,
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
    )
    selected = torch.tensor([[0, 64, 320]], dtype=torch.int32, device=device)

    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        data,
        scale,
        zero,
        torch.zeros(1, 1, dtype=torch.int32, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        rotation,
        num_splits=3,
    )

    expected, expected_lse = mixed_latent_attention_with_lse(
        query.float(),
        prefix_latent=latent[0:1].float(),
        recent_latent=latent[320:321].float(),
        history_rotated=history_rotated,
        rotation=rotation.float(),
        query_rope=query_rope.float(),
        prefix_rope=rope_values[0:1].float(),
        history_rope=rope_values[64:65].float(),
        recent_rope=rope_values[320:321].float(),
    )
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label="decode_selected_ids",
    )


@requires_cuda
@pytest.mark.parametrize("batch_size", [4, 8])
def test_sparse_decode_isolates_batched_requests(batch_size: int) -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    block_size = 16
    blocks_per_request = (seq_len + block_size - 1) // block_size
    generator = torch.Generator(device=device).manual_seed(83 + batch_size)
    latent = torch.randn(
        batch_size,
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        batch_size,
        1,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        batch_size,
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        batch_size,
        1,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(
        batch_size,
        64,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    recent = torch.zeros(
        batch_size,
        256,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device).repeat(
        batch_size
    )
    hp_rows_for_tokens = torch.arange(
        batch_size,
        dtype=torch.int32,
        device=device,
    ).repeat_interleave(seq_len)
    oscar_mla_store_bf16(
        latent.flatten(0, 1),
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        hp_rows_for_tokens,
    )

    rope_cache = torch.zeros(
        batch_size * blocks_per_request,
        block_size,
        64,
        dtype=torch.bfloat16,
        device=device,
    )
    for request_index in range(batch_size):
        request_cache = rope_cache[
            request_index * blocks_per_request : (request_index + 1)
            * blocks_per_request
        ]
        request_cache.view(-1, 64)[:seq_len].copy_(rope_values[request_index])
    rope_block_table = torch.arange(
        batch_size * blocks_per_request,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, blocks_per_request)

    history_data = torch.zeros(
        batch_size,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    history_scale = torch.zeros(
        batch_size,
        block_size,
        dim // 128,
        dtype=torch.float32,
        device=device,
    )
    history_zero = torch.zeros_like(history_scale)
    page_ids = torch.arange(batch_size, dtype=torch.int32, device=device)
    page_offsets = torch.zeros(batch_size, dtype=torch.int32, device=device)
    oscar_mla_rotate_quantize_store(
        latent[:, 64],
        rotation,
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
        clip_ratio=0.96,
    )
    history_rotated = oscar_mla_dequantize_history(
        history_data,
        history_scale,
        history_zero,
        page_ids,
        page_offsets,
    )
    selected = torch.tensor(
        [0, 64, 320],
        dtype=torch.int32,
        device=device,
    ).repeat(batch_size, 1)
    output, lse = oscar_mla_sparse_decode(
        query,
        query_rope,
        selected,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        page_ids.unsqueeze(1),
        torch.arange(batch_size, dtype=torch.int32, device=device),
        torch.full((batch_size,), seq_len, dtype=torch.int32, device=device),
        rotation,
        num_splits=3,
    )

    expected_rows = []
    expected_lse_rows = []
    for request_index in range(batch_size):
        expected_row, expected_lse_row = mixed_latent_attention_with_lse(
            query[request_index : request_index + 1].float(),
            prefix_latent=latent[request_index, 0:1].float(),
            recent_latent=latent[request_index, 320:321].float(),
            history_rotated=history_rotated[request_index : request_index + 1],
            rotation=rotation.float(),
            query_rope=query_rope[request_index : request_index + 1].float(),
            prefix_rope=rope_values[request_index, 0:1].float(),
            history_rope=rope_values[request_index, 64:65].float(),
            recent_rope=rope_values[request_index, 320:321].float(),
        )
        expected_rows.append(expected_row)
        expected_lse_rows.append(expected_lse_row)
    _assert_oracle(
        output,
        lse,
        torch.cat(expected_rows),
        torch.cat(expected_lse_rows),
        label=f"decode_batch={batch_size}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()


@requires_cuda
@pytest.mark.parametrize("num_queries", [1, 4, 6, 8])
@pytest.mark.parametrize("num_heads", [1, 8])
def test_sparse_prefill_is_causal_and_matches_three_pool_oracle(
    num_queries: int,
    num_heads: int,
) -> None:
    device = torch.device("cuda")
    dim = 512
    seq_len = 321
    block_size = 16
    generator = torch.Generator(device=device).manual_seed(97 + num_queries)
    latent = torch.randn(
        seq_len,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query = torch.randn(
        num_queries,
        num_heads,
        dim,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_values = torch.randn(
        seq_len,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    query_rope = torch.randn(
        num_queries,
        num_heads,
        64,
        generator=generator,
        dtype=torch.bfloat16,
        device=device,
    )
    rope_cache, rope_block_table = _pack_rope_cache(rope_values, block_size)
    rotation = _rotation(dim, device=device)
    prefix = torch.zeros(1, 64, dim, dtype=torch.bfloat16, device=device)
    use_mtp5_physical_recent = num_queries == 6 and num_heads == 8
    recent_capacity = 261 if use_mtp5_physical_recent else 256
    recent = torch.zeros(
        1,
        recent_capacity,
        dim,
        dtype=torch.bfloat16,
        device=device,
    )
    positions = torch.arange(seq_len, dtype=torch.int32, device=device)
    oscar_mla_store_bf16(
        latent,
        prefix,
        recent,
        positions,
        torch.full_like(positions, seq_len),
        torch.zeros_like(positions),
        recent_tokens=256,
    )

    history_data = torch.zeros(
        1,
        block_size,
        dim // 4,
        dtype=torch.uint8,
        device=device,
    )
    history_scale = torch.zeros(
        1,
        block_size,
        dim // 128,
        dtype=torch.float32,
        device=device,
    )
    history_zero = torch.zeros_like(history_scale)
    zero_index = torch.zeros(1, dtype=torch.int32, device=device)
    oscar_mla_rotate_quantize_store(
        latent[64:65],
        rotation,
        history_data,
        history_scale,
        history_zero,
        zero_index,
        zero_index,
        clip_ratio=0.96,
    )
    history_rotated = oscar_mla_dequantize_history(
        history_data,
        history_scale,
        history_zero,
        zero_index,
        zero_index,
    )

    if num_queries == 1:
        query_positions = torch.tensor([320], dtype=torch.int32, device=device)
    elif num_queries == 4:
        query_positions = torch.tensor(
            [63, 64, 319, 320],
            dtype=torch.int32,
            device=device,
        )
    elif num_queries == 6:
        query_positions = torch.tensor(
            [315, 316, 317, 318, 319, 320],
            dtype=torch.int32,
            device=device,
        )
    else:
        query_positions = torch.tensor(
            [63, 64, 100, 150, 200, 250, 319, 320],
            dtype=torch.int32,
            device=device,
        )
    selected = positions.unsqueeze(0).expand(num_queries, -1)
    output, lse = oscar_mla_sparse_prefill(
        query,
        query_rope,
        selected,
        torch.zeros(num_queries, dtype=torch.int32, device=device),
        query_positions,
        prefix,
        recent,
        rope_cache,
        rope_block_table,
        history_data,
        history_scale,
        history_zero,
        zero_index.view(1, 1),
        zero_index,
        torch.tensor([seq_len], dtype=torch.int32, device=device),
        rotation,
        num_splits=1,
        recent_tokens=256,
        group_decode_h4=False,
    )

    expected_rows = []
    expected_lse_rows = []
    for row, query_position in enumerate(query_positions.tolist()):
        causal_length = query_position + 1
        expected_row, expected_lse_row = mixed_latent_attention_with_lse(
            query[row : row + 1].float(),
            prefix_latent=latent[: min(64, causal_length)].float(),
            recent_latent=latent[65:causal_length].float(),
            history_rotated=(
                history_rotated if causal_length > 64 else history_rotated[:0]
            ),
            rotation=rotation.float(),
            query_rope=query_rope[row : row + 1].float(),
            prefix_rope=rope_values[: min(64, causal_length)].float(),
            history_rope=(
                rope_values[64:65] if causal_length > 64 else rope_values[:0]
            ).float(),
            recent_rope=rope_values[65:causal_length].float(),
        )
        expected_rows.append(expected_row)
        expected_lse_rows.append(expected_lse_row)
    expected = torch.cat(expected_rows, dim=0)
    expected_lse = torch.cat(expected_lse_rows, dim=0)
    _assert_oracle(
        output,
        lse,
        expected,
        expected_lse,
        label=f"prefill_batch={num_queries}_heads={num_heads}",
    )
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()

    if use_mtp5_physical_recent:
        mtp5_selected = torch.full(
            (num_queries, 2048),
            -1,
            dtype=torch.int32,
            device=device,
        )
        for row, query_position in enumerate(query_positions.tolist()):
            mtp5_selected[row, : query_position + 1] = torch.arange(
                query_position + 1,
                dtype=torch.int32,
                device=device,
            )
        (
            materialized_history,
            materialized_mask,
            materialized_kv,
            remapped_indices,
            _,
            _,
            _,
        ) = allocate_oscar_bf16_materialization_workspace(
            torch.empty((192, 2048), dtype=torch.int32, device=device),
            OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
        )
        materialized_rows = num_queries * 2048
        kv, remapped = materialize_oscar_mla_bf16_rows(
            positions=mtp5_selected.reshape(-1),
            num_rows=materialized_rows,
            num_requests=1,
            prefix=prefix,
            recent=recent,
            rope=rope_cache,
            rope_block_table=rope_block_table,
            history_data=history_data,
            history_scale=history_scale,
            history_zero=history_zero,
            history_page_table=zero_index.view(1, 1),
            hp_rows=zero_index,
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
            inverse_rotation=rotation.T.contiguous(),
            history_rotated=materialized_history,
            history_mask=materialized_mask,
            output_kv=materialized_kv,
            remapped_indices=remapped_indices,
            recent_tokens=256,
        )
        reference_kv = kv.clone()
        reference_mask = materialized_mask[:materialized_rows].clone()
        reference_remapped = remapped.clone()
        (
            temporal_history,
            temporal_mask,
            temporal_kv,
            temporal_remapped,
            _,
            _,
            _,
        ) = allocate_oscar_bf16_materialization_workspace(
            torch.empty((192, 2048), dtype=torch.int32, device=device),
            OSCAR_BF16_MATERIALIZATION_MAX_ROWS,
        )
        temporal_workspace = prepare_oscar_mtp_temporal_workspace(mtp5_selected)
        # The three allocation-only diagnostic caches intentionally retain
        # direct-cache-sized tag backing and have standalone storage tests.
        for allocate_temporal_cache in (allocate_oscar_mtp_temporal_cache,):
            temporal_cache = allocate_temporal_cache(mtp5_selected)
            reset_oscar_mtp_temporal_cache(temporal_cache)
            for _ in range(2):
                kv, remapped = materialize_oscar_mla_bf16_rows_temporal(
                    positions=mtp5_selected.reshape(-1),
                    num_rows=materialized_rows,
                    num_requests=1,
                    prefix=prefix,
                    recent=recent,
                    rope=rope_cache,
                    rope_block_table=rope_block_table,
                    history_data=history_data,
                    history_scale=history_scale,
                    history_zero=history_zero,
                    history_page_table=zero_index.view(1, 1),
                    hp_rows=zero_index,
                    seq_lens=torch.tensor(
                        [seq_len],
                        dtype=torch.int32,
                        device=device,
                    ),
                    inverse_rotation=rotation.T.contiguous(),
                    history_rotated=temporal_history,
                    history_mask=temporal_mask,
                    output_kv=temporal_kv,
                    remapped_indices=temporal_remapped,
                    temporal_workspace=temporal_workspace,
                    temporal_cache=temporal_cache,
                    recent_tokens=256,
                )
                assert torch.equal(kv, reference_kv)
                assert torch.equal(
                    temporal_mask[:materialized_rows],
                    reference_mask,
                )
                assert torch.equal(remapped, reference_remapped)
        direct_cache = allocate_oscar_mtp_direct_attention_cache(mtp5_selected)
        direct_workspace = prepare_oscar_mtp_temporal_workspace(mtp5_selected)
        direct_workspace = (
            direct_workspace[0],
            direct_workspace[1],
            torch.empty(
                (OSCAR_MTP_DIRECT_CACHE_ATTENTION_CAPACITY,),
                dtype=torch.int32,
                device=device,
            ),
            *direct_workspace[3:],
        )
        reset_oscar_mtp_temporal_cache(direct_cache)
        direct_miss_counts = []
        for _ in range(2):
            direct_kv, direct_remapped = (
                materialize_oscar_mla_bf16_rows_direct_attention(
                    positions=mtp5_selected.reshape(-1),
                    num_rows=materialized_rows,
                    num_requests=1,
                    prefix=prefix,
                    recent=recent,
                    rope=rope_cache,
                    rope_block_table=rope_block_table,
                    history_data=history_data,
                    history_scale=history_scale,
                    history_zero=history_zero,
                    history_page_table=zero_index.view(1, 1),
                    hp_rows=zero_index,
                    seq_lens=torch.tensor(
                        [seq_len],
                        dtype=torch.int32,
                        device=device,
                    ),
                    inverse_rotation=rotation.T.contiguous(),
                    history_rotated=temporal_history,
                    remapped_indices=temporal_remapped,
                    temporal_workspace=direct_workspace,
                    direct_cache=direct_cache,
                    recent_tokens=256,
                )
            )
            direct_miss_counts.append(direct_workspace[5].item())
            flat_direct = direct_remapped.reshape(-1)
            flat_reference = reference_remapped.reshape(-1)
            valid_rows = flat_direct >= 0
            assert torch.equal(valid_rows, flat_reference >= 0)
            assert torch.equal(
                direct_kv[flat_direct[valid_rows].long(), 0],
                reference_kv[flat_reference[valid_rows].long(), 0],
            )
            direct_output, direct_lse = triton_sparse_mla_attention(
                (query, query_rope),
                direct_kv,
                direct_remapped.view(num_queries, 1, 2048),
                sm_scale=dim**-0.5,
                num_kv_splits=16,
                assume_valid_indices=False,
                return_lse=True,
            )
            commit_oscar_mla_direct_attention_misses(
                positions=mtp5_selected.reshape(-1),
                num_rows=materialized_rows,
                temporal_workspace=direct_workspace,
                direct_cache=direct_cache,
            )
            _assert_oracle(
                direct_output.float(),
                direct_lse,
                expected,
                expected_lse,
                label="mtp5_direct_cache_attention",
            )
        assert direct_miss_counts[1] < direct_miss_counts[0]
        dual_cache = allocate_oscar_mtp_temporal_cache(mtp5_selected)
        dual_workspace = prepare_oscar_mtp_temporal_workspace(mtp5_selected)
        reset_oscar_mtp_temporal_cache(dual_cache)
        dual_miss_counts = []
        dual_outputs = []
        dual_lses = []
        for _ in range(2):
            dual_miss_values, dual_remapped = materialize_oscar_mla_bf16_rows_temporal(
                positions=mtp5_selected.reshape(-1),
                num_rows=materialized_rows,
                num_requests=1,
                prefix=prefix,
                recent=recent,
                rope=rope_cache,
                rope_block_table=rope_block_table,
                history_data=history_data,
                history_scale=history_scale,
                history_zero=history_zero,
                history_page_table=zero_index.view(1, 1),
                hp_rows=zero_index,
                seq_lens=torch.tensor(
                    [seq_len],
                    dtype=torch.int32,
                    device=device,
                ),
                inverse_rotation=rotation.T.contiguous(),
                history_rotated=temporal_history,
                history_mask=temporal_mask,
                output_kv=temporal_kv,
                remapped_indices=temporal_remapped,
                temporal_workspace=dual_workspace,
                temporal_cache=dual_cache,
                recent_tokens=256,
                dual_source_attention=True,
                two_way=True,
            )
            dual_miss_counts.append(dual_workspace[5].item())
            flat_dual = dual_remapped.reshape(-1)
            flat_reference = reference_remapped.reshape(-1)
            valid_rows = flat_dual >= 0
            assert torch.equal(valid_rows, flat_reference >= 0)
            mapped = flat_dual[valid_rows].long()
            cache_hits = mapped < OSCAR_MTP_TEMPORAL_CACHE_CAPACITY
            resolved = torch.empty(
                (mapped.shape[0], 512),
                dtype=torch.bfloat16,
                device=device,
            )
            resolved[cache_hits] = dual_cache[0][mapped[cache_hits]]
            resolved[~cache_hits] = dual_miss_values[
                mapped[~cache_hits] - OSCAR_MTP_TEMPORAL_CACHE_CAPACITY
            ]
            assert torch.equal(
                resolved,
                reference_kv[flat_reference[valid_rows].long(), 0, :512],
            )
            original_reuse_k_as_v = triton_sparse_mla_kernel._DUAL_SOURCE_REUSE_K_AS_V
            try:
                triton_sparse_mla_kernel._DUAL_SOURCE_REUSE_K_AS_V = False
                reference_dual_output, reference_dual_lse = (
                    triton_sparse_mla_attention_dual_source(
                        (query, query_rope),
                        dual_cache[0],
                        dual_miss_values,
                        rope_cache,
                        rope_block_table,
                        mtp5_selected.view(num_queries, 1, 2048),
                        dual_remapped.view(num_queries, 1, 2048),
                        sm_scale=dim**-0.5,
                        num_kv_splits=16,
                    )
                )
                triton_sparse_mla_kernel._DUAL_SOURCE_REUSE_K_AS_V = True
                dual_output, dual_lse = triton_sparse_mla_attention_dual_source(
                    (query, query_rope),
                    dual_cache[0],
                    dual_miss_values,
                    rope_cache,
                    rope_block_table,
                    mtp5_selected.view(num_queries, 1, 2048),
                    dual_remapped.view(num_queries, 1, 2048),
                    sm_scale=dim**-0.5,
                    num_kv_splits=16,
                )
            finally:
                triton_sparse_mla_kernel._DUAL_SOURCE_REUSE_K_AS_V = (
                    original_reuse_k_as_v
                )
            assert torch.equal(dual_output, reference_dual_output)
            assert torch.equal(dual_lse, reference_dual_lse)
            dual_outputs.append(dual_output.clone())
            dual_lses.append(dual_lse.clone())
            commit_oscar_mla_dual_source_attention_misses(
                positions=mtp5_selected.reshape(-1),
                num_rows=materialized_rows,
                miss_values=dual_miss_values,
                temporal_workspace=dual_workspace,
                temporal_cache=dual_cache,
                two_way=True,
            )
            _assert_oracle(
                dual_output.float(),
                dual_lse,
                expected,
                expected_lse,
                label="mtp5_dual_source_attention",
            )
        assert dual_miss_counts[1] < dual_miss_counts[0]
        materialized_output, materialized_lse = triton_sparse_mla_attention(
            (query, query_rope),
            kv,
            reference_remapped.view(num_queries, 1, 2048),
            sm_scale=dim**-0.5,
            num_kv_splits=16,
            assume_valid_indices=False,
            return_lse=True,
        )
        valid_positions = mtp5_selected.reshape(-1) >= 0
        logical_positions = mtp5_selected.reshape(-1)[valid_positions].long()
        logical_blocks = logical_positions // block_size
        rope_offsets = logical_positions % block_size
        physical_blocks = rope_block_table[0, logical_blocks].long()
        resolved_rope = rope_cache[physical_blocks, rope_offsets]
        reference_rows = reference_remapped.reshape(-1)[valid_positions].long()
        reference_rope = reference_kv[reference_rows, 0, 512:]
        assert torch.equal(resolved_rope, reference_rope)
        dual_output_error = (
            dual_outputs[-1].float() - materialized_output.float()
        ).abs()
        dual_lse_error = (dual_lses[-1] - materialized_lse).abs()
        print(
            "C127_DUAL_NUMERIC_DIAG "
            f"output_equal={torch.equal(dual_outputs[-1], materialized_output)} "
            f"output_max_abs={dual_output_error.max().item():.9g} "
            f"output_diff_count={(dual_output_error != 0).sum().item()} "
            f"lse_equal={torch.equal(dual_lses[-1], materialized_lse)} "
            f"lse_max_abs={dual_lse_error.max().item():.9g} "
            f"lse_diff_count={(dual_lse_error != 0).sum().item()} "
            f"rounds_output_equal={torch.equal(dual_outputs[0], dual_outputs[1])} "
            f"rounds_lse_equal={torch.equal(dual_lses[0], dual_lses[1])}"
        )
        assert torch.equal(dual_outputs[-1], materialized_output)
        assert torch.equal(dual_lses[-1], materialized_lse)
        assert torch.equal(dual_outputs[0], dual_outputs[1])
        assert torch.equal(dual_lses[0], dual_lses[1])
        _assert_oracle(
            materialized_output.float(),
            materialized_lse,
            expected,
            expected_lse,
            label="mtp5_selected_materialization",
        )
