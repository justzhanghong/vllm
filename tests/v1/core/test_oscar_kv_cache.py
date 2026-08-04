# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random
from types import SimpleNamespace

import pytest
import torch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.platforms import current_platform
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.core.oscar_kv_cache import (
    OscarKVCacheCapacityError,
    OscarKVCacheGeometry,
    OscarKVPageAllocator,
    plan_oscar_kv_cache,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import OscarKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    OscarKVCacheSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.worker.gpu.attn_utils import _reshape_oscar_kv_cache
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.warmup import _set_oscar_warmup_hp_rows


def qwen3_geometry() -> OscarKVCacheGeometry:
    return OscarKVCacheGeometry(
        num_layers=36,
        num_kv_heads=8,
        head_size=128,
        value_head_size=128,
        quant_slot_size=72,
    )


def qwen3_spec(
    prefix_cache_extra_tokens: int = 0, quant_slot_size: int = 72
) -> OscarKVCacheSpec:
    return OscarKVCacheSpec(
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=torch.uint8,
        quant_slot_size=quant_slot_size,
        prefix_cache_extra_tokens=prefix_cache_extra_tokens,
    )


def test_oscar_spec_exposes_both_pool_page_sizes() -> None:
    spec = qwen3_spec()
    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=8192))

    assert spec.page_size_bytes == 9216
    assert spec.hp_page_size_bytes == 65536
    assert spec.max_memory_usage_bytes(config) == 512 * spec.page_size_bytes


def test_oscar_spec_merge_rejects_different_geometry() -> None:
    spec = qwen3_spec()

    assert OscarKVCacheSpec.merge([spec, spec]) == spec
    with pytest.raises(AssertionError, match="identical geometry"):
        OscarKVCacheSpec.merge([spec, spec.copy_with_new_block_size(block_size=32)])


def test_engine_config_accounts_for_all_three_pools() -> None:
    spec = qwen3_spec()
    layers = [f"model.layers.{i}.self_attn.attn" for i in range(36)]
    config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )

    result = get_kv_cache_config_from_groups(
        config,
        [KVCacheGroupSpec(layer_names=layers, kv_cache_spec=spec)],
        10 * 1024**3,
    )

    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, 1)
    assert result.num_blocks == plan.quant_pages
    assert len(result.kv_cache_tensors) == 36
    allocated = sum(tensor.size for tensor in result.kv_cache_tensors)
    assert allocated == plan.allocated_bytes


def test_engine_config_uses_spec_quant_slot_size() -> None:
    spec = qwen3_spec(quant_slot_size=72)
    layers = [f"model.layers.{i}.self_attn.attn" for i in range(36)]
    config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
    )
    budget = 10 * 1024**3

    result = get_kv_cache_config_from_groups(
        config,
        [KVCacheGroupSpec(layer_names=layers, kv_cache_spec=spec)],
        budget,
    )

    hp_bytes_per_layer = (64 + 256) * spec.num_kv_heads * 2 * spec.head_size * 2
    quant_page_bytes_all_layers = len(layers) * spec.page_size_bytes
    expected_num_blocks = (
        budget - len(layers) * hp_bytes_per_layer
    ) // quant_page_bytes_all_layers
    expected_per_layer = expected_num_blocks * spec.page_size_bytes + hp_bytes_per_layer
    assert result.num_blocks == expected_num_blocks
    assert all(tensor.size == expected_per_layer for tensor in result.kv_cache_tensors)


def test_worker_reshape_splits_accounted_backing_into_three_tensors() -> None:
    spec = qwen3_spec()
    num_blocks = 8
    max_num_seqs = 1
    quant_bytes = num_blocks * spec.page_size_bytes
    prefix_bytes = 4 * spec.hp_page_size_bytes
    recent_bytes = 16 * spec.hp_page_size_bytes
    raw = torch.zeros(quant_bytes + prefix_bytes + recent_bytes, dtype=torch.int8)

    quant, prefix, recent = _reshape_oscar_kv_cache(
        raw,
        spec,
        (num_blocks, 16, 8, 72),
        (0, 1, 2, 3),
        num_blocks,
        max_num_seqs,
    )

    assert quant.shape == (8, 16, 8, 72)
    assert prefix.shape == (64, 8, 2, 128)
    assert recent.shape == (256, 8, 2, 128)
    assert quant.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert prefix.storage_offset() * prefix.element_size() == quant_bytes
    assert recent.storage_offset() * recent.element_size() == quant_bytes + prefix_bytes
    assert quant.numel() + prefix.numel() * 2 + recent.numel() * 2 == raw.numel()


@pytest.mark.parametrize(
    ("max_num_seqs", "prefix_slots", "recent_slots", "quant_slots", "guaranteed"),
    [
        (1, 64, 256, 515536, 515521),
        (8, 512, 2048, 499600, 499480),
        (48, 3072, 12288, 408576, 407856),
    ],
)
def test_qwen3_10_gib_plan(
    max_num_seqs: int,
    prefix_slots: int,
    recent_slots: int,
    quant_slots: int,
    guaranteed: int,
) -> None:
    geometry = qwen3_geometry()
    plan = plan_oscar_kv_cache(geometry, 10 * 1024**3, max_num_seqs)

    assert geometry.quant_token_bytes == 20736
    assert geometry.hp_token_bytes == 147456
    assert plan.prefix_slots == prefix_slots
    assert plan.recent_slots == recent_slots
    assert plan.quant_slots == quant_slots
    assert plan.guaranteed_quant_slots == guaranteed
    expected_unused = {1: 77824, 8: 225280, 48: 262144}[max_num_seqs]
    assert plan.unused_bytes == expected_unused
    assert plan.allocated_bytes + plan.unused_bytes == plan.total_memory_bytes
    assert plan.allocator_waste_fraction < 0.02


def test_single_request_capacity_target() -> None:
    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, 1)

    assert plan.bf16_slots == 72816
    assert plan.physical_capacity_ratio == pytest.approx(7.079982421)
    assert plan.guaranteed_capacity_ratio >= 6.2


def test_extra_prefix_cache_is_page_aligned() -> None:
    plan = plan_oscar_kv_cache(
        qwen3_geometry(),
        10 * 1024**3,
        max_num_seqs=1,
        prefix_cache_extra_tokens=17,
    )

    assert plan.prefix_slots == 96
    assert plan.prefix_cache_extra_tokens == 17


def test_hp_reservation_must_leave_quant_capacity() -> None:
    with pytest.raises(OscarKVCacheCapacityError, match="consume the KV budget"):
        plan_oscar_kv_cache(qwen3_geometry(), 1024, max_num_seqs=48)


def test_page_allocator_partial_page_lifecycle() -> None:
    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, 1)
    allocator = OscarKVPageAllocator(plan)
    allocator.start_request("r0")

    allocator.append_history("r0", 15)
    request = allocator.requests["r0"]
    assert request.partial_quant_slots == 15
    assert len(request.full_quant_pages) == 0

    allocator.append_history("r0", 18)
    assert request.history_tokens == 33
    assert len(request.full_quant_pages) == 2
    assert request.partial_quant_slots == 1
    allocator.assert_consistent()

    allocator.finish_request("r0")
    allocator.assert_consistent()
    assert len(allocator.prefix_pool.free_pages) == plan.prefix_pages
    assert len(allocator.recent_pool.free_pages) == plan.recent_pages
    assert len(allocator.quant_pool.free_pages) == plan.quant_pages


def test_request_start_rolls_back_on_recent_exhaustion() -> None:
    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, 1)
    allocator = OscarKVPageAllocator(plan)
    allocator.start_request("r0")

    with pytest.raises(OscarKVCacheCapacityError):
        allocator.start_request("r1")

    allocator.assert_consistent()
    assert len(allocator.prefix_pool.allocated_pages) == 4
    assert len(allocator.recent_pool.allocated_pages) == 16


def test_hp_rows_define_stable_contiguous_request_ranges() -> None:
    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, 2)
    allocator = OscarKVPageAllocator(plan)

    r0 = allocator.start_request("r0")
    r1 = allocator.start_request("r1")
    assert (r0.hp_row, r0.prefix_pages, r0.recent_pages) == (
        0,
        (0, 1, 2, 3),
        tuple(range(16)),
    )
    assert (r1.hp_row, r1.prefix_pages, r1.recent_pages) == (
        1,
        (4, 5, 6, 7),
        tuple(range(16, 32)),
    )

    allocator.finish_request("r0")
    r2 = allocator.start_request("r2")
    assert r2.hp_row == 0
    assert r2.prefix_pages == (0, 1, 2, 3)
    assert r2.recent_pages == tuple(range(16))
    allocator.assert_consistent()


def test_oscar_manager_owns_and_reuses_hp_rows_with_quant_blocks() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=32, enable_caching=False, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        max_num_seqs=2,
    )

    manager.allocate_new_blocks("r0", 16, 16)
    manager.allocate_new_blocks("r1", 16, 16)
    assert manager.get_hp_row("r0") == 0
    assert manager.get_hp_row("r1") == 1
    with pytest.raises(OscarKVCacheCapacityError, match="rows are exhausted"):
        manager.allocate_new_blocks("r2", 16, 16)

    manager.free("r0")
    manager.allocate_new_blocks("r2", 16, 16)
    assert manager.get_hp_row("r2") == 0
    manager.hp_rows.assert_consistent()


def test_oscar_prefix_hit_stops_before_request_owned_recent() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=64, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
    )
    source, branch = create_requests(
        num_requests=2,
        num_tokens=512,
        same_prompt=True,
        block_size=spec.block_size,
        req_ids=["source", "branch"],
    )

    manager.allocate_new_blocks("source", 400, 400)
    manager.cache_blocks(source, 400)
    manager.new_step_starts()
    hit = manager.find_longest_cache_hit(
        branch.block_hashes,
        511,
        [0],
        block_pool,
        spec,
        False,
        spec.block_size,
    )[0]
    assert len(hit) * spec.block_size == 144

    manager.allocate_new_blocks("source", 512, 512)
    assert manager.get_shared_hit_tokens("source") == 0
    manager.cache_blocks(source, 512)
    manager.new_step_starts()
    hit = manager.find_longest_cache_hit(
        branch.block_hashes,
        511,
        [0],
        block_pool,
        spec,
        False,
        spec.block_size,
    )[0]
    assert len(hit) * spec.block_size == 256
    assert manager.get_prefix_page_ids("source") == (0, 1, 2, 3)


def test_oscar_prefix_hit_waits_for_materialization_barrier() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
    )
    source, branch = create_requests(
        num_requests=2,
        num_tokens=64,
        same_prompt=True,
        block_size=spec.block_size,
        req_ids=["source", "branch"],
    )

    manager.allocate_new_blocks("source", 64, 64)
    manager.cache_blocks(source, 64)
    same_step_hit = manager.find_longest_cache_hit(
        branch.block_hashes,
        63,
        [0],
        block_pool,
        spec,
        False,
        spec.block_size,
    )[0]
    assert not same_step_hit

    manager.new_step_starts()
    materialized_hit = manager.find_longest_cache_hit(
        branch.block_hashes,
        63,
        [0],
        block_pool,
        spec,
        False,
        spec.block_size,
    )[0]
    assert len(materialized_hit) == 3


def test_oscar_prefix_pages_follow_quant_block_lru_eviction() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=32, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
    )
    old, new = create_requests(
        num_requests=2,
        num_tokens=64,
        block_size=spec.block_size,
        req_ids=["old", "new"],
    )

    manager.allocate_new_blocks("old", 64, 64)
    manager.cache_blocks(old, 64)
    manager.new_step_starts()
    old_blocks = tuple(manager.req_to_blocks["old"])
    manager.free("old")
    assert manager.num_free_prefix_pages == 0

    manager.allocate_new_blocks("new", 64, 64)
    assert manager.get_prefix_page_ids("new") == (0, 1, 2, 3)
    assert all(block.block_hash is None for block in old_blocks)
    assert manager.num_free_prefix_pages == 0

    manager.free("new")
    assert manager.num_free_prefix_pages == 4
    manager.assert_prefix_pages_consistent()


def test_oscar_prefix_eviction_discards_pending_readiness() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=16, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
    )
    old, new = create_requests(
        num_requests=2,
        num_tokens=64,
        block_size=spec.block_size,
        req_ids=["old", "new"],
    )

    manager.allocate_new_blocks("old", 64, 64)
    manager.cache_blocks(old, 64)
    assert len(manager.pending_ready_block_ids) == 4
    manager.free("old")

    manager.allocate_new_blocks("new", 64, 64)
    assert not manager.pending_ready_block_ids
    manager.new_step_starts()
    assert not any(block.oscar_prefix_ready for block in manager.req_to_blocks["new"])

    manager.cache_blocks(new, 64)
    manager.new_step_starts()
    assert all(block.oscar_prefix_ready for block in manager.req_to_blocks["new"])


def test_oscar_prefix_hit_shares_pages_but_recomputes_recent() -> None:
    spec = qwen3_spec()
    block_pool = BlockPool(
        num_gpu_blocks=96, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=2,
    )
    source, branch = create_requests(
        num_requests=2,
        num_tokens=512,
        same_prompt=True,
        block_size=spec.block_size,
        req_ids=["source", "branch"],
    )

    manager.allocate_new_blocks("source", 512, 512)
    manager.cache_blocks(source, 512)
    manager.new_step_starts()
    hit = manager.find_longest_cache_hit(
        branch.block_hashes,
        511,
        [0],
        block_pool,
        spec,
        False,
        spec.block_size,
    )[0]
    assert len(hit) == 16

    manager.allocate_new_computed_blocks("branch", hit, 256, 0)
    manager.allocate_new_blocks("branch", 512, 512)
    assert manager.get_shared_hit_tokens("branch") == 256
    assert manager.get_prefix_page_ids("branch") == manager.get_prefix_page_ids(
        "source"
    )
    assert all(block.ref_cnt == 2 for block in hit)
    branch_recent_blocks = {
        block.block_id for block in manager.req_to_blocks["branch"][16:]
    }
    source_recent_blocks = {
        block.block_id for block in manager.req_to_blocks["source"][16:]
    }
    assert branch_recent_blocks.isdisjoint(source_recent_blocks)
    manager.assert_prefix_pages_consistent()


def test_oscar_extra_prefix_pages_delay_lru_eviction_and_reset() -> None:
    spec = qwen3_spec(prefix_cache_extra_tokens=64)
    block_pool = BlockPool(
        num_gpu_blocks=32, enable_caching=True, hash_block_size=spec.block_size
    )
    manager = OscarKVCacheManager(
        spec,
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
    )
    old, new = create_requests(
        num_requests=2,
        num_tokens=64,
        block_size=spec.block_size,
        req_ids=["old", "new"],
    )

    manager.allocate_new_blocks("old", 64, 64)
    manager.cache_blocks(old, 64)
    manager.new_step_starts()
    old_blocks = tuple(manager.req_to_blocks["old"])
    manager.free("old")
    assert manager.num_free_prefix_pages == 4

    manager.allocate_new_blocks("new", 64, 64)
    assert manager.get_prefix_page_ids("new") == (4, 5, 6, 7)
    assert all(block.block_hash is not None for block in old_blocks)
    manager.free("new")

    assert block_pool.reset_prefix_cache()
    manager.reset_prefix_cache()
    assert manager.num_free_prefix_pages == 8
    manager.assert_prefix_pages_consistent()


def test_scheduler_preemption_releases_and_reuses_oscar_hp_rows(monkeypatch) -> None:
    monkeypatch.setattr(type(current_platform), "device_type", "cpu")
    spec = qwen3_spec()
    scheduler = create_scheduler(
        max_num_seqs=2,
        max_num_batched_tokens=100,
        block_size=spec.block_size,
        num_blocks=11,
        enable_prefix_caching=False,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=11,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["layer"], spec)],
        oscar_max_num_seqs=2,
    )
    scheduler.kv_cache_manager = KVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=100,
        hash_block_size=spec.block_size,
        enable_caching=False,
        log_stats=True,
    )
    manager = scheduler.kv_cache_manager.coordinator.single_type_managers[0]
    assert isinstance(manager, OscarKVCacheManager)

    requests = create_requests(num_requests=2, num_tokens=80, block_size=16)
    scheduler.add_request(requests[0])
    output0 = scheduler.schedule()
    assert output0.oscar_hp_row_ids == {"0": 0}
    assert output0.oscar_prefix_page_ids == {"0": (0, 1, 2, 3)}
    assert output0.oscar_shared_hit_tokens == {"0": 0}

    scheduler.add_request(requests[1])
    output1 = scheduler.schedule()
    assert output1.oscar_hp_row_ids == {"1": 1}
    assert output1.oscar_prefix_page_ids == {"1": (4, 5, 6, 7)}
    assert output1.oscar_shared_hit_tokens == {"1": 0}

    scheduler.update_from_output(
        output0,
        ModelRunnerOutput(
            req_ids=["0"],
            req_id_to_index={"0": 0},
            sampled_token_ids=[[0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.schedule()
    assert requests[1].status == RequestStatus.PREEMPTED
    assert manager.hp_rows.request_rows == {"0": 0}

    scheduler.update_from_output(
        output1,
        ModelRunnerOutput(
            req_ids=["1"],
            req_id_to_index={"1": 0},
            sampled_token_ids=[[42]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )
    scheduler.finish_requests("0", RequestStatus.FINISHED_ABORTED)
    resumed = scheduler.schedule()
    assert resumed.oscar_hp_row_ids == {"1": 0}
    assert resumed.oscar_shared_hit_tokens == {"1": 0}
    reused_prefix_pages = resumed.oscar_prefix_page_ids["1"]
    assert len(reused_prefix_pages) == 4
    assert set(reused_prefix_pages) == {0, 1, 2, 3}
    assert manager.hp_rows.request_rows == {"1": 0}
    manager.hp_rows.assert_consistent()
    manager.assert_prefix_pages_consistent()


def test_scheduler_output_carries_oscar_hp_row_ids() -> None:
    output = SchedulerOutput.make_empty()
    output.oscar_hp_row_ids = {"r0": 3, "r1": 1}
    output.oscar_prefix_page_ids = {"r0": (12, 13, 14, 15), "r1": (4, 5, 6, 7)}
    output.oscar_shared_hit_tokens = {"r0": 1840, "r1": 0}

    assert output.oscar_hp_row_ids == {"r0": 3, "r1": 1}
    assert output.oscar_prefix_page_ids == {
        "r0": (12, 13, 14, 15),
        "r1": (4, 5, 6, 7),
    }
    assert output.oscar_shared_hit_tokens == {"r0": 1840, "r1": 0}


def test_dummy_input_batch_uses_unique_oscar_hp_rows() -> None:
    input_buffers = InputBuffers(
        max_num_reqs=8,
        max_num_tokens=16,
        device=torch.device("cpu"),
    )

    batch = InputBatch.make_dummy(3, 6, input_buffers)

    assert batch.oscar_hp_row_ids is not None
    assert batch.oscar_hp_row_ids.tolist() == [0, 1, 2]
    assert batch.oscar_prefix_page_ids is not None
    assert batch.oscar_prefix_page_ids.tolist() == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [8, 9, 10, 11],
    ]
    assert batch.oscar_shared_hit_tokens is not None
    assert batch.oscar_shared_hit_tokens.tolist() == [0, 0, 0]


def test_synthetic_warmup_uses_unique_oscar_hp_rows() -> None:
    model_runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            oscar_max_num_seqs=2,
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=qwen3_spec())],
        )
    )
    output = SchedulerOutput.make_empty()

    _set_oscar_warmup_hp_rows(model_runner, output, ["warmup-0", "warmup-1"])

    assert output.oscar_hp_row_ids == {"warmup-0": 0, "warmup-1": 1}
    assert output.oscar_prefix_page_ids == {
        "warmup-0": (0, 1, 2, 3),
        "warmup-1": (4, 5, 6, 7),
    }
    assert output.oscar_shared_hit_tokens == {"warmup-0": 0, "warmup-1": 0}
    with pytest.raises(RuntimeError, match="exceed reserved HP rows"):
        _set_oscar_warmup_hp_rows(
            model_runner,
            output,
            ["warmup-0", "warmup-1", "warmup-2"],
        )


def test_unpadded_attention_metadata_preserves_oscar_hp_rows() -> None:
    metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([9, 7], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=9,
        block_table_tensor=torch.zeros((2, 1), dtype=torch.int32),
        slot_mapping=torch.tensor([0, 1], dtype=torch.int64),
        oscar_hp_row_ids=torch.tensor([3, 1], dtype=torch.int32),
        oscar_prefix_page_ids=torch.tensor(
            [[12, 13, 14, 15], [4, 5, 6, 7]], dtype=torch.int32
        ),
        oscar_shared_hit_tokens=torch.tensor([1840, 0], dtype=torch.int32),
    )

    unpadded = metadata.unpadded(num_actual_tokens=1, num_actual_reqs=1)
    assert unpadded.oscar_hp_row_ids is not None
    assert unpadded.oscar_hp_row_ids.tolist() == [3]
    assert unpadded.oscar_prefix_page_ids is not None
    assert unpadded.oscar_prefix_page_ids.tolist() == [[12, 13, 14, 15]]
    assert unpadded.oscar_shared_hit_tokens is not None
    assert unpadded.oscar_shared_hit_tokens.tolist() == [1840]


def test_randomized_page_accounting_matches_reference() -> None:
    max_num_seqs = 8
    block_size = 16
    plan = plan_oscar_kv_cache(qwen3_geometry(), 10 * 1024**3, max_num_seqs)
    allocator = OscarKVPageAllocator(plan)
    reference: dict[str, int] = {}
    rng = random.Random(20260720)

    for step in range(2000):
        available_ids = [
            f"r{i}" for i in range(max_num_seqs) if f"r{i}" not in reference
        ]
        if available_ids and (not reference or rng.random() < 0.2):
            request_id = rng.choice(available_ids)
            allocator.start_request(request_id)
            reference[request_id] = 0
        elif reference and rng.random() < 0.25:
            request_id = rng.choice(list(reference))
            allocator.finish_request(request_id)
            del reference[request_id]
        elif reference:
            request_id = rng.choice(list(reference))
            tokens = rng.randint(1, 97)
            allocator.append_history(request_id, tokens)
            reference[request_id] += tokens

        expected_quant_pages = sum(
            (tokens + block_size - 1) // block_size for tokens in reference.values()
        )
        assert len(allocator.quant_pool.allocated_pages) == expected_quant_pages, step
        assert len(allocator.prefix_pool.allocated_pages) == len(reference) * 4, step
        assert len(allocator.recent_pool.allocated_pages) == len(reference) * 16, step
        allocator.assert_consistent()
