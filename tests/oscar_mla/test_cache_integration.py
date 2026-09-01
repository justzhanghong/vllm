# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization.oscar_mla.cache import (
    MLACacheGeometry,
    WorkerCacheMetadata,
    plan_mla_runtime_cache,
)
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    generate_scheduler_kv_cache_config,
    get_kv_cache_config_from_groups,
    get_kv_cache_configs,
    get_max_concurrency_for_kv_cache_config,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import OscarMLAKVCacheManager
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    OscarMLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.oscar_mla_cache import (
    OscarMLAWorkerOwnership,
    reshape_oscar_mla_cache,
)


def _spec(*, speculative_tokens: int = 0) -> OscarMLAAttentionSpec:
    return OscarMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
        cache_dtype_str="oscar_mla_int2",
        latent_rank=512,
        rope_head_size=64,
        history_slot_size=160,
        group_size=128,
        prefix_tokens=64,
        recent_tokens=256,
        speculative_tokens=speculative_tokens,
    )


def test_oscar_mla_spec_accounts_only_history_pages() -> None:
    spec = _spec()

    assert spec.page_size_bytes == 16 * (160 + 64 * 2)
    assert spec.bf16_token_size_bytes == 512 * 2
    assert OscarMLAAttentionSpec.merge([spec, spec]) == spec

    with pytest.raises(ValueError, match="expected=160"):
        OscarMLAAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=576,
            dtype=torch.bfloat16,
            cache_dtype_str="oscar_mla_int2",
            latent_rank=512,
            rope_head_size=64,
            history_slot_size=159,
        )


def test_cache_config_matches_three_pool_plan_exactly() -> None:
    num_layers = 78
    max_num_seqs = 16
    available_memory = 14 * 1024**3
    main_layer_names = [
        f"model.layers.{i}.self_attn.mla_attn" for i in range(num_layers)
    ]
    index_layer_names = [
        f"model.layers.{i}.self_attn.indexer.k_cache"
        for i in (0, 1, 2, *range(6, 78, 4))
    ]
    per_layer_specs = {layer_name: _spec() for layer_name in main_layer_names}
    per_layer_specs.update(
        {
            layer_name: MLAAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=132,
                dtype=torch.uint8,
            )
            for layer_name in index_layer_names
        }
    )
    layer_names = main_layer_names + index_layer_names
    groups = [
        KVCacheGroupSpec(
            layer_names=layer_names,
            kv_cache_spec=UniformTypeKVCacheSpecs(
                block_size=16,
                kv_cache_specs=per_layer_specs,
            ),
        )
    ]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
    )

    config = get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory,
        suppress_log=True,
    )
    expected = plan_mla_runtime_cache(
        MLACacheGeometry(num_layers=num_layers, latent_rank=512),
        total_memory_bytes=available_memory,
        max_num_seqs=max_num_seqs,
        rope_bytes_per_layer_token=64 * 2,
        auxiliary_bytes_per_block=21 * 16 * 132,
    )

    assert config.num_blocks == expected.num_blocks
    assert config.oscar_mla_history_pages == expected.history_pages
    assert config.oscar_mla_max_num_seqs == max_num_seqs
    assert expected.fixed_prefix_slots == 1024
    assert expected.fixed_recent_slots == 4096
    assert expected.history_slots == expected.history_pages * 16
    assert expected.theoretical_history_compression_ratio == pytest.approx(6.4)
    assert expected.padded_history_compression_ratio == pytest.approx(6.4)
    assert expected.native_logical_token_slots == 162256
    assert expected.allocated_capacity_ratio == pytest.approx(
        expected.logical_token_slots / 162256
    )
    assert len(config.kv_cache_tensors) == len(layer_names)
    assert sum(tensor.size for tensor in config.kv_cache_tensors) == (
        expected.allocated_bytes
    )
    assert all(len(tensor.shared_by) == 1 for tensor in config.kv_cache_tensors)
    vllm_config.model_config = SimpleNamespace(max_model_len=32768)
    assert get_max_concurrency_for_kv_cache_config(vllm_config, config) == 16
    scheduler_config = generate_scheduler_kv_cache_config([config])
    assert isinstance(
        scheduler_config.kv_cache_groups[0].kv_cache_spec,
        OscarMLAAttentionSpec,
    )


def test_tp_workers_use_limiting_oscar_mla_three_pool_plan() -> None:
    spec = _spec()
    kv_cache_specs = [{"layer": spec}, {"layer": spec}]
    max_num_seqs = 16
    limiting_memory = 14 * 1024**3
    bytes_per_block = spec.page_size_bytes
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_seqs=max_num_seqs,
        ),
        model_config=SimpleNamespace(
            max_model_len=32768,
            original_max_model_len=32768,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    configs = get_kv_cache_configs(
        vllm_config,
        kv_cache_specs,
        [limiting_memory + 12 * bytes_per_block, limiting_memory],
    )
    expected = plan_mla_runtime_cache(
        MLACacheGeometry(num_layers=1, latent_rank=spec.latent_rank),
        total_memory_bytes=limiting_memory,
        max_num_seqs=max_num_seqs,
        rope_bytes_per_layer_token=spec.rope_head_size * 2,
        auxiliary_bytes_per_block=0,
    )

    assert [config.num_blocks for config in configs] == [
        expected.num_blocks,
        expected.num_blocks,
    ]
    assert [config.oscar_mla_history_pages for config in configs] == [
        expected.history_pages,
        expected.history_pages,
    ]
    assert configs[0].kv_cache_tensors == configs[1].kv_cache_tensors
    assert sum(tensor.size for tensor in configs[0].kv_cache_tensors) == (
        expected.allocated_bytes
    )


def test_pp_workers_use_common_logical_capacity_with_stage_local_pools() -> None:
    spec = _spec(speculative_tokens=3)
    max_num_seqs = 64
    available_memory = 4 * 1024**3
    worker_specs = [
        {"stage0.layer0": spec},
        {"stage1.layer0": spec, "stage1.layer1": spec},
    ]
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_seqs=max_num_seqs,
        ),
        model_config=SimpleNamespace(
            max_model_len=32768,
            original_max_model_len=32768,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    configs = get_kv_cache_configs(
        vllm_config,
        worker_specs,
        [available_memory, available_memory],
    )

    assert configs[0].num_blocks == configs[1].num_blocks
    assert configs[0].oscar_mla_history_pages == configs[0].num_blocks
    assert configs[1].oscar_mla_history_pages == configs[1].num_blocks
    assert len(configs[0].kv_cache_tensors) == 1
    assert len(configs[1].kv_cache_tensors) == 2
    assert sum(t.size for t in configs[0].kv_cache_tensors) <= available_memory
    assert sum(t.size for t in configs[1].kv_cache_tensors) <= available_memory


def test_native_workers_keep_existing_limiting_block_plan() -> None:
    spec = MLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
            max_num_seqs=16,
        ),
        model_config=SimpleNamespace(
            max_model_len=16,
            original_max_model_len=16,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
    )

    configs = get_kv_cache_configs(
        vllm_config,
        [{"layer": spec}, {"layer": spec}],
        [10 * spec.page_size_bytes, 20 * spec.page_size_bytes],
    )

    assert [config.num_blocks for config in configs] == [10, 10]
    assert [config.kv_cache_tensors[0].size for config in configs] == [
        10 * spec.page_size_bytes,
        10 * spec.page_size_bytes,
    ]


def test_worker_views_cover_raw_allocation_without_padding() -> None:
    spec = _spec()
    num_blocks = 3
    max_num_seqs = 2
    history_slots = num_blocks * spec.block_size
    raw_bytes = (
        history_slots * spec.history_slot_size
        + max_num_seqs
        * (spec.prefix_tokens + spec.recent_tokens)
        * spec.bf16_token_size_bytes
        + history_slots * spec.rope_head_size * 2
    )
    raw = torch.empty(raw_bytes, dtype=torch.int8)

    tensors = reshape_oscar_mla_cache(
        raw,
        spec,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
    )

    assert tensors.history_data.shape == (3, 16, 128)
    assert tensors.history_scale.shape == (3, 16, 4)
    assert tensors.history_zero.shape == (3, 16, 4)
    assert tensors.prefix.shape == (2, 64, 512)
    assert tensors.recent.shape == (2, 256, 512)
    assert tensors.rope.shape == (3, 16, 64)
    data_bytes = history_slots * 128
    metadata_bytes = history_slots * 4 * 4
    prefix_bytes = max_num_seqs * spec.prefix_tokens * 512 * 2
    assert tensors.history_data.data_ptr() == raw.data_ptr()
    assert tensors.history_scale.data_ptr() == raw.data_ptr() + data_bytes
    assert tensors.history_zero.data_ptr() == (
        raw.data_ptr() + data_bytes + metadata_bytes
    )
    assert tensors.prefix.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes
    )
    assert tensors.recent.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes + prefix_bytes
    )
    recent_bytes = max_num_seqs * spec.recent_tokens * 512 * 2
    assert tensors.rope.data_ptr() == (
        raw.data_ptr() + data_bytes + 2 * metadata_bytes + prefix_bytes + recent_bytes
    )
    assert tensors.recent.untyped_storage().nbytes() == raw.numel()


def test_mtp5_worker_views_reserve_candidate_safe_recent_capacity() -> None:
    spec = _spec(speculative_tokens=5)
    num_blocks = 3
    max_num_seqs = 2
    history_slots = num_blocks * spec.block_size
    raw_bytes = (
        history_slots * spec.history_slot_size
        + max_num_seqs
        * (spec.prefix_tokens + spec.recent_capacity_tokens)
        * spec.bf16_token_size_bytes
        + history_slots * spec.rope_head_size * 2
    )

    tensors = reshape_oscar_mla_cache(
        torch.empty(raw_bytes, dtype=torch.int8),
        spec,
        num_blocks=num_blocks,
        max_num_seqs=max_num_seqs,
    )

    assert tensors.recent.shape == (2, 261, 512)
    assert tensors.recent_tokens == 256


def test_scheduler_manager_binds_physical_blocks_and_reuses_hp_row() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=40,
        enable_caching=False,
        hash_block_size=16,
    )
    manager = OscarMLAKVCacheManager(
        _spec(),
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        max_num_seqs=1,
        history_pages=2,
    )

    assert len(manager.allocate_new_blocks("r0", 320, 320)) == 20
    initial = manager.metadata("r0")
    assert initial.hp_row == 0
    assert initial.block_ids == tuple(
        block.block_id for block in manager.req_to_blocks["r0"]
    )
    assert initial.history_pages == ()

    first = manager.allocate_new_blocks("r0", 321, 321)
    assert len(first) == 1
    assert manager.metadata("r0").partial_history_slots == 1
    second = manager.allocate_new_blocks("r0", 337, 337)
    assert len(second) == 1
    assert len(manager.allocate_new_blocks("r0", 353, 353)) == 1
    current = manager.metadata("r0")
    assert current.history_pages == current.block_ids[4:7]
    first_generation = current.generation
    manager.free("r0")
    manager.allocate_new_blocks("reused", 64, 64)
    reused = manager.metadata("reused")
    assert reused.hp_row == 0
    assert reused.generation != first_generation


def test_scheduler_and_worker_restore_prefix_cache_hit_from_physical_blocks() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=48,
        enable_caching=True,
        hash_block_size=16,
    )
    manager = OscarMLAKVCacheManager(
        _spec(),
        block_pool=block_pool,
        enable_caching=True,
        kv_cache_group_id=0,
        max_num_seqs=1,
        history_pages=48,
    )
    cached_blocks = block_pool.get_new_blocks(20)
    block_pool.free_blocks(reversed(cached_blocks))

    manager.allocate_new_computed_blocks("hit", cached_blocks, 320, 0)
    assert len(manager.allocate_new_blocks("hit", 337, 337)) == 2
    scheduler_metadata = manager.metadata("hit")
    assert scheduler_metadata.num_cached_tokens == 320
    assert scheduler_metadata.block_ids[:20] == tuple(
        block.block_id for block in cached_blocks
    )
    assert scheduler_metadata.history_pages == scheduler_metadata.block_ids[4:6]

    ownership = OscarMLAWorkerOwnership()
    output = SchedulerOutput.make_empty()
    output.oscar_mla_cache_metadata = {"hit": scheduler_metadata}
    ownership.apply(output)
    batch = ownership.build_batch_metadata(
        ["hit"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        device=torch.device("cpu"),
    )

    expected_positions = list(range(64)) + list(range(81, 320))
    assert batch.num_restore_rows == len(expected_positions)
    assert batch.restore_positions.tolist() == expected_positions
    assert batch.restore_hp_rows.tolist() == [scheduler_metadata.hp_row] * len(
        expected_positions
    )
    assert batch.restore_page_ids.tolist() == [
        scheduler_metadata.block_ids[position // 16] for position in expected_positions
    ]
    assert batch.restore_page_offsets.tolist() == [
        position % 16 for position in expected_positions
    ]

    ownership.apply(output)
    next_batch = ownership.build_batch_metadata(
        ["hit"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        device=torch.device("cpu"),
    )
    assert next_batch.num_restore_rows == 0


def test_scheduler_manager_separates_mtp5_lookahead_from_oscar_length() -> None:
    block_pool = BlockPool(
        num_gpu_blocks=40,
        enable_caching=False,
        hash_block_size=16,
    )
    manager = OscarMLAKVCacheManager(
        _spec(speculative_tokens=5),
        block_pool=block_pool,
        enable_caching=False,
        kv_cache_group_id=0,
        max_num_seqs=1,
        history_pages=2,
    )

    # The native auxiliary cache reserves five lookahead slots, while OSCAR
    # commits only the six target-verification positions (one input + drafts).
    manager.allocate_new_blocks("r0", 325, 320)
    assert manager.metadata("r0").logical_length == 320
    manager.allocate_new_blocks("r0", 331, 326)
    candidate = manager.metadata("r0")
    assert candidate.logical_length == 326
    assert candidate.history_tokens == 6

    # Rejecting all five drafts rolls back by five while retaining the history
    # high-water allocation; the next cycle can safely reuse it.
    manager.allocate_new_blocks("r0", 326, 321)
    rollback = manager.metadata("r0")
    assert rollback.logical_length == 321
    assert rollback.history_tokens == 6
    assert rollback.history_pages == candidate.history_pages


def test_kv_cache_manager_exposes_scheduler_metadata() -> None:
    config = KVCacheConfig(
        num_blocks=30,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0"], kv_cache_spec=_spec())
        ],
        oscar_mla_max_num_seqs=1,
        oscar_mla_history_pages=3,
    )
    cache_manager = KVCacheManager(
        config,
        max_model_len=32768,
        hash_block_size=16,
        enable_caching=False,
    )
    runtime_manager = cache_manager.coordinator.single_type_managers[0]
    assert isinstance(runtime_manager, OscarMLAKVCacheManager)
    runtime_manager.allocate_new_blocks("r0", 321, 321)

    metadata = cache_manager.get_oscar_mla_metadata(["r0"])

    assert metadata["r0"].logical_length == 321
    assert len(metadata["r0"].history_pages) == 1


def test_worker_ownership_rejects_stale_versions_and_handles_reuse() -> None:
    ownership = OscarMLAWorkerOwnership()
    output = SchedulerOutput.make_empty()
    current = WorkerCacheMetadata(
        request_id="r0",
        generation=1,
        cache_version=2,
        logical_length=337,
        hp_row=0,
        prefix_start=0,
        recent_start=0,
        history_pages=(1, 2),
        partial_history_slots=1,
        history_tokens=17,
    )
    output.oscar_mla_cache_metadata = {"r0": current}
    ownership.apply(output)
    assert ownership.get("r0") == current

    output.oscar_mla_cache_metadata = {
        "r0": WorkerCacheMetadata(**{**current.__dict__, "cache_version": 1})
    }
    with pytest.raises(RuntimeError, match="stale"):
        ownership.apply(output)

    output.finished_req_ids = {"r0"}
    output.oscar_mla_cache_metadata = {
        "r0": WorkerCacheMetadata(
            **{**current.__dict__, "generation": 2, "cache_version": 1}
        )
    }
    ownership.apply(output)
    assert ownership.get("r0").generation == 2


def test_worker_builds_incremental_batch_demotion_metadata() -> None:
    ownership = OscarMLAWorkerOwnership()
    output = SchedulerOutput.make_empty()
    before = WorkerCacheMetadata(
        request_id="r0",
        generation=1,
        cache_version=1,
        logical_length=320,
        hp_row=2,
        prefix_start=128,
        recent_start=512,
        history_pages=(),
        partial_history_slots=0,
        history_tokens=0,
    )
    output.oscar_mla_cache_metadata = {"r0": before}
    ownership.apply(output)

    current = WorkerCacheMetadata(
        **{
            **before.__dict__,
            "cache_version": 2,
            "logical_length": 337,
            "history_pages": (9, 11),
            "partial_history_slots": 1,
            "history_tokens": 17,
        }
    )
    output.oscar_mla_cache_metadata = {"r0": current}
    ownership.apply(output)
    metadata = ownership.build_batch_metadata(
        ["r0"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        device=torch.device("cpu"),
        padded_size=2,
    )

    assert metadata.hp_rows.tolist() == [2, -1]
    assert metadata.decode_positions.tolist() == [336, -1]
    assert metadata.final_seq_lens.tolist() == [337, 0]
    assert metadata.previous_seq_lens.tolist() == [320, 0]
    assert metadata.history_page_table.tolist() == [[9, 11], [0, 0]]
    assert metadata.demotion_hp_rows.tolist() == [2] * 17
    assert metadata.demotion_positions.tolist() == list(range(64, 81))
    assert metadata.demotion_page_ids.tolist() == [9] * 16 + [11]
    assert metadata.demotion_page_offsets.tolist() == list(range(16)) + [0]


def test_worker_mtp5_demotion_uses_history_high_water_across_rollback() -> None:
    ownership = OscarMLAWorkerOwnership()
    output = SchedulerOutput.make_empty()
    before = WorkerCacheMetadata(
        request_id="r0",
        generation=1,
        cache_version=1,
        logical_length=320,
        hp_row=0,
        prefix_start=0,
        recent_start=0,
        history_pages=(),
        partial_history_slots=0,
        history_tokens=0,
    )
    output.oscar_mla_cache_metadata = {"r0": before}
    ownership.apply(output)

    candidate = WorkerCacheMetadata(
        **{
            **before.__dict__,
            "cache_version": 2,
            "logical_length": 326,
            "history_pages": (9,),
            "partial_history_slots": 6,
            "history_tokens": 6,
        }
    )
    output.oscar_mla_cache_metadata = {"r0": candidate}
    ownership.apply(output)
    candidate_batch = ownership.build_batch_metadata(
        ["r0"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        recent_capacity_tokens=261,
        device=torch.device("cpu"),
        padded_size=1,
        cudagraph_max_history_pages=4,
        max_demotion_tokens_per_request=6,
    )
    assert candidate_batch.demotion_positions.tolist() == list(range(64, 70))
    assert candidate_batch.demotion_page_offsets.tolist() == list(range(6))

    rollback = WorkerCacheMetadata(
        **{
            **candidate.__dict__,
            "cache_version": 3,
            "logical_length": 321,
        }
    )
    output.oscar_mla_cache_metadata = {"r0": rollback}
    ownership.apply(output)
    rollback_batch = ownership.build_batch_metadata(
        ["r0"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        recent_capacity_tokens=261,
        device=torch.device("cpu"),
    )
    assert rollback_batch.demotion_positions.numel() == 0

    replay = WorkerCacheMetadata(
        **{
            **rollback.__dict__,
            "cache_version": 4,
            "logical_length": 326,
        }
    )
    output.oscar_mla_cache_metadata = {"r0": replay}
    ownership.apply(output)
    replay_batch = ownership.build_batch_metadata(
        ["r0"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        recent_capacity_tokens=261,
        device=torch.device("cpu"),
    )
    assert replay_batch.demotion_positions.numel() == 0


def test_worker_reuses_fixed_cudagraph_metadata_buffers() -> None:
    ownership = OscarMLAWorkerOwnership()
    captured = ownership.build_batch_metadata(
        [],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        device=torch.device("cpu"),
        padded_size=2,
        cudagraph_max_history_pages=4,
    )
    captured_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            captured.hp_rows,
            captured.decode_positions,
            captured.final_seq_lens,
            captured.history_page_table,
            captured.previous_seq_lens,
            captured.demotion_hp_rows,
            captured.demotion_positions,
            captured.demotion_page_ids,
            captured.demotion_page_offsets,
            captured.restore_positions,
            captured.restore_hp_rows,
            captured.restore_page_ids,
            captured.restore_page_offsets,
        )
    )
    assert captured.hp_rows.tolist() == [-1, -1]
    assert captured.decode_positions.tolist() == [-1, -1]
    assert captured.final_seq_lens.tolist() == [0, 0]
    assert captured.demotion_hp_rows.tolist() == [-1, -1]
    assert captured.demotion_positions.tolist() == [-1, -1]
    assert captured.num_restore_rows == 0
    assert captured.restore_positions.tolist() == [-1] * 640

    output = SchedulerOutput.make_empty()
    before = WorkerCacheMetadata(
        request_id="r0",
        generation=1,
        cache_version=1,
        logical_length=320,
        hp_row=2,
        prefix_start=128,
        recent_start=512,
        history_pages=(9,),
        partial_history_slots=0,
        history_tokens=0,
    )
    output.oscar_mla_cache_metadata = {"r0": before}
    ownership.apply(output)
    current = WorkerCacheMetadata(
        **{
            **before.__dict__,
            "cache_version": 2,
            "logical_length": 321,
            "history_tokens": 1,
        }
    )
    output.oscar_mla_cache_metadata = {"r0": current}
    ownership.apply(output)

    runtime = ownership.build_batch_metadata(
        ["r0"],
        block_size=16,
        prefix_tokens=64,
        recent_tokens=256,
        device=torch.device("cpu"),
        padded_size=2,
        cudagraph_max_history_pages=4,
    )
    runtime_ptrs = tuple(
        tensor.data_ptr()
        for tensor in (
            runtime.hp_rows,
            runtime.decode_positions,
            runtime.final_seq_lens,
            runtime.history_page_table,
            runtime.previous_seq_lens,
            runtime.demotion_hp_rows,
            runtime.demotion_positions,
            runtime.demotion_page_ids,
            runtime.demotion_page_offsets,
            runtime.restore_positions,
            runtime.restore_hp_rows,
            runtime.restore_page_ids,
            runtime.restore_page_offsets,
        )
    )
    assert runtime_ptrs == captured_ptrs
    assert runtime.hp_rows.tolist() == [2, -1]
    assert runtime.decode_positions.tolist() == [320, -1]
    assert runtime.final_seq_lens.tolist() == [321, 0]
    assert runtime.history_page_table.tolist() == [
        [9, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert runtime.demotion_hp_rows.tolist() == [2, -1]
    assert runtime.demotion_positions.tolist() == [64, -1]
    assert runtime.demotion_page_ids.tolist() == [9, -1]
    assert runtime.demotion_page_offsets.tolist() == [0, 0]
