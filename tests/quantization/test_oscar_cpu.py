# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from vllm.config import CompilationConfig, CompilationMode, CUDAGraphMode
from vllm.config.vllm import OptimizationLevel
from vllm.engine.arg_utils import _is_oscar_execution_mode_supported
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
from vllm.model_executor.layers.quantization.oscar.rotation import (
    _load_checkpoint,
    absorb_v_rotation_into_qkv,
    get_layer_rotation,
)
from vllm.v1.attention.backend import AttentionCGSupport, CommonAttentionMetadata
from vllm.v1.attention.backends.oscar_attn import OscarMetadataBuilder
from vllm.v1.attention.ops.triton_oscar_bulk_flush import (
    bind_oscar_bulk_flush_state,
    build_oscar_bulk_flush_plan_cpu,
    clear_oscar_bulk_flush_state,
    is_oscar_bulk_flush_supported,
    is_oscar_bulk_flush_target,
    prepare_oscar_bulk_flush_plan,
    register_oscar_bulk_flush_caches,
    update_oscar_bulk_flush_row_generations,
    validate_oscar_bulk_flush_scheduler_output,
)


class TestOscarConfigAndLayout(unittest.TestCase):
    def test_bulk_flush_plan_crosses_noncontiguous_pages(self):
        block_table = torch.tensor([[7, 2, 11, 5]], dtype=torch.int32)
        plan = build_oscar_bulk_flush_plan_cpu(
            phase=torch.tensor([7], dtype=torch.int32),
            cached_lens=torch.tensor([49], dtype=torch.int32),
            hp_row_ids=torch.tensor([0], dtype=torch.int32),
            shared_hit_tokens=torch.tensor([0], dtype=torch.int32),
            block_table=block_table,
            prefix_tokens=8,
            recent_tokens=32,
            recent_capacity=48,
            block_size=16,
            flush_interval=8,
        )

        self.assertEqual(plan.next_phase.tolist(), [0])
        self.assertEqual(plan.recent_extra.tolist(), [0])
        self.assertEqual(plan.positions.tolist(), [[10, 11, 12, 13, 14, 15, 16, 17]])
        self.assertEqual(
            plan.dst_slots.tolist(),
            [[122, 123, 124, 125, 126, 127, 32, 33]],
        )
        self.assertEqual(plan.valid.tolist(), [[True] * 8])
        self.assertEqual(plan.src_recent_slots.tolist(), [[2, 3, 4, 5, 6, 7, 8, 9]])

        for cached_len, table, expected_valid in (
            (4, [[7]], [False] * 8),
            (49, [[7, -1]], [True] * 6 + [False] * 2),
            (49, [[7]], [True] * 6 + [False] * 2),
        ):
            bounded = build_oscar_bulk_flush_plan_cpu(
                phase=torch.tensor([7], dtype=torch.int32),
                cached_lens=torch.tensor([cached_len], dtype=torch.int32),
                hp_row_ids=torch.tensor([0], dtype=torch.int32),
                shared_hit_tokens=torch.tensor([0], dtype=torch.int32),
                block_table=torch.tensor(table, dtype=torch.int32),
                prefix_tokens=8,
                recent_tokens=32,
                recent_capacity=48,
                block_size=16,
                flush_interval=8,
            )
            with self.subTest(cached_len=cached_len, table=table):
                self.assertEqual(bounded.valid[0].tolist(), expected_valid)
                self.assertTrue(torch.all(bounded.positions[~bounded.valid] == -1))
                self.assertTrue(torch.all(bounded.dst_slots[~bounded.valid] == -1))
                self.assertTrue(
                    torch.all(bounded.src_recent_slots[~bounded.valid] == -1)
                )

    def test_bulk_flush_phase_is_periodic_and_resets(self):
        phase = torch.tensor([0, 0], dtype=torch.int32)
        block_table = torch.stack(
            (
                torch.arange(64, dtype=torch.int32),
                torch.arange(64, dtype=torch.int32) + 100,
            )
        )
        flush_steps = []
        live_slots = [dict(), dict()]
        wrap_count = [0, 0]
        for step in range(1, 545):
            cached = 320 + step - 1
            plan = build_oscar_bulk_flush_plan_cpu(
                phase=phase,
                cached_lens=torch.tensor([cached, cached], dtype=torch.int32),
                hp_row_ids=torch.tensor([0, 1], dtype=torch.int32),
                shared_hit_tokens=torch.tensor([0, 0], dtype=torch.int32),
                block_table=block_table,
                prefix_tokens=64,
                recent_tokens=256,
                recent_capacity=272,
                block_size=16,
                flush_interval=8,
            )
            phase = plan.next_phase
            if plan.valid.any():
                flush_steps.append(step)
            self.assertEqual(plan.recent_extra.tolist(), [step % 8] * 2)
            for row in range(2):
                for lane in range(8):
                    position = cached - 256 - 8 + 1 + lane
                    valid = step % 8 == 0 and position >= 64
                    self.assertEqual(plan.valid[row, lane].item(), valid)
                    self.assertEqual(
                        plan.positions[row, lane].item(), position if valid else -1
                    )
                    if valid:
                        self.assertEqual(
                            plan.src_recent_slots[row, lane].item(),
                            row * 272 + (position - 64) % 272,
                        )
                        block = block_table[row, position // 16].item()
                        self.assertEqual(
                            plan.dst_slots[row, lane].item(),
                            block * 16 + position % 16,
                        )

                write_position = 64 + step - 1
                slot = row * 272 + (write_position - 64) % 272
                if slot in live_slots[row]:
                    wrap_count[row] += 1
                live_slots[row][slot] = write_position
                live_start = max(64, write_position - 271)
                live_slots[row] = {
                    physical: logical
                    for physical, logical in live_slots[row].items()
                    if logical >= live_start
                }
                self.assertEqual(len(live_slots[row]), min(step, 272))
                self.assertEqual(len(set(live_slots[row])), len(live_slots[row]))
            self.assertTrue(set(live_slots[0]).isdisjoint(live_slots[1]))

        self.assertEqual(flush_steps, list(range(8, 545, 8)))
        self.assertEqual(wrap_count, [272, 272])
        self.assertEqual([len(slots) for slots in live_slots], [272, 272])

    def test_bulk_flush_support_gate_fails_closed(self):
        def config(**overrides):
            values = {
                "max_num_seqs": 1,
                "enable_chunked_prefill": True,
                "enable_prefix_caching": False,
                "speculative_config": None,
                "tp": 1,
                "pp": 1,
                "connector": None,
                "cpu_offload_gb": 0,
            }
            values.update(overrides)
            return SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=values["max_num_seqs"],
                    enable_chunked_prefill=values["enable_chunked_prefill"],
                ),
                cache_config=SimpleNamespace(
                    enable_prefix_caching=values["enable_prefix_caching"]
                ),
                speculative_config=values["speculative_config"],
                parallel_config=SimpleNamespace(
                    tensor_parallel_size=values["tp"],
                    pipeline_parallel_size=values["pp"],
                ),
                kv_transfer_config=SimpleNamespace(
                    kv_connector=values["connector"]
                ),
                offload_config=SimpleNamespace(
                    cpu_offload_gb=values["cpu_offload_gb"]
                ),
            )

        self.assertTrue(is_oscar_bulk_flush_supported(config()))
        for unsupported in (
            config(max_num_seqs=2),
            config(enable_chunked_prefill=False),
            config(enable_prefix_caching=True),
            config(speculative_config=SimpleNamespace()),
            config(tp=2),
            config(pp=2),
            config(connector="LMCacheConnectorV1"),
            config(cpu_offload_gb=1),
        ):
            with self.subTest(config=unsupported):
                self.assertFalse(is_oscar_bulk_flush_supported(unsupported))

    def test_bulk_flush_pointer_table_preserves_layer_order(self):
        layer_names = [f"layer.{index}" for index in range(36)]
        caches = {
            name: [
                torch.empty((2, 16, 8, 72), dtype=torch.uint8),
                torch.empty((64, 8, 2, 128), dtype=torch.bfloat16),
                torch.empty((272, 8, 2, 128), dtype=torch.bfloat16),
            ]
            for name in layer_names
        }
        state = register_oscar_bulk_flush_caches(
            caches,
            layer_names,
            expected_recent_capacity=272,
            max_num_seqs=1,
        )

        self.assertEqual(state.layer_names, tuple(layer_names))
        self.assertEqual(
            state.quant_ptrs.tolist(),
            [caches[name][0].data_ptr() for name in layer_names],
        )
        self.assertEqual(
            state.recent_ptrs.tolist(),
            [caches[name][2].data_ptr() for name in layer_names],
        )
        self.assertEqual(state.recent_capacity, 272)
        self.assertEqual(state.quant_stride, caches["layer.0"][0].stride())
        self.assertEqual(state.recent_stride, caches["layer.0"][2].stride())

    def test_bulk_flush_reset_and_generation_prevent_row_aba(self):
        phase = torch.tensor([7], dtype=torch.int32)
        stored_generation = torch.tensor([4], dtype=torch.int64)
        buffers = {
            "recent_extra": torch.zeros(1, dtype=torch.int32),
            "positions": torch.full((1, 8), -1, dtype=torch.int32),
            "src_recent_slots": torch.full((1, 8), -1, dtype=torch.int64),
            "dst_slots": torch.full((1, 8), -1, dtype=torch.int64),
            "valid": torch.zeros((1, 8), dtype=torch.bool),
        }
        common = {
            "phase": phase,
            "row_generations": stored_generation,
            "cached_lens": torch.tensor([321], dtype=torch.int32),
            "hp_row_ids": torch.tensor([0], dtype=torch.int32),
            "shared_hit_tokens": torch.tensor([0], dtype=torch.int32),
            "block_table": torch.arange(64, dtype=torch.int32).view(1, -1),
            "prefix_tokens": 64,
            "recent_tokens": 256,
            "recent_capacity": 272,
            "block_size": 16,
            "flush_interval": 8,
            **buffers,
        }

        reset_plan = prepare_oscar_bulk_flush_plan(
            reset_mask=torch.tensor([True]),
            request_generations=torch.tensor([5], dtype=torch.int64),
            **common,
        )
        self.assertEqual(reset_plan.recent_extra.tolist(), [1])
        self.assertFalse(reset_plan.valid.any())
        self.assertEqual(stored_generation.tolist(), [5])

        phase.fill_(7)
        aba_plan = prepare_oscar_bulk_flush_plan(
            reset_mask=torch.tensor([False]),
            request_generations=torch.tensor([6], dtype=torch.int64),
            **common,
        )
        self.assertEqual(aba_plan.recent_extra.tolist(), [1])
        self.assertFalse(aba_plan.valid.any())
        self.assertEqual(stored_generation.tolist(), [6])

    def test_bulk_flush_capture_does_not_advance_phase(self):
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8,
                max_num_seqs=1,
                enable_chunked_prefill=True,
            ),
            cache_config=SimpleNamespace(
                kv_cache_memory_bytes=None,
                enable_prefix_caching=False,
            ),
            model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=8),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            speculative_config=None,
            kv_transfer_config=SimpleNamespace(kv_connector=None),
            offload_config=SimpleNamespace(cpu_offload_gb=0),
        )
        spec = SimpleNamespace(
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            block_size=16,
            quant_slot_size=72,
            group_size=128,
            prefix_tokens=64,
            recent_tokens=256,
            recent_row_capacity=272,
            flush_interval=8,
            hp_dtype=torch.bfloat16,
        )
        layer_names = [f"layer.{index}" for index in range(36)]
        builder = OscarMetadataBuilder(spec, layer_names, vllm_config, "cpu")
        metadata = CommonAttentionMetadata(
            query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
            query_start_loc_cpu=torch.tensor([0, 1], dtype=torch.int32),
            seq_lens=torch.tensor([7], dtype=torch.int32),
            _seq_lens_cpu=torch.tensor([7], dtype=torch.int32),
            seq_lens_cpu_upper_bound=torch.tensor([7], dtype=torch.int32),
            num_reqs=1,
            num_actual_tokens=1,
            max_query_len=1,
            max_seq_len=7,
            block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
            slot_mapping=torch.tensor([6], dtype=torch.int64),
            oscar_hp_row_ids=torch.tensor([0], dtype=torch.int32),
            oscar_prefix_page_ids=torch.tensor(
                [[0, 1, 2, 3]], dtype=torch.int32
            ),
            oscar_shared_hit_tokens=torch.tensor([0], dtype=torch.int32),
            oscar_reset_mask=torch.tensor([True]),
            oscar_row_generations=torch.tensor([1], dtype=torch.int64),
        )
        builder.flush_phase.fill_(7)

        with patch(
            "vllm.v1.attention.backends.oscar_attn.materialize_oscar_slot_ids"
        ):
            captured = builder.build_for_cudagraph_capture(metadata)

        self.assertEqual(builder.flush_phase.tolist(), [7])
        self.assertEqual(builder.row_generations.tolist(), [-1])
        self.assertEqual(captured.bulk_flush_plan.recent_extra.tolist(), [0])
        self.assertFalse(captured.bulk_flush_plan.valid.any())

    def test_bulk_flush_registration_fails_closed(self):
        quant = torch.empty((2, 16, 8, 72), dtype=torch.uint8)
        prefix = torch.empty((128, 8, 2, 128), dtype=torch.bfloat16)
        recent = torch.empty((544, 8, 2, 128), dtype=torch.bfloat16)
        caches = {"layer.0": [quant, prefix, recent]}

        with self.assertRaisesRegex(ValueError, "unique"):
            register_oscar_bulk_flush_caches(
                caches,
                ["layer.0", "layer.0"],
                expected_recent_capacity=272,
                max_num_seqs=2,
            )
        with self.assertRaisesRegex(ValueError, "missing"):
            register_oscar_bulk_flush_caches(
                caches,
                ["layer.1"],
                expected_recent_capacity=272,
                max_num_seqs=2,
            )
        with self.assertRaisesRegex(ValueError, "physical row capacity"):
            register_oscar_bulk_flush_caches(
                caches,
                ["layer.0"],
                expected_recent_capacity=256,
                max_num_seqs=2,
            )
        qwen_quant = torch.empty((2, 16, 8, 72), dtype=torch.uint8)
        qwen_prefix = torch.empty((64, 8, 2, 128), dtype=torch.bfloat16)
        qwen_recent = torch.empty((272, 8, 2, 128), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "uint8"):
            register_oscar_bulk_flush_caches(
                {
                    "layer.0": [
                        qwen_quant.to(torch.int8),
                        qwen_prefix,
                        qwen_recent,
                    ]
                },
                ["layer.0"],
                expected_recent_capacity=272,
                max_num_seqs=1,
            )
        with self.assertRaisesRegex(ValueError, "shapes"):
            register_oscar_bulk_flush_caches(
                {
                    "layer.0": [
                        qwen_quant[:, :, :4],
                        qwen_prefix,
                        qwen_recent,
                    ]
                },
                ["layer.0"],
                expected_recent_capacity=272,
                max_num_seqs=1,
            )
        padded_recent = torch.empty(
            (272, 8, 2, 129), dtype=torch.bfloat16
        )[..., :128]
        with self.assertRaisesRegex(ValueError, "combined K/V offset"):
            register_oscar_bulk_flush_caches(
                {"layer.0": [qwen_quant, qwen_prefix, padded_recent]},
                ["layer.0"],
                expected_recent_capacity=272,
                max_num_seqs=1,
            )

    def test_bulk_flush_binding_has_exactly_one_owner(self):
        layer_names = [f"layer.{index}" for index in range(36)]
        caches = {
            name: [
                torch.empty((2, 16, 8, 72), dtype=torch.uint8),
                torch.empty((64, 8, 2, 128), dtype=torch.bfloat16),
                torch.empty((272, 8, 2, 128), dtype=torch.bfloat16),
            ]
            for name in layer_names
        }
        state = register_oscar_bulk_flush_caches(
            caches,
            layer_names,
            expected_recent_capacity=272,
            max_num_seqs=1,
        )
        context = {name: SimpleNamespace() for name in layer_names}

        pointers_before = {
            name: tuple(tensor.data_ptr() for tensor in caches[name])
            for name in layer_names
        }
        versions_before = {
            name: tuple(tensor._version for tensor in caches[name])
            for name in layer_names
        }
        with patch("torch.empty", side_effect=AssertionError("unexpected allocate")):
            bind_oscar_bulk_flush_state(context, state)

        self.assertTrue(context["layer.0"]._oscar_bulk_flush_owner)
        self.assertEqual(
            sum(context[name]._oscar_bulk_flush_owner for name in layer_names), 1
        )
        for name in layer_names:
            self.assertIs(context[name]._oscar_bulk_flush_state, state)
            self.assertEqual(
                tuple(tensor.data_ptr() for tensor in caches[name]),
                pointers_before[name],
            )
            self.assertEqual(
                tuple(tensor._version for tensor in caches[name]),
                versions_before[name],
            )

        clear_oscar_bulk_flush_state(context)
        self.assertEqual(
            sum(context[name]._oscar_bulk_flush_owner for name in layer_names), 0
        )
        self.assertTrue(
            all(
                context[name]._oscar_bulk_flush_state is None
                for name in layer_names
            )
        )

    def test_bulk_flush_target_gate_is_shape_exact(self):
        spec = SimpleNamespace(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            quant_slot_size=72,
            group_size=128,
            prefix_tokens=64,
            recent_tokens=256,
            prefix_cache_extra_tokens=0,
            flush_interval=8,
            recent_row_capacity=272,
            hp_dtype=torch.bfloat16,
        )
        layers = [f"layer.{index}" for index in range(36)]

        self.assertTrue(is_oscar_bulk_flush_target(spec, layers))
        for field, value in (
            ("block_size", 32),
            ("num_kv_heads", 4),
            ("head_size", 64),
            ("quant_slot_size", 80),
            ("group_size", 64),
            ("prefix_cache_extra_tokens", 16),
            ("recent_row_capacity", 256),
        ):
            original = getattr(spec, field)
            setattr(spec, field, value)
            with self.subTest(field=field):
                self.assertFalse(is_oscar_bulk_flush_target(spec, layers))
            setattr(spec, field, original)
        self.assertFalse(is_oscar_bulk_flush_target(spec, layers[:-1]))

    def test_bulk_flush_scheduler_lifecycle_gate_rejects_unsupported_ops(self):
        config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_seqs=1, enable_chunked_prefill=True
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            speculative_config=None,
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            kv_transfer_config=SimpleNamespace(kv_connector=None),
            offload_config=SimpleNamespace(cpu_offload_gb=0),
        )

        def output(*, preempted=False, resumed=False, request=None):
            return SimpleNamespace(
                preempted_req_ids={"r0"} if preempted else set(),
                scheduled_cached_reqs=SimpleNamespace(
                    resumed_req_ids={"r0"} if resumed else set()
                ),
                scheduled_new_reqs=[] if request is None else [request],
            )

        validate_oscar_bulk_flush_scheduler_output(config, output(), enabled=True)
        generations = np.zeros(1, dtype=np.int64)
        reset, request_generations = update_oscar_bulk_flush_row_generations(
            generations, ["new-a"], [0], {"new-a"}
        )
        self.assertEqual((reset[0], request_generations[0]), (True, 1))
        reset, request_generations = update_oscar_bulk_flush_row_generations(
            generations, ["new-a"], [0], set()
        )
        self.assertEqual((reset[0], request_generations[0]), (False, 1))
        reset, request_generations = update_oscar_bulk_flush_row_generations(
            generations, ["new-b"], [0], {"new-b"}
        )
        self.assertEqual((reset[0], request_generations[0]), (True, 2))

        for label, unsafe in (
            ("preempt", output(preempted=True)),
            ("resume", output(resumed=True)),
            (
                "fanout",
                output(
                    request=SimpleNamespace(
                        sampling_params=SimpleNamespace(n=2),
                        num_computed_tokens=0,
                    )
                )
            ),
            (
                "copy",
                output(
                    request=SimpleNamespace(
                        sampling_params=SimpleNamespace(n=1),
                        num_computed_tokens=8,
                    )
                )
            ),
            (
                "fork",
                output(
                    request=SimpleNamespace(
                        sampling_params=SimpleNamespace(n=1),
                        num_computed_tokens=16,
                    )
                )
            ),
        ):
            with self.subTest(case=label):
                before = generations.copy()
                with self.assertRaises(RuntimeError):
                    validate_oscar_bulk_flush_scheduler_output(
                        config, unsafe, enabled=True
                    )
                np.testing.assert_array_equal(generations, before)

        from vllm.v1.worker.gpu.model_runner import (
            GPUModelRunner as ModularGPUModelRunner,
        )
        from vllm.v1.worker.gpu_model_runner import (
            GPUModelRunner as LegacyGPUModelRunner,
        )

        for runner_cls, prepare_name, first_mutation in (
            (ModularGPUModelRunner, "prepare_inputs", "update_pp_decode_requests"),
            (LegacyGPUModelRunner, "_prepare_inputs", "execute_model_state"),
        ):
            execute_source = inspect.getsource(runner_cls.execute_model)
            self.assertLess(
                execute_source.index("validate_oscar_bulk_flush_scheduler_output"),
                execute_source.index(first_mutation),
            )
            prepare_source = inspect.getsource(getattr(runner_cls, prepare_name))
            self.assertIn(
                "update_oscar_bulk_flush_row_generations", prepare_source
            )
            runner = object.__new__(runner_cls)
            runner.vllm_config = config
            runner._oscar_bulk_flush_enabled = True
            sentinel = object()
            runner.pre_validation_sentinel = sentinel
            for label, unsafe in (
                ("preempt", output(preempted=True)),
                ("resume", output(resumed=True)),
                (
                    "fanout",
                    output(
                        request=SimpleNamespace(
                            sampling_params=SimpleNamespace(n=2),
                            num_computed_tokens=0,
                        )
                    ),
                ),
                (
                    "copy",
                    output(
                        request=SimpleNamespace(
                            sampling_params=SimpleNamespace(n=1),
                            num_computed_tokens=8,
                        )
                    ),
                ),
                (
                    "fork",
                    output(
                        request=SimpleNamespace(
                            sampling_params=SimpleNamespace(n=1),
                            num_computed_tokens=16,
                        )
                    ),
                ),
            ):
                before = {
                    name: id(value) for name, value in runner.__dict__.items()
                }
                with self.subTest(runner=runner_cls.__name__, case=label):
                    with self.assertRaises(RuntimeError):
                        runner_cls.execute_model(runner, unsafe)
                    self.assertEqual(
                        {
                            name: id(value)
                            for name, value in runner.__dict__.items()
                        },
                        before,
                    )
                    self.assertIs(runner.pre_validation_sentinel, sentinel)

    def test_bulk_flush_shared_hit_lower_bound_masks_plan(self):
        plan = build_oscar_bulk_flush_plan_cpu(
            phase=torch.tensor([7], dtype=torch.int32),
            cached_lens=torch.tensor([337], dtype=torch.int32),
            hp_row_ids=torch.tensor([1], dtype=torch.int32),
            shared_hit_tokens=torch.tensor([78], dtype=torch.int32),
            block_table=torch.arange(32, dtype=torch.int32).view(1, -1),
            prefix_tokens=64,
            recent_tokens=256,
            recent_capacity=272,
            block_size=16,
            flush_interval=8,
        )

        self.assertEqual(
            plan.positions.tolist(), [[-1, -1, -1, -1, 78, 79, 80, 81]]
        )
        self.assertEqual(plan.valid.tolist(), [[False] * 4 + [True] * 4])
        self.assertEqual(plan.src_recent_slots[0, 4:].tolist(), [286, 287, 288, 289])

    def test_execution_mode_support(self):
        self.assertTrue(
            _is_oscar_execution_mode_supported(
                True,
                CompilationConfig(
                    mode=CompilationMode.NONE,
                    cudagraph_mode=CUDAGraphMode.NONE,
                ),
            )
        )
        self.assertTrue(
            _is_oscar_execution_mode_supported(
                False,
                CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                    cudagraph_mode=CUDAGraphMode.PIECEWISE,
                ),
            )
        )
        self.assertTrue(
            _is_oscar_execution_mode_supported(
                False,
                CompilationConfig(),
                OptimizationLevel.O2,
            )
        )
        self.assertFalse(
            _is_oscar_execution_mode_supported(
                False,
                CompilationConfig(),
                OptimizationLevel.O0,
            )
        )
        self.assertTrue(
            _is_oscar_execution_mode_supported(
                False,
                CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                    cudagraph_mode=CUDAGraphMode.FULL,
                ),
            )
        )
        self.assertTrue(
            _is_oscar_execution_mode_supported(
                False,
                CompilationConfig(
                    mode=CompilationMode.VLLM_COMPILE,
                    cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                ),
            )
        )
        for mode, cudagraph_mode in (
            (CompilationMode.NONE, CUDAGraphMode.NONE),
            (CompilationMode.VLLM_COMPILE, CUDAGraphMode.NONE),
            (CompilationMode.VLLM_COMPILE, CUDAGraphMode.FULL_DECODE_ONLY),
        ):
            with self.subTest(mode=mode, cudagraph_mode=cudagraph_mode):
                self.assertFalse(
                    _is_oscar_execution_mode_supported(
                        False,
                        CompilationConfig(
                            mode=mode,
                            cudagraph_mode=cudagraph_mode,
                        ),
                    )
                )

    def test_full_cudagraph_support_is_fail_closed(self):
        def config(*, max_num_seqs=1, chunked=False, prefix=False, spec=None):
            return SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    max_num_seqs=max_num_seqs,
                    enable_chunked_prefill=chunked,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=prefix),
                speculative_config=spec,
            )

        kv_cache_spec = SimpleNamespace()
        self.assertEqual(
            OscarMetadataBuilder.get_cudagraph_support(config(), kv_cache_spec),
            AttentionCGSupport.ALWAYS,
        )
        self.assertEqual(
            OscarMetadataBuilder.get_cudagraph_support(
                config(chunked=True), kv_cache_spec
            ),
            AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE,
        )
        for unsafe_config in (
            config(max_num_seqs=2),
            config(max_num_seqs=2, chunked=True),
            config(prefix=True),
            config(spec=SimpleNamespace()),
        ):
            with self.subTest(config=unsafe_config):
                self.assertEqual(
                    OscarMetadataBuilder.get_cudagraph_support(
                        unsafe_config, kv_cache_spec
                    ),
                    AttentionCGSupport.NEVER,
                )

    def test_full_cudagraph_metadata_reuses_persistent_buffers(self):
        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8,
                max_num_seqs=1,
                enable_chunked_prefill=False,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(dtype=torch.bfloat16, max_model_len=8),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                use_ubatching=False,
            ),
            speculative_config=None,
        )
        kv_cache_spec = SimpleNamespace(
            num_kv_heads=1,
            head_size=128,
            block_size=4,
        )
        layer_names = [
            f"model.layers.{index}.self_attn.attn" for index in range(36)
        ]
        builder = OscarMetadataBuilder(
            kv_cache_spec, layer_names, vllm_config, "cpu"
        )

        def common_metadata(seq_len, query_len, padded_tokens, blocks=(3, 1)):
            return CommonAttentionMetadata(
                query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
                query_start_loc_cpu=torch.tensor(
                    [0, query_len], dtype=torch.int32
                ),
                seq_lens=torch.tensor([seq_len], dtype=torch.int32),
                _seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
                seq_lens_cpu_upper_bound=torch.tensor(
                    [seq_len], dtype=torch.int32
                ),
                num_reqs=1,
                num_actual_tokens=padded_tokens,
                max_query_len=query_len,
                max_seq_len=seq_len,
                block_table_tensor=torch.tensor([blocks], dtype=torch.int32),
                slot_mapping=torch.full(
                    (padded_tokens,), -1, dtype=torch.int64
                ),
                oscar_hp_row_ids=torch.tensor([0], dtype=torch.int32),
                oscar_prefix_page_ids=torch.zeros((1, 4), dtype=torch.int32),
                oscar_shared_hit_tokens=torch.tensor([0], dtype=torch.int32),
            )

        materialize_calls = []

        def materialize(block_table, seq_lens, physical_slot_ids, block_size):
            materialize_calls.append(
                (physical_slot_ids.data_ptr(), tuple(builder.layer_names))
            )
            for req_idx, seq_len in enumerate(seq_lens.tolist()):
                logical = torch.arange(seq_len, dtype=torch.int64)
                blocks = block_table[req_idx, logical // block_size].to(torch.int64)
                physical_slot_ids[req_idx, :seq_len] = (
                    blocks * block_size + logical % block_size
                )

        with (
            patch(
                "vllm.v1.attention.backend.np_to_pinned_tensor",
                side_effect=torch.from_numpy,
            ),
            patch(
                "vllm.v1.attention.backends.oscar_attn."
                "materialize_oscar_slot_ids",
                side_effect=materialize,
            ),
        ):
            captured = builder.build_for_cudagraph_capture(
                common_metadata(seq_len=4, query_len=4, padded_tokens=4)
            )
            self.assertIsNone(captured.physical_slot_ids)
            pointers = (
                captured.token_to_req_indices.data_ptr(),
                captured.seq_start_loc.data_ptr(),
                captured.cached_lens.data_ptr(),
                builder.physical_slot_ids.data_ptr(),
            )
            self.assertEqual(captured.seq_lens.tolist(), [4])

            decode_capture = builder.build_for_cudagraph_capture(
                common_metadata(seq_len=7, query_len=1, padded_tokens=1)
            )
            self.assertEqual(decode_capture.seq_lens.tolist(), [1])
            self.assertEqual(
                decode_capture.physical_slot_ids[0, :7].tolist(),
                [12, 13, 14, 15, 4, 5, 6],
            )
            self.assertEqual(decode_capture.physical_slot_ids.dtype, torch.int64)
            self.assertEqual(decode_capture.physical_slot_ids.shape, (1, 8))
            self.assertEqual(decode_capture.physical_slot_ids.stride(), (8, 1))
            capture_storage_ptr = (
                decode_capture.physical_slot_ids.untyped_storage().data_ptr()
            )

            replay = builder.build(
                0,
                common_metadata(
                    seq_len=8,
                    query_len=1,
                    padded_tokens=1,
                    blocks=(3, 5),
                ),
            )
        self.assertEqual(
            pointers,
            (
                replay.token_to_req_indices.data_ptr(),
                replay.seq_start_loc.data_ptr(),
                replay.cached_lens.data_ptr(),
                replay.physical_slot_ids.data_ptr(),
            ),
        )
        self.assertEqual(replay.seq_start_loc.tolist(), [0, 8])
        self.assertEqual(replay.cached_lens.tolist(), [7])
        self.assertEqual(
            replay.physical_slot_ids[0, :8].tolist(),
            [12, 13, 14, 15, 20, 21, 22, 23],
        )
        self.assertEqual(replay.physical_slot_ids.shape, (1, 8))
        self.assertEqual(replay.physical_slot_ids.stride(), (8, 1))
        self.assertEqual(replay.physical_slot_ids.dtype, torch.int64)
        self.assertEqual(
            replay.physical_slot_ids.untyped_storage().data_ptr(),
            capture_storage_ptr,
        )
        self.assertEqual(len(materialize_calls), 2)
        self.assertEqual(
            {call[0] for call in materialize_calls},
            {builder.physical_slot_ids.data_ptr()},
        )
        self.assertTrue(
            all(call[1] == tuple(layer_names) for call in materialize_calls)
        )

    def test_task_defaults_and_geometry(self):
        cfg = OscarConfig()
        self.assertEqual((cfg.prefix_tokens, cfg.recent_tokens), (64, 256))
        self.assertEqual((cfg.k_clip_ratio, cfg.v_clip_ratio), (0.96, 0.92))
        self.assertEqual(cfg.slot_size_aligned, 72)
        self.assertEqual(cfg.bf16_slot_size, 512)

    def test_mixed_layout_boundaries(self):
        expected = {
            32: (32, 0, 0),
            63: (63, 0, 0),
            64: (64, 0, 0),
            65: (64, 0, 1),
            200: (64, 0, 136),
            319: (64, 0, 255),
            320: (64, 0, 256),
            321: (64, 1, 256),
            1024: (64, 704, 256),
        }
        for seq_len, counts in expected.items():
            with self.subTest(seq_len=seq_len):
                part = partition_tokens(seq_len, prefix_tokens=64, recent_tokens=256)
                self.assertEqual(
                    (part.prefix_count, part.history_count, part.recent_count), counts
                )
                self.assertEqual(part.total_count, seq_len)

    def test_mixed_bytes_include_windows(self):
        cfg = OscarConfig()
        self.assertEqual(cfg.mixed_bytes_per_token_per_layer(64, 8), 4096)
        self.assertEqual(cfg.mixed_bytes_per_token_per_layer(320, 8), 4096)
        expected = ((320 * 512) + (704 * 72)) * 8
        self.assertEqual(cfg.mixed_bytes_per_layer(1024, 8), expected)

    def test_padded_page_reserves_bounded_bf16_arena(self):
        cfg = OscarConfig()
        self.assertEqual(cfg.hp_slots_per_block(16, 1024), 5)
        self.assertEqual(cfg.hp_slots_per_block(16, 4096), 2)
        self.assertEqual(cfg.hp_slots_per_block(16, 8192), 1)
        self.assertEqual(
            cfg.padded_page_size_bytes(16, 8, 8192),
            16 * 8 * 72 + 1 * 8 * 512,
        )

    def test_invalid_layout(self):
        for args in [(-1, 64, 256), (1, -1, 256), (1, 64, -1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                partition_tokens(args[0], prefix_tokens=args[1], recent_tokens=args[2])

    def test_prototype_validation_rejects_missing_calibration(self):
        with self.assertRaisesRegex(ValueError, "K_ROTATION_PATH"):
            OscarConfig().validate_prototype_settings()

    def test_prototype_validation_allows_block_aligned_recent_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            k_path = Path(tmp) / "k.pt"
            v_path = Path(tmp) / "v.pt"
            k_path.touch()
            v_path.touch()
            OscarConfig(
                recent_tokens=512,
                k_rotation_path=str(k_path),
                v_rotation_path=str(v_path),
            ).validate_prototype_settings()

    def test_prototype_validation_rejects_unaligned_recent_window(self):
        with self.assertRaisesRegex(ValueError, "recent_tokens.*block aligned"):
            OscarConfig(recent_tokens=500).validate_prototype_settings()

    def test_absorb_v_rotation_env_flag(self):
        with patch.dict(os.environ, {"VLLM_OSCAR_ABSORB_V_ROTATION": "1"}):
            self.assertTrue(
                OscarConfig.from_cache_dtype("oscar_int2", 128).absorb_v_rotation
            )
        with patch.dict(os.environ, {"VLLM_OSCAR_ABSORB_V_ROTATION": "false"}):
            self.assertFalse(
                OscarConfig.from_cache_dtype("oscar_int2", 128).absorb_v_rotation
            )
        with (
            patch.dict(os.environ, {"VLLM_OSCAR_ABSORB_V_ROTATION": "invalid"}),
            self.assertRaisesRegex(ValueError, "VLLM_OSCAR_ABSORB_V_ROTATION"),
        ):
            OscarConfig.from_cache_dtype("oscar_int2", 128)

    def test_serving_materialize_capacity_is_bounded_and_fail_closed(self):
        from vllm.v1.attention.backends import oscar_attn

        def config(
            *,
            chunked=True,
            max_num_seqs=1,
            prefix=False,
            spec=None,
            use_ubatching=False,
            max_model_len=33792,
        ):
            return SimpleNamespace(
                scheduler_config=SimpleNamespace(
                    enable_chunked_prefill=chunked,
                    max_num_seqs=max_num_seqs,
                ),
                cache_config=SimpleNamespace(enable_prefix_caching=prefix),
                speculative_config=spec,
                parallel_config=SimpleNamespace(use_ubatching=use_ubatching),
                model_config=SimpleNamespace(max_model_len=max_model_len),
            )

        with patch.object(oscar_attn, "_HAS_FLASH_ATTN", True):
            capacity = oscar_attn._materialize_token_capacity(
                config(), 8, 128, torch.bfloat16
            )
            self.assertEqual(capacity, 33792)
            self.assertEqual(
                capacity * 2 * 8 * 128 * torch.bfloat16.itemsize,
                132 * 1024**2,
            )

            for unsafe_config in (
                config(chunked=False),
                config(max_num_seqs=2),
                config(prefix=True),
                config(spec=SimpleNamespace()),
                config(use_ubatching=True),
                config(max_model_len=0),
                config(max_model_len=-1),
                config(max_model_len=33792.0),
                config(max_model_len=True),
                config(max_model_len=None),
            ):
                with self.subTest(config=unsafe_config):
                    self.assertEqual(
                        oscar_attn._materialize_token_capacity(
                            unsafe_config, 8, 128, torch.bfloat16
                        ),
                        0,
                    )

        with patch.object(oscar_attn, "_HAS_FLASH_ATTN", False):
            self.assertEqual(
                oscar_attn._materialize_token_capacity(
                    config(), 8, 128, torch.bfloat16
                ),
                0,
            )

    def test_eligible_materialize_builder_reserves_two_bf16_views(self):
        from vllm.v1.attention.backends import oscar_attn

        vllm_config = SimpleNamespace(
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8192,
                max_num_seqs=1,
                enable_chunked_prefill=True,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            model_config=SimpleNamespace(
                dtype=torch.bfloat16,
                max_model_len=33792,
            ),
            parallel_config=SimpleNamespace(
                decode_context_parallel_size=1,
                use_ubatching=False,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
            speculative_config=None,
        )
        kv_cache_spec = SimpleNamespace(
            num_kv_heads=8,
            head_size=128,
            block_size=16,
        )
        calls = []
        workspace = SimpleNamespace(
            get_simultaneous=lambda *args: calls.append(args)
        )

        with (
            patch.object(oscar_attn, "_HAS_FLASH_ATTN", True),
            patch.object(
                oscar_attn,
                "is_workspace_manager_initialized",
                return_value=True,
            ),
            patch.object(
                oscar_attn,
                "current_workspace_manager",
                return_value=workspace,
            ),
        ):
            OscarMetadataBuilder(
                kv_cache_spec,
                ["model.layers.0.self_attn.attn"],
                vllm_config,
                "cpu",
            )

        workspace_spec = ((33792, 8, 128), torch.bfloat16)
        self.assertEqual(calls, [(workspace_spec, workspace_spec)])

    def test_grouped_h4_decode_stage1_dispatch_is_narrow(self):
        from vllm.v1.attention.ops.triton_oscar_decode import (
            _has_linear_physical_slot_layout,
            _use_grouped_h4_stage1,
        )

        self.assertTrue(_use_grouped_h4_stage1(32, 8, 128))
        self.assertTrue(_use_grouped_h4_stage1(8, 2, 128))
        for shape in ((32, 8, 64), (32, 4, 128), (8, 8, 128), (7, 2, 128)):
            with self.subTest(shape=shape):
                self.assertFalse(_use_grouped_h4_stage1(*shape))

        contiguous = torch.empty(4, 16, 8, 72, dtype=torch.uint8)
        self.assertTrue(_has_linear_physical_slot_layout(contiguous))
        self.assertFalse(_has_linear_physical_slot_layout(contiguous[:, ::2]))

    def test_d128_quarter_layout_and_d64_legacy_layout_are_explicit(self):
        from vllm.v1.attention.ops.triton_oscar_store import (
            _int2_byte_index_and_shift,
        )

        for dim in range(128):
            byte_idx, shift = _int2_byte_index_and_shift(dim, 128)
            self.assertEqual(byte_idx, dim % 32)
            self.assertEqual(shift, (dim // 32) * 2)
        for dim in range(64):
            byte_idx, shift = _int2_byte_index_and_shift(dim, 64)
            self.assertEqual(byte_idx, dim // 4)
            self.assertEqual(shift, (dim % 4) * 2)

    def test_grouped_h4_split_contract_is_mixed_20(self):
        from vllm.v1.attention.ops.triton_oscar_decode import (
            _grouped_h4_partial_counts,
        )

        self.assertEqual(_grouped_h4_partial_counts(12, True), (8, 20))
        self.assertEqual(_grouped_h4_partial_counts(12, False), (1, 13))

    def test_grouped_h4_quant_reader_uses_preindexed_slots(self):
        from vllm.v1.attention.ops import triton_oscar_decode

        source = Path(triton_oscar_decode.__file__).read_text()
        kernel_start = source.index(
            "def _oscar_decode_quant_stage1_grouped_h4("
        )
        kernel_end = source.index("\n\n@triton.jit", kernel_start)
        kernel_source = source[kernel_start:kernel_end]
        self.assertIn("Physical_slot_ids_ptr", kernel_source)
        self.assertIn("physical_slots * stride_cache_pos", kernel_source)
        self.assertNotIn("Block_table_ptr", kernel_source)
        self.assertNotIn("kv_offs // BLOCK_SIZE", kernel_source)
        self.assertNotIn("kv_offs % BLOCK_SIZE", kernel_source)

        materialize_start = source.index("def _materialize_oscar_slot_ids_kernel(")
        materialize_end = source.index("\n\ndef materialize_oscar_slot_ids(")
        materialize_source = source[materialize_start:materialize_end]
        self.assertIn("offsets // BLOCK_SIZE", materialize_source)
        self.assertIn("page_offsets = offsets % BLOCK_SIZE", materialize_source)
        self.assertIn(
            "physical_blocks * BLOCK_SIZE + page_offsets.to(tl.int64)",
            materialize_source,
        )

        dispatch_source = source[source.index("def oscar_decode_attention(") :]
        self.assertIn(
            "_GROUPED_H4_PREINDEXED_QUANT_SPLITS if grouped_h4",
            dispatch_source,
        )
        self.assertIn("physical_slot_ids.stride(0)", dispatch_source)
        self.assertIn("_has_linear_physical_slot_layout(kv_cache)", dispatch_source)

        backend_source = (
            Path(triton_oscar_decode.__file__).parents[1] / "backends" / "oscar_attn.py"
        ).read_text()
        build_start = backend_source.index("    def build(")
        build_end = backend_source.index("\n\n\nclass OscarAttentionImpl", build_start)
        build_source = backend_source[build_start:build_end]
        self.assertIn("materialize_oscar_slot_ids(", build_source)
        self.assertNotIn("synchronize", build_source)

    def test_grouped_h4_hp_stage1_uses_dot_and_eight_splits(self):
        from vllm.v1.attention.ops import triton_oscar_decode

        source = Path(triton_oscar_decode.__file__).read_text()
        kernel_start = source.index("def _oscar_decode_hp_stage1(")
        kernel_end = source.index("\n\n@triton.jit", kernel_start)
        kernel_source = source[kernel_start:kernel_end]
        self.assertIn("sid = tl.program_id(2)", kernel_source)
        self.assertIn("heads = head0 + tl.arange(0, BLOCK_H)", kernel_source)
        self.assertIn("head_mask = heads < head0 + KV_GROUP_SIZE", kernel_source)
        self.assertIn(
            "tl.cdiv(tl.cdiv(hp_len, NUM_HP_SPLITS), BLOCK_N) * BLOCK_N",
            kernel_source,
        )
        hp_tokens = 320
        split_len = ((hp_tokens + 7) // 8 + 31) // 32 * 32
        effective_splits = (hp_tokens + split_len - 1) // split_len
        self.assertEqual((split_len, effective_splits), (64, 5))
        self.assertEqual(effective_splits * (split_len // 32), 10)
        self.assertIn("scores = tl.dot(query, keys)", kernel_source)
        self.assertIn("tl.dot(probs.to(tl.bfloat16), values)", kernel_source)

        dispatch_source = source[source.index("def oscar_decode_attention(") :]
        self.assertIn(
            "_oscar_decode_quant_stage1_grouped_h4[(B, Hk, NUM_KV_SPLITS)](",
            dispatch_source,
        )
        self.assertIn(
            "_oscar_decode_hp_stage1[(B, Hk, NUM_HP_SPLITS)](",
            dispatch_source,
        )
        self.assertIn("HP_PARTIAL_START=NUM_KV_SPLITS", dispatch_source)
        self.assertIn("BLOCK_N=32", dispatch_source)
        self.assertIn("BLOCK_H=16", dispatch_source)
        self.assertIn(
            "_grouped_h4_partial_counts(NUM_KV_SPLITS, mixed_kv)",
            dispatch_source,
        )
        self.assertIn("NUM_PARTIALS=NUM_TOTAL_SPLITS", dispatch_source)

    def test_grouped_h4_uses_private_finite_lse_reducer(self):
        from vllm.v1.attention.ops import triton_oscar_decode

        source = Path(triton_oscar_decode.__file__).read_text()
        kernel_start = source.index("def _oscar_finite_lse_stage2(")
        kernel_end = source.index("\n\n@triton.jit", kernel_start)
        kernel_source = source[kernel_start:kernel_end]
        self.assertIn("for partial_idx in range(0, NUM_PARTIALS):", kernel_source)
        self.assertIn("is_finite =", kernel_source)
        self.assertIn("tlogic > -float(\"inf\")", kernel_source)
        self.assertIn("tlogic < float(\"inf\")", kernel_source)
        self.assertIn("safe_e_sum = tl.where(e_sum > 0.0, e_sum, 1.0)", kernel_source)
        self.assertIn(
            "out = tl.where(e_sum > 0.0, acc / safe_e_sum, 0.0)",
            kernel_source,
        )
        self.assertIn(
            "lse_out = tl.where(\n"
            '        e_sum > 0.0, e_max + tl.log(safe_e_sum), -float("inf")\n'
            "    )",
            kernel_source,
        )

        dispatch_source = source[source.index("def oscar_decode_attention(") :]
        self.assertIn(
            "if grouped_h4:\n        _oscar_finite_lse_stage2[grid2](",
            dispatch_source,
        )
        self.assertIn("else:\n        _fwd_kernel_stage2[grid2](", dispatch_source)


class TestOscarRotationLoading(unittest.TestCase):
    def tearDown(self):
        _load_checkpoint.cache_clear()

    def test_rotation_loader_requires_every_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotation.pt"
            torch.save({"layers": {0: {"rotation": torch.eye(4)}}}, path)
            got = get_layer_rotation(
                str(path),
                "model.layers.0.self_attn.attn",
                4,
                torch.device("cpu"),
                torch.float32,
            )
            torch.testing.assert_close(got, torch.eye(4))
            with self.assertRaises(KeyError):
                get_layer_rotation(
                    str(path),
                    "model.layers.1.self_attn.attn",
                    4,
                    torch.device("cpu"),
                    torch.float32,
                )

    @unittest.skipUnless(
        os.environ.get("OSCAR_TEST_ROTATION_DIR"),
        "OSCAR_TEST_ROTATION_DIR is required for calibration artifact validation",
    )
    def test_qwen3_calibration_artifacts(self):
        root = Path(os.environ["OSCAR_TEST_ROTATION_DIR"])
        OscarConfig(
            k_rotation_path=str(root / "k_rotation_qqt_r_h_pbr.pt"),
            v_rotation_path=str(root / "v_rotation_sst_r_h_pbr.pt"),
        ).validate_prototype_settings()
        for name in (
            "k_rotation_qqt_r_h_pbr.pt",
            "v_rotation_sst_r_h_pbr.pt",
        ):
            table = _load_checkpoint(str(root / name))
            self.assertEqual(len(table), 36)
            identity = torch.eye(128)
            for rotation in table.values():
                self.assertEqual(tuple(rotation.shape), (128, 128))
                self.assertEqual(rotation.dtype, torch.float32)
                self.assertEqual(rotation.device.type, "cpu")
                torch.testing.assert_close(
                    rotation.T @ rotation,
                    identity,
                    atol=1e-5,
                    rtol=1e-5,
                )


class TestOscarVRotationAbsorption(unittest.TestCase):
    def test_dense_qkv_weight_and_bias_match_runtime_rotation(self):
        torch.manual_seed(7)
        num_query_heads, num_kv_heads, head_dim, hidden_size = 4, 2, 8, 24
        q_size = num_query_heads * head_dim
        kv_size = num_kv_heads * head_dim
        weight = torch.randn(q_size + 2 * kv_size, hidden_size)
        bias = torch.randn(q_size + 2 * kv_size)
        original_weight = weight.clone()
        original_bias = bias.clone()
        rotation = torch.linalg.qr(torch.randn(head_dim, head_dim)).Q.contiguous()
        attn = SimpleNamespace(
            q_size=q_size,
            kv_size=kv_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            qkv_proj=SimpleNamespace(weight=weight, bias=bias, quant_method=None),
        )

        absorb_v_rotation_into_qkv(attn, rotation)

        torch.testing.assert_close(
            weight[: q_size + kv_size], original_weight[: q_size + kv_size]
        )
        torch.testing.assert_close(
            bias[: q_size + kv_size], original_bias[: q_size + kv_size]
        )
        inputs = torch.randn(11, hidden_size)
        v_offset = q_size + kv_size
        original_v = inputs @ original_weight[v_offset:].T + original_bias[v_offset:]
        folded_v = inputs @ weight[v_offset:].T + bias[v_offset:]
        expected = torch.matmul(
            original_v.view(11, num_kv_heads, head_dim), rotation
        ).view_as(folded_v)
        torch.testing.assert_close(folded_v, expected, atol=1e-5, rtol=1e-5)

    def test_dense_qkv_rejects_unsupported_layout(self):
        attn = SimpleNamespace(
            q_size=8,
            kv_size=4,
            num_kv_heads=1,
            head_dim=4,
            qkv_proj=SimpleNamespace(
                weight=torch.ones(16, 8, dtype=torch.int8),
                bias=None,
                quant_method=object(),
            ),
        )
        with self.assertRaisesRegex(RuntimeError, "dense floating-point"):
            absorb_v_rotation_into_qkv(attn, torch.eye(4))


if __name__ == "__main__":
    unittest.main()
