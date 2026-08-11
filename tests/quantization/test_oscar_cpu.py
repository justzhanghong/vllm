# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

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


class TestOscarConfigAndLayout(unittest.TestCase):
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
        kv_cache_spec = SimpleNamespace(num_kv_heads=1, head_size=128)
        builder = OscarMetadataBuilder(
            kv_cache_spec, ["model.layers.0.self_attn.attn"], vllm_config, "cpu"
        )

        def common_metadata(seq_len, query_len, padded_tokens):
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
                block_table_tensor=torch.zeros((1, 1), dtype=torch.int32),
                slot_mapping=torch.full(
                    (padded_tokens,), -1, dtype=torch.int64
                ),
                oscar_hp_row_ids=torch.tensor([0], dtype=torch.int32),
                oscar_prefix_page_ids=torch.zeros((1, 4), dtype=torch.int32),
                oscar_shared_hit_tokens=torch.tensor([0], dtype=torch.int32),
            )

        with patch(
            "vllm.v1.attention.backend.np_to_pinned_tensor",
            side_effect=torch.from_numpy,
        ):
            captured = builder.build_for_cudagraph_capture(
                common_metadata(seq_len=4, query_len=4, padded_tokens=4)
            )
            pointers = (
                captured.token_to_req_indices.data_ptr(),
                captured.seq_start_loc.data_ptr(),
                captured.cached_lens.data_ptr(),
            )
            self.assertEqual(captured.seq_lens.tolist(), [4])

            decode_capture = builder.build_for_cudagraph_capture(
                common_metadata(seq_len=7, query_len=1, padded_tokens=1)
            )
            self.assertEqual(decode_capture.seq_lens.tolist(), [1])

            replay = builder.build(
                0, common_metadata(seq_len=7, query_len=1, padded_tokens=1)
            )
        self.assertEqual(
            pointers,
            (
                replay.token_to_req_indices.data_ptr(),
                replay.seq_start_loc.data_ptr(),
                replay.cached_lens.data_ptr(),
            ),
        )
        self.assertEqual(replay.seq_start_loc.tolist(), [0, 7])
        self.assertEqual(replay.cached_lens.tolist(), [6])

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
            ),
            speculative_config=None,
        )
        kv_cache_spec = SimpleNamespace(num_kv_heads=8, head_size=128)
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
            _use_grouped_h4_stage1,
        )

        self.assertTrue(_use_grouped_h4_stage1(32, 8, 128))
        self.assertTrue(_use_grouped_h4_stage1(8, 2, 128))
        for shape in ((32, 8, 64), (32, 4, 128), (8, 8, 128), (7, 2, 128)):
            with self.subTest(shape=shape):
                self.assertFalse(_use_grouped_h4_stage1(*shape))

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

    def test_grouped_h4_split_contract_is_mixed_40_and_quant_only_33(self):
        from vllm.v1.attention.ops.triton_oscar_decode import (
            _grouped_h4_partial_counts,
        )

        self.assertEqual(_grouped_h4_partial_counts(32, True), (8, 40))
        self.assertEqual(_grouped_h4_partial_counts(32, False), (1, 33))

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
