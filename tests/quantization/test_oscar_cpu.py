# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import inspect
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.quantization.oscar.config import OscarConfig
from vllm.model_executor.layers.quantization.oscar.layout import partition_tokens
from vllm.model_executor.layers.quantization.oscar.rotation import (
    _load_checkpoint,
    absorb_v_rotation_into_qkv,
    get_layer_rotation,
)
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.platforms.interface import Platform
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, OscarKVCacheSpec


class TestOscarConfigAndLayout(unittest.TestCase):
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

    def test_serving_materialize_capacity_is_disabled(self):
        from vllm.v1.attention.backends.oscar_attn import (
            OscarAttentionBackend,
            _materialize_token_capacity,
        )

        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kv_cache_memory_bytes=10 * 1024**3)
        )
        self.assertEqual(
            _materialize_token_capacity(vllm_config, 8, 128, torch.bfloat16), 0
        )
        self.assertTrue(OscarAttentionBackend.supports_kv_cache_dtype("oscar_int2"))
        self.assertFalse(
            OscarAttentionBackend.supports_kv_cache_dtype("oscar_mla_int2")
        )

    def test_full_attention_and_mla_dtypes_route_to_distinct_specs(self):
        attn = Attention.__new__(Attention)
        attn.attn_type = AttentionType.DECODER
        attn.sliding_window = None
        attn.num_kv_heads = 8
        attn.head_size = 128
        attn.head_size_v = 128
        attn.kv_cache_torch_dtype = torch.uint8
        vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16),
            model_config=SimpleNamespace(use_mla=False),
        )

        attn.kv_cache_dtype = "oscar_int2"
        self.assertIsInstance(attn.get_kv_cache_spec(vllm_config), OscarKVCacheSpec)

        attn.kv_cache_dtype = "oscar_mla_int2"
        mla_route_spec = attn.get_kv_cache_spec(vllm_config)
        self.assertIsInstance(mla_route_spec, FullAttentionSpec)
        self.assertNotIsInstance(mla_route_spec, OscarKVCacheSpec)

    def test_qwen3_full_attention_hook_ignores_mla_dtype(self):
        model = SimpleNamespace(
            vllm_config=SimpleNamespace(
                cache_config=SimpleNamespace(cache_dtype="oscar_mla_int2")
            )
        )
        with patch.object(
            OscarConfig,
            "from_cache_dtype",
            side_effect=AssertionError("MLA route must not enter full OSCAR hook"),
        ):
            Qwen3ForCausalLM._maybe_absorb_oscar_v_rotation(model)

    def test_hybrid_page_size_uses_exact_full_attention_dtype(self):
        source = inspect.getsource(Platform._align_hybrid_block_size)
        self.assertIn('cache_config.cache_dtype == "oscar_int2"', source)
        self.assertNotIn('cache_config.cache_dtype.startswith("oscar_")', source)


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
