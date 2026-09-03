# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
import torch

from vllm.config.cache import CacheConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.config.vllm import VllmConfig
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.models.deepseek_v2 import _get_mla_cache_config_for_layer
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseBackend,
)
from vllm.v1.kv_cache_interface import OscarMLAAttentionSpec


def test_oscar_cache_dtype_supports_prefix_caching() -> None:
    config = CacheConfig(
        cache_dtype="oscar_mla_int2",
        enable_prefix_caching=False,
    )

    assert config.cache_dtype == "oscar_mla_int2"
    assert STR_DTYPE_TO_TORCH_DTYPE[config.cache_dtype] is torch.uint8
    assert TritonMLASparseBackend.supports_kv_cache_dtype(config.cache_dtype)

    prefix_config = CacheConfig(
        cache_dtype="oscar_mla_int2",
        enable_prefix_caching=True,
    )
    assert prefix_config.enable_prefix_caching


def test_mtp_draft_layer_uses_native_mla_cache_without_mutating_target(
    monkeypatch,
) -> None:
    monkeypatch.delenv("VLLM_OSCAR_MTP_DRAFT_INT2", raising=False)
    cache_config = CacheConfig(
        cache_dtype="oscar_mla_int2",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
    )
    model_config = SimpleNamespace(
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
    )

    target_config = _get_mla_cache_config_for_layer(
        vllm_config,
        model_config,
        cache_config,
        "model.layers.77.self_attn",
    )
    draft_config = _get_mla_cache_config_for_layer(
        vllm_config,
        model_config,
        cache_config,
        "model.layers.78.self_attn",
    )

    assert target_config is cache_config
    assert target_config.cache_dtype == "oscar_mla_int2"
    assert draft_config is not cache_config
    assert draft_config.cache_dtype == "auto"
    assert cache_config.cache_dtype == "oscar_mla_int2"


def test_mtp_draft_layer_uses_oscar_with_calibrated_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    cache_config = CacheConfig(
        cache_dtype="oscar_mla_int2",
        enable_prefix_caching=False,
    )
    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(method="mtp"),
    )
    model_config = SimpleNamespace(
        num_hidden_layers=78,
        num_nextn_predict_layers=1,
    )

    draft_config = _get_mla_cache_config_for_layer(
        vllm_config,
        model_config,
        cache_config,
        "model.layers.78.self_attn",
    )

    assert draft_config is cache_config
    assert draft_config.cache_dtype == "oscar_mla_int2"


def test_mla_layer_builds_oscar_three_pool_spec() -> None:
    layer = SimpleNamespace(
        kv_cache_dtype="oscar_mla_int2",
        use_sparse=True,
        attn_backend=TritonMLASparseBackend,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        head_size=576,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=True,
        ),
        speculative_config=None,
    )

    spec = MLAAttention.get_kv_cache_spec(layer, vllm_config)

    assert isinstance(spec, OscarMLAAttentionSpec)
    assert spec.latent_rank == 512
    assert spec.rope_head_size == 64
    assert spec.history_slot_size == 160
    assert spec.prefix_tokens == 64
    assert spec.recent_tokens == 256
    assert spec.speculative_tokens == 0
    assert spec.recent_capacity_tokens == 256


def test_mla_layer_reserves_mtp5_candidate_recent_capacity() -> None:
    layer = SimpleNamespace(
        kv_cache_dtype="oscar_mla_int2",
        use_sparse=True,
        attn_backend=TritonMLASparseBackend,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        head_size=576,
    )
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            enable_prefix_caching=False,
        ),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
        ),
    )

    spec = MLAAttention.get_kv_cache_spec(layer, vllm_config)

    assert isinstance(spec, OscarMLAAttentionSpec)
    assert spec.recent_tokens == 256
    assert spec.speculative_tokens == 5
    assert spec.recent_capacity_tokens == 261


def test_oscar_runtime_accepts_mtp5() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    VllmConfig._validate_oscar_mla_runtime(config)


def test_oscar_runtime_accepts_async_scheduling() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=True),
    )

    VllmConfig._validate_oscar_mla_runtime(config)


def test_oscar_runtime_accepts_nixl_fail_closed_transfer() -> None:
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector",
            kv_load_failure_policy="fail",
            kv_buffer_device="cuda",
        ),
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    VllmConfig._validate_oscar_mla_runtime(config)


def test_oscar_runtime_accepts_nixl_recompute_transfer() -> None:
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=SimpleNamespace(
            kv_connector="NixlConnector",
            kv_load_failure_policy="recompute",
            kv_buffer_device="cuda",
        ),
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    VllmConfig._validate_oscar_mla_runtime(config)


def test_oscar_runtime_rejects_non_mtp_speculative_method() -> None:
    config = SimpleNamespace(
        speculative_config=SimpleNamespace(
            method="ngram",
            num_speculative_tokens=5,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )

    with pytest.raises(ValueError, match="speculative decoding"):
        VllmConfig._validate_oscar_mla_runtime(config)


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        (
            {
                "speculative_config": SimpleNamespace(
                    method="ngram",
                    num_speculative_tokens=5,
                )
            },
            "speculative",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=2,
                    prefill_context_parallel_size=1,
                    enable_dbo=False,
                )
            },
            "decode context",
        ),
        (
            {
                "parallel_config": SimpleNamespace(
                    decode_context_parallel_size=1,
                    prefill_context_parallel_size=2,
                    enable_dbo=False,
                )
            },
            "prefill context",
        ),
        (
            {
                "kv_transfer_config": SimpleNamespace(
                    kv_connector="MooncakeConnector",
                    kv_load_failure_policy="fail",
                    kv_buffer_device="cuda",
                )
            },
            "other than NixlConnector",
        ),
        (
            {
                "kv_transfer_config": SimpleNamespace(
                    kv_connector="NixlConnector",
                    kv_load_failure_policy="fail",
                    kv_buffer_device="cpu",
                )
            },
            "non-CUDA",
        ),
        (
            {"cache_config": SimpleNamespace(kv_offloading_size=8)},
            "KV offloading",
        ),
    ],
)
def test_oscar_runtime_rejects_unimplemented_modes(
    override: dict[str, object],
    reason: str,
) -> None:
    config = SimpleNamespace(
        model_config=SimpleNamespace(enforce_eager=True),
        compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
        ),
        kv_transfer_config=None,
        cache_config=SimpleNamespace(kv_offloading_size=None),
        scheduler_config=SimpleNamespace(async_scheduling=False),
    )
    for name, value in override.items():
        setattr(config, name, value)

    with pytest.raises(ValueError, match=reason):
        VllmConfig._validate_oscar_mla_runtime(config)
