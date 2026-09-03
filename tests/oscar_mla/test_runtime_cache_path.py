# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch

from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.layers.attention import mla_attention
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.mla import triton_mla_sparse
from vllm.v1.attention.backends.mla.triton_mla_sparse import (
    TritonMLASparseImpl,
    TritonMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.spec_decode import llm_base_proposer
from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer
from vllm.v1.worker import gpu_model_runner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.worker.gpu_worker import (
    _collect_oscar_mla_call_counts,
    _find_oscar_mla_impls,
)
from vllm.v1.worker.oscar_mla_cache import (
    OscarMLABatchMetadata,
    OscarMLACacheTensors,
)


def _cache(
    *,
    raw_size: int = 0,
    recent_capacity: int = 256,
) -> OscarMLACacheTensors:
    return OscarMLACacheTensors(
        raw=torch.empty(raw_size, dtype=torch.int8),
        history_data=torch.zeros(12, 16, 128, dtype=torch.uint8),
        history_scale=torch.zeros(12, 16, 4),
        history_zero=torch.zeros(12, 16, 4),
        prefix=torch.zeros(3, 64, 512, dtype=torch.bfloat16),
        recent=torch.zeros(3, recent_capacity, 512, dtype=torch.bfloat16),
        recent_tokens=256,
        rope=torch.zeros(32, 16, 64, dtype=torch.bfloat16),
    )


def _metadata(
    *,
    query_start: int,
    seq_len: int,
    num_tokens: int,
    demote: bool,
) -> XPUMLASparseMetadata:
    demotion_positions = (
        torch.arange(64, 81, dtype=torch.int32)
        if demote
        else torch.empty(0, dtype=torch.int32)
    )
    oscar = OscarMLABatchMetadata(
        hp_rows=torch.tensor([2], dtype=torch.int32),
        decode_positions=torch.tensor([seq_len - 1], dtype=torch.int32),
        final_seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        history_page_table=torch.tensor([[9, 11]], dtype=torch.int32),
        previous_seq_lens=torch.tensor(
            [320 if demote else 0],
            dtype=torch.int32,
        ),
        demotion_hp_rows=torch.full(
            (demotion_positions.numel(),),
            2,
            dtype=torch.int32,
        ),
        demotion_positions=demotion_positions,
        demotion_page_ids=torch.tensor(
            [9] * 16 + [11] if demote else [],
            dtype=torch.int32,
        ),
        demotion_page_offsets=torch.tensor(
            list(range(16)) + [0] if demote else [],
            dtype=torch.int32,
        ),
        restore_positions=torch.empty(0, dtype=torch.int32),
        restore_hp_rows=torch.empty(0, dtype=torch.int32),
        restore_page_ids=torch.empty(0, dtype=torch.int32),
        restore_page_offsets=torch.empty(0, dtype=torch.int32),
        num_restore_rows=0,
    )
    return XPUMLASparseMetadata(
        num_reqs=1,
        max_query_len=num_tokens,
        max_seq_len=seq_len,
        num_actual_tokens=num_tokens,
        query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        block_table=torch.arange(32, dtype=torch.int32).unsqueeze(0),
        req_id_per_token=torch.zeros(num_tokens, dtype=torch.int32),
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        oscar_mla=oscar,
        block_size=16,
        base_seq_len=query_start,
    )


def _impl() -> TritonMLASparseImpl:
    impl = object.__new__(TritonMLASparseImpl)
    impl.oscar_write_calls = 0
    impl.oscar_demotion_calls = 0
    impl.oscar_read_calls = 0
    impl.oscar_restore_calls = 0
    impl._oscar_demotion_ksplit_workspace = None
    impl._oscar_bf16_materialization_workspace = None
    impl._oscar_mtp_temporal_cache = None
    impl._oscar_mtp_temporal_workspace = None
    impl._oscar_mtp_direct_compare_count = 0
    impl._oscar_capability_major = 8
    impl._oscar_grouped_h4_score_workspace = None
    impl._sm_count = None
    impl.kv_lora_rank = 512
    impl.num_heads = 2
    impl.need_to_return_lse_for_decode = False
    return impl


def _layer() -> SimpleNamespace:
    rotation = torch.eye(512)
    return SimpleNamespace(
        layer_name="model.layers.0.self_attn.attn",
        _oscar_rotation=rotation,
        _oscar_inverse_rotation=rotation,
        _oscar_inverse_rotation_bf16=rotation.to(torch.bfloat16),
    )


def test_zero_length_graph_profile_returns_empty_attention() -> None:
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.topk_indices_buffer = torch.empty((4, 2048), dtype=torch.int32)
    metadata = _metadata(query_start=0, seq_len=0, num_tokens=4, demote=False)
    q_nope = torch.ones((4, 2, 512), dtype=torch.bfloat16)
    q_pe = torch.ones((4, 2, 64), dtype=torch.bfloat16)

    output, lse = impl.forward_mqa(
        (q_nope, q_pe),
        _cache(raw_size=1),
        metadata,
        _layer(),
    )

    assert torch.equal(output, torch.zeros_like(q_nope))
    assert lse is None


def test_runtime_call_counts_aggregate_all_oscar_layers() -> None:
    first = SimpleNamespace(
        kv_cache_dtype="oscar_mla_int2",
        oscar_write_calls=7,
        oscar_demotion_calls=3,
        oscar_read_calls=6,
    )
    second = SimpleNamespace(
        kv_cache_dtype="oscar_mla_int2",
        oscar_write_calls=7,
        oscar_demotion_calls=2,
        oscar_read_calls=6,
    )
    native = SimpleNamespace(
        kv_cache_dtype="auto",
        oscar_write_calls=100,
        oscar_demotion_calls=100,
        oscar_read_calls=100,
    )
    model = SimpleNamespace(
        modules=lambda: (
            SimpleNamespace(impl=first),
            SimpleNamespace(impl=second),
            SimpleNamespace(impl=native),
            SimpleNamespace(),
        )
    )

    impls = _find_oscar_mla_impls(model)
    counts = _collect_oscar_mla_call_counts(impls)

    assert impls == (first, second)
    assert counts == {
        "layers": 2,
        "store_total": 14,
        "store_min": 7,
        "store_max": 7,
        "demotion_total": 5,
        "demotion_min": 2,
        "demotion_max": 3,
        "read_total": 12,
        "read_min": 6,
        "read_max": 6,
    }


def test_unified_update_handles_empty_oscar_cache(monkeypatch) -> None:
    layer = SimpleNamespace(kv_cache=_cache())
    context = SimpleNamespace(no_compile_layers={"layer": layer})
    monkeypatch.setattr(mla_attention, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)

    result = mla_attention.unified_mla_kv_cache_update(
        torch.empty(0, 512),
        torch.empty(0, 1, 64),
        "layer",
        "oscar_mla_int2",
        torch.tensor(1.0),
    )

    assert result.numel() == 0


def test_unified_update_handles_empty_oscar_profile_tensor(monkeypatch) -> None:
    layer = SimpleNamespace(kv_cache=torch.empty(0, dtype=torch.int8))
    context = SimpleNamespace(no_compile_layers={"layer": layer})
    monkeypatch.setattr(mla_attention, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)

    result = mla_attention.unified_mla_kv_cache_update(
        torch.empty(0, 512),
        torch.empty(0, 1, 64),
        "layer",
        "oscar_mla_int2",
        torch.tensor(1.0),
    )

    assert result.numel() == 0


def test_unified_update_skips_oscar_compile_warmup_without_metadata(
    monkeypatch,
) -> None:
    def unexpected_update(*args, **kwargs) -> None:
        raise AssertionError("compile warmup must not update the OSCAR cache")

    layer = SimpleNamespace(
        kv_cache=_cache(raw_size=1),
        impl=SimpleNamespace(do_oscar_kv_cache_update=unexpected_update),
    )
    context = SimpleNamespace(
        no_compile_layers={"layer": layer},
        slot_mapping={"layer": torch.empty(0, dtype=torch.int64)},
        attn_metadata=None,
    )
    monkeypatch.setattr(mla_attention, "_resolve_layer_name", lambda name: name)
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)

    result = mla_attention.unified_mla_kv_cache_update(
        torch.empty(0, 512),
        torch.empty(0, 1, 64),
        "layer",
        "oscar_mla_int2",
        torch.tensor(1.0),
    )

    assert result.numel() == 0


def test_direct_update_skips_oscar_compile_warmup_without_metadata(
    monkeypatch,
) -> None:
    def unexpected_update(*args, **kwargs) -> None:
        raise AssertionError("compile warmup must not update the OSCAR cache")

    layer = SimpleNamespace(
        calculate_kv_scales=False,
        use_direct_call=True,
        layer_name="layer",
        kv_cache_dtype="oscar_mla_int2",
        kv_cache=_cache(raw_size=1),
        impl=SimpleNamespace(do_oscar_kv_cache_update=unexpected_update),
        forward_impl=lambda *args, **kwargs: kwargs["output"].fill_(1),
    )
    context = SimpleNamespace(
        attn_metadata=None,
        slot_mapping={"layer": torch.empty(0, dtype=torch.int64)},
    )
    monkeypatch.setattr(mla_attention, "get_forward_context", lambda: context)

    result = mla_attention.MLAAttention.forward(
        layer,
        torch.empty(0, 512),
        torch.empty(0, 512),
        torch.empty(0, 1, 64),
        output_shape=torch.Size([1, 2]),
    )

    torch.testing.assert_close(result, torch.ones_like(result))


def test_runtime_write_demotes_before_overwriting_recent(monkeypatch) -> None:
    events: list[tuple[str, object]] = []
    demotion: dict[str, torch.Tensor] = {}
    demotion_workspaces: list[torch.Tensor | None] = []
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_rope",
        lambda *args: events.append(("rope", args[2].clone())),
    )

    def _capture_demotion(*args, **kwargs) -> None:
        demotion["positions"] = args[5].clone()
        demotion["hp_rows"] = args[6].clone()
        demotion_workspaces.append(kwargs["partial_workspace"])
        events.append(("demote", None))

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        _capture_demotion,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        lambda *args, **kwargs: events.append(("history", args[0].shape[0])),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        lambda *args: events.append(("bf16", args[3].clone())),
    )
    metadata = _metadata(
        query_start=320,
        seq_len=337,
        num_tokens=17,
        demote=True,
    )

    impl = _impl()
    impl.do_oscar_kv_cache_update(
        torch.randn(17, 512, dtype=torch.bfloat16),
        torch.randn(17, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert [event[0] for event in events] == [
        "rope",
        "demote",
        "history",
        "bf16",
    ]
    assert torch.equal(demotion["positions"], torch.arange(64, 81, dtype=torch.int32))
    assert demotion["hp_rows"].tolist() == [2] * 17
    assert demotion_workspaces == [None]
    assert events[2][1] == 17
    assert torch.equal(events[3][1], torch.arange(320, 337, dtype=torch.int32))
    assert impl.oscar_write_calls == 1
    assert impl.oscar_demotion_calls == 1


def test_prefix_caching_eagerly_writes_decode_latent_to_canonical_slot(
    monkeypatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_bf16", lambda *args: None)

    def _capture_canonical(*args, **kwargs) -> None:
        captured["pages"] = args[5].clone()
        captured["offsets"] = args[6].clone()

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        _capture_canonical,
    )
    metadata = replace(
        _metadata(
            query_start=336,
            seq_len=337,
            num_tokens=1,
            demote=False,
        ),
        enable_prefix_caching=True,
    )

    _impl().do_oscar_kv_cache_update(
        torch.randn(1, 512, dtype=torch.bfloat16),
        torch.randn(1, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert captured["pages"].tolist() == [0]
    assert captured["offsets"].tolist() == [0]


def test_runtime_incremental_mtp_routes_prequant_demotion_cache(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)

    def _capture_demotion(*args, **kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        _capture_demotion,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        lambda *args, **kwargs: None,
    )
    metadata = _metadata(
        query_start=320,
        seq_len=324,
        num_tokens=4,
        demote=True,
    )
    impl = _impl()
    impl._oscar_mtp_temporal_cache = (torch.empty(0), torch.empty(0))

    impl.do_oscar_kv_cache_update(
        torch.randn(4, 512, dtype=torch.bfloat16),
        torch.randn(4, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert captured["prequant_temporal_cache"] is impl._oscar_mtp_temporal_cache
    assert captured["temporal_two_way"] is True


def test_runtime_incremental_mtp_disables_prequant_route_without_cache(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_PREQUANT_DEMOTION_CACHE_ENABLED",
        True,
        raising=False,
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: captured.update(kwargs),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        lambda *args, **kwargs: None,
    )
    metadata = _metadata(
        query_start=320,
        seq_len=324,
        num_tokens=4,
        demote=True,
    )
    impl = _impl()
    impl._oscar_mtp_temporal_cache = None

    impl.do_oscar_kv_cache_update(
        torch.randn(4, 512, dtype=torch.bfloat16),
        torch.randn(4, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert captured["prequant_temporal_cache"] is None
    assert captured["temporal_two_way"] is False


def test_runtime_write_directly_stores_current_history(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: None,
    )

    def _capture_history(*args, **kwargs) -> None:
        captured["latent"] = args[0]
        captured["pages"] = args[5]
        captured["offsets"] = args[6]

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        _capture_history,
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_bf16", lambda *args: None)
    metadata = _metadata(
        query_start=0,
        seq_len=337,
        num_tokens=337,
        demote=False,
    )

    impl = _impl()
    impl.do_oscar_kv_cache_update(
        torch.arange(337 * 512, dtype=torch.float32).view(337, 512),
        torch.zeros(337, 1, 64),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert captured["latent"].shape == (337, 512)
    assert captured["pages"].tolist() == [-1] * 64 + [9] * 16 + [11] + [-1] * 256
    assert captured["offsets"].tolist() == list(range(16)) * 21 + [0]


def test_runtime_prefill_seed_resets_once_and_seeds_both_sources(
    monkeypatch,
) -> None:
    events: list[tuple[str, object]] = []
    monkeypatch.setattr(triton_mla_sparse, "_OSCAR_MTP_PREFILL_SEED_ENABLED", True)
    monkeypatch.setattr(
        triton_mla_sparse,
        "reset_oscar_mtp_temporal_cache",
        lambda cache: events.append(("reset", cache)),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "seed_oscar_mtp_temporal_cache_recent",
        lambda *args, **kwargs: events.append(("seed_recent", args[1].clone())),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "seed_oscar_mtp_temporal_cache_rows",
        lambda *args: events.append(("seed_rows", args[2].clone())),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_rope",
        lambda *args: events.append(("rope", None)),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: events.append(("demote", None)),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        lambda *args, **kwargs: events.append(("history", None)),
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        lambda *args: events.append(("bf16", None)),
    )

    impl = _impl()
    impl._oscar_mtp_temporal_cache = (torch.empty(0), torch.empty(0))
    cache = _cache()
    for query_start, seq_len, demote in ((0, 337, False), (337, 674, True)):
        metadata = _metadata(
            query_start=query_start,
            seq_len=seq_len,
            num_tokens=337,
            demote=demote,
        )
        assert metadata.oscar_mla is not None
        metadata.oscar_mla = replace(
            metadata.oscar_mla,
            history_page_table=torch.zeros((1, 64), dtype=torch.int32),
        )
        impl.do_oscar_kv_cache_update(
            torch.randn(337, 512, dtype=torch.bfloat16),
            torch.randn(337, 1, 64, dtype=torch.bfloat16),
            cache,
            metadata,
            torch.eye(512),
            clip_ratio=0.96,
        )

    assert [event[0] for event in events] == [
        "reset",
        "rope",
        "history",
        "seed_rows",
        "bf16",
        "rope",
        "demote",
        "seed_recent",
        "history",
        "seed_rows",
        "bf16",
    ]
    row_seed_masks = []
    for name, payload in events:
        if name == "seed_rows":
            assert isinstance(payload, torch.Tensor)
            row_seed_masks.append(payload)
    assert [int(mask.sum()) for mask in row_seed_masks] == [17, 81]


def test_runtime_mtp5_write_uses_per_token_layout(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    captured_recent_tokens: list[int] = []
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: None,
    )

    def _unexpected_history_store(*args, **kwargs) -> None:
        raise AssertionError(
            "incremental MTP targets are always in the BF16 recent pool"
        )

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        _unexpected_history_store,
    )

    def _capture_bf16(*args) -> None:
        captured["positions"] = args[3]
        captured["seq_lens"] = args[4]
        captured["hp_rows"] = args[5]
        captured_recent_tokens.append(args[6])

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        _capture_bf16,
    )
    metadata = _metadata(
        query_start=320,
        seq_len=326,
        num_tokens=6,
        demote=False,
    )
    assert metadata.oscar_mla is not None
    metadata.oscar_mla = replace(
        metadata.oscar_mla,
        decode_positions=torch.tensor([-777], dtype=torch.int32),
        final_seq_lens=torch.tensor([-888], dtype=torch.int32),
    )

    _impl().do_oscar_kv_cache_update(
        torch.randn(6, 512, dtype=torch.bfloat16),
        torch.randn(6, 1, 64, dtype=torch.bfloat16),
        _cache(recent_capacity=261),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert torch.equal(
        captured["positions"],
        torch.arange(320, 326, dtype=torch.int32),
    )
    assert captured["seq_lens"].tolist() == [326] * 6
    assert captured["hp_rows"].tolist() == [2] * 6
    assert captured_recent_tokens == [256]


def test_runtime_decode_uses_precomputed_metadata_and_skips_history(
    monkeypatch,
) -> None:
    captured: dict[str, torch.Tensor] = {}
    demotion_positions: list[list[int]] = []
    demotion_hp_rows: list[list[int]] = []
    demotion_kwargs: list[dict[str, object]] = []
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)

    def _capture_decode_demotion(*args, **kwargs) -> None:
        demotion_positions.append(args[5].tolist())
        demotion_hp_rows.append(args[6].tolist())
        demotion_kwargs.append(kwargs)

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        _capture_decode_demotion,
    )

    def _unexpected_history(*args, **kwargs) -> None:
        raise AssertionError("decode token cannot belong to current history")

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_rotate_quantize_store",
        _unexpected_history,
    )

    def _capture_bf16(*args) -> None:
        captured["positions"] = args[3]
        captured["seq_lens"] = args[4]
        captured["hp_rows"] = args[5]

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        _capture_bf16,
    )
    metadata = _metadata(
        query_start=336,
        seq_len=337,
        num_tokens=1,
        demote=False,
    )
    metadata.num_reqs = 2
    metadata.num_actual_tokens = 2
    metadata.query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    metadata.slot_mapping = torch.tensor([336, -1], dtype=torch.int32)
    metadata.req_id_per_token = torch.tensor([0, 1], dtype=torch.int32)
    metadata.seq_lens = torch.tensor([999, 999], dtype=torch.int32)
    assert metadata.oscar_mla is not None
    metadata.oscar_mla = replace(
        metadata.oscar_mla,
        hp_rows=torch.tensor([2, -1], dtype=torch.int32),
        decode_positions=torch.tensor([336, -1], dtype=torch.int32),
        final_seq_lens=torch.tensor([337, 0], dtype=torch.int32),
        history_page_table=torch.tensor(
            [[9, 11], [0, 0]],
            dtype=torch.int32,
        ),
        demotion_hp_rows=torch.tensor([2, -1], dtype=torch.int32),
        demotion_positions=torch.tensor([64, -1], dtype=torch.int32),
        demotion_page_ids=torch.tensor([9, -1], dtype=torch.int32),
        demotion_page_offsets=torch.tensor([0, 0], dtype=torch.int32),
    )

    impl = _impl()
    cache = _cache()
    for _ in range(2):
        impl.do_oscar_kv_cache_update(
            torch.randn(2, 512, dtype=torch.bfloat16),
            torch.randn(2, 1, 64, dtype=torch.bfloat16),
            cache,
            metadata,
            torch.eye(512),
            clip_ratio=0.96,
        )

    assert captured["positions"].tolist() == [336, -1]
    assert captured["seq_lens"].tolist() == [337, 0]
    assert captured["hp_rows"].tolist() == [2, -1]
    assert demotion_positions == [[64, -1], [64, -1]]
    assert demotion_hp_rows == [[2, -1], [2, -1]]
    assert demotion_kwargs == [
        {
            "prefix_tokens": 64,
            "clip_ratio": 0.96,
            "partial_workspace": None,
            "prequant_temporal_cache": None,
            "temporal_two_way": False,
        },
        {
            "prefix_tokens": 64,
            "clip_ratio": 0.96,
            "partial_workspace": None,
            "prequant_temporal_cache": None,
            "temporal_two_way": False,
        },
    ]


def test_runtime_batch1_decode_passes_instance_ksplit_workspace(monkeypatch) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_bf16", lambda *args: None)
    metadata = _metadata(
        query_start=336,
        seq_len=337,
        num_tokens=1,
        demote=False,
    )
    assert metadata.oscar_mla is not None
    metadata.oscar_mla = replace(
        metadata.oscar_mla,
        demotion_hp_rows=torch.tensor([2], dtype=torch.int32),
        demotion_positions=torch.tensor([64], dtype=torch.int32),
        demotion_page_ids=torch.tensor([9], dtype=torch.int32),
        demotion_page_offsets=torch.tensor([0], dtype=torch.int32),
    )
    workspace = torch.empty((1, 4, 8, 128), dtype=torch.float32)
    impl = _impl()
    impl._oscar_demotion_ksplit_workspace = workspace

    impl.do_oscar_kv_cache_update(
        torch.randn(1, 512, dtype=torch.bfloat16),
        torch.randn(1, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert len(captured) == 1
    assert captured[0]["prefix_tokens"] == 64
    assert captured[0]["clip_ratio"] == 0.96
    assert captured[0]["partial_workspace"] is workspace


def test_runtime_batch1_decode_skips_ksplit_for_multirow_demotion(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)
    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        lambda *args, **kwargs: captured.append(kwargs),
    )
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_bf16", lambda *args: None)
    metadata = _metadata(
        query_start=336,
        seq_len=337,
        num_tokens=1,
        demote=True,
    )
    workspace = torch.empty((1, 4, 8, 128), dtype=torch.float32)
    impl = _impl()
    impl._oscar_demotion_ksplit_workspace = workspace

    impl.do_oscar_kv_cache_update(
        torch.randn(1, 512, dtype=torch.bfloat16),
        torch.randn(1, 1, 64, dtype=torch.bfloat16),
        _cache(),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    assert len(captured) == 1
    assert captured[0]["partial_workspace"] is None


def test_runtime_draft_step_uses_current_sequence_length(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_store_rope", lambda *args: None)

    def _unexpected_demotion(*args, **kwargs) -> None:
        raise AssertionError("later draft steps must not repeat target demotions")

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_demote_recent",
        _unexpected_demotion,
    )

    def _capture_bf16(*args) -> None:
        captured["positions"] = args[3]
        captured["seq_lens"] = args[4]
        captured["hp_rows"] = args[5]
        captured["recent_tokens"] = args[6]

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_store_bf16",
        _capture_bf16,
    )
    metadata = _metadata(
        query_start=336,
        seq_len=337,
        num_tokens=1,
        demote=False,
    )
    metadata.seq_lens = torch.tensor([341], dtype=torch.int32)
    metadata.oscar_mla_draft_step = True

    _impl().do_oscar_kv_cache_update(
        torch.randn(1, 512, dtype=torch.bfloat16),
        torch.randn(1, 1, 64, dtype=torch.bfloat16),
        _cache(recent_capacity=261),
        metadata,
        torch.eye(512),
        clip_ratio=0.96,
    )

    positions = captured["positions"]
    seq_lens = captured["seq_lens"]
    hp_rows = captured["hp_rows"]
    assert isinstance(positions, torch.Tensor)
    assert isinstance(seq_lens, torch.Tensor)
    assert isinstance(hp_rows, torch.Tensor)
    assert positions.tolist() == [340]
    assert seq_lens.tolist() == [341]
    assert hp_rows.tolist() == [2]
    assert captured["recent_tokens"] == 261


def test_runtime_builder_clears_replayed_demotion_after_first_draft(
    monkeypatch,
) -> None:
    metadata = _metadata(
        query_start=320,
        seq_len=337,
        num_tokens=1,
        demote=True,
    )

    def _build_for_drafting(*args, **kwargs):
        return metadata

    monkeypatch.setattr(
        XPUMLASparseMetadataBuilder,
        "build_for_drafting",
        _build_for_drafting,
    )
    builder = object.__new__(TritonMLASparseMetadataBuilder)

    first = builder.build_for_drafting(SimpleNamespace(), draft_index=0)
    assert first.oscar_mla is not None
    assert first.oscar_mla.demotion_positions.numel() == 17
    assert not first.oscar_mla_draft_step

    later = builder.build_for_drafting(SimpleNamespace(), draft_index=1)
    assert later.oscar_mla is not None
    assert later.oscar_mla.demotion_hp_rows.numel() == 0
    assert later.oscar_mla.demotion_positions.numel() == 0
    assert later.oscar_mla.demotion_page_ids.numel() == 0
    assert later.oscar_mla.demotion_page_offsets.numel() == 0
    assert later.oscar_mla_draft_step


def test_runtime_demotion_has_no_persistent_scratch() -> None:
    impl = _impl()
    assert not hasattr(impl, "_get_oscar_demotion_scratch")
    assert not hasattr(impl, "_oscar_demotion_gathered")
    assert not hasattr(impl, "_oscar_demotion_rotated")
    assert impl._oscar_demotion_ksplit_workspace is None


def test_runtime_read_uses_local_dsa_ids_and_three_pool_cache(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    captured_num_splits: list[int] = []

    def _read(*args, **kwargs):
        captured["selected"] = args[2]
        captured["query_positions"] = args[4]
        captured["block_table"] = args[8]
        captured_num_splits.append(kwargs["num_splits"])
        return torch.ones(1, 2, 512), torch.zeros(1, 2)

    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_sparse_prefill", _read)
    metadata = _metadata(
        query_start=320,
        seq_len=321,
        num_tokens=1,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.topk_indices_buffer = torch.arange(2048, dtype=torch.int32).view(1, 2048)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(1, 2, 512, dtype=torch.bfloat16),
            torch.zeros(1, 2, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        _layer(),
    )

    assert output.dtype == torch.bfloat16
    assert lse is None
    assert captured["selected"].shape == (1, 2048)
    assert torch.equal(captured["selected"], impl.topk_indices_buffer)
    assert captured_num_splits == [16]
    assert captured["query_positions"].tolist() == [320]
    assert captured["block_table"] is metadata.block_table
    assert impl.oscar_read_calls == 1


def test_runtime_read_maps_multiple_requests_to_local_positions(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    captured_num_splits: list[int] = []

    def _read(*args, **kwargs):
        captured["selected"] = args[2]
        captured["request_indices"] = args[3]
        captured["query_positions"] = args[4]
        captured["block_table"] = args[8]
        captured["history_page_table"] = args[12]
        captured["hp_rows"] = args[13]
        captured_num_splits.append(kwargs["num_splits"])
        return torch.ones(3, 2, 512), torch.zeros(3, 2)

    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_sparse_prefill", _read)
    empty = torch.empty(0, dtype=torch.int32)
    oscar = OscarMLABatchMetadata(
        hp_rows=torch.tensor([2, 1], dtype=torch.int32),
        decode_positions=torch.tensor([320, 336], dtype=torch.int32),
        final_seq_lens=torch.tensor([321, 337], dtype=torch.int32),
        history_page_table=torch.tensor([[9, 11], [4, 5]], dtype=torch.int32),
        previous_seq_lens=torch.tensor([320, 335], dtype=torch.int32),
        demotion_hp_rows=empty,
        demotion_positions=empty,
        demotion_page_ids=empty,
        demotion_page_offsets=empty,
    )
    metadata = XPUMLASparseMetadata(
        num_reqs=2,
        max_query_len=2,
        max_seq_len=337,
        num_actual_tokens=3,
        query_start_loc=torch.tensor([0, 1, 3], dtype=torch.int32),
        slot_mapping=torch.arange(3, dtype=torch.int32),
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        req_id_per_token=torch.tensor([0, 1, 1], dtype=torch.int32),
        seq_lens=torch.tensor([321, 337], dtype=torch.int32),
        oscar_mla=oscar,
        block_size=16,
        base_seq_len=320,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.topk_indices_buffer = torch.tensor(
        [[0, 64, 320], [0, 64, 334], [0, 64, 336]],
        dtype=torch.int32,
    )

    output, lse = impl.forward_mqa(
        (
            torch.zeros(3, 2, 512, dtype=torch.bfloat16),
            torch.zeros(3, 2, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        _layer(),
    )

    assert output.shape == (3, 2, 512)
    assert lse is None
    assert captured["selected"].tolist() == impl.topk_indices_buffer.tolist()
    assert captured["request_indices"].tolist() == [0, 1, 1]
    assert captured["query_positions"].tolist() == [320, 335, 336]
    assert captured["block_table"] is metadata.block_table
    assert captured["history_page_table"] is oscar.history_page_table
    assert captured["hp_rows"] is oscar.hp_rows
    assert captured_num_splits == [1]


def test_runtime_mtp5_read_uses_selected_materialized_path(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _unexpected_mixed_read(*args, **kwargs):
        raise AssertionError("MTP5 target must not use the mixed read path")

    def _materialize(**kwargs):
        captured["positions"] = kwargs["positions"]
        captured["num_rows"] = kwargs["num_rows"]
        captured["recent_tokens"] = kwargs["recent_tokens"]
        return (
            torch.empty(6 * 2048, 1, 576, dtype=torch.bfloat16),
            torch.arange(6 * 2048, dtype=torch.int32).view(1, 1, -1),
        )

    def _sparse_attention(q, kv, indices, **kwargs):
        captured["indices"] = indices
        captured["num_kv_splits"] = kwargs["num_kv_splits"]
        return torch.ones(6, 8, 512), torch.zeros(6, 8)

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_sparse_prefill",
        _unexpected_mixed_read,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows",
        _materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "triton_sparse_mla_attention",
        _sparse_attention,
    )
    metadata = _metadata(
        query_start=2048,
        seq_len=2054,
        num_tokens=6,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.num_heads = 8
    impl._oscar_bf16_materialization_workspace = tuple(torch.empty(0) for _ in range(7))
    impl.topk_indices_buffer = torch.arange(
        6 * 2048,
        dtype=torch.int32,
    ).view(6, 2048)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(6, 8, 512, dtype=torch.bfloat16),
            torch.zeros(6, 8, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        _layer(),
    )

    assert output.shape == (6, 8, 512)
    assert lse is None
    positions = captured["positions"]
    assert isinstance(positions, torch.Tensor)
    assert torch.equal(positions, impl.topk_indices_buffer.reshape(-1))
    assert captured["num_rows"] == 6 * 2048
    assert captured["recent_tokens"] == 256
    indices = captured["indices"]
    assert isinstance(indices, torch.Tensor)
    assert indices.shape == (6, 1, 2048)
    assert captured["num_kv_splits"] == 16


def test_runtime_later_draft_read_uses_candidate_safe_recent_capacity(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _unexpected_mixed_read(*args, **kwargs):
        raise AssertionError("later MTP draft must use the materialized read path")

    def _materialize(**kwargs):
        captured["recent_tokens"] = kwargs["recent_tokens"]
        return (
            torch.empty(2048, 1, 576, dtype=torch.bfloat16),
            torch.arange(2048, dtype=torch.int32).view(1, 1, -1),
        )

    def _sparse_attention(q, kv, indices, **kwargs):
        return torch.ones(1, 8, 512), torch.zeros(1, 8)

    monkeypatch.setattr(
        triton_mla_sparse,
        "oscar_mla_sparse_prefill",
        _unexpected_mixed_read,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows",
        _materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "triton_sparse_mla_attention",
        _sparse_attention,
    )
    metadata = _metadata(
        query_start=24576,
        seq_len=24577,
        num_tokens=1,
        demote=False,
    )
    metadata.oscar_mla_draft_step = True
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.num_heads = 8
    impl._oscar_bf16_materialization_workspace = tuple(torch.empty(0) for _ in range(7))
    impl.topk_indices_buffer = torch.arange(2048, dtype=torch.int32).view(1, 2048)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(1, 8, 512, dtype=torch.bfloat16),
            torch.zeros(1, 8, 64, dtype=torch.bfloat16),
        ),
        _cache(recent_capacity=261),
        metadata,
        _layer(),
    )

    assert output.shape == (1, 8, 512)
    assert lse is None
    assert captured["recent_tokens"] == 261


def test_runtime_mtp5_temporal_cache_uses_exact_selected_route(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED",
        False,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED",
        False,
    )

    def _unexpected_materialize(**kwargs):
        raise AssertionError(
            "enabled MTP temporal route must not use full materializer"
        )

    def _temporal_materialize(**kwargs):
        captured.update(kwargs)
        return (
            torch.empty(6 * 2048, 1, 576, dtype=torch.bfloat16),
            torch.arange(6 * 2048, dtype=torch.int32),
        )

    def _sparse_attention(q, kv, indices, **kwargs):
        return torch.ones(6, 8, 512), torch.zeros(6, 8)

    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows",
        _unexpected_materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows_temporal",
        _temporal_materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "triton_sparse_mla_attention",
        _sparse_attention,
    )
    metadata = _metadata(
        query_start=2048,
        seq_len=2054,
        num_tokens=6,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.num_heads = 8
    impl._oscar_bf16_materialization_workspace = tuple(torch.empty(0) for _ in range(7))
    impl._oscar_mtp_temporal_cache = (torch.empty(0), torch.empty(0))
    impl._oscar_mtp_temporal_workspace = tuple(torch.empty(0) for _ in range(7))
    impl.topk_indices_buffer = torch.arange(
        6 * 2048,
        dtype=torch.int32,
    ).view(6, 2048)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(6, 8, 512, dtype=torch.bfloat16),
            torch.zeros(6, 8, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        _layer(),
    )

    assert output.shape == (6, 8, 512)
    assert lse is None
    assert captured["positions"] is not None
    assert captured["num_rows"] == 6 * 2048
    assert captured["temporal_cache"] is impl._oscar_mtp_temporal_cache
    assert captured["temporal_workspace"] is impl._oscar_mtp_temporal_workspace


def test_runtime_mtp5_direct_cache_commits_after_attention(monkeypatch) -> None:
    captured: dict[str, object] = {}
    events: list[str] = []
    compares: list[dict[str, object]] = []
    direct_kv = torch.empty(1)
    reference_kv = torch.empty(2)
    direct_output = torch.ones(6, 8, 512, dtype=torch.bfloat16)
    direct_lse = torch.zeros(6, 8)
    reference_output = torch.full((6, 8, 512), 2.0, dtype=torch.bfloat16)
    reference_lse = torch.ones(6, 8)

    def _reference_materialize(**kwargs):
        events.append("reference_materialize")
        return (
            reference_kv,
            torch.ones(6 * 2048, dtype=torch.int32),
        )

    def _unexpected_temporal(**kwargs):
        raise AssertionError("direct comparison must use the full reference")

    def _direct_materialize(**kwargs):
        events.append("direct_materialize")
        captured.update(kwargs)
        return (
            direct_kv,
            torch.arange(6 * 2048, dtype=torch.int32),
        )

    def _sparse_attention(q, kv, indices, **kwargs):
        if kv is direct_kv:
            events.append("direct_attention")
            return direct_output, direct_lse
        assert kv is reference_kv
        events.append("reference_attention")
        return reference_output, reference_lse

    def _commit(**kwargs):
        events.append("commit")
        captured["commit"] = kwargs

    def _reset(cache):
        events.append("reset")
        captured["reset_cache"] = cache

    def _compare(**kwargs):
        events.append("compare")
        compares.append(kwargs)

    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DIRECT_CACHE_ATTENTION_ENABLED",
        True,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DUAL_SOURCE_ATTENTION_ENABLED",
        False,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DIRECT_RESET_EACH_STEP_ENABLED",
        True,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_ENABLED",
        True,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_OSCAR_MTP_DIRECT_COMPARE_REFERENCE_STEPS",
        2,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "reset_oscar_mtp_temporal_cache",
        _reset,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows",
        _reference_materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows_temporal",
        _unexpected_temporal,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "materialize_oscar_mla_bf16_rows_direct_attention",
        _direct_materialize,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "triton_sparse_mla_attention",
        _sparse_attention,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "commit_oscar_mla_direct_attention_misses",
        _commit,
    )
    monkeypatch.setattr(
        triton_mla_sparse,
        "_log_oscar_mtp_direct_reference_diff",
        _compare,
    )
    metadata = _metadata(
        query_start=2048,
        seq_len=2054,
        num_tokens=6,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.num_heads = 8
    impl._oscar_bf16_materialization_workspace = tuple(torch.empty(0) for _ in range(7))
    impl._oscar_mtp_temporal_cache = (torch.empty(0), torch.empty(0))
    impl._oscar_mtp_temporal_workspace = tuple(torch.empty(0) for _ in range(7))
    impl.topk_indices_buffer = torch.arange(
        6 * 2048,
        dtype=torch.int32,
    ).view(6, 2048)

    outputs = [
        impl.forward_mqa(
            (
                torch.zeros(6, 8, 512, dtype=torch.bfloat16),
                torch.zeros(6, 8, 64, dtype=torch.bfloat16),
            ),
            _cache(),
            metadata,
            _layer(),
        )
        for _ in range(2)
    ]

    for output, lse in outputs:
        assert output is direct_output
        assert lse is None
    expected_events = [
        "reset",
        "direct_materialize",
        "direct_attention",
        "commit",
        "reference_materialize",
        "reference_attention",
        "compare",
    ]
    assert events == expected_events * 2
    assert captured["reset_cache"] is impl._oscar_mtp_temporal_cache
    assert captured["direct_cache"] is impl._oscar_mtp_temporal_cache
    assert captured["temporal_workspace"] is impl._oscar_mtp_temporal_workspace
    commit = captured["commit"]
    assert isinstance(commit, dict)
    assert commit["positions"] is captured["positions"]
    assert commit["num_rows"] == 6 * 2048
    assert commit["direct_cache"] is impl._oscar_mtp_temporal_cache
    assert commit["temporal_workspace"] is impl._oscar_mtp_temporal_workspace
    compare = compares[0]
    assert compare["direct_output"] is direct_output
    assert compare["direct_lse"] is direct_lse
    assert compare["reference_output"] is reference_output
    assert compare["reference_lse"] is reference_lse
    assert compare["layer_name"] == "model.layers.0.self_attn.attn"
    assert [row["compare_step"] for row in compares] == [0, 1]
    assert impl._oscar_mtp_direct_compare_count == 2


def test_runtime_short_prefill_crops_invalid_topk_tail(monkeypatch) -> None:
    captured: dict[str, torch.Tensor] = {}
    captured_num_splits: list[int] = []

    def _read(*args, **kwargs):
        captured["selected"] = args[2]
        captured_num_splits.append(kwargs["num_splits"])
        return torch.ones(3, 2, 512), torch.zeros(3, 2)

    monkeypatch.setattr(triton_mla_sparse, "oscar_mla_sparse_prefill", _read)
    metadata = _metadata(
        query_start=0,
        seq_len=3,
        num_tokens=3,
        demote=False,
    )
    impl = _impl()
    impl.kv_cache_dtype = "oscar_mla_int2"
    impl.softmax_scale = 576**-0.5
    impl.topk_indices_buffer = torch.arange(15, dtype=torch.int32).view(3, 5)

    output, lse = impl.forward_mqa(
        (
            torch.zeros(3, 2, 512, dtype=torch.bfloat16),
            torch.zeros(3, 2, 64, dtype=torch.bfloat16),
        ),
        _cache(),
        metadata,
        _layer(),
    )

    assert output.shape == (3, 2, 512)
    assert lse is None
    selected = captured["selected"]
    assert isinstance(selected, torch.Tensor)
    assert selected.shape == (3, 3)
    assert torch.equal(selected, impl.topk_indices_buffer[:, :3])
    assert captured_num_splits == [1]


def _common_spec_metadata(
    *,
    query_start_loc: list[int],
    seq_lens: list[int],
    oscar_mla: object,
) -> CommonAttentionMetadata:
    query_start = torch.tensor(query_start_loc, dtype=torch.int32)
    lengths = torch.tensor(seq_lens, dtype=torch.int32)
    num_tokens = query_start_loc[-1]
    return CommonAttentionMetadata(
        query_start_loc=query_start,
        query_start_loc_cpu=query_start.clone(),
        seq_lens=lengths,
        _seq_lens_cpu=lengths.clone(),
        _num_computed_tokens_cpu=torch.zeros_like(lengths),
        seq_lens_cpu_upper_bound=lengths.clone(),
        num_reqs=len(seq_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(
            end - start for start, end in zip(query_start_loc, query_start_loc[1:])
        ),
        max_seq_len=max(seq_lens),
        block_table_tensor=torch.zeros(len(seq_lens), 1, dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        oscar_mla=oscar_mla,
    )


def test_prepare_inputs_preserves_oscar_metadata(monkeypatch) -> None:
    monkeypatch.setattr(llm_base_proposer, "is_pin_memory_available", lambda: False)
    oscar_mla = object()
    metadata = _common_spec_metadata(
        query_start_loc=[0, 4, 11, 16],
        seq_lens=[4, 7, 5],
        oscar_mla=oscar_mla,
    )
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(token_arange_np=np.arange(16)),
    )

    updated, _ = SpecDecodeBaseProposer.prepare_inputs(
        proposer,
        metadata,
        sampled_token_ids=[[0, 0, 1], [0, 0, 0, 1], [0, 0, 1]],
        num_draft_tokens=[3, 6, 4],
    )

    assert updated.oscar_mla is oscar_mla


def test_prepare_inputs_padded_preserves_oscar_metadata(monkeypatch) -> None:
    class _FakeKernel:
        def __getitem__(self, grid):
            assert grid == (3,)

            def _launch(
                cu_num_draft_tokens,
                valid_sampled_tokens_count,
                query_start_loc,
                token_indices_to_sample,
                num_rejected_tokens_gpu,
                num_reqs,
            ) -> None:
                assert num_reqs == 3
                token_indices_to_sample.copy_(torch.tensor([1, 5, 6]))
                num_rejected_tokens_gpu.copy_(torch.tensor([1, 0, 2]))

            return _launch

    monkeypatch.setattr(
        llm_base_proposer,
        "eagle_prepare_inputs_padded_kernel",
        _FakeKernel(),
    )
    oscar_mla = object()
    metadata = _common_spec_metadata(
        query_start_loc=[0, 3, 6, 9],
        seq_lens=[3, 3, 3],
        oscar_mla=oscar_mla,
    )

    updated, _, _ = SpecDecodeBaseProposer.prepare_inputs_padded(
        cast(SpecDecodeBaseProposer, SimpleNamespace()),
        metadata,
        SimpleNamespace(cu_num_draft_tokens=torch.tensor([3, 6, 9])),
        torch.tensor([2, 3, 1], dtype=torch.int32),
    )

    assert updated.oscar_mla is oscar_mla


def test_mtp_draft_int2_full_cudagraph_is_explicit_opt_in(monkeypatch) -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.uniform_decode_query_len = 6
            self.calls: list[tuple[CUDAGraphMode, int, list[int] | None]] = []

        def initialize_cudagraph_keys(
            self,
            mode: CUDAGraphMode,
            uniform_decode_query_len: int = 1,
            cudagraph_capture_sizes: list[int] | None = None,
        ) -> None:
            self.calls.append((mode, uniform_decode_query_len, cudagraph_capture_sizes))

    dispatcher = _FakeDispatcher()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
            max_batch_size=8,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            speculative_config=SimpleNamespace(enforce_eager=False),
            cudagraph_dispatcher=dispatcher,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    SpecDecodeBaseProposer.initialize_cudagraph_keys(
        proposer,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    assert dispatcher.uniform_decode_query_len == 1
    assert dispatcher.calls == [
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            1,
            [1, 2, 4, 6, 8, 12, 24, 48],
        )
    ]
    assert proposer._oscar_mtp_draft_decode_capture_sizes == (1, 2, 4, 6, 8)


def test_mtp_native_bf16_draft_full_cudagraph_uses_exact_graph_sizes(
    monkeypatch,
) -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.uniform_decode_query_len = 6
            self.calls: list[tuple[CUDAGraphMode, int, list[int] | None]] = []

        def initialize_cudagraph_keys(
            self,
            mode: CUDAGraphMode,
            uniform_decode_query_len: int = 1,
            cudagraph_capture_sizes: list[int] | None = None,
        ) -> None:
            self.calls.append((mode, uniform_decode_query_len, cudagraph_capture_sizes))

    dispatcher = _FakeDispatcher()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
            max_batch_size=8,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            speculative_config=SimpleNamespace(enforce_eager=False),
            cudagraph_dispatcher=dispatcher,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "0")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    SpecDecodeBaseProposer.initialize_cudagraph_keys(
        proposer,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    assert dispatcher.uniform_decode_query_len == 1
    assert dispatcher.calls == [
        (
            CUDAGraphMode.FULL_AND_PIECEWISE,
            1,
            [1, 2, 4, 6, 8, 12, 24, 48],
        )
    ]
    assert proposer._oscar_mtp_draft_decode_capture_sizes == (1, 2, 4, 6, 8)


def test_mtp_dynamic_draft_full_cudagraph_captures_all_batch1_widths(
    monkeypatch,
) -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.uniform_decode_query_len = 6
            self.capture_sizes: list[int] | None = None

        def initialize_cudagraph_keys(
            self,
            mode: CUDAGraphMode,
            uniform_decode_query_len: int = 1,
            cudagraph_capture_sizes: list[int] | None = None,
        ) -> None:
            del mode, uniform_decode_query_len
            self.capture_sizes = cudagraph_capture_sizes

    dispatcher = _FakeDispatcher()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
            max_batch_size=8,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            speculative_config=SimpleNamespace(enforce_eager=False),
            cudagraph_dispatcher=dispatcher,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_MARGIN_EARLY_STOP", "1")

    SpecDecodeBaseProposer.initialize_cudagraph_keys(
        proposer,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    assert dispatcher.capture_sizes == [1, 2, 3, 4, 5, 6, 8, 12, 24, 48]


def test_mtp_periodic_draft_full_cudagraph_captures_all_batch1_widths(
    monkeypatch,
) -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.uniform_decode_query_len = 6
            self.capture_sizes: list[int] | None = None

        def initialize_cudagraph_keys(
            self,
            mode: CUDAGraphMode,
            uniform_decode_query_len: int = 1,
            cudagraph_capture_sizes: list[int] | None = None,
        ) -> None:
            del mode, uniform_decode_query_len
            self.capture_sizes = cudagraph_capture_sizes

    dispatcher = _FakeDispatcher()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            num_speculative_tokens=5,
            max_batch_size=8,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            speculative_config=SimpleNamespace(enforce_eager=False),
            cudagraph_dispatcher=dispatcher,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_PERIODIC_DRAFT", "1")

    SpecDecodeBaseProposer.initialize_cudagraph_keys(
        proposer,
        CUDAGraphMode.FULL_AND_PIECEWISE,
    )

    assert dispatcher.capture_sizes == [1, 2, 3, 4, 5, 6, 8, 12, 24, 48]


def test_mtp_dynamic_draft_captures_first_pass_piecewise_widths(
    monkeypatch,
) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            num_spec_tokens=5,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            drafter=SimpleNamespace(method="mtp"),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_MARGIN_EARLY_STOP", "1")

    capture_sizes = GPUModelRunner._mtp_dynamic_main_capture_sizes(runner)

    assert capture_sizes == [1, 2, 3, 4, 5, 6, 12, 24, 48]


def test_mtp_acceptance_adaptive_captures_variable_widths(monkeypatch) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            num_spec_tokens=5,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            drafter=SimpleNamespace(method="mtp"),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_ACCEPTANCE_ADAPTIVE", "1")

    capture_sizes = GPUModelRunner._mtp_dynamic_main_capture_sizes(runner)

    assert capture_sizes == [1, 2, 3, 4, 5, 6, 12, 24, 48]


def test_mtp_periodic_draft_captures_variable_widths(monkeypatch) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            num_spec_tokens=5,
            compilation_config=SimpleNamespace(cudagraph_capture_sizes=[6, 12, 24, 48]),
            drafter=SimpleNamespace(method="mtp"),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_PERIODIC_DRAFT", "1")

    capture_sizes = GPUModelRunner._mtp_dynamic_main_capture_sizes(runner)

    assert capture_sizes == [1, 2, 3, 4, 5, 6, 12, 24, 48]


def test_mtp_periodic_target_uses_actual_uniform_query_width(monkeypatch) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            num_spec_tokens=5,
            speculative_config=SimpleNamespace(method="mtp"),
            uniform_decode_query_len=6,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_PERIODIC_DRAFT", "1")

    assert GPUModelRunner._mtp_dynamic_uniform_decode_query_len(runner, 4, 4, 1) == 4
    assert GPUModelRunner._mtp_dynamic_uniform_decode_query_len(runner, 6, 6, 1) == 6
    assert GPUModelRunner._mtp_dynamic_uniform_decode_query_len(runner, 4, 8, 2) == 6


def test_mtp_periodic_target_registers_batch1_full_graph_key(monkeypatch) -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[CUDAGraphMode, BatchDescriptor]] = []

        def add_cudagraph_key(
            self, mode: CUDAGraphMode, descriptor: BatchDescriptor
        ) -> None:
            self.calls.append((mode, descriptor))

    dispatcher = _FakeDispatcher()
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            num_spec_tokens=5,
            speculative_config=SimpleNamespace(method="mtp"),
            cudagraph_dispatcher=dispatcher,
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_PERIODIC_DRAFT", "1")

    GPUModelRunner._register_mtp_dynamic_target_full_cudagraph_keys(runner)

    assert dispatcher.calls == [
        (
            CUDAGraphMode.FULL,
            BatchDescriptor(
                num_tokens=4,
                num_reqs=1,
                uniform=True,
                uniform_query_len=4,
            ),
        )
    ]


def test_cudagraph_dispatcher_accepts_dynamic_uniform_query_width() -> None:
    dispatcher = CudagraphDispatcher.__new__(CudagraphDispatcher)
    dispatcher.vllm_config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_seqs=16),
        lora_config=None,
    )
    dispatcher.compilation_config = SimpleNamespace(max_cudagraph_capture_size=48)
    dispatcher.uniform_decode_query_len = 6
    dispatcher.cudagraph_mode = CUDAGraphMode.FULL_AND_PIECEWISE
    dispatcher.cudagraph_keys = {
        CUDAGraphMode.PIECEWISE: set(),
        CUDAGraphMode.FULL: {
            BatchDescriptor(
                num_tokens=4,
                num_reqs=1,
                uniform=True,
                uniform_query_len=4,
            )
        },
    }
    dispatcher.keys_initialized = True
    dispatcher._bs_to_padded_graph_size = list(range(49))
    dispatcher.specialize_lora_count = False
    dispatcher.captured_lora_counts = []

    mode, descriptor = dispatcher.dispatch(
        num_tokens=4,
        uniform_decode=True,
        uniform_decode_query_len=4,
    )

    assert mode == CUDAGraphMode.FULL
    assert descriptor.uniform_query_len == 4


def test_mtp_draft_exact_graph_sizes_avoid_six_lane_padding() -> None:
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY,
        max_cudagraph_capture_size=48,
        cudagraph_capture_sizes=[6, 12, 24, 48],
        compile_sizes=[],
        cudagraph_specialize_lora=False,
        is_attention_compiled_piecewise=lambda: False,
    )
    vllm_config = SimpleNamespace(
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=8),
        speculative_config=None,
        lora_config=None,
    )
    dispatcher = CudagraphDispatcher(cast(VllmConfig, vllm_config))
    dispatcher.initialize_cudagraph_keys(
        CUDAGraphMode.FULL_DECODE_ONLY,
        uniform_decode_query_len=1,
        cudagraph_capture_sizes=[1, 2, 4, 6, 8, 12, 24, 48],
    )

    mode, descriptor = dispatcher.dispatch(1, uniform_decode=True)

    assert mode == CUDAGraphMode.FULL
    assert descriptor.num_tokens == 1
    assert descriptor.num_reqs == 1
    assert descriptor.uniform
    assert compilation_config.cudagraph_capture_sizes == [6, 12, 24, 48]


def test_mtp_later_draft_dispatches_as_uniform_decode() -> None:
    class _FakeDispatcher:
        def __init__(self) -> None:
            self.calls: list[tuple[int, bool, object]] = []

        def dispatch(
            self,
            num_tokens: int,
            *,
            uniform_decode: bool = False,
            valid_modes=None,
        ):
            self.calls.append((num_tokens, uniform_decode, valid_modes))
            return CUDAGraphMode.FULL, SimpleNamespace(num_tokens=num_tokens)

    dispatcher = _FakeDispatcher()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            cudagraph_dispatcher=dispatcher,
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_size=1)
            ),
        ),
    )

    mode, padded_tokens, across_dp = (
        SpecDecodeBaseProposer._determine_batch_execution_and_padding(
            proposer,
            1,
            uniform_decode=True,
        )
    )

    assert mode == CUDAGraphMode.FULL
    assert padded_tokens == 1
    assert across_dp is None
    assert dispatcher.calls == [(1, True, None)]


def test_mtp_draft_full_cudagraph_capture_runs_piecewise_and_full(
    monkeypatch,
) -> None:
    class _FakeMTPModel:
        def __init__(self) -> None:
            self.model = self
            self.skip_topk = False

        def set_skip_topk(self, skip: bool) -> None:
            self.skip_topk = skip

        def __call__(self, **kwargs) -> None:
            model_calls.append((kwargs["input_ids"].shape[0], self.skip_topk))

    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer._share_mtp_indices = True
    proposer.parallel_drafting = False
    proposer.num_speculative_tokens = 5
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer._draft_attn_layer_names = {"draft.layer"}
    proposer.vllm_config = cast(
        VllmConfig,
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=3)),
    )
    proposer._oscar_mtp_draft_decode_capture_sizes = (1, 2, 3)
    proposer.input_ids = torch.zeros(12, dtype=torch.int32)
    proposer.positions = torch.zeros(12, dtype=torch.int64)
    proposer._slot_mapping_buffer = torch.zeros(12, dtype=torch.int64)
    proposer.arange = torch.arange(13, dtype=torch.int32)
    proposer.token_arange_np = np.arange(13, dtype=np.int32)
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0

    empty = torch.empty(0, dtype=torch.int32)
    common_oscar = OscarMLABatchMetadata(
        hp_rows=torch.tensor([3, 4, 5], dtype=torch.int32),
        decode_positions=torch.tensor([127, 127, 127], dtype=torch.int32),
        final_seq_lens=torch.tensor([128, 128, 128], dtype=torch.int32),
        history_page_table=torch.arange(24, dtype=torch.int32).view(3, 8),
        previous_seq_lens=torch.tensor([124, 124, 124], dtype=torch.int32),
        demotion_hp_rows=empty,
        demotion_positions=empty,
        demotion_page_ids=empty,
        demotion_page_offsets=empty,
    )
    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 4, 8, 12], dtype=torch.int32),
        seq_lens=torch.tensor([128, 128, 128], dtype=torch.int32),
        num_reqs=3,
        num_actual_tokens=12,
        max_query_len=4,
        max_seq_len=128,
        block_table_tensor=torch.zeros((3, 8), dtype=torch.int32),
        slot_mapping=torch.arange(12, dtype=torch.int32),
        oscar_mla=cast(OscarMLABatchMetadata, common_oscar),
    )

    dispatch_calls: list[tuple[int, bool, bool]] = []
    model_calls: list[tuple[int, bool]] = []
    metadata_build_calls: list[
        tuple[int, int, int, int, int, int, bool, int, int, int, int, int]
    ] = []
    context_metadata: list[object] = []
    context_modes: list[CUDAGraphMode] = []
    captured_oscar = SimpleNamespace(hp_rows=torch.tensor([4], dtype=torch.int32))
    captured_layer_metadata = SimpleNamespace(oscar_mla=captured_oscar)

    def _determine(
        num_tokens: int,
        use_cudagraphs: bool = True,
        uniform_decode: bool = False,
    ):
        dispatch_calls.append((num_tokens, use_cudagraphs, uniform_decode))
        mode = CUDAGraphMode.FULL if uniform_decode else CUDAGraphMode.PIECEWISE
        return mode, num_tokens, None

    monkeypatch.setattr(
        proposer,
        "_determine_batch_execution_and_padding",
        _determine,
    )

    def _build_metadata(common, draft_index=0):
        metadata_build_calls.append(
            (
                draft_index,
                common.num_actual_tokens,
                common.max_query_len,
                common.num_reqs,
                common.seq_lens.shape[0],
                common.block_table_tensor.shape[0],
                common.oscar_mla is not None,
                common.oscar_mla.hp_rows.shape[0],
                common.oscar_mla.decode_positions.shape[0],
                common.oscar_mla.final_seq_lens.shape[0],
                common.oscar_mla.history_page_table.shape[0],
                common.oscar_mla.previous_seq_lens.shape[0],
            )
        )
        layer_metadata = captured_layer_metadata if draft_index == 1 else "layer-0"
        return [f"group-{draft_index}"], {"draft.layer": layer_metadata}

    monkeypatch.setattr(
        proposer,
        "build_per_group_and_layer_attn_metadata",
        _build_metadata,
    )
    proposer.model = _FakeMTPModel()

    def _set_context(metadata, *args, **kwargs):
        context_metadata.append(metadata)
        context_modes.append(kwargs["cudagraph_runtime_mode"])
        return nullcontext()

    monkeypatch.setattr(
        llm_base_proposer,
        "set_forward_context",
        _set_context,
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")

    SpecDecodeBaseProposer.dummy_run(
        proposer,
        12,
        use_cudagraphs=True,
        is_graph_capturing=True,
        common_attn_metadata=common_attn_metadata,
    )

    assert dispatch_calls == [
        (12, True, False),
        (3, True, True),
        (2, True, True),
        (1, True, True),
    ]
    assert model_calls == [
        (12, False),
        (3, True),
        (3, True),
        (2, True),
        (2, True),
        (1, True),
        (1, True),
    ]
    assert metadata_build_calls == [
        (0, 12, 4, 3, 3, 3, True, 3, 3, 3, 3, 3),
        (1, 3, 1, 3, 3, 3, True, 3, 3, 3, 3, 3),
        (1, 2, 1, 2, 2, 2, True, 2, 2, 2, 2, 2),
        (1, 1, 1, 1, 1, 1, True, 1, 1, 1, 1, 1),
    ]
    assert context_metadata[0] == {"draft.layer": "layer-0"}
    for metadata in context_metadata[1:]:
        assert (
            cast(dict[str, object], metadata)["draft.layer"] is captured_layer_metadata
        )
    assert context_modes == [
        CUDAGraphMode.PIECEWISE,
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
        CUDAGraphMode.NONE,
        CUDAGraphMode.FULL,
    ]
    assert proposer._oscar_mtp_draft_graph_capture_metadata == {
        1: captured_oscar,
        2: captured_oscar,
        3: captured_oscar,
    }


def test_oscar_mtp_draft_piecewise_capture_requests_common_metadata(
    monkeypatch,
) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            drafter=SimpleNamespace(method="mtp"),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    should_build = GPUModelRunner._should_build_mtp_draft_capture_metadata
    assert should_build(runner, True, CUDAGraphMode.PIECEWISE, 12)
    assert not should_build(runner, True, CUDAGraphMode.FULL, 12)
    assert not should_build(runner, False, CUDAGraphMode.PIECEWISE, 12)
    assert should_build(runner, True, CUDAGraphMode.PIECEWISE, 17)


def test_mtp_native_bf16_draft_piecewise_capture_requests_common_metadata(
    monkeypatch,
) -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            drafter=SimpleNamespace(method="mtp"),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "0")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    should_build = GPUModelRunner._should_build_mtp_draft_capture_metadata
    assert should_build(runner, True, CUDAGraphMode.PIECEWISE, 12)
    assert not should_build(runner, True, CUDAGraphMode.FULL, 12)
    assert not should_build(runner, False, CUDAGraphMode.PIECEWISE, 12)


def test_mtp_draft_full_cudagraph_capture_is_disabled_by_default(
    monkeypatch,
) -> None:
    proposer = object.__new__(SpecDecodeBaseProposer)
    proposer.method = "mtp"
    proposer.parallel_drafting = False
    proposer.num_speculative_tokens = 5
    proposer.supports_mm_inputs = False
    proposer.pass_hidden_states_to_model = False
    proposer._draft_attn_layer_names = set()
    proposer.vllm_config = cast(
        VllmConfig,
        SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=16)),
    )
    proposer.input_ids = torch.zeros(8, dtype=torch.int32)
    proposer.positions = torch.zeros(8, dtype=torch.int64)
    proposer.uses_mrope = False
    proposer.uses_xdrope_dim = 0

    dispatch_calls: list[tuple[int, bool, bool]] = []

    def _determine(
        num_tokens: int,
        use_cudagraphs: bool = True,
        uniform_decode: bool = False,
    ):
        dispatch_calls.append((num_tokens, use_cudagraphs, uniform_decode))
        return CUDAGraphMode.PIECEWISE, num_tokens, None

    monkeypatch.setattr(
        proposer,
        "_determine_batch_execution_and_padding",
        _determine,
    )
    proposer.model = lambda **kwargs: None
    monkeypatch.setattr(
        llm_base_proposer,
        "set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.delenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", raising=False)

    SpecDecodeBaseProposer.dummy_run(
        proposer,
        8,
        use_cudagraphs=True,
        is_graph_capturing=True,
    )

    assert dispatch_calls == [(8, True, False)]


def test_mtp_profile_dummy_covers_configured_capture_bound() -> None:
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            drafter=SimpleNamespace(method="mtp"),
            max_num_reqs=32,
            compilation_config=SimpleNamespace(
                max_cudagraph_capture_size=256,
            ),
        ),
    )

    get_num_tokens = GPUModelRunner._get_draft_dummy_num_tokens
    assert get_num_tokens(runner, 8192, True) == 256
    assert get_num_tokens(runner, 128, True) == 128
    assert get_num_tokens(runner, 8192, False) == 8192

    runner.drafter.method = "ngram"
    assert get_num_tokens(runner, 8192, True) == 8192


def test_mtp_draft_full_cudagraph_remembers_capture_bound_oscar_metadata() -> None:
    proposer = cast(SpecDecodeBaseProposer, SimpleNamespace())
    captured_oscar = SimpleNamespace(hp_rows=torch.tensor([4], dtype=torch.int32))
    per_layer_attn_metadata = {"draft.layer": SimpleNamespace(oscar_mla=captured_oscar)}

    SpecDecodeBaseProposer._remember_oscar_mtp_draft_graph_metadata(
        proposer,
        6,
        per_layer_attn_metadata,
    )

    assert proposer._oscar_mtp_draft_graph_capture_metadata == {6: captured_oscar}


def test_profile_cudagraph_memory_clears_mtp_draft_capture_metadata(
    monkeypatch,
) -> None:
    descriptor = BatchDescriptor(num_tokens=4)
    dispatcher = SimpleNamespace(
        cudagraph_keys={
            CUDAGraphMode.PIECEWISE: {descriptor},
            CUDAGraphMode.FULL: set(),
        },
        get_capture_descs=lambda: [(CUDAGraphMode.PIECEWISE, [descriptor])],
        keys_initialized=True,
    )
    drafter = SimpleNamespace(_oscar_mtp_draft_graph_capture_metadata={})
    runner = cast(
        GPUModelRunner,
        SimpleNamespace(
            vllm_config=SimpleNamespace(),
            cudagraph_dispatcher=dispatcher,
            device=torch.device("cpu"),
            max_model_len=128,
            max_num_tokens=128,
            drafter=drafter,
            lora_config=None,
            _init_minimal_kv_cache_for_profiling=lambda: None,
            _freeze_gc=lambda: nullcontext(),
            maybe_remove_all_loras=lambda _: None,
            _cleanup_profiling_kv_cache=lambda: None,
        ),
    )

    def _profile_capture(*args, **kwargs) -> None:
        drafter._oscar_mtp_draft_graph_capture_metadata[1] = object()

    runner._warmup_and_capture = _profile_capture
    monkeypatch.setattr(
        gpu_model_runner,
        "set_current_vllm_config",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        gpu_model_runner,
        "graph_capture",
        lambda **kwargs: nullcontext(),
    )
    monkeypatch.setattr(
        gpu_model_runner,
        "set_cudagraph_capturing_enabled",
        lambda _: None,
    )
    monkeypatch.setattr(
        gpu_model_runner.current_platform,
        "graph_pool_handle",
        lambda: object(),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (8 << 30, 8 << 30))
    monkeypatch.setattr(
        gpu_model_runner.CUDAGraphWrapper,
        "clear_all_graphs",
        lambda: None,
    )

    GPUModelRunner.profile_cudagraph_memory(runner)

    assert drafter._oscar_mtp_draft_graph_capture_metadata == {}


def test_mtp_draft_full_cudagraph_binds_runtime_values_to_capture_buffers() -> None:
    empty = torch.empty(0, dtype=torch.int32)
    captured = OscarMLABatchMetadata(
        hp_rows=torch.full((6,), 99, dtype=torch.int32),
        decode_positions=torch.full((6,), 99, dtype=torch.int32),
        final_seq_lens=torch.full((6,), 99, dtype=torch.int32),
        history_page_table=torch.full((6, 4), 99, dtype=torch.int32),
        previous_seq_lens=torch.full((6,), 99, dtype=torch.int32),
        demotion_hp_rows=empty,
        demotion_positions=empty,
        demotion_page_ids=empty,
        demotion_page_offsets=empty,
    )
    runtime = OscarMLABatchMetadata(
        hp_rows=torch.tensor([7, 8], dtype=torch.int32),
        decode_positions=torch.tensor([319, 511], dtype=torch.int32),
        final_seq_lens=torch.tensor([320, 512], dtype=torch.int32),
        history_page_table=torch.tensor([[3, 4], [5, 6]], dtype=torch.int32),
        previous_seq_lens=torch.tensor([318, 510], dtype=torch.int32),
        demotion_hp_rows=torch.tensor([7], dtype=torch.int32),
        demotion_positions=torch.tensor([256], dtype=torch.int32),
        demotion_page_ids=torch.tensor([3], dtype=torch.int32),
        demotion_page_offsets=torch.tensor([0], dtype=torch.int32),
    )
    captured_ptrs = {
        name: getattr(captured, name).data_ptr()
        for name in (
            "hp_rows",
            "decode_positions",
            "final_seq_lens",
            "history_page_table",
            "previous_seq_lens",
        )
    }
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(_oscar_mtp_draft_graph_capture_metadata={6: captured}),
    )
    common_attn_metadata = cast(
        CommonAttentionMetadata,
        SimpleNamespace(oscar_mla=runtime),
    )

    SpecDecodeBaseProposer._bind_oscar_mtp_draft_graph_metadata(
        proposer,
        common_attn_metadata,
        6,
    )

    assert common_attn_metadata.oscar_mla is captured
    assert captured.hp_rows.tolist() == [7, 8, -1, -1, -1, -1]
    assert captured.decode_positions.tolist() == [319, 511, -1, -1, -1, -1]
    assert captured.final_seq_lens.tolist() == [320, 512, 0, 0, 0, 0]
    assert captured.previous_seq_lens.tolist() == [318, 510, 0, 0, 0, 0]
    assert captured.history_page_table.tolist() == [
        [3, 4, 0, 0],
        [5, 6, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert {
        name: getattr(captured, name).data_ptr() for name in captured_ptrs
    } == captured_ptrs
    assert captured.demotion_positions.numel() == 0
    assert runtime.demotion_positions.tolist() == [256]


def test_mtp_draft_full_cudagraph_wraps_model_when_explicitly_enabled(
    monkeypatch,
) -> None:
    model = object()
    wrapper_calls: list[tuple[object, object, CUDAGraphMode, CUDAGraphOptions]] = []

    class _FakeCUDAGraphWrapper:
        def __init__(
            self,
            wrapped_model,
            vllm_config,
            *,
            runtime_mode,
            cudagraph_options,
        ) -> None:
            wrapper_calls.append(
                (
                    wrapped_model,
                    vllm_config,
                    runtime_mode,
                    cudagraph_options,
                )
            )

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            model=model,
            speculative_config=SimpleNamespace(enforce_eager=False),
            vllm_config=vllm_config,
        ),
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "CUDAGraphWrapper",
        _FakeCUDAGraphWrapper,
        raising=False,
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    SpecDecodeBaseProposer._maybe_wrap_oscar_mtp_draft_full_cudagraph(proposer)
    SpecDecodeBaseProposer._maybe_wrap_oscar_mtp_draft_full_cudagraph(proposer)

    assert len(wrapper_calls) == 1
    wrapped_model, wrapped_config, runtime_mode, cudagraph_options = wrapper_calls[0]
    assert wrapped_model is model
    assert wrapped_config is vllm_config
    assert runtime_mode == CUDAGraphMode.FULL
    assert cudagraph_options.weak_ref_output is False
    assert cudagraph_options.strong_ref_entry_output is True
    assert isinstance(proposer.model, _FakeCUDAGraphWrapper)


def test_mtp_native_bf16_draft_full_cudagraph_wraps_model_when_enabled(
    monkeypatch,
) -> None:
    model = object()
    wrapper_calls: list[tuple[object, object, CUDAGraphMode, CUDAGraphOptions]] = []

    class _FakeCUDAGraphWrapper:
        def __init__(
            self,
            wrapped_model,
            vllm_config,
            *,
            runtime_mode,
            cudagraph_options,
        ) -> None:
            wrapper_calls.append(
                (
                    wrapped_model,
                    vllm_config,
                    runtime_mode,
                    cudagraph_options,
                )
            )

    vllm_config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
        ),
        parallel_config=SimpleNamespace(use_ubatching=False),
    )
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            model=model,
            speculative_config=SimpleNamespace(enforce_eager=False),
            vllm_config=vllm_config,
        ),
    )
    monkeypatch.setattr(
        llm_base_proposer,
        "CUDAGraphWrapper",
        _FakeCUDAGraphWrapper,
        raising=False,
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "0")
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", "1")

    SpecDecodeBaseProposer._maybe_wrap_oscar_mtp_draft_full_cudagraph(proposer)

    assert len(wrapper_calls) == 1
    assert wrapper_calls[0][0] is model
    assert isinstance(proposer.model, _FakeCUDAGraphWrapper)


def test_mtp_draft_full_cudagraph_does_not_wrap_model_by_default(
    monkeypatch,
) -> None:
    model = object()
    proposer = cast(
        SpecDecodeBaseProposer,
        SimpleNamespace(
            method="mtp",
            model=model,
            speculative_config=SimpleNamespace(enforce_eager=False),
            vllm_config=SimpleNamespace(
                compilation_config=SimpleNamespace(
                    cudagraph_mode=CUDAGraphMode.FULL_AND_PIECEWISE,
                ),
                parallel_config=SimpleNamespace(use_ubatching=False),
            ),
        ),
    )
    monkeypatch.setenv("VLLM_OSCAR_MTP_DRAFT_INT2", "1")
    monkeypatch.delenv("VLLM_OSCAR_MTP_DRAFT_FULL_CUDAGRAPH", raising=False)

    SpecDecodeBaseProposer._maybe_wrap_oscar_mtp_draft_full_cudagraph(proposer)

    assert proposer.model is model
