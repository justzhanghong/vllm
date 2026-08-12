# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm.v1.attention.ops.triton_oscar_bulk_flush import (
    OscarBulkFlushPlan,
    bind_oscar_bulk_flush_state,
    build_oscar_bulk_flush_plan_cpu,
    oscar_bulk_flush,
    prepare_oscar_bulk_flush_plan,
    register_oscar_bulk_flush_caches,
    validate_oscar_bulk_flush_scheduler_output,
)
from vllm.v1.attention.ops.triton_oscar_store import oscar_store

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

DEVICE = "cuda"
BLOCK_SIZE = 16
NUM_HEADS = 8
HEAD_DIM = 128
DATA_BYTES = 32
META_WIDTH = 2
RECENT_CAPACITY = 272


def _make_caches(num_layers: int, *, max_num_seqs: int = 1):
    caches = {}
    for index in range(num_layers):
        caches[f"layer.{index}"] = [
            torch.zeros(
                (16, BLOCK_SIZE, NUM_HEADS, DATA_BYTES),
                dtype=torch.uint8,
                device=DEVICE,
            ),
            torch.zeros(
                (16, BLOCK_SIZE, NUM_HEADS, DATA_BYTES),
                dtype=torch.uint8,
                device=DEVICE,
            ),
            torch.zeros(
                (16, BLOCK_SIZE, NUM_HEADS, META_WIDTH),
                dtype=torch.bfloat16,
                device=DEVICE,
            ),
            torch.zeros(
                (16, BLOCK_SIZE, NUM_HEADS, META_WIDTH),
                dtype=torch.bfloat16,
                device=DEVICE,
            ),
            torch.zeros(
                (64 * max_num_seqs, NUM_HEADS, 2, HEAD_DIM),
                dtype=torch.bfloat16,
                device=DEVICE,
            ),
            torch.randn(
                (
                    RECENT_CAPACITY * max_num_seqs,
                    NUM_HEADS,
                    2,
                    HEAD_DIM,
                ),
                dtype=torch.bfloat16,
                device=DEVICE,
            ),
        ]
    return caches


def _full_plan(*, dst_start: int = 0) -> OscarBulkFlushPlan:
    positions = torch.arange(8, dtype=torch.int32, device=DEVICE).view(1, 8)
    slots = torch.arange(8, dtype=torch.int64, device=DEVICE).view(1, 8)
    return OscarBulkFlushPlan(
        next_phase=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        recent_extra=torch.zeros(1, dtype=torch.int32, device=DEVICE),
        positions=positions,
        src_recent_slots=slots,
        dst_slots=slots + dst_start,
        valid=torch.ones((1, 8), dtype=torch.bool, device=DEVICE),
    )


def _run_bulk(caches, plan, *, max_num_seqs: int = 1):
    layer_names = tuple(caches)
    state = register_oscar_bulk_flush_caches(
        caches,
        layer_names,
        expected_recent_capacity=RECENT_CAPACITY,
        max_num_seqs=max_num_seqs,
    )
    oscar_bulk_flush(
        state,
        plan,
        key_levels=4,
        value_levels=4,
        key_packed_size=36,
        data_bytes=32,
        k_clip_ratio=0.96,
        v_clip_ratio=0.92,
    )


def test_gpu_plan_matches_cpu_reference_across_pages():
    phase = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    generations = torch.tensor([3], dtype=torch.int64, device=DEVICE)
    cached_lens = torch.tensor([330], dtype=torch.int32, device=DEVICE)
    hp_rows = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    shared = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    blocks = list(range(64))
    blocks[4], blocks[5] = 11, 3
    block_table = torch.tensor([blocks], dtype=torch.int32, device=DEVICE)
    buffers = {
        "recent_extra": torch.zeros(1, dtype=torch.int32, device=DEVICE),
        "positions": torch.full((1, 8), -1, dtype=torch.int32, device=DEVICE),
        "src_recent_slots": torch.full(
            (1, 8), -1, dtype=torch.int64, device=DEVICE
        ),
        "dst_slots": torch.full(
            (1, 8), -1, dtype=torch.int64, device=DEVICE
        ),
        "valid": torch.zeros((1, 8), dtype=torch.bool, device=DEVICE),
    }
    caches = _make_caches(1)
    state = register_oscar_bulk_flush_caches(
        caches,
        ("layer.0",),
        expected_recent_capacity=RECENT_CAPACITY,
        max_num_seqs=1,
    )
    cpu_phase = torch.tensor([0], dtype=torch.int32)
    written_steps = []
    checkpoints = {}
    for step in range(1, 17):
        cached_lens.fill_(330 + step - 1)
        before = tuple(tensor.clone() for tensor in caches["layer.0"][:4])
        gpu = prepare_oscar_bulk_flush_plan(
            phase=phase,
            row_generations=generations,
            reset_mask=torch.tensor([False], device=DEVICE),
            request_generations=generations.clone(),
            cached_lens=cached_lens,
            hp_row_ids=hp_rows,
            shared_hit_tokens=shared,
            block_table=block_table,
            prefix_tokens=64,
            recent_tokens=256,
            recent_capacity=RECENT_CAPACITY,
            block_size=BLOCK_SIZE,
            flush_interval=8,
            **buffers,
        )
        cpu = build_oscar_bulk_flush_plan_cpu(
            phase=cpu_phase,
            cached_lens=cached_lens.cpu(),
            hp_row_ids=hp_rows.cpu(),
            shared_hit_tokens=shared.cpu(),
            block_table=block_table.cpu(),
            prefix_tokens=64,
            recent_tokens=256,
            recent_capacity=RECENT_CAPACITY,
            block_size=BLOCK_SIZE,
            flush_interval=8,
        )
        cpu_phase = cpu.next_phase
        assert torch.equal(gpu.positions.cpu(), cpu.positions)
        assert torch.equal(gpu.src_recent_slots.cpu(), cpu.src_recent_slots)
        assert torch.equal(gpu.dst_slots.cpu(), cpu.dst_slots)
        assert torch.equal(gpu.valid.cpu(), cpu.valid)
        oscar_bulk_flush(
            state,
            gpu,
            key_levels=4,
            value_levels=4,
            key_packed_size=36,
            data_bytes=32,
            k_clip_ratio=0.96,
            v_clip_ratio=0.92,
        )
        torch.cuda.synchronize()
        if any(
            not torch.equal(old, new)
            for old, new in zip(before, caches["layer.0"][:4])
        ):
            written_steps.append(step)
        if step in (1, 7, 8, 9, 15, 16):
            checkpoints[step] = (
                gpu.recent_extra.item(),
                gpu.valid.sum().item(),
                gpu.positions.cpu().tolist(),
            )

    assert written_steps == [8, 16]
    assert {step: value[:2] for step, value in checkpoints.items()} == {
        1: (1, 0),
        7: (7, 0),
        8: (0, 8),
        9: (1, 0),
        15: (7, 0),
        16: (0, 8),
    }
    assert checkpoints[8][2] == [list(range(74, 82))]
    assert checkpoints[16][2] == [list(range(82, 90))]


@pytest.mark.parametrize("num_layers", [2, 36])
def test_bulk_quant_matches_production_single_token_store(num_layers: int):
    caches = _make_caches(num_layers, max_num_seqs=2)
    plan = _full_plan(dst_start=12)
    plan.src_recent_slots.add_(RECENT_CAPACITY)
    plan.valid[0] = torch.tensor(
        [True, False, True, True, False, True, False, True], device=DEVICE
    )

    _run_bulk(caches, plan, max_num_seqs=2)
    for cache in caches.values():
        reference = [torch.zeros_like(tensor) for tensor in cache[:4]]
        valid = plan.valid.reshape(-1)
        src = plan.src_recent_slots.reshape(-1)[valid]
        dst = plan.dst_slots.reshape(-1)[valid]
        oscar_store(
            cache[5][src, :, 0, :],
            cache[5][src, :, 1, :],
            *reference,
            dst,
            key_levels=4,
            value_levels=4,
            data_bytes=32,
            k_clip_ratio=0.96,
            v_clip_ratio=0.92,
        )

        assert all(
            torch.equal(actual, expected)
            for actual, expected in zip(cache[:4], reference)
        )


def test_bulk_quant_respects_partial_valid_mask():
    caches = _make_caches(1, max_num_seqs=2)
    caches["layer.0"][5][:RECENT_CAPACITY].zero_()
    plan = _full_plan()
    plan.src_recent_slots.add_(RECENT_CAPACITY)
    plan.valid[0, 1::2] = False

    _run_bulk(caches, plan, max_num_seqs=2)

    written = torch.stack(
        [
            tensor.view(tensor.shape[0] * BLOCK_SIZE, NUM_HEADS, -1)
            .ne(0)
            .any(dim=(1, 2))
            for tensor in caches["layer.0"][:4]
        ]
    ).any(dim=0)
    assert torch.all(written[0:8:2])
    assert torch.all(~written[1:8:2])
    reference = [
        torch.zeros_like(tensor) for tensor in caches["layer.0"][:4]
    ]
    valid = plan.valid.reshape(-1)
    src = plan.src_recent_slots.reshape(-1)[valid]
    dst = plan.dst_slots.reshape(-1)[valid]
    recent = caches["layer.0"][5]
    oscar_store(
        recent[src, :, 0, :],
        recent[src, :, 1, :],
        *reference,
        dst,
        key_levels=4,
        value_levels=4,
        data_bytes=32,
        k_clip_ratio=0.96,
        v_clip_ratio=0.92,
    )
    assert all(
        torch.equal(actual, expected)
        for actual, expected in zip(caches["layer.0"][:4], reference)
    )


def test_bulk_quant_flushes_all_registered_layers():
    caches = _make_caches(36)
    plan = _full_plan(dst_start=8)

    malformed = dict(caches)
    malformed["layer.0"] = [
        caches["layer.0"][0].to(torch.int8),
        *caches["layer.0"][1:],
    ]
    with patch.object(torch.cuda, "CUDAGraph") as graph_constructor:
        with pytest.raises(ValueError, match="uint8"):
            register_oscar_bulk_flush_caches(
                malformed,
                tuple(malformed),
                expected_recent_capacity=RECENT_CAPACITY,
                max_num_seqs=1,
            )
        graph_constructor.assert_not_called()

    state = register_oscar_bulk_flush_caches(
        caches,
        tuple(caches),
        expected_recent_capacity=RECENT_CAPACITY,
        max_num_seqs=1,
    )
    incomplete_context = {
        name: SimpleNamespace() for name in state.layer_names[:-1]
    }
    with patch.object(torch.cuda, "CUDAGraph") as graph_constructor:
        with pytest.raises(ValueError, match="missing"):
            bind_oscar_bulk_flush_state(incomplete_context, state)
        graph_constructor.assert_not_called()
    context = {name: SimpleNamespace() for name in state.layer_names}
    bind_oscar_bulk_flush_state(context, state)
    assert sum(layer._oscar_bulk_flush_owner for layer in context.values()) == 1

    _run_bulk(caches, plan)

    for cache in caches.values():
        written = torch.stack(
            [
                tensor.view(tensor.shape[0] * BLOCK_SIZE, NUM_HEADS, -1)
                .ne(0)
                .any(dim=(1, 2))
                for tensor in cache[:4]
            ]
        ).any(dim=0)
        assert torch.all(written[8:16])


def test_bulk_quant_cuda_graph_replay_smoke():
    caches = _make_caches(2)
    plan = _full_plan()
    plan.valid.zero_()
    layer_names = tuple(caches)
    state = register_oscar_bulk_flush_caches(
        caches,
        layer_names,
        expected_recent_capacity=RECENT_CAPACITY,
        max_num_seqs=1,
    )

    def run():
        oscar_bulk_flush(
            state,
            plan,
            key_levels=4,
            value_levels=4,
            key_packed_size=36,
            data_bytes=32,
            k_clip_ratio=0.96,
            v_clip_ratio=0.92,
        )

    stable_pointers = (
        state.k_data_ptrs.data_ptr(),
        state.v_data_ptrs.data_ptr(),
        state.k_meta_ptrs.data_ptr(),
        state.v_meta_ptrs.data_ptr(),
        state.recent_ptrs.data_ptr(),
        plan.src_recent_slots.data_ptr(),
        plan.dst_slots.data_ptr(),
        plan.valid.data_ptr(),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for cache in caches.values():
        for tensor in cache[:4]:
            tensor.zero_()

    graph.replay()
    torch.cuda.synchronize()
    for cache in caches.values():
        assert sum(torch.count_nonzero(tensor).item() for tensor in cache[:4]) == 0

    plan.valid.fill_(True)
    graph.replay()
    torch.cuda.synchronize()

    for cache in caches.values():
        assert any(torch.any(tensor != 0) for tensor in cache[:4])
    assert stable_pointers == (
        state.k_data_ptrs.data_ptr(),
        state.v_data_ptrs.data_ptr(),
        state.k_meta_ptrs.data_ptr(),
        state.v_meta_ptrs.data_ptr(),
        state.recent_ptrs.data_ptr(),
        plan.src_recent_slots.data_ptr(),
        plan.dst_slots.data_ptr(),
        plan.valid.data_ptr(),
    )

    config = SimpleNamespace()
    resumed = SimpleNamespace(
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"r0"}),
        scheduled_new_reqs=[],
    )
    with patch(
        "vllm.v1.attention.backends.oscar_attn.oscar_bulk_flush"
    ) as bulk_call:
        with pytest.raises(RuntimeError, match="resume"):
            validate_oscar_bulk_flush_scheduler_output(
                config, resumed, enabled=True
            )
        bulk_call.assert_not_called()
        validate_oscar_bulk_flush_scheduler_output(
            config, resumed, enabled=False
        )
        bulk_call.assert_not_called()
