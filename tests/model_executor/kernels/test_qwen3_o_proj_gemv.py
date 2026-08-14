# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from vllm.model_executor.kernels.linear.qwen3_o_proj_gemv import (
    QWEN3_4B_O_PROJ_INPUT_SIZE,
    QWEN3_4B_O_PROJ_OUTPUT_SIZE,
    qwen3_batch1_o_proj_gemv_fake,
    qwen3_o_proj,
)
from vllm.model_executor.layers.linear import UnquantizedLinearMethod


class _FakeRowParallelLinear:
    def __init__(self) -> None:
        self.weight = torch.empty(
            QWEN3_4B_O_PROJ_OUTPUT_SIZE,
            QWEN3_4B_O_PROJ_INPUT_SIZE,
            dtype=torch.bfloat16,
        )
        self.bias = None
        self.skip_bias_add = False
        self.return_bias = True
        self.tp_size = 1
        self.input_size_per_partition = QWEN3_4B_O_PROJ_INPUT_SIZE
        self.output_size = QWEN3_4B_O_PROJ_OUTPUT_SIZE
        self.quant_method = UnquantizedLinearMethod()
        self.fallback_calls = 0

    def __call__(self, input_: torch.Tensor):
        self.fallback_calls += 1
        output = torch.full(
            (*input_.shape[:-1], QWEN3_4B_O_PROJ_OUTPUT_SIZE),
            7,
            dtype=input_.dtype,
        )
        return output, None


def _eager_cond(pred, true_fn, false_fn, operands):
    return true_fn(*operands) if pred else false_fn(*operands)


def test_fake_impl_preserves_leading_shape_dtype_and_device() -> None:
    input_ = torch.empty((1, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    weight = torch.empty(
        (QWEN3_4B_O_PROJ_OUTPUT_SIZE, QWEN3_4B_O_PROJ_INPUT_SIZE),
        dtype=torch.bfloat16,
    )

    output = qwen3_batch1_o_proj_gemv_fake(input_, weight)

    assert output.shape == (1, QWEN3_4B_O_PROJ_OUTPUT_SIZE)
    assert output.dtype == input_.dtype
    assert output.device == input_.device


def test_registered_custom_op_uses_fake_impl() -> None:
    mode = FakeTensorMode()
    input_ = mode.from_tensor(
        torch.empty((1, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    )
    weight = mode.from_tensor(
        torch.empty(
            (QWEN3_4B_O_PROJ_OUTPUT_SIZE, QWEN3_4B_O_PROJ_INPUT_SIZE),
            dtype=torch.bfloat16,
        )
    )

    output = torch.ops.vllm.qwen3_batch1_o_proj_gemv(input_, weight)

    assert isinstance(output, FakeTensor)
    assert output.shape == (1, QWEN3_4B_O_PROJ_OUTPUT_SIZE)
    assert output.dtype == torch.bfloat16


def test_single_token_eligible_path_selects_opaque_gemv(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.qwen3_o_proj_gemv as module

    layer = _FakeRowParallelLinear()
    input_ = torch.empty((1, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    expected = torch.full((1, QWEN3_4B_O_PROJ_OUTPUT_SIZE), 3, dtype=torch.bfloat16)
    calls = []

    monkeypatch.setattr(module, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(module, "_is_cuda_pair", lambda *_: True)
    monkeypatch.setattr(module.torch, "cond", _eager_cond)
    monkeypatch.setattr(
        module,
        "_qwen3_batch1_o_proj_gemv_custom_op",
        lambda actual_input, actual_weight: (
            calls.append((actual_input, actual_weight)) or expected
        ),
    )

    output = qwen3_o_proj(input_, layer)

    assert output is expected
    assert calls == [(input_, layer.weight)]
    assert layer.fallback_calls == 0


def test_multi_token_eligible_path_selects_dense_linear(monkeypatch) -> None:
    import vllm.model_executor.kernels.linear.qwen3_o_proj_gemv as module

    layer = _FakeRowParallelLinear()
    input_ = torch.empty((2, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    expected = torch.full((2, QWEN3_4B_O_PROJ_OUTPUT_SIZE), 5, dtype=torch.bfloat16)
    calls = []

    monkeypatch.setattr(module, "RowParallelLinear", _FakeRowParallelLinear)
    monkeypatch.setattr(module, "_is_cuda_pair", lambda *_: True)
    monkeypatch.setattr(module.torch, "cond", _eager_cond)
    monkeypatch.setattr(
        module,
        "_qwen3_batch1_o_proj_gemv_custom_op",
        lambda *_: pytest.fail("M>1 must not enter the opaque GEMV"),
    )
    monkeypatch.setattr(
        module.torch.nn.functional,
        "linear",
        lambda actual_input, actual_weight: calls.append((actual_input, actual_weight))
        or expected,
    )

    output = qwen3_o_proj(input_, layer)

    assert output is expected
    assert calls == [(input_, layer.weight)]
    assert layer.fallback_calls == 0


def test_shape_dispatch_exports_both_cond_branches() -> None:
    from torch.export import Dim

    from vllm.model_executor.kernels.linear.qwen3_o_proj_gemv import (
        _qwen3_o_proj_shape_dispatch,
    )

    class _Dispatch(torch.nn.Module):
        def forward(self, input_: torch.Tensor, weight: torch.Tensor):
            return _qwen3_o_proj_shape_dispatch(input_, weight)

    input_ = torch.empty((2, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    weight = torch.empty(
        (QWEN3_4B_O_PROJ_OUTPUT_SIZE, QWEN3_4B_O_PROJ_INPUT_SIZE),
        dtype=torch.bfloat16,
    )

    exported = torch.export.export(
        _Dispatch(),
        (input_, weight),
        dynamic_shapes=({0: Dim("tokens", min=1, max=8192)}, None),
        strict=False,
    )
    graph_text = str(exported.graph_module.graph)
    true_graph_text = str(exported.graph_module.true_graph_0.graph)
    false_graph_text = str(exported.graph_module.false_graph_0.graph)

    assert "torch.ops.higher_order.cond" in graph_text
    assert "true_graph_0" in graph_text
    assert "false_graph_0" in graph_text
    assert "torch.ops.vllm.qwen3_batch1_o_proj_gemv" in true_graph_text
    assert "torch.ops.aten.linear" in false_graph_text


def test_qwen3_attention_binds_specialized_o_proj_at_attention_boundary(
    monkeypatch,
) -> None:
    import vllm.model_executor.models.qwen3 as qwen3_module

    class _QKV(torch.nn.Module):
        def forward(self, hidden_states: torch.Tensor):
            return hidden_states.new_zeros((1, 8)), None

    class _Rope(torch.nn.Module):
        def forward(self, positions, q, k):
            return q, k

    class _Attention(torch.nn.Module):
        def forward(self, q, k, v):
            return v.new_full((1, QWEN3_4B_O_PROJ_INPUT_SIZE), 5)

    attention = qwen3_module.Qwen3Attention.__new__(qwen3_module.Qwen3Attention)
    torch.nn.Module.__init__(attention)
    attention.q_size = 4
    attention.kv_size = 2
    attention.head_dim = 2
    attention.qkv_proj = _QKV()
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.rotary_emb = _Rope()
    attention.attn = _Attention()
    attention.o_proj = _FakeRowParallelLinear()
    expected = torch.full((1, QWEN3_4B_O_PROJ_OUTPUT_SIZE), 11)
    calls = []
    monkeypatch.setattr(
        qwen3_module,
        "qwen3_o_proj",
        lambda input_, layer: calls.append((input_, layer)) or expected,
    )

    output = attention(torch.zeros(1), torch.zeros((1, 8)))

    assert output is expected
    assert calls[0][0].shape == (1, QWEN3_4B_O_PROJ_INPUT_SIZE)
    assert calls[0][1] is attention.o_proj


@pytest.mark.parametrize(
    "case",
    [
        "input_dtype",
        "input_noncontiguous",
        "input_geometry",
        "weight_dtype",
        "weight_noncontiguous",
        "weight_geometry",
        "cpu_device",
        "tp2",
        "bias",
        "skip_bias_add",
        "quantized",
        "lora_wrapper",
    ],
)
def test_non_target_conditions_fall_back(monkeypatch, case: str) -> None:
    import vllm.model_executor.kernels.linear.qwen3_o_proj_gemv as module

    layer = _FakeRowParallelLinear()
    input_ = torch.empty((1, QWEN3_4B_O_PROJ_INPUT_SIZE), dtype=torch.bfloat16)
    monkeypatch.setattr(module, "RowParallelLinear", _FakeRowParallelLinear)
    if case != "cpu_device":
        monkeypatch.setattr(module, "_is_cuda_pair", lambda *_: True)

    if case == "input_dtype":
        input_ = input_.float()
    elif case == "input_noncontiguous":
        input_ = torch.empty((QWEN3_4B_O_PROJ_INPUT_SIZE, 2), dtype=torch.bfloat16).t()[
            :1
        ]
    elif case == "input_geometry":
        input_ = torch.empty((1, 2048), dtype=torch.bfloat16)
    elif case == "weight_dtype":
        layer.weight = layer.weight.float()
    elif case == "weight_noncontiguous":
        layer.weight = torch.empty(
            (QWEN3_4B_O_PROJ_INPUT_SIZE, QWEN3_4B_O_PROJ_OUTPUT_SIZE),
            dtype=torch.bfloat16,
        ).t()
    elif case == "weight_geometry":
        layer.weight = torch.empty((2048, 4096), dtype=torch.bfloat16)
    elif case == "cpu_device":
        pass
    elif case == "tp2":
        layer.tp_size = 2
    elif case == "bias":
        layer.bias = torch.empty(QWEN3_4B_O_PROJ_OUTPUT_SIZE)
    elif case == "skip_bias_add":
        layer.skip_bias_add = True
    elif case == "quantized":
        layer.quant_method = SimpleNamespace()
    elif case == "lora_wrapper":
        monkeypatch.setattr(module, "RowParallelLinear", type("OtherLayer", (), {}))
    else:  # pragma: no cover
        raise AssertionError(case)

    monkeypatch.setattr(
        module.torch,
        "cond",
        lambda *_: pytest.fail("non-target conditions must not enter torch.cond"),
    )
    monkeypatch.setattr(
        module,
        "_qwen3_batch1_o_proj_gemv_custom_op",
        lambda *_: pytest.fail("target custom op must not run"),
        raising=False,
    )

    output = qwen3_o_proj(input_, layer)

    assert output.shape == (*input_.shape[:-1], QWEN3_4B_O_PROJ_OUTPUT_SIZE)
    assert layer.fallback_calls == 1
