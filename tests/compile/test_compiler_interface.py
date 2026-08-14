# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch
from torch.export import Dim

from vllm.compilation.compiler_interface import InductorStandaloneAdaptor
from vllm.config.utils import Range


class _ConditionalLinear(torch.nn.Module):
    def forward(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.cond(
            x.shape[0] == 1,
            lambda input_, weight_: torch.nn.functional.linear(input_, weight_),
            lambda input_, weight_: torch.nn.functional.linear(input_, weight_) + 1,
            (x, weight),
        )


def _first_output(output: object) -> torch.Tensor:
    if isinstance(output, (list, tuple)):
        output = output[0]
    assert isinstance(output, torch.Tensor)
    return output


def test_standalone_adaptor_saves_and_loads_higher_order_cond(
    tmp_path: Path,
) -> None:
    model = _ConditionalLinear()
    input_ = torch.randn(2, 4)
    weight = torch.randn(3, 4)
    graph = torch.export.export(
        model,
        (input_, weight),
        dynamic_shapes=({0: Dim("num_tokens", min=1, max=8)}, None),
    ).graph_module

    original_cacheable_functions = dict(
        torch._inductor.config.unsafe_marked_cacheable_functions
    )
    compile_range = Range(start=1, end=8)
    adaptor = InductorStandaloneAdaptor(save_format="binary")
    adaptor.initialize_cache(str(tmp_path))
    compiled, handle = adaptor.compile(
        graph,
        [input_, weight],
        compiler_config={},
        compile_range=compile_range,
        key="cond_artifact",
    )

    assert compiled is not None
    assert handle == ("cond_artifact", str(tmp_path / "cond_artifact"))
    assert (tmp_path / "cond_artifact").is_file()
    assert compiled._artifacts is not None
    _, cache_info = compiled._artifacts
    assert len(cache_info.aot_autograd_artifacts) == 1
    assert (
        torch._inductor.config.unsafe_marked_cacheable_functions
        == original_cacheable_functions
    )

    loaded = adaptor.load(handle, graph, [input_, weight], 0, compile_range)
    for num_tokens in (1, 2, 4):
        test_input = torch.randn(num_tokens, 4)
        expected = model(test_input, weight)
        assert torch.allclose(_first_output(compiled(test_input, weight)), expected)
        assert torch.allclose(_first_output(loaded(test_input, weight)), expected)
