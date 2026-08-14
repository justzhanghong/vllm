# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import fx

from vllm.compilation.piecewise_backend import (
    create_concrete_args,
    get_fake_args_from_graph,
)


def _graph_with_lifted_placeholder() -> tuple[fx.GraphModule, list[torch.Tensor]]:
    graph = fx.Graph()
    input_node = graph.placeholder("input")
    negated = graph.call_function(torch.neg, (input_node,))
    lifted_weight = graph.placeholder("lifted_weight")
    graph.output((negated, lifted_weight))
    graph_module = fx.GraphModule({}, graph)

    example_values = [torch.empty(2), torch.empty(3)]
    placeholders = graph_module.graph.find_nodes(op="placeholder")
    for node, value in zip(placeholders, example_values, strict=True):
        node.meta["example_value"] = value
    return graph_module, example_values


def test_piecewise_args_include_placeholders_after_compute_nodes() -> None:
    graph, example_values = _graph_with_lifted_placeholder()

    fake_args = get_fake_args_from_graph(graph)
    concrete_args = create_concrete_args(graph, size=1)

    assert fake_args == example_values
    assert len(concrete_args) == 2
    assert [tuple(arg.shape) for arg in concrete_args] == [(2,), (3,)]
    assert [arg.dtype for arg in concrete_args] == [torch.float32, torch.float32]
