# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3-4B batch-one BF16 output-projection GEMV."""

from __future__ import annotations

import torch

from vllm.model_executor.layers.linear import (
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

QWEN3_4B_O_PROJ_INPUT_SIZE = 4096
QWEN3_4B_O_PROJ_OUTPUT_SIZE = 2560
_QWEN3_4B_O_PROJ_NUM_WARPS = 8


@triton.jit
def qwen3_batch1_o_proj_gemv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    BLOCK_K: tl.constexpr,
):
    """Compute one output row per CTA for a single input token."""
    output_idx = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    input_values = tl.load(input_ptr + k_offsets)
    weight_values = tl.load(weight_ptr + output_idx * BLOCK_K + k_offsets)
    accumulator = tl.sum(
        input_values.to(tl.float32) * weight_values.to(tl.float32), axis=0
    )
    tl.store(output_ptr + output_idx, accumulator)


def qwen3_batch1_o_proj_gemv(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (1, QWEN3_4B_O_PROJ_OUTPUT_SIZE),
        dtype=input_.dtype,
        device=input_.device,
    )
    qwen3_batch1_o_proj_gemv_kernel[(QWEN3_4B_O_PROJ_OUTPUT_SIZE,)](
        input_,
        weight,
        output,
        BLOCK_K=QWEN3_4B_O_PROJ_INPUT_SIZE,
        num_warps=_QWEN3_4B_O_PROJ_NUM_WARPS,
        num_stages=1,
    )
    return output


def qwen3_batch1_o_proj_gemv_fake(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return input_.new_empty((*input_.shape[:-1], weight.size(0)))


direct_register_custom_op(
    op_name="qwen3_batch1_o_proj_gemv",
    op_func=qwen3_batch1_o_proj_gemv,
    fake_impl=qwen3_batch1_o_proj_gemv_fake,
)


def _qwen3_batch1_o_proj_gemv_custom_op(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.vllm.qwen3_batch1_o_proj_gemv(input_, weight)


def _qwen3_o_proj_dense_linear(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.nn.functional.linear(input_, weight)


def _qwen3_o_proj_shape_dispatch(
    input_: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return torch.cond(
        input_.shape[0] == 1,
        _qwen3_batch1_o_proj_gemv_custom_op,
        _qwen3_o_proj_dense_linear,
        (input_, weight),
    )


def _is_cuda_pair(input_: torch.Tensor, weight: torch.Tensor) -> bool:
    return input_.is_cuda and weight.is_cuda and input_.device == weight.device


def _is_eligible(input_: torch.Tensor, layer: torch.nn.Module) -> bool:
    if not isinstance(layer, RowParallelLinear):
        return False
    weight = layer.weight
    return (
        layer.tp_size == 1
        and layer.input_size_per_partition == QWEN3_4B_O_PROJ_INPUT_SIZE
        and layer.output_size == QWEN3_4B_O_PROJ_OUTPUT_SIZE
        and layer.bias is None
        and not layer.skip_bias_add
        and isinstance(layer.quant_method, UnquantizedLinearMethod)
        and input_.ndim == 2
        and input_.shape[-1] == QWEN3_4B_O_PROJ_INPUT_SIZE
        and input_.dtype == torch.bfloat16
        and input_.is_contiguous()
        and tuple(weight.shape)
        == (QWEN3_4B_O_PROJ_OUTPUT_SIZE, QWEN3_4B_O_PROJ_INPUT_SIZE)
        and weight.dtype == torch.bfloat16
        and weight.is_contiguous()
        and _is_cuda_pair(input_, weight)
    )


def qwen3_o_proj(input_: torch.Tensor, layer: torch.nn.Module) -> torch.Tensor:
    """Dispatch eligible shapes in-graph; preserve RowParallelLinear otherwise."""
    if _is_eligible(input_, layer):
        return _qwen3_o_proj_shape_dispatch(input_, layer.weight)
    output, _ = layer(input_)
    return output
