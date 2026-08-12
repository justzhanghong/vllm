# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host-side shape and stride contract for separated OSCAR quant arenas."""

import torch


def _is_linear_arena(tensor: torch.Tensor) -> bool:
    return (
        tensor.ndim == 4
        and tensor.stride(3) == 1
        and tensor.stride(2) == tensor.shape[3]
        and tensor.stride(1) == tensor.shape[2] * tensor.stride(2)
        and tensor.stride(0) == tensor.shape[1] * tensor.stride(1)
    )


def has_linear_oscar_arena_layout(
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
) -> bool:
    return all(
        _is_linear_arena(tensor)
        for tensor in (k_data, v_data, k_meta, v_meta)
    )


def validate_oscar_separated_arenas(
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_meta: torch.Tensor,
    v_meta: torch.Tensor,
    *,
    data_bytes: int | None = None,
    require_linear: bool = False,
) -> None:
    """Validate metadata without synchronizing or reading device contents."""
    arenas = (k_data, v_data, k_meta, v_meta)
    if any(tensor.ndim != 4 for tensor in arenas):
        raise ValueError("OSCAR separated arenas must be four-dimensional")
    if any(tensor.stride(3) != 1 for tensor in arenas):
        raise ValueError(
            "OSCAR separated arena innermost dimensions must be contiguous"
        )
    if any(tensor.device != k_data.device for tensor in arenas[1:]):
        raise ValueError("OSCAR separated arenas must share one device")
    if k_data.dtype != torch.uint8 or v_data.dtype != torch.uint8:
        raise ValueError("OSCAR separated data arenas must use uint8")
    if k_meta.dtype != torch.bfloat16 or v_meta.dtype != torch.bfloat16:
        raise ValueError("OSCAR separated metadata arenas must use bfloat16")
    if v_data.shape != k_data.shape:
        raise ValueError("OSCAR K/V data arena shapes must match")
    if v_meta.shape != k_meta.shape:
        raise ValueError("OSCAR K/V metadata arena shapes must match")
    if k_data.shape[:3] != k_meta.shape[:3]:
        raise ValueError("OSCAR separated arena leading shapes must match")
    if data_bytes is not None and k_data.shape[3] != data_bytes:
        raise ValueError("OSCAR data arena width does not match data_bytes")
    if k_meta.shape[3] != 2:
        raise ValueError("OSCAR metadata arenas require one BF16 scale/zero pair")
    if v_data.stride() != k_data.stride():
        raise ValueError("OSCAR K/V data strides must match")
    if v_meta.stride() != k_meta.stride():
        raise ValueError("OSCAR K/V metadata strides must match")
    if require_linear and not has_linear_oscar_arena_layout(*arenas):
        raise ValueError("OSCAR Grouped-H4 requires linear separated arenas")
