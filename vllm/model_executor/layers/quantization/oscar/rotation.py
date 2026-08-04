# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loading and caching of OSCAR per-layer rotation matrices.

The reference exporter (``rotation/compute_kv_rotation.py``) saves a
checkpoint of the form::

    {
        "format_version": 1,
        "objective": "qqt_r_h_pbr",
        "source_grouping": "layer",
        "layers": {
            <layer_id>: {
                "layer_id": int,
                "rotation": Tensor[head_dim, head_dim] (fp32, orthogonal),
                "eigenvalues": Tensor[head_dim] (fp32),
            },
            ...
        },
    }

Layer ids may be stored as ``int`` or ``str``. A stacked ``[num_layers,
head_dim, head_dim]`` tensor (one matrix per layer) is also accepted.
"""

from __future__ import annotations

from functools import lru_cache

import regex as re
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


def layer_index_from_name(layer_name: str) -> int | None:
    """Best-effort extraction of the global decoder layer index.

    vLLM attention layer names look like
    ``model.layers.<idx>.self_attn.attn``. Returns ``None`` when no index
    can be parsed (e.g. non-standard naming) so the caller can fall back to
    an identity rotation.
    """
    m = _LAYER_IDX_RE.search(layer_name)
    if m is not None:
        return int(m.group(1))
    # Fallback: last integer token in the name.
    ints = re.findall(r"\d+", layer_name)
    return int(ints[-1]) if ints else None


@lru_cache(maxsize=8)
def _load_checkpoint(path: str) -> dict[int, torch.Tensor]:
    """Load and normalize a rotation checkpoint into ``{layer_id: matrix}``.

    Cached per path: the same file is shared across all attention layers and
    loaded once. Returns fp32 CPU tensors; callers move/cast as needed.
    """
    obj = torch.load(path, map_location="cpu", weights_only=False)

    out: dict[int, torch.Tensor] = {}
    if isinstance(obj, dict) and "layers" in obj:
        for k, entry in obj["layers"].items():
            lid = int(k)
            rot = entry["rotation"] if isinstance(entry, dict) else entry
            out[lid] = rot.float().contiguous()
    elif isinstance(obj, dict):
        # Flat ``{layer_id: matrix}`` mapping.
        for k, rot in obj.items():
            out[int(k)] = rot.float().contiguous()
    elif torch.is_tensor(obj) and obj.dim() == 3:
        # Stacked [num_layers, head_dim, head_dim].
        for lid in range(obj.shape[0]):
            out[lid] = obj[lid].float().contiguous()
    else:
        raise ValueError(
            f"Unrecognized OSCAR rotation checkpoint structure at {path!r}: {type(obj)}"
        )
    logger.info("Loaded OSCAR rotation checkpoint %s (%d layers)", path, len(out))
    return out


def get_layer_rotation(
    path: str,
    layer_name: str,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the ``[head_dim, head_dim]`` rotation for one attention layer.

    Missing calibration data is an error. Silently falling back to identity
    would run naive clipped INT2 while claiming the OSCAR path is active.
    """
    if not path:
        raise ValueError("OSCAR rotation path must not be empty")

    table = _load_checkpoint(path)
    lid = layer_index_from_name(layer_name)
    if lid is None:
        raise ValueError(f"Cannot determine OSCAR layer index from {layer_name!r}")
    if lid not in table:
        raise KeyError(
            f"OSCAR rotation checkpoint {path!r} has no entry for layer {lid}"
        )
    rot = table[lid]
    if rot.shape != (head_dim, head_dim):
        raise ValueError(
            f"OSCAR rotation for layer {lid} has shape {tuple(rot.shape)}, "
            f"expected ({head_dim}, {head_dim})."
        )
    return rot.to(device=device, dtype=dtype).contiguous()


def absorb_v_rotation_into_qkv(attn: torch.nn.Module, rotation: torch.Tensor) -> None:
    """Fold one V rotation into a dense fused-QKV projection in place."""
    qkv_proj = attn.qkv_proj
    weight = getattr(qkv_proj, "weight", None)
    supported_dtypes = {torch.float32, torch.float16, torch.bfloat16}
    if weight is None or weight.ndim != 2 or weight.dtype not in supported_dtypes:
        raise RuntimeError(
            "OSCAR V rotation absorption requires a dense floating-point "
            "2D qkv_proj.weight"
        )
    if attn.kv_size != attn.num_kv_heads * attn.head_dim:
        raise RuntimeError("OSCAR QKV V slice geometry is inconsistent")
    expected_rows = attn.q_size + 2 * attn.kv_size
    if weight.shape[0] != expected_rows:
        raise RuntimeError(
            f"OSCAR fused QKV weight has {weight.shape[0]} rows, expected "
            f"{expected_rows}"
        )
    if rotation.shape != (attn.head_dim, attn.head_dim):
        raise RuntimeError(
            f"OSCAR V rotation has shape {tuple(rotation.shape)}, expected "
            f"({attn.head_dim}, {attn.head_dim})"
        )

    rotation = rotation.to(device=weight.device, dtype=torch.float32)
    v_offset = attn.q_size + attn.kv_size
    with torch.no_grad():
        v_weight = weight.data.narrow(0, v_offset, attn.kv_size)
        v_weight_by_head = v_weight.reshape(attn.num_kv_heads, attn.head_dim, -1)
        folded_weight = torch.matmul(rotation.T, v_weight_by_head.to(torch.float32)).to(
            weight.dtype
        )
        v_weight.copy_(folded_weight.reshape_as(v_weight))

        bias = getattr(qkv_proj, "bias", None)
        if bias is not None:
            if bias.ndim != 1 or bias.shape[0] != expected_rows:
                raise RuntimeError("OSCAR fused QKV bias geometry is inconsistent")
            v_bias = bias.data.narrow(0, v_offset, attn.kv_size)
            v_bias_by_head = v_bias.reshape(attn.num_kv_heads, attn.head_dim)
            folded_bias = torch.matmul(v_bias_by_head.to(torch.float32), rotation).to(
                bias.dtype
            )
            v_bias.copy_(folded_bias.reshape_as(v_bias))
