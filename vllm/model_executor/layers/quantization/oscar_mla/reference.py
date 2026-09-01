"""PyTorch reference operations for shared-latent OSCAR MLA."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Int2Quantized:
    """Asymmetric INT2 values and their per-group metadata."""

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    clipped: torch.Tensor


@dataclass(frozen=True)
class MixedTokenPartition:
    """Prefix, recent, and history token slices for one sequence."""

    prefix: slice
    recent: slice
    history: slice
    total_tokens: int


def partition_mixed_tokens(
    seq_len: int,
    *,
    prefix_tokens: int,
    recent_tokens: int,
) -> MixedTokenPartition:
    """Partition a sequence into disjoint prefix, history, and recent tiers."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if prefix_tokens < 0 or recent_tokens < 0:
        raise ValueError("prefix_tokens and recent_tokens must be non-negative")

    prefix_end = min(seq_len, prefix_tokens)
    recent_start = max(prefix_end, seq_len - recent_tokens)
    return MixedTokenPartition(
        prefix=slice(0, prefix_end),
        history=slice(prefix_end, recent_start),
        recent=slice(recent_start, seq_len),
        total_tokens=seq_len,
    )


def quantize_int2(
    values: torch.Tensor,
    *,
    group_size: int,
    clip_ratio: float,
    eps: float = 1e-8,
) -> Int2Quantized:
    """Clip and asymmetrically quantize the last dimension to unsigned INT2."""
    if values.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {values.shape[-1]} is not divisible by {group_size}"
        )
    if not 0 < clip_ratio <= 1:
        raise ValueError(f"clip_ratio must be in (0, 1], got {clip_ratio}")
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")

    groups = values.reshape(*values.shape[:-1], -1, group_size)
    clip_index = min(group_size - 1, int(clip_ratio * group_size))
    threshold = groups.abs().sort(dim=-1).values[..., clip_index : clip_index + 1]
    clipped_groups = groups.clamp(-threshold, threshold)
    value_min = clipped_groups.amin(dim=-1, keepdim=True)
    value_max = clipped_groups.amax(dim=-1, keepdim=True)
    scale = (value_max - value_min).clamp(min=eps) / 3
    zero_point = -value_min / scale
    quantized = (
        (clipped_groups / scale + zero_point + 0.5)
        .to(torch.int32)
        .clamp_(0, 3)
        .to(torch.uint8)
    )
    return Int2Quantized(
        data=quantized.reshape(values.shape),
        scale=scale,
        zero_point=zero_point,
        clipped=clipped_groups.reshape(values.shape),
    )


def dequantize_int2(
    data: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    *,
    group_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize unsigned INT2 values using FP32 scale and zero point."""
    if data.shape[-1] % group_size:
        raise ValueError(
            f"last dimension {data.shape[-1]} is not divisible by {group_size}"
        )
    groups = data.reshape(*data.shape[:-1], -1, group_size).float()
    restored = (groups - zero_point.float()) * scale.float()
    return restored.reshape(data.shape).to(dtype)


def pack_int2(data: torch.Tensor) -> torch.Tensor:
    """Pack four unsigned INT2 values into each byte, low bits first."""
    if data.shape[-1] % 4:
        raise ValueError(f"last dimension {data.shape[-1]} is not divisible by 4")
    if data.dtype != torch.uint8:
        raise TypeError(f"INT2 data must use uint8 storage, got {data.dtype}")
    if data.numel() and bool(torch.any(data > 3)):
        raise ValueError("INT2 data contains a value greater than 3")

    values = data.reshape(*data.shape[:-1], -1, 4)
    return (
        values[..., 0]
        | (values[..., 1] << 2)
        | (values[..., 2] << 4)
        | (values[..., 3] << 6)
    )


def unpack_int2(packed: torch.Tensor, *, original_dim: int) -> torch.Tensor:
    """Unpack low-bit-first INT2 bytes and trim to the original dimension."""
    if packed.dtype != torch.uint8:
        raise TypeError(f"packed INT2 data must use uint8, got {packed.dtype}")
    available = packed.shape[-1] * 4
    if original_dim < 0 or original_dim > available:
        raise ValueError(
            f"original_dim must be in [0, {available}], got {original_dim}"
        )
    values = torch.stack(
        tuple((packed >> shift) & 0x03 for shift in (0, 2, 4, 6)),
        dim=-1,
    ).flatten(start_dim=-2)
    return values[..., :original_dim]


def _attention_scale(query: torch.Tensor, scale: float | None) -> float:
    return query.shape[-1] ** -0.5 if scale is None else scale


def native_latent_attention(
    query: torch.Tensor,
    latent: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute dense attention directly in the unrotated latent basis."""
    logits = torch.einsum("...hd,sd->...hs", query, latent)
    weights = torch.softmax(logits * _attention_scale(query, scale), dim=-1)
    return torch.einsum("...hs,sd->...hd", weights, latent)


def rotated_latent_attention(
    query: torch.Tensor,
    latent: torch.Tensor,
    rotation: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute latent attention in a shared rotated basis and invert output."""
    query_rotated = query @ rotation
    latent_rotated = latent @ rotation
    output_rotated = native_latent_attention(
        query_rotated,
        latent_rotated,
        scale=scale,
    )
    return output_rotated @ rotation.T


def mixed_latent_attention_with_lse(
    query: torch.Tensor,
    *,
    prefix_latent: torch.Tensor,
    recent_latent: torch.Tensor,
    history_rotated: torch.Tensor,
    rotation: torch.Tensor,
    query_rope: torch.Tensor | None = None,
    prefix_rope: torch.Tensor | None = None,
    history_rope: torch.Tensor | None = None,
    recent_rope: torch.Tensor | None = None,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute mixed-tier output and natural-log sum-exp."""
    query_rotated = query @ rotation
    rope_tensors = (prefix_rope, history_rope, recent_rope)
    if query_rope is None:
        if any(tensor is not None for tensor in rope_tensors):
            raise ValueError("query_rope is required when cached RoPE values are given")
        rope_rank = 0
    else:
        if any(tensor is None for tensor in rope_tensors):
            raise ValueError("all cached RoPE tiers are required with query_rope")
        rope_rank = query_rope.shape[-1]
    factor = (query.shape[-1] + rope_rank) ** -0.5 if scale is None else scale
    prefix_logits = torch.einsum("...hd,sd->...hs", query, prefix_latent)
    history_logits = torch.einsum(
        "...hd,sd->...hs",
        query_rotated,
        history_rotated,
    )
    recent_logits = torch.einsum("...hd,sd->...hs", query, recent_latent)
    if query_rope is not None:
        assert prefix_rope is not None
        assert history_rope is not None
        assert recent_rope is not None
        prefix_logits += torch.einsum("...hd,sd->...hs", query_rope, prefix_rope)
        history_logits += torch.einsum("...hd,sd->...hs", query_rope, history_rope)
        recent_logits += torch.einsum("...hd,sd->...hs", query_rope, recent_rope)
    logits = torch.cat((prefix_logits, history_logits, recent_logits), dim=-1)
    scaled_logits = logits * factor
    weights = torch.softmax(scaled_logits, dim=-1)

    prefix_end = prefix_latent.shape[0]
    history_end = prefix_end + history_rotated.shape[0]
    prefix_output = torch.einsum(
        "...hs,sd->...hd",
        weights[..., :prefix_end],
        prefix_latent,
    )
    history_output = (
        torch.einsum(
            "...hs,sd->...hd",
            weights[..., prefix_end:history_end],
            history_rotated,
        )
        @ rotation.T
    )
    recent_output = torch.einsum(
        "...hs,sd->...hd",
        weights[..., history_end:],
        recent_latent,
    )
    output = prefix_output + history_output + recent_output
    return output, torch.logsumexp(scaled_logits, dim=-1)


def mixed_latent_attention(
    query: torch.Tensor,
    *,
    prefix_latent: torch.Tensor,
    recent_latent: torch.Tensor,
    history_rotated: torch.Tensor,
    rotation: torch.Tensor,
    query_rope: torch.Tensor | None = None,
    prefix_rope: torch.Tensor | None = None,
    history_rope: torch.Tensor | None = None,
    recent_rope: torch.Tensor | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute one softmax across BF16 tiers and rotated history."""
    output, _ = mixed_latent_attention_with_lse(
        query,
        prefix_latent=prefix_latent,
        recent_latent=recent_latent,
        history_rotated=history_rotated,
        rotation=rotation,
        query_rope=query_rope,
        prefix_rope=prefix_rope,
        history_rope=history_rope,
        recent_rope=recent_rope,
        scale=scale,
    )
    return output
