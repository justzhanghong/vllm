"""Merge TP capture statistics and search shared OSCAR MLA rotations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .calibration import (
    build_shared_covariance,
    normalize_covariance,
    oscar_covariance_rotation,
)
from .reference import dequantize_int2, quantize_int2

DEFAULT_ALPHA_GRID = (0.25, 0.50, 0.75)
DEFAULT_CLIP_GRID = (0.92, 0.94, 0.96, 0.98, 0.99)


@dataclass(frozen=True)
class LayerCaptureStatistics:
    """Merged covariance and holdout latent rows for one layer."""

    score_covariance: torch.Tensor
    value_covariance: torch.Tensor
    latent_covariance: torch.Tensor | None
    latent_samples: torch.Tensor
    score_samples: int
    value_samples: int
    latent_samples_count: int


@dataclass(frozen=True)
class LayerSearchResult:
    rotation: torch.Tensor
    clip_ratio: float
    normalized_loss: float


@dataclass(frozen=True)
class SharedSearchResult:
    alpha: float
    layers: dict[int, LayerSearchResult]
    normalized_loss: float
    alpha_losses: dict[float, float]


def load_and_merge_capture_layer(
    capture_dir: str | Path,
    layer_name: str,
    *,
    tp_size: int,
    split: str,
    latent_rank: int,
    token_budget: int,
) -> LayerCaptureStatistics:
    """Load one complete layer from every TP rank and merge its statistics."""
    payloads = []
    for tp_rank in range(tp_size):
        path = (
            Path(capture_dir)
            / f"tp_rank_{tp_rank:02d}"
            / "layers"
            / f"{layer_name.replace('/', '_')}.pt"
        )
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except FileNotFoundError as error:
            raise ValueError(f"missing capture layer: {path}") from error
        payloads.append(payload)
    return merge_capture_payloads(
        payloads,
        layer_name=layer_name,
        tp_size=tp_size,
        split=split,
        latent_rank=latent_rank,
        token_budget=token_budget,
    )


def merge_capture_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    layer_name: str,
    tp_size: int,
    split: str,
    latent_rank: int,
    token_budget: int,
) -> LayerCaptureStatistics:
    """Fail closed while merging per-rank covariance sums."""
    if len(payloads) != tp_size:
        raise ValueError(f"expected {tp_size} TP payloads, got {len(payloads)}")
    expected_ranks = set(range(tp_size))
    actual_ranks = {payload.get("tp_rank") for payload in payloads}
    if actual_ranks != expected_ranks:
        raise ValueError(
            f"capture TP ranks mismatch; expected={expected_ranks}, got={actual_ranks}"
        )
    score_sum = torch.zeros(latent_rank, latent_rank, dtype=torch.float64)
    value_sum = torch.zeros_like(score_sum)
    score_samples = 0
    value_samples = 0
    rank_zero: Mapping[str, Any] | None = None
    required = {
        "format_version",
        "layer_name",
        "split",
        "tp_rank",
        "latent_rank",
        "token_budget",
        "captured_tokens",
        "score_covariance_samples",
        "value_covariance_samples",
        "score_second_moment_sum",
        "value_second_moment_sum",
        "latent_samples",
    }
    for payload in payloads:
        missing = required - set(payload)
        if missing:
            raise ValueError(f"capture payload is missing fields: {sorted(missing)}")
        if payload["format_version"] != 1:
            raise ValueError("unsupported capture format version")
        for name, expected in (
            ("layer_name", layer_name),
            ("split", split),
            ("latent_rank", latent_rank),
            ("token_budget", token_budget),
            ("captured_tokens", token_budget),
        ):
            if payload[name] != expected:
                raise ValueError(
                    f"capture {name} mismatch: {payload[name]!r} != {expected!r}"
                )
        score_matrix = _validate_covariance_sum(
            payload["score_second_moment_sum"],
            latent_rank,
            "score",
        )
        value_matrix = _validate_covariance_sum(
            payload["value_second_moment_sum"],
            latent_rank,
            "value",
        )
        score_count = int(payload["score_covariance_samples"])
        value_count = int(payload["value_covariance_samples"])
        if score_count <= 0 or value_count <= 0:
            raise ValueError("capture covariance sample counts must be positive")
        score_sum += score_matrix
        value_sum += value_matrix
        score_samples += score_count
        value_samples += value_count
        if payload["tp_rank"] == 0:
            rank_zero = payload

    assert rank_zero is not None
    latent_sum = rank_zero.get("latent_second_moment_sum")
    latent_count = int(rank_zero.get("latent_covariance_samples", 0))
    if latent_sum is None or latent_count <= 0:
        raise ValueError("TP rank 0 capture is missing latent covariance")
    latent_sum = _validate_covariance_sum(latent_sum, latent_rank, "latent")
    latent_samples = rank_zero["latent_samples"]
    if (
        not isinstance(latent_samples, torch.Tensor)
        or latent_samples.ndim != 2
        or latent_samples.shape[1] != latent_rank
    ):
        raise ValueError("rank 0 latent_samples have an invalid shape")
    return LayerCaptureStatistics(
        score_covariance=score_sum / score_samples,
        value_covariance=value_sum / value_samples,
        latent_covariance=latent_sum / latent_count,
        latent_samples=latent_samples.float(),
        score_samples=score_samples,
        value_samples=value_samples,
        latent_samples_count=latent_count,
    )


def search_shared_rotations(
    train_layers: Mapping[int, LayerCaptureStatistics],
    holdout_layers: Mapping[int, LayerCaptureStatistics],
    *,
    group_size: int,
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    clip_grid: Sequence[float] = DEFAULT_CLIP_GRID,
) -> SharedSearchResult:
    """Select one global alpha and one holdout-optimal clip per layer."""
    if set(train_layers) != set(holdout_layers) or not train_layers:
        raise ValueError("train and holdout layer mappings must be non-empty and equal")
    if not alpha_grid or not clip_grid:
        raise ValueError("alpha_grid and clip_grid must not be empty")
    candidates: dict[float, tuple[dict[int, LayerSearchResult], float]] = {}
    for alpha in alpha_grid:
        layer_results: dict[int, LayerSearchResult] = {}
        for layer in sorted(train_layers):
            train = train_layers[layer]
            holdout = holdout_layers[layer]
            shared = build_shared_covariance(
                train.score_covariance,
                train.value_covariance,
                alpha=alpha,
            )
            rotation = oscar_covariance_rotation(shared)
            losses = [
                _quantization_sensitivity_loss(
                    holdout,
                    rotation,
                    alpha=alpha,
                    clip_ratio=clip_ratio,
                    group_size=group_size,
                )
                for clip_ratio in clip_grid
            ]
            best_index = min(range(len(losses)), key=losses.__getitem__)
            layer_results[layer] = LayerSearchResult(
                rotation=rotation.float(),
                clip_ratio=float(clip_grid[best_index]),
                normalized_loss=losses[best_index],
            )
        total_loss = sum(
            result.normalized_loss for result in layer_results.values()
        ) / len(layer_results)
        candidates[float(alpha)] = layer_results, total_loss
    best_alpha = min(candidates, key=lambda alpha: candidates[alpha][1])
    best_layers, best_loss = candidates[best_alpha]
    return SharedSearchResult(
        alpha=best_alpha,
        layers=best_layers,
        normalized_loss=best_loss,
        alpha_losses={
            alpha: candidate_loss for alpha, (_, candidate_loss) in candidates.items()
        },
    )


def _quantization_sensitivity_loss(
    holdout: LayerCaptureStatistics,
    rotation: torch.Tensor,
    *,
    alpha: float,
    clip_ratio: float,
    group_size: int,
) -> float:
    if not holdout.latent_samples.numel():
        raise ValueError("holdout latent_samples must not be empty")
    latent = holdout.latent_samples.double()
    rotation = rotation.double()
    rotated = latent @ rotation
    quantized = quantize_int2(
        rotated,
        group_size=group_size,
        clip_ratio=clip_ratio,
    )
    restored = dequantize_int2(
        quantized.data,
        quantized.scale,
        quantized.zero_point,
        group_size=group_size,
        dtype=torch.float64,
    )
    error = restored - rotated
    shared = build_shared_covariance(
        holdout.score_covariance,
        holdout.value_covariance,
        alpha=alpha,
    )
    rotated_sensitivity = rotation.T @ normalize_covariance(shared) @ rotation
    error_energy = torch.einsum(
        "nd,df,nf->",
        error,
        rotated_sensitivity,
        error,
    )
    signal_energy = torch.einsum(
        "nd,df,nf->",
        rotated,
        rotated_sensitivity,
        rotated,
    ).clamp(min=torch.finfo(torch.float64).eps)
    return float((error_energy / signal_energy).item())


def _validate_covariance_sum(
    value: Any,
    latent_rank: int,
    name: str,
) -> torch.Tensor:
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (latent_rank, latent_rank)
        or not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"{name} covariance sum has an invalid tensor")
    return value.double()
