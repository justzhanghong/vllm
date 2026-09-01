"""Fail-closed rotation artifact contract for OSCAR MLA."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

ROTATION_FORMAT_VERSION = 1
_MANIFEST_FILENAME = "manifest.json"
_ROTATION_FILENAME = "rotations.pt"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


@dataclass(frozen=True)
class ArtifactMetadata:
    """Identity and serving parameters bound to a rotation artifact."""

    model_config_sha256: str
    checkpoint_manifest_sha256: str
    expert_mapping_sha256: str
    calibration_code_commit: str
    calibration_manifest_sha256: str
    seed: int
    num_layers: int
    latent_rank: int
    group_size: int
    alpha: float
    prefix_tokens: int
    recent_tokens: int
    clip_ratios: tuple[float, ...]
    format_version: int = ROTATION_FORMAT_VERSION

    def __post_init__(self) -> None:
        for name in (
            "model_config_sha256",
            "checkpoint_manifest_sha256",
            "expert_mapping_sha256",
            "calibration_manifest_sha256",
        ):
            value = getattr(self, name)
            if not _SHA256_PATTERN.fullmatch(value):
                raise ValueError(f"{name} must be a lowercase SHA256")
        if not _COMMIT_PATTERN.fullmatch(self.calibration_code_commit):
            raise ValueError(
                "calibration_code_commit must be a 40-64 character lowercase hash"
            )
        if self.format_version != ROTATION_FORMAT_VERSION:
            raise ValueError(
                f"unsupported rotation format version {self.format_version}"
            )
        if self.num_layers <= 0 or self.latent_rank <= 0:
            raise ValueError("num_layers and latent_rank must be positive")
        if self.group_size <= 0 or self.latent_rank % self.group_size:
            raise ValueError("group_size must exactly divide latent_rank")
        if not 0 <= self.alpha <= 1:
            raise ValueError("alpha must be in [0, 1]")
        if self.prefix_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("prefix_tokens and recent_tokens must be non-negative")
        if len(self.clip_ratios) != self.num_layers:
            raise ValueError("clip_ratios must contain one value per layer")
        if any(not 0 < ratio <= 1 for ratio in self.clip_ratios):
            raise ValueError("every clip ratio must be in (0, 1]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ArtifactMetadata:
        fields = dict(data)
        try:
            fields["clip_ratios"] = tuple(fields["clip_ratios"])
        except KeyError as error:
            raise ValueError("artifact metadata is missing clip_ratios") from error
        try:
            return cls(**fields)
        except TypeError as error:
            raise ValueError(f"invalid artifact metadata: {error}") from error


@dataclass(frozen=True)
class ArtifactExpectation:
    """Runtime values that must match before serving may start."""

    model_config_sha256: str
    checkpoint_manifest_sha256: str
    expert_mapping_sha256: str
    num_layers: int
    latent_rank: int
    group_size: int
    prefix_tokens: int
    recent_tokens: int

    def validate(self, metadata: ArtifactMetadata) -> None:
        mismatches = [
            name
            for name in (
                "model_config_sha256",
                "checkpoint_manifest_sha256",
                "expert_mapping_sha256",
                "num_layers",
                "latent_rank",
                "group_size",
                "prefix_tokens",
                "recent_tokens",
            )
            if getattr(self, name) != getattr(metadata, name)
        ]
        if mismatches:
            raise ValueError(
                "rotation artifact does not match runtime: " + ", ".join(mismatches)
            )


@dataclass(frozen=True)
class LoadedRotationArtifact:
    metadata: ArtifactMetadata
    rotations: dict[int, torch.Tensor]
    manifest_sha256: str
    rotations_sha256: str


def write_rotation_artifact(
    output_dir: str | Path,
    metadata: ArtifactMetadata,
    rotations: Mapping[int, torch.Tensor],
) -> LoadedRotationArtifact:
    """Validate and atomically write one complete rotation artifact."""
    normalized = _validate_rotations(metadata, rotations)
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    rotation_path = artifact_dir / _ROTATION_FILENAME
    rotation_temp = rotation_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "format_version": metadata.format_version,
            "rotations": {str(layer): value for layer, value in normalized.items()},
        },
        rotation_temp,
    )
    os.replace(rotation_temp, rotation_path)
    rotations_sha256 = _sha256_file(rotation_path)

    manifest = {
        "metadata": asdict(metadata),
        "rotation_file": _ROTATION_FILENAME,
        "rotations_sha256": rotations_sha256,
        "layer_ids": list(range(metadata.num_layers)),
    }
    manifest_path = artifact_dir / _MANIFEST_FILENAME
    manifest_temp = manifest_path.with_suffix(".json.tmp")
    manifest_temp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(manifest_temp, manifest_path)
    return LoadedRotationArtifact(
        metadata=metadata,
        rotations=normalized,
        manifest_sha256=_sha256_file(manifest_path),
        rotations_sha256=rotations_sha256,
    )


def load_rotation_artifact(
    artifact_dir: str | Path,
    *,
    expectation: ArtifactExpectation | None = None,
) -> LoadedRotationArtifact:
    """Load an artifact only when every identity, layer, and tensor check passes."""
    root = Path(artifact_dir)
    manifest_path = root / _MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"rotation artifact is missing {manifest_path}") from error
    except json.JSONDecodeError as error:
        raise ValueError("rotation manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("rotation manifest must be a JSON object")

    try:
        metadata = ArtifactMetadata.from_dict(manifest["metadata"])
        rotation_filename = manifest["rotation_file"]
        expected_rotation_hash = manifest["rotations_sha256"]
        layer_ids = manifest["layer_ids"]
    except KeyError as error:
        raise ValueError(f"rotation manifest is missing {error.args[0]}") from error
    if rotation_filename != _ROTATION_FILENAME:
        raise ValueError(f"unsupported rotation filename {rotation_filename!r}")
    if layer_ids != list(range(metadata.num_layers)):
        raise ValueError("rotation manifest layer_ids are incomplete or reordered")
    if expectation is not None:
        expectation.validate(metadata)

    rotation_path = root / rotation_filename
    actual_rotation_hash = _sha256_file(rotation_path)
    if actual_rotation_hash != expected_rotation_hash:
        raise ValueError("rotation tensor SHA256 does not match manifest")
    try:
        payload = torch.load(rotation_path, map_location="cpu", weights_only=True)
    except FileNotFoundError as error:
        raise ValueError(f"rotation artifact is missing {rotation_path}") from error
    if not isinstance(payload, dict) or set(payload) != {
        "format_version",
        "rotations",
    }:
        raise ValueError("rotation tensor payload has an unsupported schema")
    if payload["format_version"] != metadata.format_version:
        raise ValueError("rotation tensor format version does not match manifest")
    if not isinstance(payload["rotations"], dict):
        raise ValueError("rotation tensor payload must contain a rotations mapping")
    try:
        rotations = {int(layer): value for layer, value in payload["rotations"].items()}
    except (TypeError, ValueError) as error:
        raise ValueError("rotation layer keys must be integers") from error
    normalized = _validate_rotations(metadata, rotations)
    return LoadedRotationArtifact(
        metadata=metadata,
        rotations=normalized,
        manifest_sha256=_sha256_file(manifest_path),
        rotations_sha256=actual_rotation_hash,
    )


def _validate_rotations(
    metadata: ArtifactMetadata,
    rotations: Mapping[int, torch.Tensor],
) -> dict[int, torch.Tensor]:
    expected_layers = set(range(metadata.num_layers))
    if set(rotations) != expected_layers:
        missing = sorted(expected_layers - set(rotations))
        extra = sorted(set(rotations) - expected_layers)
        raise ValueError(f"rotation layers mismatch; missing={missing}, extra={extra}")
    identity = torch.eye(
        metadata.latent_rank,
        dtype=torch.float64,
        device="cpu",
    )
    normalized: dict[int, torch.Tensor] = {}
    for layer in range(metadata.num_layers):
        rotation = rotations[layer]
        if not isinstance(rotation, torch.Tensor):
            raise TypeError(f"rotation layer {layer} must be a tensor")
        if rotation.shape != (metadata.latent_rank, metadata.latent_rank):
            raise ValueError(
                f"rotation layer {layer} has shape {tuple(rotation.shape)}, "
                f"expected {(metadata.latent_rank, metadata.latent_rank)}"
            )
        if not bool(torch.isfinite(rotation).all()):
            raise ValueError(f"rotation layer {layer} contains non-finite values")
        rotation64 = rotation.detach().double().cpu()
        if not torch.allclose(
            rotation64.T @ rotation64,
            identity,
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError(f"rotation layer {layer} is not orthogonal")
        normalized[layer] = rotation.detach().float().cpu().contiguous()
    return normalized


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as error:
        raise ValueError(f"artifact file is missing: {path}") from error
    return digest.hexdigest()
