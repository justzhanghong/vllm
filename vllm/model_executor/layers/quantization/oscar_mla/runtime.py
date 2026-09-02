"""Fail-closed runtime binding for OSCAR MLA rotation artifacts."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from vllm.model_executor.layers.quantization.oscar_mla.artifact import (
    ArtifactExpectation,
    LoadedRotationArtifact,
    load_rotation_artifact,
)

_ARTIFACT_ENV = "VLLM_OSCAR_MLA_ROTATION_ARTIFACT"
_EXPECTATION_ENV = "VLLM_OSCAR_MLA_RUNTIME_EXPECTATION"
_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class LayerRuntimeParameters:
    rotation: torch.Tensor
    clip_ratio: float
    manifest_sha256: str
    rotations_sha256: str


@dataclass(frozen=True)
class RuntimeArtifactIdentity:
    manifest_sha256: str
    rotations_sha256: str


def load_runtime_artifact_identity() -> RuntimeArtifactIdentity:
    """Return the identity of the same verified artifact used by attention."""
    artifact_path = os.getenv(_ARTIFACT_ENV)
    expectation_path = os.getenv(_EXPECTATION_ENV)
    if not artifact_path or not expectation_path:
        raise ValueError(
            f"oscar_mla_int2 requires {_ARTIFACT_ENV} and {_EXPECTATION_ENV}"
        )
    loaded = _load_runtime_artifact(artifact_path, expectation_path)
    return RuntimeArtifactIdentity(
        manifest_sha256=loaded.manifest_sha256,
        rotations_sha256=loaded.rotations_sha256,
    )


def load_layer_runtime_parameters(
    layer_name: str,
    *,
    latent_rank: int,
    prefix_tokens: int,
    recent_tokens: int,
) -> LayerRuntimeParameters:
    """Load one layer only after artifact identity and geometry match runtime."""
    artifact_path = os.getenv(_ARTIFACT_ENV)
    expectation_path = os.getenv(_EXPECTATION_ENV)
    if not artifact_path or not expectation_path:
        raise ValueError(
            f"oscar_mla_int2 requires {_ARTIFACT_ENV} and {_EXPECTATION_ENV}"
        )
    match = _LAYER_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(f"cannot resolve OSCAR MLA layer ID from {layer_name!r}")
    layer_id = int(match.group(1))

    loaded = _load_runtime_artifact(artifact_path, expectation_path)
    metadata = loaded.metadata
    mismatches = []
    for name, actual in (
        ("latent_rank", latent_rank),
        ("prefix_tokens", prefix_tokens),
        ("recent_tokens", recent_tokens),
    ):
        if getattr(metadata, name) != actual:
            mismatches.append(name)
    if mismatches:
        raise ValueError(
            "OSCAR MLA layer geometry does not match rotation artifact: "
            + ", ".join(mismatches)
        )
    if layer_id not in loaded.rotations:
        raise ValueError(f"rotation artifact does not contain layer {layer_id}")
    return LayerRuntimeParameters(
        rotation=loaded.rotations[layer_id],
        clip_ratio=metadata.clip_ratios[layer_id],
        manifest_sha256=loaded.manifest_sha256,
        rotations_sha256=loaded.rotations_sha256,
    )


@lru_cache(maxsize=4)
def _load_runtime_artifact(
    artifact_path: str,
    expectation_path: str,
) -> LoadedRotationArtifact:
    try:
        payload = json.loads(Path(expectation_path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"OSCAR MLA runtime expectation is missing {expectation_path}"
        ) from error
    except json.JSONDecodeError as error:
        raise ValueError("OSCAR MLA runtime expectation is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("OSCAR MLA runtime expectation must be a JSON object")
    try:
        expectation = ArtifactExpectation(
            model_config_sha256=payload["model_config_sha256"],
            checkpoint_manifest_sha256=payload["checkpoint_manifest_sha256"],
            expert_mapping_sha256=payload["expert_mapping_sha256"],
            num_layers=payload["num_layers"],
            latent_rank=payload["latent_rank"],
            group_size=payload["group_size"],
            prefix_tokens=payload["prefix_tokens"],
            recent_tokens=payload["recent_tokens"],
        )
    except KeyError as error:
        raise ValueError(
            f"OSCAR MLA runtime expectation is missing {error.args[0]}"
        ) from error
    return load_rotation_artifact(artifact_path, expectation=expectation)
