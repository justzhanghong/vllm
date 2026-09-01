"""Read-only activation capture for shared-latent OSCAR calibration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch

_CAPTURE_CONFIG_ENV = "VLLM_OSCAR_MLA_CAPTURE_CONFIG"


@dataclass(frozen=True)
class CaptureConfig:
    """Configuration shared by every tensor-parallel capture worker."""

    output_dir: str
    token_budget: int
    latent_rank: int
    seed: int
    split: str
    flush_interval_rows: int = 65536
    reservoir_rows: int = 0
    dsa_sample_rows: int = 0
    capture_layers: list[str] | None = None
    tp_rank: int = 0
    capture_latent: bool = True

    def __post_init__(self) -> None:
        if not self.output_dir:
            raise ValueError("output_dir must not be empty")
        if self.token_budget <= 0:
            raise ValueError("token_budget must be positive")
        if self.latent_rank <= 0:
            raise ValueError("latent_rank must be positive")
        if self.flush_interval_rows <= 0:
            raise ValueError("flush_interval_rows must be positive")
        if not 0 <= self.reservoir_rows <= self.token_budget:
            raise ValueError("reservoir_rows must be in [0, token_budget]")
        if not 0 <= self.dsa_sample_rows <= self.reservoir_rows:
            raise ValueError("dsa_sample_rows must be in [0, reservoir_rows]")
        if self.tp_rank < 0:
            raise ValueError("tp_rank must be non-negative")
        if not self.split:
            raise ValueError("split must not be empty")
        if self.capture_layers is not None and (
            not self.capture_layers
            or any(not layer for layer in self.capture_layers)
            or len(self.capture_layers) != len(set(self.capture_layers))
        ):
            raise ValueError("capture_layers must contain unique non-empty names")

    @classmethod
    def from_json(cls, path: str | Path) -> CaptureConfig:
        with Path(path).open(encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise TypeError("capture config must be a JSON object")
        return cls(**data)


class _GramAccumulator:
    """Accumulate X^T X on-device and periodically merge it into CPU FP64."""

    def __init__(self, rank: int, flush_interval_rows: int) -> None:
        self.rank = rank
        self.flush_interval_rows = flush_interval_rows
        self.samples = 0
        self._pending_rows = 0
        self._device_sum: torch.Tensor | None = None
        self._cpu_sum = torch.zeros(rank, rank, dtype=torch.float64)

    def update(self, rows: torch.Tensor) -> None:
        rows = rows.detach().reshape(-1, self.rank).float()
        if not rows.numel():
            return
        if self._device_sum is None:
            self._device_sum = torch.zeros(
                self.rank,
                self.rank,
                dtype=torch.float32,
                device=rows.device,
            )
        elif self._device_sum.device != rows.device:
            raise ValueError(
                "capture accumulator received tensors from different devices"
            )
        self._device_sum.addmm_(rows.T, rows)
        self.samples += rows.shape[0]
        self._pending_rows += rows.shape[0]
        if self._pending_rows >= self.flush_interval_rows:
            self.flush()

    def flush(self) -> None:
        if self._device_sum is None or self._pending_rows == 0:
            return
        self._cpu_sum.add_(self._device_sum.detach().double().cpu())
        self._device_sum.zero_()
        self._pending_rows = 0

    def sum(self) -> torch.Tensor:
        self.flush()
        return self._cpu_sum.clone()


class _LayerCapture:
    def __init__(
        self,
        config: CaptureConfig,
        sample_positions: torch.Tensor,
    ) -> None:
        self.config = config
        self.sample_positions = sample_positions
        self.tokens = 0
        self.score = _GramAccumulator(
            config.latent_rank,
            config.flush_interval_rows,
        )
        self.value = _GramAccumulator(
            config.latent_rank,
            config.flush_interval_rows,
        )
        self.latent = (
            _GramAccumulator(config.latent_rank, config.flush_interval_rows)
            if config.capture_latent
            else None
        )
        self._sample_token_positions: list[torch.Tensor] = []
        self._latent_samples: list[torch.Tensor] = []
        self._query_samples: list[torch.Tensor] = []
        self._value_samples: list[torch.Tensor] = []
        self._dsa_samples: list[torch.Tensor] = []

    def update(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        value_output: torch.Tensor,
        dsa_indices: torch.Tensor | None,
    ) -> bool:
        remaining = self.config.token_budget - self.tokens
        if remaining <= 0:
            return True
        rows = min(remaining, latent.shape[0])
        latent = latent[:rows]
        query = query[:rows]
        value_output = value_output[:rows]
        if dsa_indices is not None:
            dsa_indices = dsa_indices[:rows]

        self._validate_shapes(latent, query, value_output, dsa_indices)
        row_indices = torch.arange(rows, device=query.device)
        global_positions = row_indices + self.tokens
        head_indices = global_positions.remainder(query.shape[1])
        sampled_query = query[row_indices, head_indices]
        sampled_value = value_output[row_indices, head_indices]

        self.score.update(sampled_query)
        self.value.update(sampled_value)
        if self.latent is not None:
            self.latent.update(latent)
        self._capture_reservoir(
            latent,
            sampled_query,
            sampled_value,
            dsa_indices,
            self.tokens,
            self.tokens + rows,
        )
        self.tokens += rows
        return self.tokens == self.config.token_budget

    def payload(self, layer_name: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "format_version": 1,
            "layer_name": layer_name,
            "split": self.config.split,
            "tp_rank": self.config.tp_rank,
            "latent_rank": self.config.latent_rank,
            "token_budget": self.config.token_budget,
            "captured_tokens": self.tokens,
            "seed": self.config.seed,
            "score_covariance_samples": self.score.samples,
            "value_covariance_samples": self.value.samples,
            "score_second_moment_sum": self.score.sum(),
            "value_second_moment_sum": self.value.sum(),
            "sample_token_positions": self._cat(
                self._sample_token_positions,
                dtype=torch.int64,
            ),
            "latent_samples": self._cat(
                self._latent_samples,
                shape=(0, self.config.latent_rank),
            ),
            "query_samples": self._cat(
                self._query_samples,
                shape=(0, self.config.latent_rank),
            ),
            "value_samples": self._cat(
                self._value_samples,
                shape=(0, self.config.latent_rank),
            ),
            "dsa_samples": self._cat(self._dsa_samples),
        }
        if self.latent is not None:
            payload["latent_covariance_samples"] = self.latent.samples
            payload["latent_second_moment_sum"] = self.latent.sum()
        return payload

    def _capture_reservoir(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        value_output: torch.Tensor,
        dsa_indices: torch.Tensor | None,
        start: int,
        end: int,
    ) -> None:
        selected = self.sample_positions[
            (self.sample_positions >= start) & (self.sample_positions < end)
        ]
        if not selected.numel():
            return
        offsets = (selected - start).to(device=latent.device)
        self._sample_token_positions.append(selected.clone())
        self._latent_samples.append(latent[offsets].detach().cpu())
        self._query_samples.append(query[offsets].detach().cpu())
        self._value_samples.append(value_output[offsets].detach().cpu())
        if dsa_indices is not None:
            captured = sum(chunk.shape[0] for chunk in self._dsa_samples)
            remaining = self.config.dsa_sample_rows - captured
            if remaining > 0:
                self._dsa_samples.append(
                    dsa_indices[offsets[:remaining]].detach().cpu()
                )

    def _validate_shapes(
        self,
        latent: torch.Tensor,
        query: torch.Tensor,
        value_output: torch.Tensor,
        dsa_indices: torch.Tensor | None,
    ) -> None:
        rank = self.config.latent_rank
        if latent.ndim != 2 or latent.shape[1] != rank:
            raise ValueError(f"latent must have shape [tokens, {rank}]")
        if query.ndim != 3 or query.shape[2] != rank:
            raise ValueError(f"query must have shape [tokens, heads, {rank}]")
        if value_output.shape != query.shape:
            raise ValueError("value_output must have the same shape as query")
        if latent.shape[0] != query.shape[0]:
            raise ValueError("latent and query token counts differ")
        if dsa_indices is not None and (
            dsa_indices.ndim != 2 or dsa_indices.shape[0] != latent.shape[0]
        ):
            raise ValueError("dsa_indices must have shape [tokens, topk]")

    @staticmethod
    def _cat(
        chunks: list[torch.Tensor],
        *,
        dtype: torch.dtype | None = None,
        shape: tuple[int, ...] = (0, 0),
    ) -> torch.Tensor:
        if chunks:
            return torch.cat(chunks)
        return torch.empty(shape, dtype=dtype)


class ActivationCaptureSession:
    """Capture one deterministic token budget for each MLA layer."""

    def __init__(self, config: CaptureConfig) -> None:
        self.config = config
        generator = torch.Generator().manual_seed(config.seed)
        self.sample_positions = torch.sort(
            torch.randperm(config.token_budget, generator=generator)[
                : config.reservoir_rows
            ]
        ).values
        self._layers: dict[str, _LayerCapture] = {}
        self._completed: set[str] = set()

    @torch.no_grad()
    def capture(
        self,
        layer_name: str,
        latent: torch.Tensor,
        query: torch.Tensor,
        value_output: torch.Tensor,
        dsa_indices: torch.Tensor | None = None,
    ) -> None:
        """Observe tensors without changing their values, shapes, or storage."""
        if (
            self.config.capture_layers is not None
            and layer_name not in self.config.capture_layers
        ):
            return
        if layer_name in self._completed:
            return
        layer = self._layers.setdefault(
            layer_name,
            _LayerCapture(self.config, self.sample_positions),
        )
        if layer.update(latent, query, value_output, dsa_indices):
            self._write_layer(layer_name, layer.payload(layer_name))
            self._completed.add(layer_name)
            del self._layers[layer_name]

    def _write_layer(self, layer_name: str, payload: dict[str, Any]) -> None:
        layer_dir = (
            Path(self.config.output_dir)
            / f"tp_rank_{self.config.tp_rank:02d}"
            / "layers"
        )
        layer_dir.mkdir(parents=True, exist_ok=True)
        safe_name = layer_name.replace("/", "_")
        output_path = layer_dir / f"{safe_name}.pt"
        temporary_path = output_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary_path)
        os.replace(temporary_path, output_path)


_capture_session: ActivationCaptureSession | None = None
_capture_config_path: str | None = None


def capture_mla_activations(
    layer_name: str,
    latent: torch.Tensor,
    query: torch.Tensor,
    value_output: torch.Tensor,
    dsa_indices: torch.Tensor | None,
) -> None:
    """Dispatch to the process-local capture session when explicitly enabled."""
    global _capture_config_path, _capture_session

    config_path = os.getenv(_CAPTURE_CONFIG_ENV)
    if not config_path:
        return
    if _capture_session is None:
        if _capture_config_path is not None and _capture_config_path != config_path:
            raise RuntimeError("OSCAR MLA capture config changed after initialization")
        config = CaptureConfig.from_json(config_path)
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            tp_rank = get_tensor_model_parallel_rank()
        except (AssertionError, RuntimeError):
            tp_rank = config.tp_rank
        config = replace(
            config,
            tp_rank=tp_rank,
            capture_latent=tp_rank == 0,
        )
        _capture_config_path = config_path
        _capture_session = ActivationCaptureSession(config)
    _capture_session.capture(
        layer_name,
        latent,
        query,
        value_output,
        dsa_indices,
    )
