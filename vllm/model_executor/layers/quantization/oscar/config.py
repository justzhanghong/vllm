# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OSCAR configuration."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Named OSCAR presets, selected via --kv-cache-dtype. Each maps to a frozen
# (key_bits, value_bits) pair. The remaining, deployment-specific knobs
# (rotation matrix paths, clip ratios, mixed-precision window sizes) are read
# from environment variables, mirroring the SGLang reference UX so that a
# checkpoint's RotationZoo artifacts can be pointed at without re-exporting a
# model. See ``OscarConfig.from_env``.
OSCAR_PRESETS: dict[str, dict] = {
    # OSCAR's headline 2-bit KV configuration (BPE ~2.28 in the paper).
    "oscar_int2": {"key_quant_bits": 2, "value_quant_bits": 2},
}

# Environment knobs (defaults match the OSCAR README serving recipe).
_ENV_K_ROTATION = "VLLM_OSCAR_K_ROTATION_PATH"
_ENV_V_ROTATION = "VLLM_OSCAR_V_ROTATION_PATH"
_ENV_K_CLIP = "VLLM_OSCAR_K_CLIP_RATIO"
_ENV_V_CLIP = "VLLM_OSCAR_V_CLIP_RATIO"
_ENV_GROUP_SIZE = "VLLM_OSCAR_GROUP_SIZE"
_ENV_PREFIX_TOKENS = "VLLM_OSCAR_PREFIX_TOKENS"
_ENV_RECENT_TOKENS = "VLLM_OSCAR_RECENT_TOKENS"
_ENV_PREFIX_CACHE_EXTRA_TOKENS = "VLLM_OSCAR_PREFIX_CACHE_EXTRA_TOKENS"
_ENV_ABSORB_V_ROTATION = "VLLM_OSCAR_ABSORB_V_ROTATION"


def _parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}"
    )


@dataclass
class OscarConfig:
    """Configuration for OSCAR INT2 KV-cache quantization.

    Applies a *calibrated* per-layer orthogonal rotation (loaded from disk)
    to keys and values, optional percentile clipping, then per-group
    asymmetric INT2 scalar quantization.

    Because the rotation is orthogonal, the attention scores are invariant
    under it: keys are stored as ``K @ R_k`` and the query is rotated by the
    same ``R_k`` at score time, so ``(Q R_k)(K R_k)^T = Q K^T``. Values are
    stored as ``V @ R_v`` and the attention output is mapped back with
    ``R_v^T`` after the weighted sum. When ``absorb_v_rotation`` is enabled,
    the QKV projection directly produces ``V @ R_v``; the output inverse is
    still required, matching the reference implementation.

    Args:
        head_dim: Attention head dimension (e.g. 128).
        key_quant_bits: Bits per key element (2 for the headline preset).
        value_quant_bits: Bits per value element (2 for the headline preset).
        group_size: Quantization group size along head_dim. One scale/zero
            pair is stored per group. With ``head_dim <= group_size`` this is
            a single group per vector (the Qwen3 ``head_dim=128`` case).
        k_clip_ratio: Fraction of the per-vector dynamic range retained for
            keys (0 disables clipping). 0.96 in the README recipe.
        v_clip_ratio: As ``k_clip_ratio`` for values. 0.92 in the recipe.
        k_rotation_path: Path to the ``[num_layers, head_dim, head_dim]`` (or
            per-layer) key rotation tensor. The prototype requires this path.
        v_rotation_path: As ``k_rotation_path`` for values.
        absorb_v_rotation: Fold ``R_v`` into the QKV V slice after loading.
    """

    head_dim: int = 128
    key_quant_bits: int = 2
    value_quant_bits: int = 2
    group_size: int = 128
    k_clip_ratio: float = 0.96
    v_clip_ratio: float = 0.92
    k_rotation_path: str = ""
    v_rotation_path: str = ""
    prefix_tokens: int = 64
    recent_tokens: int = 256
    prefix_cache_extra_tokens: int = 0
    absorb_v_rotation: bool = False

    # ----- derived geometry ------------------------------------------------
    @property
    def num_groups(self) -> int:
        return math.ceil(self.head_dim / self.group_size)

    @property
    def key_levels(self) -> int:
        return 2**self.key_quant_bits

    @property
    def value_levels(self) -> int:
        return 2**self.value_quant_bits

    @property
    def key_data_bytes(self) -> int:
        """Packed index bytes for one key vector (4 INT2 values per byte)."""
        return math.ceil(self.head_dim * self.key_quant_bits / 8)

    @property
    def value_data_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_quant_bits / 8)

    @property
    def meta_bytes(self) -> int:
        """Per-vector metadata: BF16 scale + zero point per group."""
        return self.num_groups * 4

    @property
    def key_packed_size(self) -> int:
        return self.key_data_bytes + self.meta_bytes

    @property
    def value_packed_size(self) -> int:
        return self.value_data_bytes + self.meta_bytes

    @property
    def slot_size(self) -> int:
        """Combined per-head per-position bytes: [key_packed | value_packed]."""
        return self.key_packed_size + self.value_packed_size

    @property
    def slot_size_aligned(self) -> int:
        """Slot size rounded up to an even number so ``slot // 2`` is integral
        (vLLM derives ``effective_head_size = slot_size_aligned // 2``)."""
        s = self.slot_size
        return s + (s % 2)

    @property
    def bf16_slot_size(self) -> int:
        """Uncompressed K+V bytes for one token and one KV head."""
        return 2 * self.head_dim * 2

    def hp_slots_per_block(self, block_size: int, max_model_len: int) -> int:
        """BF16 arena slots contributed by each physical cache block."""
        if block_size <= 0 or max_model_len <= 0:
            raise ValueError(
                "block_size and max_model_len must be positive, got "
                f"{block_size} and {max_model_len}"
            )
        min_blocks = math.ceil(max_model_len / block_size)
        hp_tokens = min(self.prefix_tokens + self.recent_tokens, max_model_len)
        return math.ceil(hp_tokens / min_blocks)

    def padded_page_size_bytes(
        self,
        block_size: int,
        num_kv_heads: int,
        max_model_len: int,
    ) -> int:
        """INT2 logical page plus its contribution to the shared BF16 arena."""
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        quant_page = block_size * num_kv_heads * self.slot_size_aligned
        hp_page = (
            self.hp_slots_per_block(block_size, max_model_len)
            * num_kv_heads
            * self.bf16_slot_size
        )
        return quant_page + hp_page

    def mixed_bytes_per_layer(self, seq_len: int, num_kv_heads: int) -> int:
        """Bytes occupied by the task's mixed layout for one model layer."""
        from vllm.model_executor.layers.quantization.oscar.layout import (
            partition_tokens,
        )

        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {num_kv_heads}")
        part = partition_tokens(
            seq_len,
            prefix_tokens=self.prefix_tokens,
            recent_tokens=self.recent_tokens,
        )
        bf16_tokens = part.prefix_count + part.recent_count
        bytes_per_head = (
            bf16_tokens * self.bf16_slot_size
            + part.history_count * self.slot_size_aligned
        )
        return bytes_per_head * num_kv_heads

    def mixed_bytes_per_token_per_layer(self, seq_len: int, num_kv_heads: int) -> float:
        if seq_len <= 0:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        return self.mixed_bytes_per_layer(seq_len, num_kv_heads) / seq_len

    def validate_prototype_settings(self) -> None:
        """Reject configurations outside the task's verified prototype scope."""
        expected = {
            "head_dim": (self.head_dim, 128),
            "group_size": (self.group_size, 128),
            "key_quant_bits": (self.key_quant_bits, 2),
            "value_quant_bits": (self.value_quant_bits, 2),
            "prefix_tokens": (self.prefix_tokens, 64),
            "recent_tokens": (self.recent_tokens, 256),
        }
        mismatches = [
            f"{name}={actual} (expected {wanted})"
            for name, (actual, wanted) in expected.items()
            if actual != wanted
        ]
        if not math.isclose(self.k_clip_ratio, 0.96, abs_tol=1e-9):
            mismatches.append(f"k_clip_ratio={self.k_clip_ratio} (expected 0.96)")
        if not math.isclose(self.v_clip_ratio, 0.92, abs_tol=1e-9):
            mismatches.append(f"v_clip_ratio={self.v_clip_ratio} (expected 0.92)")
        if mismatches:
            raise ValueError(
                "Unsupported OSCAR prototype configuration: " + ", ".join(mismatches)
            )

        for kind, path in (
            ("K", self.k_rotation_path),
            ("V", self.v_rotation_path),
        ):
            if not path:
                raise ValueError(f"VLLM_OSCAR_{kind}_ROTATION_PATH must be set")
            if not Path(path).is_file():
                raise ValueError(f"OSCAR {kind} rotation file does not exist: {path}")

    # ----- constructors ----------------------------------------------------
    @staticmethod
    def from_cache_dtype(cache_dtype: str, head_dim: int) -> OscarConfig:
        """Create a config from a named preset plus environment knobs."""
        if cache_dtype not in OSCAR_PRESETS:
            valid = ", ".join(OSCAR_PRESETS.keys())
            raise ValueError(
                f"Unknown OSCAR cache dtype: {cache_dtype!r}. Valid presets: {valid}"
            )
        preset = OSCAR_PRESETS[cache_dtype]
        group_size = int(os.environ.get(_ENV_GROUP_SIZE, "128"))
        return OscarConfig(
            head_dim=head_dim,
            key_quant_bits=preset["key_quant_bits"],
            value_quant_bits=preset["value_quant_bits"],
            group_size=group_size,
            k_clip_ratio=float(os.environ.get(_ENV_K_CLIP, "0.96")),
            v_clip_ratio=float(os.environ.get(_ENV_V_CLIP, "0.92")),
            k_rotation_path=os.environ.get(_ENV_K_ROTATION, ""),
            v_rotation_path=os.environ.get(_ENV_V_ROTATION, ""),
            prefix_tokens=int(os.environ.get(_ENV_PREFIX_TOKENS, "64")),
            recent_tokens=int(os.environ.get(_ENV_RECENT_TOKENS, "256")),
            prefix_cache_extra_tokens=int(
                os.environ.get(_ENV_PREFIX_CACHE_EXTRA_TOKENS, "0")
            ),
            absorb_v_rotation=_parse_bool_env(_ENV_ABSORB_V_ROTATION),
        )
