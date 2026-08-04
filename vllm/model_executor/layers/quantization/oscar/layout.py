# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token partitioning for OSCAR mixed-precision KV cache."""

from dataclasses import dataclass


@dataclass(frozen=True)
class OscarTokenPartition:
    prefix: range
    history: range
    recent: range

    @property
    def prefix_count(self) -> int:
        return len(self.prefix)

    @property
    def history_count(self) -> int:
        return len(self.history)

    @property
    def recent_count(self) -> int:
        return len(self.recent)

    @property
    def total_count(self) -> int:
        return self.prefix_count + self.history_count + self.recent_count


def partition_tokens(
    seq_len: int,
    *,
    prefix_tokens: int,
    recent_tokens: int,
) -> OscarTokenPartition:
    """Partition ``[0, seq_len)`` into BF16 prefix, INT2 history, BF16 recent."""
    if seq_len < 0:
        raise ValueError(f"seq_len must be non-negative, got {seq_len}")
    if prefix_tokens < 0:
        raise ValueError(f"prefix_tokens must be non-negative, got {prefix_tokens}")
    if recent_tokens < 0:
        raise ValueError(f"recent_tokens must be non-negative, got {recent_tokens}")

    prefix_end = min(seq_len, prefix_tokens)
    recent_start = max(prefix_end, seq_len - recent_tokens)
    return OscarTokenPartition(
        prefix=range(0, prefix_end),
        history=range(prefix_end, recent_start),
        recent=range(recent_start, seq_len),
    )
