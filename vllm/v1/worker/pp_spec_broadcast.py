# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PP speculative-decoding sampled-token helpers."""

import torch


def count_valid_sampled_tokens_per_req(
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Count non-padding sampled tokens per request."""
    return (sampled_token_ids != -1).sum(dim=1)


def select_latest_sampled_token_per_req(
    sampled_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Select the latest valid sampled token from each request row."""
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids)
    last_idx = (counts - 1).clamp(min=0)
    return sampled_token_ids.gather(1, last_idx.unsqueeze(1)).squeeze(1)


def gather_valid_sampled_tokens_per_req(
    sampled_token_ids: torch.Tensor,
) -> list[list[int]]:
    """Return all leading valid sampled tokens for each request row."""
    counts = count_valid_sampled_tokens_per_req(sampled_token_ids).tolist()
    rows = sampled_token_ids.tolist()
    return [row[:count] for row, count in zip(rows, counts)]


def num_computed_tokens_drift_correction(
    prev_num_draft_len: int,
    valid_sampled_count: int,
) -> int:
    """Rejected-draft count to subtract from optimistic num_computed_tokens."""
    return (1 + prev_num_draft_len) - valid_sampled_count
