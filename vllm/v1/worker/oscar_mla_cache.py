# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side views and ownership state for OSCAR MLA three-pool caches."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.v1.kv_cache_interface import OscarMLAAttentionSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization.oscar_mla.cache import (
        WorkerCacheMetadata,
    )
    from vllm.v1.core.sched.output import SchedulerOutput


@dataclass(frozen=True)
class OscarMLACacheTensors:
    """Non-overlapping views over one exact per-layer cache allocation."""

    raw: torch.Tensor
    history_data: torch.Tensor
    history_scale: torch.Tensor
    history_zero: torch.Tensor
    prefix: torch.Tensor
    recent: torch.Tensor
    recent_tokens: int
    rope: torch.Tensor


@dataclass(frozen=True)
class OscarMLABatchMetadata:
    """GPU metadata for one scheduled batch of three-pool cache operations."""

    hp_rows: torch.Tensor
    decode_positions: torch.Tensor
    final_seq_lens: torch.Tensor
    history_page_table: torch.Tensor
    previous_seq_lens: torch.Tensor
    demotion_hp_rows: torch.Tensor
    demotion_positions: torch.Tensor
    demotion_page_ids: torch.Tensor
    demotion_page_offsets: torch.Tensor
    restore_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    restore_hp_rows: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    restore_page_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    restore_page_offsets: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.int32)
    )
    num_restore_rows: int = 0


def reshape_oscar_mla_cache(
    raw: torch.Tensor,
    spec: OscarMLAAttentionSpec,
    *,
    num_blocks: int,
    max_num_seqs: int,
) -> OscarMLACacheTensors:
    """Carve the raw byte allocation into the three logical cache pools."""
    if raw.dtype != torch.int8 or raw.ndim != 1:
        raise ValueError("OSCAR MLA raw cache must be a flat int8 tensor")
    if num_blocks <= 0 or max_num_seqs <= 0:
        raise ValueError("OSCAR MLA cache dimensions must be positive")

    block_size = spec.block_size
    latent_rank = spec.latent_rank
    num_groups = latent_rank // spec.group_size
    history_slots = num_blocks * block_size
    packed_bytes = latent_rank * 2 // 8

    data_bytes = history_slots * packed_bytes
    metadata_elements = history_slots * num_groups
    metadata_bytes = (
        metadata_elements * torch.tensor([], dtype=torch.float32).element_size()
    )
    prefix_elements = max_num_seqs * spec.prefix_tokens * latent_rank
    recent_elements = max_num_seqs * spec.recent_capacity_tokens * latent_rank
    rope_elements = num_blocks * block_size * spec.rope_head_size
    hp_element_size = torch.tensor([], dtype=spec.hp_dtype).element_size()
    expected_bytes = (
        data_bytes
        + 2 * metadata_bytes
        + (prefix_elements + recent_elements + rope_elements) * hp_element_size
    )
    if raw.numel() != expected_bytes:
        raise ValueError(
            "OSCAR MLA raw cache size does not match the planner: "
            f"expected={expected_bytes}, actual={raw.numel()}"
        )

    offset = 0
    history_data = (
        raw.narrow(0, offset, data_bytes)
        .view(torch.uint8)
        .view(num_blocks, block_size, packed_bytes)
    )
    offset += data_bytes
    history_scale = (
        raw.narrow(0, offset, metadata_bytes)
        .view(torch.float32)
        .view(num_blocks, block_size, num_groups)
    )
    offset += metadata_bytes
    history_zero = (
        raw.narrow(0, offset, metadata_bytes)
        .view(torch.float32)
        .view(num_blocks, block_size, num_groups)
    )
    offset += metadata_bytes
    prefix_bytes = prefix_elements * hp_element_size
    prefix = (
        raw.narrow(0, offset, prefix_bytes)
        .view(spec.hp_dtype)
        .view(max_num_seqs, spec.prefix_tokens, latent_rank)
    )
    offset += prefix_bytes
    recent_bytes = recent_elements * hp_element_size
    recent = (
        raw.narrow(0, offset, recent_bytes)
        .view(spec.hp_dtype)
        .view(max_num_seqs, spec.recent_capacity_tokens, latent_rank)
    )
    offset += recent_bytes
    rope_bytes = rope_elements * hp_element_size
    rope = (
        raw.narrow(0, offset, rope_bytes)
        .view(spec.hp_dtype)
        .view(num_blocks, block_size, spec.rope_head_size)
    )
    offset += rope_bytes
    assert offset == raw.numel()

    return OscarMLACacheTensors(
        raw=raw,
        history_data=history_data,
        history_scale=history_scale,
        history_zero=history_zero,
        prefix=prefix,
        recent=recent,
        recent_tokens=spec.recent_tokens,
        rope=rope,
    )


class OscarMLAWorkerOwnership:
    """Request-keyed worker mirror of scheduler-owned three-pool metadata."""

    def __init__(self) -> None:
        self._metadata: dict[str, WorkerCacheMetadata] = {}
        self._previous_lengths: dict[str, int] = {}
        self._previous_history_tokens: dict[str, int] = {}
        self._cache_hit_lengths: dict[str, int] = {}
        self._cudagraph_metadata: dict[
            tuple[str, int, int, int, int], OscarMLABatchMetadata
        ] = {}

    def apply(self, scheduler_output: SchedulerOutput) -> None:
        released = set(scheduler_output.finished_req_ids)
        if scheduler_output.preempted_req_ids:
            released.update(scheduler_output.preempted_req_ids)
        candidate = {
            request_id: metadata
            for request_id, metadata in self._metadata.items()
            if request_id not in released
        }
        previous_lengths: dict[str, int] = {}
        previous_history_tokens: dict[str, int] = {}
        cache_hit_lengths: dict[str, int] = {}

        for request_id, metadata in scheduler_output.oscar_mla_cache_metadata.items():
            if metadata.request_id != request_id:
                raise RuntimeError("OSCAR MLA metadata request ID mismatch")
            previous = candidate.get(request_id)
            if previous is not None:
                if metadata.generation != previous.generation:
                    raise RuntimeError(
                        "OSCAR MLA generation changed without a release event"
                    )
                if metadata.cache_version < previous.cache_version:
                    raise RuntimeError("stale OSCAR MLA cache metadata")
            previous_lengths[request_id] = (
                previous.logical_length if previous is not None else 0
            )
            previous_history_tokens[request_id] = (
                previous.history_tokens if previous is not None else 0
            )
            cache_hit_lengths[request_id] = (
                metadata.num_cached_tokens if previous is None else 0
            )
            candidate[request_id] = metadata
        self._metadata = candidate
        self._previous_lengths = previous_lengths
        self._previous_history_tokens = previous_history_tokens
        self._cache_hit_lengths = cache_hit_lengths

    def build_batch_metadata(
        self,
        request_ids: list[str],
        *,
        block_size: int,
        prefix_tokens: int,
        recent_tokens: int,
        recent_capacity_tokens: int | None = None,
        device: torch.device,
        padded_size: int | None = None,
        cudagraph_max_history_pages: int | None = None,
        max_demotion_tokens_per_request: int = 1,
    ) -> OscarMLABatchMetadata:
        """Materialize scheduler ownership and incremental demotions on a device."""
        if recent_capacity_tokens is None:
            recent_capacity_tokens = recent_tokens
        if (
            block_size <= 0
            or prefix_tokens <= 0
            or recent_tokens <= 0
            or recent_capacity_tokens < recent_tokens
        ):
            raise ValueError("OSCAR MLA cache geometry must be positive")
        if padded_size is None:
            padded_size = len(request_ids)
        if padded_size < len(request_ids):
            raise ValueError("padded_size cannot be smaller than the request batch")
        if max_demotion_tokens_per_request <= 0:
            raise ValueError("max demotion tokens per request must be positive")

        rows: list[int] = []
        previous_seq_lens: list[int] = []
        previous_history_tokens: list[int] = []
        metadata_rows: list[WorkerCacheMetadata] = []
        for request_id in request_ids:
            metadata = self._metadata[request_id]
            if metadata.prefix_start != metadata.hp_row * prefix_tokens:
                raise RuntimeError("OSCAR MLA prefix ownership is inconsistent")
            if metadata.recent_start != metadata.hp_row * recent_capacity_tokens:
                raise RuntimeError("OSCAR MLA recent ownership is inconsistent")
            rows.append(metadata.hp_row)
            previous_seq_lens.append(
                self._previous_lengths.get(request_id, metadata.logical_length)
            )
            previous_history_tokens.append(
                self._previous_history_tokens.get(
                    request_id,
                    metadata.history_tokens,
                )
            )
            metadata_rows.append(metadata)

        actual_max_history_pages = max(
            [1, *(len(metadata.history_pages) for metadata in metadata_rows)]
        )
        max_history_pages = (
            actual_max_history_pages
            if cudagraph_max_history_pages is None
            else cudagraph_max_history_pages
        )
        if max_history_pages < actual_max_history_pages:
            raise ValueError("cudagraph history page capacity is too small")

        demotion_hp_rows: list[int] = []
        demotion_positions: list[int] = []
        demotion_pages: list[int] = []
        demotion_offsets: list[int] = []
        for request_index, metadata in enumerate(metadata_rows):
            previous_length = previous_seq_lens[request_index]
            old_history_end = prefix_tokens + previous_history_tokens[request_index]
            new_history_end = prefix_tokens + metadata.history_tokens
            demotion_end = min(new_history_end, previous_length)
            for position in range(old_history_end, demotion_end):
                history_index = position - prefix_tokens
                logical_page, page_offset = divmod(history_index, block_size)
                if logical_page >= len(metadata.history_pages):
                    raise RuntimeError("OSCAR MLA history ownership is incomplete")
                demotion_hp_rows.append(metadata.hp_row)
                demotion_positions.append(position)
                demotion_pages.append(metadata.history_pages[logical_page])
                demotion_offsets.append(page_offset)

        restore_positions: list[int] = []
        restore_hp_rows: list[int] = []
        restore_pages: list[int] = []
        restore_offsets: list[int] = []
        for metadata in metadata_rows:
            cache_hit_length = self._cache_hit_lengths.get(metadata.request_id, 0)
            if cache_hit_length <= 0:
                continue
            recent_begin = max(prefix_tokens, metadata.logical_length - recent_tokens)
            positions = list(range(min(prefix_tokens, cache_hit_length)))
            positions.extend(range(recent_begin, cache_hit_length))
            for position in positions:
                block_index, block_offset = divmod(position, block_size)
                if block_index >= len(metadata.block_ids):
                    raise RuntimeError("OSCAR MLA cached block ownership is incomplete")
                restore_positions.append(position)
                restore_hp_rows.append(metadata.hp_row)
                restore_pages.append(metadata.block_ids[block_index])
                restore_offsets.append(block_offset)

        def _device_tensor(values: list[int]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.int32, device=device)

        hp_rows = rows + [-1] * (padded_size - len(rows))
        decode_positions = [metadata.logical_length - 1 for metadata in metadata_rows]
        decode_positions += [-1] * (padded_size - len(decode_positions))
        final_seq_lens = [metadata.logical_length for metadata in metadata_rows]
        final_seq_lens += [0] * (padded_size - len(final_seq_lens))
        padded_previous_seq_lens = previous_seq_lens + [0] * (
            padded_size - len(previous_seq_lens)
        )
        if cudagraph_max_history_pages is None:
            history_page_table = torch.zeros(
                (padded_size, max_history_pages),
                dtype=torch.int32,
                device=device,
            )
            for row, metadata in enumerate(metadata_rows):
                if metadata.history_pages:
                    history_page_table[row, : len(metadata.history_pages)] = (
                        _device_tensor(list(metadata.history_pages))
                    )
            return OscarMLABatchMetadata(
                hp_rows=_device_tensor(hp_rows),
                decode_positions=_device_tensor(decode_positions),
                final_seq_lens=_device_tensor(final_seq_lens),
                history_page_table=history_page_table,
                previous_seq_lens=_device_tensor(padded_previous_seq_lens),
                demotion_hp_rows=_device_tensor(demotion_hp_rows),
                demotion_positions=_device_tensor(demotion_positions),
                demotion_page_ids=_device_tensor(demotion_pages),
                demotion_page_offsets=_device_tensor(demotion_offsets),
                restore_positions=_device_tensor(restore_positions),
                restore_hp_rows=_device_tensor(restore_hp_rows),
                restore_page_ids=_device_tensor(restore_pages),
                restore_page_offsets=_device_tensor(restore_offsets),
                num_restore_rows=len(restore_positions),
            )

        demotion_capacity = padded_size * max_demotion_tokens_per_request
        if len(demotion_positions) > demotion_capacity:
            raise RuntimeError(
                "OSCAR MLA CUDA graph decode exceeded its demotion capacity"
            )
        graph_key = (
            str(device),
            padded_size,
            max_history_pages,
            demotion_capacity,
            padded_size * (prefix_tokens + recent_tokens),
        )
        graph_metadata = self._cudagraph_metadata.get(graph_key)
        if graph_metadata is None:
            default_vector_size = int(padded_size)

            def _empty_vector(size: int | None = None) -> torch.Tensor:
                vector_size = default_vector_size if size is None else size
                return torch.empty(vector_size, dtype=torch.int32, device=device)

            graph_metadata = OscarMLABatchMetadata(
                hp_rows=_empty_vector(),
                decode_positions=_empty_vector(),
                final_seq_lens=_empty_vector(),
                history_page_table=torch.empty(
                    (padded_size, max_history_pages),
                    dtype=torch.int32,
                    device=device,
                ),
                previous_seq_lens=_empty_vector(),
                demotion_hp_rows=_empty_vector(demotion_capacity),
                demotion_positions=_empty_vector(demotion_capacity),
                demotion_page_ids=_empty_vector(demotion_capacity),
                demotion_page_offsets=_empty_vector(demotion_capacity),
                restore_positions=_empty_vector(
                    padded_size * (prefix_tokens + recent_tokens)
                ),
                restore_hp_rows=_empty_vector(
                    padded_size * (prefix_tokens + recent_tokens)
                ),
                restore_page_ids=_empty_vector(
                    padded_size * (prefix_tokens + recent_tokens)
                ),
                restore_page_offsets=_empty_vector(
                    padded_size * (prefix_tokens + recent_tokens)
                ),
                num_restore_rows=0,
            )
            self._cudagraph_metadata[graph_key] = graph_metadata

        graph_metadata.hp_rows.copy_(_device_tensor(hp_rows))
        graph_metadata.decode_positions.copy_(_device_tensor(decode_positions))
        graph_metadata.final_seq_lens.copy_(_device_tensor(final_seq_lens))
        graph_metadata.previous_seq_lens.copy_(_device_tensor(padded_previous_seq_lens))
        graph_metadata.history_page_table.zero_()
        for row, metadata in enumerate(metadata_rows):
            if metadata.history_pages:
                graph_metadata.history_page_table[
                    row, : len(metadata.history_pages)
                ].copy_(_device_tensor(list(metadata.history_pages)))

        demotion_padding = demotion_capacity - len(demotion_positions)
        graph_metadata.demotion_hp_rows.copy_(
            _device_tensor(demotion_hp_rows + [-1] * demotion_padding)
        )
        graph_metadata.demotion_positions.copy_(
            _device_tensor(demotion_positions + [-1] * demotion_padding)
        )
        graph_metadata.demotion_page_ids.copy_(
            _device_tensor(demotion_pages + [-1] * demotion_padding)
        )
        graph_metadata.demotion_page_offsets.copy_(
            _device_tensor(demotion_offsets + [0] * demotion_padding)
        )
        restore_capacity = padded_size * (prefix_tokens + recent_tokens)
        restore_padding = restore_capacity - len(restore_positions)
        graph_metadata.restore_positions.copy_(
            _device_tensor(restore_positions + [-1] * restore_padding)
        )
        graph_metadata.restore_hp_rows.copy_(
            _device_tensor(restore_hp_rows + [-1] * restore_padding)
        )
        graph_metadata.restore_page_ids.copy_(
            _device_tensor(restore_pages + [-1] * restore_padding)
        )
        graph_metadata.restore_page_offsets.copy_(
            _device_tensor(restore_offsets + [0] * restore_padding)
        )
        if restore_positions:
            raise RuntimeError("OSCAR MLA cache-hit restore cannot run in a CUDA graph")
        return graph_metadata

    def get(self, request_id: str) -> WorkerCacheMetadata:
        return self._metadata[request_id]

    def __contains__(self, request_id: str) -> bool:
        return request_id in self._metadata
