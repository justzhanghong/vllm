# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Wire-contract helpers for OSCAR MLA native NIXL transfers."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

OSCAR_MLA_NIXL_PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class OscarMLARequestMetadata:
    """Request-local ownership exported by one OSCAR MLA cache allocator."""

    generation: int
    cache_version: int
    logical_length: int
    hp_row: int
    block_ids: tuple[int, ...]
    history_pages: tuple[int, ...]
    partial_history_slots: int
    history_tokens: int

    def trim_speculative_tail_blocks(
        self, block_size: int
    ) -> OscarMLARequestMetadata:
        """Drop scheduler-only lookahead blocks from the wire ownership.

        MTP may reserve one physical tail block beyond the request's confirmed
        logical length.  That block belongs to Decode-local draft state and
        must not be exported as target prompt KV.
        """
        if block_size <= 0:
            raise ValueError("invalid OSCAR MLA block size")
        expected_blocks = (self.logical_length + block_size - 1) // block_size
        if len(self.block_ids) < expected_blocks:
            raise ValueError("OSCAR MLA physical block ownership is incomplete")
        if len(self.block_ids) == expected_blocks:
            return self
        return replace(self, block_ids=self.block_ids[:expected_blocks])

    def to_wire(self) -> dict[str, Any]:
        return {
            "generation": self.generation,
            "cache_version": self.cache_version,
            "logical_length": self.logical_length,
            "hp_row": self.hp_row,
            "block_ids": list(self.block_ids),
            "history_pages": list(self.history_pages),
            "partial_history_slots": self.partial_history_slots,
            "history_tokens": self.history_tokens,
        }

    @classmethod
    def from_wire(cls, value: dict[str, Any]) -> OscarMLARequestMetadata:
        required = {
            "generation",
            "cache_version",
            "logical_length",
            "hp_row",
            "block_ids",
            "history_pages",
            "partial_history_slots",
            "history_tokens",
        }
        if set(value) != required:
            raise ValueError("invalid OSCAR MLA request metadata fields")
        return cls(
            generation=int(value["generation"]),
            cache_version=int(value["cache_version"]),
            logical_length=int(value["logical_length"]),
            hp_row=int(value["hp_row"]),
            block_ids=tuple(int(block) for block in value["block_ids"]),
            history_pages=tuple(int(page) for page in value["history_pages"]),
            partial_history_slots=int(value["partial_history_slots"]),
            history_tokens=int(value["history_tokens"]),
        )


@dataclass(frozen=True)
class OscarMLAAgentMetadata:
    """Process-wide OSCAR MLA cache ABI advertised during NIXL handshake."""

    protocol_version: int
    layer_names: tuple[str, ...]
    auxiliary_layer_names: tuple[str, ...]
    auxiliary_page_bytes: tuple[int, ...]
    num_blocks: int
    max_num_seqs: int
    block_size: int
    latent_rank: int
    rope_head_size: int
    group_size: int
    prefix_tokens: int
    recent_tokens: int
    speculative_tokens: int
    hp_dtype: str
    artifact_manifest_sha256: str
    artifact_rotations_sha256: str

    def __post_init__(self) -> None:
        if self.protocol_version != OSCAR_MLA_NIXL_PROTOCOL_VERSION:
            raise ValueError(
                "unsupported OSCAR MLA NIXL protocol version: "
                f"{self.protocol_version}"
            )
        positive = (
            self.num_blocks,
            self.max_num_seqs,
            self.block_size,
            self.latent_rank,
            self.rope_head_size,
            self.group_size,
        )
        if not self.layer_names or any(value <= 0 for value in positive):
            raise ValueError("invalid OSCAR MLA transfer geometry")
        if len(self.auxiliary_layer_names) != len(self.auxiliary_page_bytes):
            raise ValueError("invalid OSCAR MLA auxiliary cache geometry")
        if any(value <= 0 for value in self.auxiliary_page_bytes):
            raise ValueError("invalid OSCAR MLA auxiliary cache geometry")
        all_layer_names = self.layer_names + self.auxiliary_layer_names
        if len(set(all_layer_names)) != len(all_layer_names):
            raise ValueError("duplicate OSCAR MLA transfer layer name")
        if self.prefix_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("invalid OSCAR MLA transfer geometry")
        if self.speculative_tokens < 0:
            raise ValueError("invalid OSCAR MLA transfer geometry")
        if self.latent_rank % self.group_size:
            raise ValueError("invalid OSCAR MLA transfer geometry")
        if self.hp_dtype != "bfloat16":
            raise ValueError("OSCAR MLA transfer requires BF16 hot pools")
        if not self.artifact_manifest_sha256 or not self.artifact_rotations_sha256:
            raise ValueError("OSCAR MLA transfer requires verified artifact hashes")

    @property
    def recent_capacity_tokens(self) -> int:
        return self.recent_tokens + self.speculative_tokens

    @property
    def geometry_fingerprint(self) -> tuple[object, ...]:
        # Cache capacities are intentionally excluded. Producer and consumer
        # descriptor spaces use their own strides, and request validation
        # independently bounds every block and HP row against the owning
        # agent. Requiring identical capacities would reject otherwise
        # compatible engines when CUDA Graph capture leaves a few blocks of
        # machine-dependent free memory.
        return (
            self.protocol_version,
            self.layer_names,
            self.auxiliary_layer_names,
            self.auxiliary_page_bytes,
            self.block_size,
            self.latent_rank,
            self.rope_head_size,
            self.group_size,
            self.prefix_tokens,
            self.recent_tokens,
            self.speculative_tokens,
            self.hp_dtype,
        )

    @property
    def artifact_fingerprint(self) -> tuple[str, str]:
        return (
            self.artifact_manifest_sha256,
            self.artifact_rotations_sha256,
        )

    @property
    def descriptors_per_layer(self) -> int:
        return 4 * self.num_blocks + 2 * self.max_num_seqs

    def transfer_bytes(self, request: OscarMLARequestMetadata) -> int:
        return sum(self.transfer_byte_breakdown(request).values())

    def transfer_byte_breakdown(
        self, request: OscarMLARequestMetadata
    ) -> dict[str, int]:
        """Return the exact bytes posted to NIXL for each OSCAR pool."""
        validate_oscar_mla_request(self, request)
        packed_bytes = self.latent_rank * 2 // 8
        groups = self.latent_rank // self.group_size
        layers = len(self.layer_names)
        history_pages = len(request.history_pages)
        blocks = len(request.block_ids)
        return {
            "history_data": (
                layers * history_pages * self.block_size * packed_bytes
            ),
            "history_scale": (
                layers * history_pages * self.block_size * groups * 4
            ),
            "history_zero": (
                layers * history_pages * self.block_size * groups * 4
            ),
            "rope": (
                layers * blocks * self.block_size * self.rope_head_size * 2
            ),
            "prefix": layers * self.prefix_tokens * self.latent_rank * 2,
            "recent": (
                layers * self.recent_capacity_tokens * self.latent_rank * 2
            ),
            "auxiliary": blocks * sum(self.auxiliary_page_bytes),
        }

    def bf16_reference_bytes(self, request: OscarMLARequestMetadata) -> int:
        validate_oscar_mla_request(self, request)
        return (
            len(self.layer_names)
            * request.logical_length
            * (self.latent_rank + self.rope_head_size)
            * 2
            + len(request.block_ids) * sum(self.auxiliary_page_bytes)
        )


def validate_oscar_mla_request(
    agent: OscarMLAAgentMetadata,
    request: OscarMLARequestMetadata,
    *,
    expected_generation: int | None = None,
) -> None:
    """Validate ownership before a request is allowed to become visible."""
    if request.generation <= 0 or (
        expected_generation is not None
        and request.generation != expected_generation
    ):
        raise ValueError("OSCAR MLA request generation mismatch")
    if request.cache_version < 0 or request.logical_length < 0:
        raise ValueError("invalid OSCAR MLA request version or length")
    if not 0 <= request.hp_row < agent.max_num_seqs:
        raise ValueError("OSCAR MLA hp row is outside cache geometry")
    expected_blocks = (
        request.logical_length + agent.block_size - 1
    ) // agent.block_size
    if len(request.block_ids) != expected_blocks:
        raise ValueError("OSCAR MLA block count does not match geometry")
    if len(set(request.block_ids)) != len(request.block_ids) or any(
        block < 0 or block >= agent.num_blocks for block in request.block_ids
    ):
        raise ValueError("invalid OSCAR MLA block ownership")

    expected_history = max(
        0,
        request.logical_length - agent.prefix_tokens - agent.recent_tokens,
    )
    if request.history_tokens != expected_history:
        raise ValueError("OSCAR MLA history token count does not match geometry")
    expected_pages = (
        expected_history + agent.block_size - 1
    ) // agent.block_size
    if len(request.history_pages) != expected_pages:
        raise ValueError("OSCAR MLA history page count does not match geometry")
    if len(set(request.history_pages)) != len(request.history_pages) or any(
        page < 0 or page >= agent.num_blocks for page in request.history_pages
    ):
        raise ValueError("invalid OSCAR MLA history page ownership")
    if request.partial_history_slots != expected_history % agent.block_size:
        raise ValueError("OSCAR MLA partial history slots do not match geometry")
    prefix_blocks = agent.prefix_tokens // agent.block_size
    if request.history_pages != request.block_ids[
        prefix_blocks : prefix_blocks + expected_pages
    ]:
        raise ValueError("OSCAR MLA history pages do not match block ownership")


def project_oscar_mla_request_prefix(
    agent: OscarMLAAgentMetadata,
    request: OscarMLARequestMetadata,
    logical_length: int,
) -> OscarMLARequestMetadata:
    """Project producer ownership to the prefix consumed by Decode."""
    request = request.trim_speculative_tail_blocks(agent.block_size)
    validate_oscar_mla_request(agent, request)
    if not 0 <= logical_length <= request.logical_length:
        raise ValueError("invalid OSCAR MLA projected logical length")

    block_count = (logical_length + agent.block_size - 1) // agent.block_size
    block_ids = request.block_ids[:block_count]
    history_tokens = max(
        0,
        logical_length - agent.prefix_tokens - agent.recent_tokens,
    )
    history_page_count = (
        history_tokens + agent.block_size - 1
    ) // agent.block_size
    prefix_blocks = agent.prefix_tokens // agent.block_size
    projected = OscarMLARequestMetadata(
        generation=request.generation,
        cache_version=request.cache_version,
        logical_length=logical_length,
        hp_row=request.hp_row,
        block_ids=block_ids,
        history_pages=block_ids[
            prefix_blocks : prefix_blocks + history_page_count
        ],
        partial_history_slots=history_tokens % agent.block_size,
        history_tokens=history_tokens,
    )
    validate_oscar_mla_request(agent, projected)
    return projected


def _request_descriptor_ids(
    agent: OscarMLAAgentMetadata,
    request: OscarMLARequestMetadata,
) -> tuple[int, ...]:
    validate_oscar_mla_request(agent, request)
    result: list[int] = []
    layer_stride = agent.descriptors_per_layer
    for layer_idx in range(len(agent.layer_names)):
        layer_base = layer_idx * layer_stride
        for region_idx in range(3):
            region_base = layer_base + region_idx * agent.num_blocks
            result.extend(region_base + page for page in request.history_pages)
        rope_base = layer_base + 3 * agent.num_blocks
        result.extend(rope_base + block for block in request.block_ids)
        prefix_base = layer_base + 4 * agent.num_blocks
        recent_base = prefix_base + agent.max_num_seqs
        result.append(prefix_base + request.hp_row)
        result.append(recent_base + request.hp_row)
    auxiliary_base = len(agent.layer_names) * layer_stride
    for layer_idx in range(len(agent.auxiliary_layer_names)):
        layer_base = auxiliary_base + layer_idx * agent.num_blocks
        result.extend(layer_base + block for block in request.block_ids)
    return tuple(result)


def build_oscar_mla_descriptor_pairs(
    local_agent: OscarMLAAgentMetadata,
    local_request: OscarMLARequestMetadata,
    remote_agent: OscarMLAAgentMetadata,
    remote_request: OscarMLARequestMetadata,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build matching local/remote descriptor IDs with local page remapping."""
    if local_agent.geometry_fingerprint != remote_agent.geometry_fingerprint:
        raise ValueError("OSCAR MLA transfer geometry mismatch")
    if local_agent.artifact_fingerprint != remote_agent.artifact_fingerprint:
        raise ValueError("OSCAR MLA artifact mismatch")
    if local_request.logical_length != remote_request.logical_length:
        raise ValueError("OSCAR MLA request logical length mismatch")
    if local_request.history_tokens != remote_request.history_tokens:
        raise ValueError("OSCAR MLA request history partition mismatch")
    if len(local_request.history_pages) != len(remote_request.history_pages):
        raise ValueError("OSCAR MLA request history page mapping mismatch")

    local_ids = _request_descriptor_ids(local_agent, local_request)
    remote_ids = _request_descriptor_ids(remote_agent, remote_request)
    if len(local_ids) != len(remote_ids):
        raise ValueError("OSCAR MLA descriptor count mismatch")
    return local_ids, remote_ids
