# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Capacity planning and page ownership for OSCAR mixed KV caches."""

from __future__ import annotations

from dataclasses import dataclass, field

from vllm.utils.math_utils import cdiv


class OscarKVCacheCapacityError(RuntimeError):
    """Raised when an OSCAR tier cannot satisfy an allocation."""


@dataclass(frozen=True)
class OscarKVCacheGeometry:
    num_layers: int
    num_kv_heads: int
    head_size: int
    value_head_size: int
    quant_slot_size: int
    group_size: int = 128
    block_size: int = 16
    prefix_tokens: int = 64
    recent_tokens: int = 256
    flush_interval: int = 8
    key_bits: int = 2
    value_bits: int = 2
    hp_element_size: int = 2

    def __post_init__(self) -> None:
        positive = {
            "num_layers": self.num_layers,
            "num_kv_heads": self.num_kv_heads,
            "head_size": self.head_size,
            "value_head_size": self.value_head_size,
            "quant_slot_size": self.quant_slot_size,
            "group_size": self.group_size,
            "block_size": self.block_size,
            "hp_element_size": self.hp_element_size,
            "flush_interval": self.flush_interval,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"OSCAR geometry values must be positive: {invalid}")
        if self.key_bits != 2 or self.value_bits != 2:
            raise ValueError("OSCAR pool geometry currently supports INT2 K/V only")
        if self.prefix_tokens < 0 or self.recent_tokens < 0:
            raise ValueError("OSCAR prefix and recent windows must be non-negative")
        if self.prefix_tokens % self.block_size != 0:
            raise ValueError("OSCAR prefix window must be block aligned")
        if self.recent_tokens % self.block_size != 0:
            raise ValueError("OSCAR recent window must be block aligned")

    @property
    def quant_token_bytes(self) -> int:
        return self.num_layers * self.num_kv_heads * self.quant_slot_size

    @property
    def hp_token_bytes(self) -> int:
        per_head = (self.head_size + self.value_head_size) * self.hp_element_size
        return self.num_layers * self.num_kv_heads * per_head

    @property
    def quant_page_bytes(self) -> int:
        return self.block_size * self.quant_token_bytes

    @property
    def hp_page_bytes(self) -> int:
        return self.block_size * self.hp_token_bytes

    @property
    def recent_row_capacity(self) -> int:
        """Physical BF16 row capacity; logical recent remains unchanged."""
        return cdiv(
            self.recent_tokens + self.flush_interval - 1, self.block_size
        ) * self.block_size


@dataclass(frozen=True)
class OscarKVCachePlan:
    geometry: OscarKVCacheGeometry
    total_memory_bytes: int
    max_num_seqs: int
    prefix_cache_extra_tokens: int
    prefix_pages: int
    recent_pages: int
    quant_pages: int
    unused_bytes: int

    @property
    def prefix_slots(self) -> int:
        return self.prefix_pages * self.geometry.block_size

    @property
    def recent_slots(self) -> int:
        return self.recent_pages * self.geometry.block_size

    @property
    def quant_slots(self) -> int:
        return self.quant_pages * self.geometry.block_size

    @property
    def guaranteed_quant_slots(self) -> int:
        fragmentation = self.max_num_seqs * (self.geometry.block_size - 1)
        return max(0, self.quant_slots - fragmentation)

    @property
    def hp_bytes(self) -> int:
        return (self.prefix_slots + self.recent_slots) * self.geometry.hp_token_bytes

    @property
    def quant_bytes(self) -> int:
        return self.quant_slots * self.geometry.quant_token_bytes

    @property
    def allocated_bytes(self) -> int:
        return self.hp_bytes + self.quant_bytes

    @property
    def bf16_slots(self) -> int:
        pages = self.total_memory_bytes // self.geometry.hp_page_bytes
        return pages * self.geometry.block_size

    @property
    def physical_capacity_ratio(self) -> float:
        return self.quant_slots / self.bf16_slots

    @property
    def guaranteed_capacity_ratio(self) -> float:
        return self.guaranteed_quant_slots / self.bf16_slots

    @property
    def allocator_waste_fraction(self) -> float:
        remaining = self.total_memory_bytes - self.hp_bytes
        ideal_quant_slots = remaining // self.geometry.quant_token_bytes
        if ideal_quant_slots == 0:
            return 0.0
        return (ideal_quant_slots - self.guaranteed_quant_slots) / ideal_quant_slots


def plan_oscar_kv_cache(
    geometry: OscarKVCacheGeometry,
    total_memory_bytes: int,
    max_num_seqs: int,
    prefix_cache_extra_tokens: int = 0,
) -> OscarKVCachePlan:
    if total_memory_bytes <= 0:
        raise ValueError("total_memory_bytes must be positive")
    if max_num_seqs <= 0:
        raise ValueError("max_num_seqs must be positive")
    if prefix_cache_extra_tokens < 0:
        raise ValueError("prefix_cache_extra_tokens must be non-negative")

    block_size = geometry.block_size
    extra_prefix_pages = cdiv(prefix_cache_extra_tokens, block_size)
    prefix_pages = (
        max_num_seqs * (geometry.prefix_tokens // block_size) + extra_prefix_pages
    )
    recent_pages = max_num_seqs * (geometry.recent_row_capacity // block_size)
    hp_bytes = (prefix_pages + recent_pages) * geometry.hp_page_bytes
    if hp_bytes >= total_memory_bytes:
        raise OscarKVCacheCapacityError(
            "OSCAR BF16 pools consume the KV budget: "
            f"required={hp_bytes}, budget={total_memory_bytes}"
        )

    quant_pages = (total_memory_bytes - hp_bytes) // geometry.quant_page_bytes
    if quant_pages == 0:
        raise OscarKVCacheCapacityError("OSCAR KV budget has no room for INT2 pages")
    allocated_bytes = hp_bytes + quant_pages * geometry.quant_page_bytes
    return OscarKVCachePlan(
        geometry=geometry,
        total_memory_bytes=total_memory_bytes,
        max_num_seqs=max_num_seqs,
        prefix_cache_extra_tokens=prefix_cache_extra_tokens,
        prefix_pages=prefix_pages,
        recent_pages=recent_pages,
        quant_pages=quant_pages,
        unused_bytes=total_memory_bytes - allocated_bytes,
    )


@dataclass
class _PagePool:
    num_pages: int
    free_pages: list[int] = field(init=False)
    allocated_pages: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        self.free_pages = list(range(self.num_pages - 1, -1, -1))

    def allocate(self, count: int) -> tuple[int, ...]:
        if count < 0:
            raise ValueError("page count must be non-negative")
        if count > len(self.free_pages):
            raise OscarKVCacheCapacityError(
                f"requested {count} pages with {len(self.free_pages)} free"
            )
        pages = tuple(self.free_pages.pop() for _ in range(count))
        self.allocated_pages.update(pages)
        return pages

    def reserve(self, pages: tuple[int, ...]) -> None:
        page_set = set(pages)
        if len(page_set) != len(pages):
            raise RuntimeError("duplicate page in one reserve operation")
        free = set(self.free_pages)
        if not page_set.issubset(free):
            unavailable = sorted(page_set - free)
            raise OscarKVCacheCapacityError(
                f"OSCAR pages are unavailable: {unavailable}"
            )
        self.free_pages = [page for page in self.free_pages if page not in page_set]
        self.allocated_pages.update(page_set)

    def free(self, pages: tuple[int, ...] | list[int]) -> None:
        page_set = set(pages)
        if len(page_set) != len(pages):
            raise RuntimeError("duplicate page in one free operation")
        if not page_set.issubset(self.allocated_pages):
            unknown = sorted(page_set - self.allocated_pages)
            raise RuntimeError(f"free of unallocated OSCAR pages: {unknown}")
        self.allocated_pages.difference_update(page_set)
        self.free_pages.extend(page_set)

    def assert_consistent(self) -> None:
        free = set(self.free_pages)
        if len(free) != len(self.free_pages):
            raise RuntimeError("duplicate OSCAR free page")
        if free & self.allocated_pages:
            raise RuntimeError("OSCAR page is both free and allocated")
        if len(free) + len(self.allocated_pages) != self.num_pages:
            raise RuntimeError("OSCAR page accounting is not conserved")


class OscarHPRowAllocator:
    """Assign stable request rows in the fixed prefix and recent pools."""

    def __init__(self, max_num_seqs: int) -> None:
        if max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        self.max_num_seqs = max_num_seqs
        self.free_rows = list(range(max_num_seqs - 1, -1, -1))
        self.request_rows: dict[str, int] = {}

    def allocate(self, request_id: str) -> int:
        if request_id in self.request_rows:
            raise RuntimeError(f"duplicate OSCAR request: {request_id}")
        if not self.free_rows:
            raise OscarKVCacheCapacityError("OSCAR BF16 request rows are exhausted")
        row = self.free_rows.pop()
        self.request_rows[request_id] = row
        return row

    def free(self, request_id: str) -> int | None:
        row = self.request_rows.pop(request_id, None)
        if row is not None:
            self.free_rows.append(row)
        return row

    def get(self, request_id: str) -> int:
        return self.request_rows[request_id]

    def assert_consistent(self) -> None:
        free = set(self.free_rows)
        allocated = set(self.request_rows.values())
        if len(free) != len(self.free_rows):
            raise RuntimeError("duplicate OSCAR free HP row")
        if len(allocated) != len(self.request_rows):
            raise RuntimeError("OSCAR HP row is owned by multiple requests")
        if free & allocated:
            raise RuntimeError("OSCAR HP row is both free and allocated")
        if free | allocated != set(range(self.max_num_seqs)):
            raise RuntimeError("OSCAR HP row accounting is not conserved")


@dataclass
class OscarRequestPages:
    hp_row: int
    prefix_pages: tuple[int, ...]
    recent_pages: tuple[int, ...]
    full_quant_pages: list[int] = field(default_factory=list)
    partial_quant_page: int | None = None
    partial_quant_slots: int = 0
    history_tokens: int = 0


class OscarKVPageAllocator:
    """CPU ownership model used by the scheduler-side OSCAR manager."""

    def __init__(self, plan: OscarKVCachePlan) -> None:
        self.plan = plan
        self.prefix_pool = _PagePool(plan.prefix_pages)
        self.recent_pool = _PagePool(plan.recent_pages)
        self.quant_pool = _PagePool(plan.quant_pages)
        self.hp_rows = OscarHPRowAllocator(plan.max_num_seqs)
        self.requests: dict[str, OscarRequestPages] = {}

    def start_request(self, request_id: str) -> OscarRequestPages:
        if request_id in self.requests:
            raise RuntimeError(f"duplicate OSCAR request: {request_id}")
        geometry = self.plan.geometry
        prefix_count = geometry.prefix_tokens // geometry.block_size
        recent_count = geometry.recent_row_capacity // geometry.block_size
        hp_row = self.hp_rows.allocate(request_id)
        prefix_start = hp_row * prefix_count
        recent_start = hp_row * recent_count
        prefix_pages = tuple(range(prefix_start, prefix_start + prefix_count))
        recent_pages = tuple(range(recent_start, recent_start + recent_count))
        prefix_reserved = False
        try:
            self.prefix_pool.reserve(prefix_pages)
            prefix_reserved = True
            self.recent_pool.reserve(recent_pages)
        except Exception:
            if prefix_reserved:
                self.prefix_pool.free(prefix_pages)
            self.hp_rows.free(request_id)
            raise
        pages = OscarRequestPages(hp_row, prefix_pages, recent_pages)
        self.requests[request_id] = pages
        return pages

    def append_history(self, request_id: str, num_tokens: int) -> None:
        if num_tokens < 0:
            raise ValueError("num_tokens must be non-negative")
        if num_tokens == 0:
            return
        request = self.requests[request_id]
        block_size = self.plan.geometry.block_size
        available = (
            block_size - request.partial_quant_slots
            if request.partial_quant_page is not None
            else 0
        )
        remaining_after_partial = max(0, num_tokens - available)
        new_page_count = cdiv(remaining_after_partial, block_size)
        new_pages = self.quant_pool.allocate(new_page_count)

        remaining = num_tokens
        if request.partial_quant_page is not None:
            consumed = min(remaining, available)
            request.partial_quant_slots += consumed
            remaining -= consumed
            if request.partial_quant_slots == block_size:
                request.full_quant_pages.append(request.partial_quant_page)
                request.partial_quant_page = None
                request.partial_quant_slots = 0

        page_index = 0
        while remaining >= block_size:
            request.full_quant_pages.append(new_pages[page_index])
            page_index += 1
            remaining -= block_size
        if remaining:
            request.partial_quant_page = new_pages[page_index]
            request.partial_quant_slots = remaining
        request.history_tokens += num_tokens

    def finish_request(self, request_id: str) -> None:
        request = self.requests.pop(request_id)
        self.prefix_pool.free(request.prefix_pages)
        self.recent_pool.free(request.recent_pages)
        quant_pages = list(request.full_quant_pages)
        if request.partial_quant_page is not None:
            quant_pages.append(request.partial_quant_page)
        self.quant_pool.free(quant_pages)
        released_row = self.hp_rows.free(request_id)
        if released_row != request.hp_row:
            raise RuntimeError("OSCAR request released the wrong HP row")

    def assert_consistent(self) -> None:
        self.prefix_pool.assert_consistent()
        self.recent_pool.assert_consistent()
        self.quant_pool.assert_consistent()
        self.hp_rows.assert_consistent()
        block_size = self.plan.geometry.block_size
        for request_id, request in self.requests.items():
            if self.hp_rows.get(request_id) != request.hp_row:
                raise RuntimeError("OSCAR request HP row ownership mismatch")
            if not 0 <= request.partial_quant_slots < block_size:
                raise RuntimeError("invalid OSCAR partial-page occupancy")
            if (request.partial_quant_page is None) != (
                request.partial_quant_slots == 0
            ):
                raise RuntimeError("inconsistent OSCAR partial-page state")
            expected_pages = cdiv(request.history_tokens, block_size)
            actual_pages = len(request.full_quant_pages) + int(
                request.partial_quant_page is not None
            )
            if expected_pages != actual_pages:
                raise RuntimeError("OSCAR history page count does not match tokens")
