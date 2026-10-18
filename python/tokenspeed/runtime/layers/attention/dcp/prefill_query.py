# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Host plans for backend-local Prefill query sharding.

No device reads, collectives or model-wide query layout changes live here.
"""

from dataclasses import dataclass

import numpy as np

ASCEND_DSA_PREFILL_Q8 = "ascend_dsa_prefill_q8"


@dataclass(frozen=True)
class PrefillRequestShard:
    request: int
    query_start: int
    query_length: int
    prefix_length: int
    local_start: int
    local_end: int

    @property
    def local_length(self) -> int:
        return self.local_end - self.local_start

    @property
    def history_length(self) -> int:
        return self.prefix_length + self.query_length

    @property
    def visible_key_length(self) -> int:
        return self.prefix_length + self.local_end


@dataclass(frozen=True)
class PrefillQueryPlan:
    total_queries: int
    degree: int
    rank: int
    padded_queries_per_rank: int
    local_start: int
    local_end: int
    requests: tuple[PrefillRequestShard, ...]


def plan_prefill_queries(
    query_lengths: tuple[int, ...],
    prefix_lengths: tuple[int, ...],
    *,
    degree: int,
    rank: int,
) -> PrefillQueryPlan:
    if degree <= 0 or not 0 <= rank < degree:
        raise ValueError("invalid Prefill query shard rank/degree")
    if len(query_lengths) != len(prefix_lengths):
        raise ValueError("Prefill query and prefix request counts differ")
    if any(q <= 0 for q in query_lengths) or any(p < 0 for p in prefix_lengths):
        raise ValueError("Prefill requests require positive Q and nonnegative prefix")
    total = sum(query_lengths)
    width = (total + degree - 1) // degree
    first, last = min(rank * width, total), min((rank + 1) * width, total)
    requests = []
    start = 0
    for index, (q, prefix) in enumerate(zip(query_lengths, prefix_lengths)):
        local_a = min(max(first - start, 0), q)
        local_b = min(max(last - start, 0), q)
        requests.append(PrefillRequestShard(index, start, q, prefix, local_a, local_b))
        start += q
    return PrefillQueryPlan(total, degree, rank, width, first, last, tuple(requests))


@dataclass(frozen=True)
class HistoryPageSegment:
    logical_start: int
    rows: int
    owner: int
    local_start: int
    virtual_start: int


@dataclass(frozen=True)
class PrefillHistoryPlan:
    rows: int
    segments: tuple[HistoryPageSegment, ...]
    owned_rows: tuple[int, ...]


def plan_prefill_history(
    pages: tuple[int, ...],
    *,
    rows: int,
    page_size: int,
    block_granularity: int,
    virtual_block_count: int,
    degree: int,
) -> PrefillHistoryPlan:
    """Build stable owner order from host pages, without a device Sort."""
    if rows <= 0 or page_size <= 0 or degree <= 0:
        raise ValueError("history requires positive rows/page/degree")
    if block_granularity <= 0 or block_granularity % page_size:
        raise ValueError("cache blocks must contain whole kernel pages")
    if virtual_block_count <= 1:
        raise ValueError("history cache must include usable blocks after null block")
    if len(pages) * page_size < rows:
        raise ValueError("history page table does not cover its rows")
    segments = []
    counts = [0] * degree
    for column in range((rows + page_size - 1) // page_size):
        slot = pages[column] * page_size
        block, offset = divmod(slot, block_granularity)
        if not 0 < block < virtual_block_count:
            raise ValueError("history has a hole or out-of-range virtual page")
        owner = (block - 1) % degree
        length = min(page_size, rows - column * page_size)
        local = ((block - 1) // degree + 1) * block_granularity + offset
        segments.append(
            HistoryPageSegment(column * page_size, length, owner, local, slot)
        )
        counts[owner] += length
    return PrefillHistoryPlan(rows, tuple(segments), tuple(counts))


def history_owner_vectors(plan: PrefillHistoryPlan, *, rank: int):
    """Expand the current host page plan into contiguous INT64 CPU arrays.

    Each owner keeps the original logical page order. Arrays are rebuilt from
    this forward's plan, including the exact final partial page; no cache of
    mutable request-slot/page-table metadata is retained.
    """
    if not 0 <= rank < len(plan.owned_rows):
        raise ValueError("history rank is outside owner group")
    segments = np.asarray(
        [(s.logical_start, s.rows, s.owner, s.local_start, s.virtual_start)
         for s in plan.segments],
        dtype=np.int64,
    ).reshape(-1, 5)

    def expand(selected: np.ndarray, column: int) -> np.ndarray:
        lengths = selected[:, 1]
        ends = np.cumsum(lengths, dtype=np.int64)
        offsets = ends - lengths
        rows = int(ends[-1]) if len(ends) else 0
        return np.repeat(selected[:, column] - offsets, lengths) + np.arange(
            rows, dtype=np.int64
        )

    virtual = expand(segments, 4)
    local = expand(segments, 3)
    local[np.repeat(segments[:, 2] != rank, segments[:, 1])] = 0
    order = np.empty(plan.rows, dtype=np.int64)
    first = 0
    for owner, rows in enumerate(plan.owned_rows):
        selected = segments[segments[:, 2] == owner]
        order[first : first + rows] = expand(selected, 0)
        first += rows
    fetch = expand(segments[segments[:, 2] == rank], 3)
    return virtual, local, order, fetch
