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

"""Tensor movement for backend-local Prefill; no persistent cache ownership."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch

from tokenspeed.runtime.layers.attention.dcp.prefill_query import PrefillQueryPlan

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.dcp.cache import HistoryGatherPlan


class PrefillCollectives(Protocol):
    def exchange(self, output: torch.Tensor, source: torch.Tensor) -> None: ...

    def gather(self, output: torch.Tensor, source: torch.Tensor) -> None: ...

    def broadcast(self, tensor: torch.Tensor, *, owner: int) -> None: ...


def head_to_query(
    query: torch.Tensor, plan: PrefillQueryPlan, *, collectives: PrefillCollectives
) -> torch.Tensor:
    """[T, H/P, D] -> [ceil(T/P), H, D], including padded rows."""
    if query.ndim != 3 or query.shape[0] != plan.total_queries:
        raise ValueError("head-sharded query does not match its Prefill plan")
    width = plan.padded_queries_per_rank
    source = query.new_zeros(plan.degree * width, *query.shape[1:])
    source[: plan.total_queries].copy_(query)
    received = torch.empty_like(source)
    collectives.exchange(received, source)
    return (
        received.view(plan.degree, width, *query.shape[1:])
        .permute(1, 0, 2, 3)
        .reshape(width, plan.degree * query.shape[1], query.shape[2])
        .contiguous()
    )


def query_to_head(
    output: torch.Tensor, plan: PrefillQueryPlan, *, collectives: PrefillCollectives
) -> torch.Tensor:
    """[ceil(T/P), H, D] -> [H/P, T, D], dropping only padded rows."""
    if (
        output.ndim != 3
        or output.shape[0] != plan.padded_queries_per_rank
        or output.shape[1] % plan.degree
    ):
        raise ValueError("token-sharded output does not match its Prefill plan")
    heads = output.shape[1] // plan.degree
    source = (
        output.view(plan.padded_queries_per_rank, plan.degree, heads, output.shape[2])
        .permute(1, 0, 2, 3)
        .contiguous()
    )
    received = torch.empty_like(source)
    collectives.exchange(received, source)
    return (
        received.view(-1, heads, output.shape[2])[: plan.total_queries]
        .transpose(0, 1)
        .contiguous()
    )


@dataclass(frozen=True)
class PrefillHistoryTensors:
    plan: "HistoryGatherPlan"
    fetch_slots: torch.Tensor


class PrefillHistoryWorkspace:
    """One bounded request history, collective staging and split SFA keys.

    No request-slot state is retained. All ranks visit all request histories,
    including ranks with zero local queries. Uneven/fragmented ownership can
    use exact-count broadcasts when padded all-gather would exceed staging.
    """

    def __init__(
        self,
        *,
        rows: int,
        degree: int,
        page_size: int,
        kv_dim: int,
        rope_dim: int,
        max_requests: int,
        device: torch.device,
    ):
        if min(rows, degree, page_size, kv_dim, rope_dim, max_requests) <= 0:
            raise ValueError("Prefill history workspace dimensions must be positive")
        if rope_dim >= kv_dim:
            raise ValueError("invalid Prefill key dimensions")
        self.rows = rows
        self.degree = degree
        self.kv = torch.empty((rows, kv_dim), dtype=torch.bfloat16, device=device)
        self.staging = torch.empty(
            ((rows + degree * page_size) * kv_dim,), dtype=torch.bfloat16, device=device
        )
        self.key = torch.empty(
            (rows, 1, kv_dim - rope_dim), dtype=torch.bfloat16, device=device
        )
        self.rope = torch.empty(
            (rows, 1, rope_dim), dtype=torch.bfloat16, device=device
        )
        # Common HistoryGatherPlan vectors plus local fetch slots. Preallocate
        # their worst-case size too; there is no unbudgeted per-request NPU plan.
        self.virtual_slots = torch.empty(
            (max_requests, rows), dtype=torch.int64, device=device
        )
        self.local_slots = torch.empty_like(self.virtual_slots)
        self.order = torch.empty_like(self.virtual_slots)
        self.fetch_slots = torch.empty_like(self.virtual_slots)
        self.q_ends = torch.empty((max_requests, 1), dtype=torch.int32, device=device)
        self.kv_ends = torch.empty_like(self.q_ends)

    @property
    def nbytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for t in (
                self.kv,
                self.staging,
                self.key,
                self.rope,
                self.virtual_slots,
                self.local_slots,
                self.order,
                self.fetch_slots,
                self.q_ends,
                self.kv_ends,
            )
        )

    def gather_history(
        self,
        source: torch.Tensor,
        history: PrefillHistoryTensors,
        *,
        collectives: PrefillCollectives,
    ) -> torch.Tensor:
        """Read packed physical owner rows, then restore request token order."""
        plan = history.plan
        target = self.kv
        if source.dtype != torch.bfloat16 or not source.is_contiguous():
            raise ValueError(
                "Prefill history requires contiguous BF16 packed cache rows"
            )
        if source.ndim != 2 or source.shape[1] != target.shape[1]:
            raise ValueError("Prefill cache row width does not match the workspace")
        if plan.rows > self.rows or len(plan.group) != self.degree:
            raise ValueError("Prefill history exceeds its reserved workspace/topology")
        counts = plan.owned_rows_per_rank
        local_count = counts[plan.rank]
        if history.fetch_slots.numel() != local_count:
            raise ValueError("Prefill local cache slots do not match owner row count")
        torch.index_select(source, 0, history.fetch_slots, out=target[:local_count])
        width = target.shape[1]
        max_count = max(counts)
        padded_rows = max_count * self.degree
        stage_capacity = self.staging.numel() // width
        if padded_rows <= stage_capacity and max_count <= self.rows:
            # The common nearly balanced page distribution uses one HCCL AG.
            target[local_count:max_count].zero_()
            stage = self.staging[: padded_rows * width].view(padded_rows, width)
            collectives.gather(stage, target[:max_count])
            first = 0
            for owner, count in enumerate(counts):
                target.index_copy_(
                    0,
                    plan.order[first : first + count],
                    stage[owner * max_count : owner * max_count + count],
                )
                first += count
        else:
            # Arbitrarily fragmented pages can all belong to one owner. Do
            # not allocate P * history bytes just to pad an all-gather.
            stage = self.staging[: plan.rows * width].view(plan.rows, width)
            first = 0
            for owner, count in enumerate(counts):
                if count:
                    segment = stage[first : first + count]
                    if plan.rank == owner:
                        segment.copy_(target[:count])
                    collectives.broadcast(segment, owner=owner)
                first += count
            target.index_copy_(0, plan.order, stage)
        return target[: plan.rows]

    def split_keys(self, packed: torch.Tensor, *, visible_rows: int):
        if not 0 < visible_rows <= packed.shape[0]:
            raise ValueError("Prefill visible key endpoint is outside gathered history")
        nope_dim = self.key.shape[-1]
        key = self.key[:visible_rows]
        rope = self.rope[:visible_rows]
        key[:, 0].copy_(packed[:visible_rows, :nope_dim])
        rope[:, 0].copy_(packed[:visible_rows, nope_dim:])
        return key, rope
