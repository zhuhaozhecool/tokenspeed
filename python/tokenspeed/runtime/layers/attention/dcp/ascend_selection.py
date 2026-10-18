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

"""The existing BF16 KVP top-k order, with request-logical ids for Q8."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.dcp.prefill_query import PrefillQueryPlan
    from tokenspeed.runtime.layers.attention.dcp.prefill_query_ops import (
        PrefillCollectives,
        PrefillHistoryTensors,
    )


def kvp_rank_major_topk(
    values: torch.Tensor,
    *,
    degree: int,
    queries_per_rank: int,
    candidates: int,
    topk: int,
    ordered_values: torch.Tensor | None,
    topk_output: tuple[torch.Tensor, torch.Tensor] | None,
) -> torch.Tensor:
    """Use the original owner-major BF16 candidate order and torch.topk.

    Explicit None outputs retain the baseline's ordinary tensor allocation;
    Q8 supplies its fixed workspace. No dtype conversion or tie perturbation.
    """
    ordered = values.view(degree, queries_per_rank, candidates).transpose(0, 1)
    if ordered_values is None:
        ordered = ordered.contiguous().view(queries_per_rank, degree * candidates)
    else:
        ordered_values.view(queries_per_rank, degree, candidates).copy_(ordered)
        ordered = ordered_values
    if topk_output is None:
        _, positions = ordered.topk(topk, dim=1)
    else:
        _, positions = torch.topk(ordered, topk, dim=1, out=topk_output)
    return positions


class PrefillKVPSelectionWorkspace:
    """Bounded PA/KVP candidate exchange; no full-history index-K gather."""

    def __init__(self, *, max_queries, degree, topk, history_rows, device):
        if min(max_queries, degree, topk, history_rows) <= 0:
            raise ValueError("KVP selection workspace dimensions must be positive")
        if history_rows > torch.iinfo(torch.int32).max:
            raise ValueError("Request-logical KVP token ids must fit INT32")
        self.max_queries = max_queries
        self.degree = degree
        self.topk = topk
        self.width = (max_queries + degree - 1) // degree
        self.rows = self.width * degree
        shape = (self.rows, topk)
        self.send_ids = torch.empty(shape, dtype=torch.int32, device=device)
        self.received_ids = torch.empty_like(self.send_ids)
        self.send_values = torch.empty(shape, dtype=torch.bfloat16, device=device)
        self.received_values = torch.empty_like(self.send_values)
        self.invalid = torch.empty(shape, dtype=torch.bool, device=device)
        self.ordered_ids = torch.empty(
            (self.width, degree * topk), dtype=torch.int32, device=device
        )
        self.ordered_values = torch.empty_like(self.ordered_ids, dtype=torch.bfloat16)
        self.positions = torch.empty(
            (self.width, topk), dtype=torch.int64, device=device
        )
        self.topk_values = torch.empty_like(self.positions, dtype=torch.bfloat16)
        self.selected_ids = torch.empty_like(self.positions, dtype=torch.int32)
        # The existing INT64 plan.order remains the single owner map; this
        # bounded cast scratch avoids a new [requests, context] inverse map.
        self.order_map = torch.empty(history_rows, dtype=torch.int32, device=device)

    @property
    def nbytes(self):
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                self.send_ids,
                self.received_ids,
                self.send_values,
                self.received_values,
                self.invalid,
                self.ordered_ids,
                self.ordered_values,
                self.positions,
                self.topk_values,
                self.selected_ids,
                self.order_map,
            )
        )

    def select(
        self,
        local_indices: torch.Tensor,
        local_values: torch.Tensor,
        plan: "PrefillQueryPlan",
        histories: tuple["PrefillHistoryTensors", ...],
        *,
        collectives: "PrefillCollectives",
    ) -> torch.Tensor:
        """Exchange old PA results and return Q8 request-local INT32 ids.

        local_indices are compact *token* offsets, separately for each
        request. Native PA selection already applied each query's causal
        endpoint and its owner-local forced-window counts.
        """
        tokens, width = plan.total_queries, plan.padded_queries_per_rank
        expected = (tokens, 1, self.topk)
        if (
            tokens > self.max_queries
            or plan.degree != self.degree
            or len(histories) != len(plan.requests)
            or local_indices.shape != expected
            or local_values.shape != expected
            or local_indices.dtype != torch.int32
            or local_values.dtype != torch.bfloat16
        ):
            raise ValueError("PA KVP candidates do not match their Q8 workspace")
        padded = width * self.degree
        send_ids = self.send_ids[:padded]
        send_values = self.send_values[:padded]
        received_ids = self.received_ids[:padded]
        received_values = self.received_values[:padded]
        send_ids.fill_(-1)
        send_values.fill_(float("-inf"))
        send_values[:tokens].copy_(local_values[:, 0])
        for request, history in zip(plan.requests, histories):
            first, last = (
                request.query_start,
                request.query_start + request.query_length,
            )
            owner_count = history.plan.owned_rows_per_rank[plan.rank]
            if owner_count == 0:
                continue
            offset = sum(history.plan.owned_rows_per_rank[: plan.rank])
            self.order_map[:owner_count].copy_(
                history.plan.order[offset : offset + owner_count]
            )
            ids = local_indices[first:last, 0]
            invalid = self.invalid[first:last]
            safe = received_ids[first:last]
            torch.lt(ids, 0, out=invalid)
            torch.clamp(ids, min=0, out=safe)
            torch.index_select(
                self.order_map[:owner_count],
                0,
                safe.flatten(),
                out=send_ids[first:last].flatten(),
            )
            send_ids[first:last].masked_fill_(invalid, -1)
        # Identical call order on every owner, including empty-query ranks.
        collectives.exchange(received_values, send_values)
        collectives.exchange(received_ids, send_ids)
        positions = kvp_rank_major_topk(
            received_values,
            degree=self.degree,
            queries_per_rank=width,
            candidates=self.topk,
            topk=self.topk,
            ordered_values=self.ordered_values[:width],
            topk_output=(self.topk_values[:width], self.positions[:width]),
        )
        self.ordered_ids[:width].view(width, self.degree, self.topk).copy_(
            received_ids.view(self.degree, width, self.topk).transpose(0, 1)
        )
        torch.gather(
            self.ordered_ids[:width], 1, positions, out=self.selected_ids[:width]
        )
        return self.selected_ids[:width].unsqueeze(1)
