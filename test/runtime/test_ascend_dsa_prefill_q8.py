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

"""Host behavior; the separate real eight-card gate validates native kernels."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from tokenspeed.runtime.layers.attention.dcp.ascend_selection import (
    PrefillKVPSelectionWorkspace,
    kvp_rank_major_topk,
)
from tokenspeed.runtime.layers.attention.dcp.cache import HistoryGatherPlan
from tokenspeed.runtime.layers.attention.dcp.prefill_query import (
    history_owner_vectors,
    plan_prefill_history,
    plan_prefill_queries,
)
from tokenspeed.runtime.layers.attention.dcp.prefill_query_ops import (
    PrefillHistoryTensors,
    PrefillHistoryWorkspace,
    head_to_query,
    query_to_head,
)


class Network:
    """In-memory replacement only for unavailable eight-rank communication."""

    def __init__(self, degree):
        self.degree = degree
        self.barrier = Barrier(degree, timeout=20)
        self.entries = {}

    def run(self, functions):
        def invoke(rank):
            try:
                return functions(rank, Endpoint(self, rank))
            except BaseException:
                self.barrier.abort()
                raise

        with ThreadPoolExecutor(max_workers=self.degree) as pool:
            return list(pool.map(invoke, range(self.degree)))


class Endpoint:
    def __init__(self, network, rank):
        self.network, self.rank = network, rank

    def _call(self, kind, output, source, owner):
        net = self.network
        net.entries[self.rank] = (kind, output, source.clone(), owner)
        net.barrier.wait()
        if self.rank == 0:
            entries = [net.entries[i] for i in range(net.degree)]
            if any(e[0] != kind or e[3] != owner for e in entries):
                raise RuntimeError("ranks entered different collectives")
            if kind == "exchange":
                for rank, (_, target, _, _) in enumerate(entries):
                    chunks = [
                        entry[2].reshape(net.degree, -1)[rank] for entry in entries
                    ]
                    target.copy_(torch.cat(chunks).reshape_as(target))
            elif kind == "gather":
                value = torch.cat([entry[2] for entry in entries])
                for _, target, _, _ in entries:
                    target.copy_(value)
            else:
                for _, target, _, _ in entries:
                    target.copy_(entries[owner][2])
        net.barrier.wait()
        net.barrier.wait()
        if self.rank == 0:
            net.entries.clear()
        net.barrier.wait()

    def exchange(self, output, source):
        self._call("exchange", output, source, None)

    def gather(self, output, source):
        self._call("gather", output, source, None)

    def broadcast(self, tensor, *, owner):
        self._call("broadcast", tensor, tensor, owner)


@pytest.mark.parametrize(
    "lengths,prefixes", [((3, 4), (129, 7)), ((9, 13), (0, 65)), ((8192,), (117964,))]
)
def test_query_cover_and_causal_endpoints(lengths, prefixes):
    covered = [[] for _ in lengths]
    for rank in range(8):
        plan = plan_prefill_queries(lengths, prefixes, degree=8, rank=rank)
        for shard in plan.requests:
            covered[shard.request].extend(range(shard.local_start, shard.local_end))
            if shard.local_length:
                # The native right-aligned causal row j is KV-Q+j+1.
                assert (
                    shard.visible_key_length - shard.local_length + 1
                    == shard.prefix_length + shard.local_start + 1
                )
                assert shard.visible_key_length == shard.prefix_length + shard.local_end
    assert covered == [list(range(length)) for length in lengths]


@pytest.mark.parametrize("lengths", [(3, 4), (9, 13), (17,)])
def test_head_query_roundtrip_and_exact_padding(lengths):
    degree, heads, dim = 8, 2, 4
    total = sum(lengths)
    whole = (
        torch.arange(total * degree * heads * dim)
        .reshape(total, degree * heads, dim)
        .float()
    )
    net = Network(degree)

    def execute(rank, endpoint):
        plan = plan_prefill_queries(
            lengths, (0,) * len(lengths), degree=degree, rank=rank
        )
        local = whole[:, rank * heads : (rank + 1) * heads].contiguous()
        sharded = head_to_query(local, plan, collectives=endpoint)
        count = plan.local_end - plan.local_start
        torch.testing.assert_close(
            sharded[:count], whole[plan.local_start : plan.local_end]
        )
        assert torch.count_nonzero(sharded[count:]) == 0
        return query_to_head(sharded * 2 + 1, plan, collectives=endpoint)

    outputs = net.run(execute)
    for rank, output in enumerate(outputs):
        torch.testing.assert_close(
            output,
            (whole[:, rank * heads : (rank + 1) * heads] * 2 + 1).transpose(0, 1),
        )


@pytest.mark.parametrize("fragmented", [False, True])
def test_owner_history_restoration_with_empty_query_owner(fragmented):
    degree, page, block, rows = 8, 128, 256, 1027
    # Fragmented histories can all live on rank7 despite its empty Q shard.
    pages = (
        tuple((1 + i * degree) * 2 for i in range(9))
        if fragmented
        else (2, 17, 4, 19, 6, 21, 8, 23, 10)
    )
    if fragmented:
        pages = tuple((8 + i * degree) * 2 for i in range(9))
    host = plan_prefill_history(
        pages,
        rows=rows,
        page_size=page,
        block_granularity=block,
        virtual_block_count=80,
        degree=degree,
    )
    cap = 12 * block
    caches, expected = [], torch.empty((rows, 576), dtype=torch.bfloat16)
    for rank in range(degree):
        cache = torch.zeros((cap, 576), dtype=torch.bfloat16)
        cache[:, 0] = rank
        cache[:, 1] = torch.arange(cap) % 128
        cache[:, 2] = torch.arange(cap) // 128
        cache[:, 512:] = rank + 0.5
        caches.append(cache)
    for seg in host.segments:
        expected[seg.logical_start : seg.logical_start + seg.rows] = caches[seg.owner][
            seg.local_start : seg.local_start + seg.rows
        ]

    def execute(rank, endpoint):
        ws = PrefillHistoryWorkspace(
            rows=1152,
            degree=degree,
            page_size=page,
            kv_dim=576,
            rope_dim=64,
            max_requests=2,
            device=torch.device("cpu"),
        )
        virtual, local, order, fetch = history_owner_vectors(host, rank=rank)
        plan = HistoryGatherPlan(
            torch.tensor(virtual),
            torch.tensor(local),
            torch.tensor(order),
            host.owned_rows,
            tuple(range(degree)),
            rank,
        )
        history = PrefillHistoryTensors(plan, torch.tensor(fetch, dtype=torch.int64))
        query = plan_prefill_queries((7,), (rows - 7,), degree=degree, rank=rank)
        if rank == 7:
            assert query.local_start == query.local_end
            if fragmented:
                assert history.fetch_slots.numel() == rows
        gathered = ws.gather_history(caches[rank], history, collectives=endpoint)
        torch.testing.assert_close(gathered, expected)
        key, rope = ws.split_keys(gathered, visible_rows=1001)
        assert key.is_contiguous() and rope.is_contiguous()
        torch.testing.assert_close(key[:, 0], expected[:1001, :512])
        torch.testing.assert_close(rope[:, 0], expected[:1001, 512:])
        # Fixed workspace includes all BF16 buffers and four INT64 plans.
        assert ws.nbytes == 2 * (1152 * (2 * 576) + (1152 + 8 * 128) * 576) + 2 * (
            1152 * 4 * 8 + 2 * 4
        )

    Network(degree).run(execute)


def test_history_holes_and_bounds_rejected():
    for pages, rows in [((0,), 1), ((2,), 129), ((200,), 1)]:
        with pytest.raises(ValueError):
            plan_prefill_history(
                pages,
                rows=rows,
                page_size=128,
                block_granularity=256,
                virtual_block_count=80,
                degree=8,
            )


def test_empty_batch_plan_has_no_queries():
    plan = plan_prefill_queries((), (), degree=8, rank=7)
    assert plan.total_queries == plan.padded_queries_per_rank == 0
    assert not plan.requests


@pytest.mark.parametrize(
    "lengths,prefixes,fragmented",
    [
        ((5, 8), (261, 1), False),
        ((3,), (258,), True),
        ((17,), (129,), False),
    ],
)
def test_kvp_bf16_selection_retains_owner_order_and_request_ids(
    lengths, prefixes, fragmented
):
    degree, page, block, topk = 8, 4, 8, 4
    hosts = []
    for qlen, prefix in zip(lengths, prefixes):
        count = (qlen + prefix + page - 1) // page
        pages = tuple(
            2 * (8 + j * degree) + j % 2
            if fragmented
            else 2 * (1 + (j * 5) % 127) + j % 2
            for j in range(count)
        )
        hosts.append(
            plan_prefill_history(
                pages,
                rows=qlen + prefix,
                page_size=page,
                block_granularity=block,
                virtual_block_count=1024,
                degree=degree,
            )
        )
    total = sum(lengths)
    ids, values, logical_maps = [], [], []
    for rank in range(degree):
        local_ids = torch.full((total, 1, topk), -1, dtype=torch.int32)
        local_values = torch.full((total, 1, topk), -float("inf"), dtype=torch.bfloat16)
        owner_maps = []
        first = 0
        for length, prefix, host in zip(lengths, prefixes, hosts):
            logical = [
                i
                for segment in host.segments
                if segment.owner == rank
                for i in range(
                    segment.logical_start, segment.logical_start + segment.rows
                )
            ]
            owner_maps.append(logical)
            for row in range(length):
                available = [
                    i for i, token in enumerate(logical) if token < prefix + row + 1
                ]
                if available:
                    # BF16 ties cross owners and local candidate boundaries.
                    scores = torch.tensor(
                        [1 + (logical[i] % 7) * 0.001 for i in available]
                    ).bfloat16()
                    chosen_values, positions = scores.topk(min(topk, len(available)))
                    chosen = torch.tensor(available, dtype=torch.int32)[positions]
                    local_ids[first + row, 0, : len(chosen)] = chosen
                    local_values[first + row, 0, : len(chosen)] = chosen_values
            first += length
        ids.append(local_ids)
        values.append(local_values)
        logical_maps.append(owner_maps)

    # Reference is the frozen baseline's rank-major BF16 global topk, before
    # mapping positions back through each actual owner/request candidate list.
    expected_values = (
        torch.stack([v[:, 0] for v in values])
        .transpose(0, 1)
        .contiguous()
        .view(total, degree * topk)
    )
    expected_positions = expected_values.topk(topk, dim=1).indices
    expected_ids = torch.full((total, topk), -1, dtype=torch.int32)
    first = 0
    for request, length in enumerate(lengths):
        for row in range(length):
            for column, position in enumerate(expected_positions[first + row].tolist()):
                owner, candidate = divmod(position, topk)
                compact = int(ids[owner][first + row, 0, candidate])
                if compact >= 0:
                    expected_ids[first + row, column] = logical_maps[owner][request][
                        compact
                    ]
        first += length

    def execute(rank, endpoint):
        plan = plan_prefill_queries(lengths, prefixes, degree=degree, rank=rank)
        histories = []
        for host in hosts:
            virtual, local, order, fetch = history_owner_vectors(host, rank=rank)
            history = HistoryGatherPlan(
                torch.tensor(virtual),
                torch.tensor(local),
                torch.tensor(order),
                host.owned_rows,
                tuple(range(degree)),
                rank,
            )
            histories.append(PrefillHistoryTensors(history, torch.tensor(fetch)))
        workspace = PrefillKVPSelectionWorkspace(
            max_queries=total + 3,
            degree=degree,
            topk=topk,
            history_rows=max(q + p for q, p in zip(lengths, prefixes)),
            device="cpu",
        )
        actual = workspace.select(
            ids[rank], values[rank], plan, tuple(histories), collectives=endpoint
        )
        count = plan.local_end - plan.local_start
        torch.testing.assert_close(
            actual[:count, 0],
            expected_ids[plan.local_start : plan.local_end],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            workspace.positions[:count],
            expected_positions[plan.local_start : plan.local_end],
            rtol=0,
            atol=0,
        )
        for request in plan.requests:
            for query_row in range(request.local_start, request.local_end):
                local = request.query_start + query_row - plan.local_start
                live = actual[local, 0]
                assert bool(
                    torch.all(
                        (live < request.prefix_length + query_row + 1) | (live == -1)
                    )
                )
        if total < degree and rank == degree - 1:
            assert count == 0 and histories[0].plan.owned_rows_per_rank[rank] > 0
        return actual.clone()

    Network(degree).run(execute)


def test_kvp_topk_workspace_matches_original_bf16_ties():
    values = torch.tensor(
        [
            [1.0, 1.001, 0.999, -float("inf")],
            [1.0, 1.001, 0.999, -float("inf")],
        ],
        dtype=torch.bfloat16,
    )
    reference = (
        values.view(2, 1, 4).transpose(0, 1).contiguous().view(1, 8).topk(4).indices
    )
    actual = kvp_rank_major_topk(
        values,
        degree=2,
        queries_per_rank=1,
        candidates=4,
        topk=4,
        ordered_values=torch.empty((1, 8), dtype=torch.bfloat16),
        topk_output=(
            torch.empty((1, 4), dtype=torch.bfloat16),
            torch.empty((1, 4), dtype=torch.int64),
        ),
    )
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
