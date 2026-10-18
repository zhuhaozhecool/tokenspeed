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

"""Exact owned-page metadata and window counts after changed-input capture."""

import os

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.ascend_metadata import compact_kvp_dsa_metadata
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import compact_kvp_pages

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENSPEED_DSA_METADATA_LIBRARY"),
    reason="requires the Ascend metadata native library",
)


@pytest.mark.parametrize("rank", [0, 1, 7])
@pytest.mark.parametrize("subpages", [1, 2, 4])
@pytest.mark.parametrize("rows,cols", [(8, 65), (64, 1041)])
def test_dsa_metadata_changed_input_graph(rank, subpages, rows, cols):
    torch.npu.set_device(0)
    page = 128
    table = torch.zeros((rows, cols), dtype=torch.int32, device="npu")
    lengths = torch.zeros(rows, dtype=torch.int32, device="npu")
    out = torch.empty_like(table)
    local = torch.empty_like(lengths)
    init = torch.empty_like(lengths)
    window = torch.empty_like(lengths)

    def run():
        compact_kvp_dsa_metadata(
            table,
            lengths,
            page_size=page,
            block_granularity=page * subpages,
            virtual_block_count=49,
            degree=8,
            rank=rank,
            init_tokens=126,
            local_tokens=127,
            out=out,
            local_lengths=local,
            init_counts=init,
            local_counts=window,
        )

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for seed in (19, 27, 0):
        generator = torch.Generator().manual_seed(seed)
        cpu_table = torch.randint(
            -subpages,
            50 * subpages,
            (rows, cols),
            generator=generator,
            dtype=torch.int32,
        )
        cpu_lengths = torch.tensor(
            [0, 125, 126, 127, 128, 129, 6144, cols * page] * (rows // 8), dtype=torch.int32
        )
        if seed == 0:
            cpu_table.zero_()
            cpu_lengths.zero_()
        table.copy_(cpu_table)
        lengths.copy_(cpu_lengths)
        graph.replay()
        torch.npu.synchronize()
        packed = torch.empty_like(cpu_table)
        prefix = torch.empty((rows, cols + 1), dtype=torch.int32)
        count = torch.empty_like(cpu_lengths)
        compact_kvp_pages(
            cpu_table,
            cpu_lengths,
            page_size=page,
            block_granularity=page * subpages,
            virtual_block_count=49,
            degree=8,
            rank=rank,
            out=packed,
            local_lengths=count,
            page_prefix=prefix,
        )
        positions = torch.arange(cols * page)
        owner = cpu_table[:, positions // page] // subpages > 0
        blocks = cpu_table[:, positions // page] // subpages
        owner &= (blocks < 49) & ((blocks - 1) % 8 == rank)
        visible = positions[None, :] < cpu_lengths[:, None]
        expected_init = (owner & visible & (positions[None, :] < 126)).sum(1)
        expected_window = (
            owner
            & visible
            & (positions[None, :] >= (cpu_lengths - 127).clamp_min(0)[:, None])
        ).sum(1)
        for actual, expected in (
            (out, packed),
            (local, count),
            (init, expected_init),
            (window, expected_window),
        ):
            torch.testing.assert_close(
                actual.cpu(), expected.to(actual.dtype), rtol=0, atol=0
            )


def test_metadata_rejects_task_queue_after_warmup(monkeypatch):
    from tokenspeed_kernel.ops.kvcache.ascend_metadata import _library

    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    _library(os.environ["TOKENSPEED_DSA_METADATA_LIBRARY"])
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "1")
    table = torch.ones((1, 1), dtype=torch.int32, device="npu")
    count = torch.ones(1, dtype=torch.int32, device="npu")
    with pytest.raises(ValueError, match="TASK_QUEUE_ENABLE=0"):
        compact_kvp_dsa_metadata(
            table,
            count,
            page_size=128,
            block_granularity=128,
            virtual_block_count=49,
            degree=8,
            rank=0,
            init_tokens=126,
            local_tokens=127,
            out=torch.empty_like(table),
            local_lengths=torch.empty_like(count),
            init_counts=torch.empty_like(count),
            local_counts=torch.empty_like(count),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"virtual_block_count": 2**31},
        {"degree": 2**31},
        {"init_tokens": 2**31},
        {"local_tokens": 2**31},
        {"page_size": 2**30, "block_granularity": 2**30},
    ],
)
def test_metadata_rejects_int32_overflow(override):
    table = torch.ones((1, 2), dtype=torch.int32, device="npu")
    count = torch.ones(1, dtype=torch.int32, device="npu")
    geometry = dict(
        page_size=128,
        block_granularity=128,
        virtual_block_count=49,
        degree=8,
        rank=0,
        init_tokens=126,
        local_tokens=127,
    )
    geometry.update(override)
    with pytest.raises(ValueError, match="fit INT32"):
        compact_kvp_dsa_metadata(
            table,
            count,
            **geometry,
            out=torch.empty_like(table),
            local_lengths=torch.empty_like(count),
            init_counts=torch.empty_like(count),
            local_counts=torch.empty_like(count),
        )
