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

"""Cache-write addresses retain exact integer semantics under Graph replay."""

import os

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.ascend_slots import ascend_write_slots
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import virtual_slots_to_local

pytestmark = pytest.mark.skipif(
    not os.environ.get("TOKENSPEED_DSA_SLOTS_LIBRARY"),
    reason="requires the Ascend write-slot native library",
)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("rows", [1, 125, 128, 256])
@pytest.mark.parametrize("rank", [0, 1, 7])
def test_write_slots_changed_input_graph(dtype, rows, rank):
    torch.npu.set_device(0)
    source = torch.empty(65, dtype=dtype, device="npu")
    out = torch.empty_like(source)

    def run():
        ascend_write_slots(
            source,
            rows_per_page=rows,
            virtual_block_count=49,
            degree=8,
            rank=rank,
            out=out,
        )

    run()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for seed in (7, 0, 19):
        values = torch.randint(
            -rows * 2,
            50 * rows,
            (65,),
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
        )
        values[:8] = torch.tensor(
            [
                -1,
                0,
                rows - 1,
                rows,
                49 * rows - 1,
                49 * rows,
                torch.iinfo(dtype).min,
                torch.iinfo(dtype).max,
            ],
            dtype=dtype,
        )
        if seed == 0:
            values.zero_()
        source.copy_(values)
        graph.replay()
        torch.npu.synchronize()
        local, owner = virtual_slots_to_local(
            values,
            rows_per_page=rows,
            virtual_block_count=49,
            degree=8,
            rank=rank,
        )
        expected = torch.where(owner, local, -1)
        torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_slots_reject_task_queue_after_warmup(monkeypatch):
    from tokenspeed_kernel.ops.kvcache.ascend_slots import _library

    monkeypatch.setenv("TASK_QUEUE_ENABLE", "0")
    _library(os.environ["TOKENSPEED_DSA_SLOTS_LIBRARY"])
    monkeypatch.setenv("TASK_QUEUE_ENABLE", "1")
    source = torch.ones(1, dtype=torch.int64, device="npu")
    with pytest.raises(ValueError, match="TASK_QUEUE_ENABLE=0"):
        ascend_write_slots(
            source,
            rows_per_page=128,
            virtual_block_count=49,
            degree=8,
            rank=0,
            out=torch.empty_like(source),
        )
