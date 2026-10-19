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

"""Long one-query-per-row attention agrees across independent native call boundaries."""

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("flash_ops")
from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


@pytest.mark.parametrize("total", [1024, 1025, 2049])
def test_sfa_request_limit_preserves_query_order(total):
    if not torch.npu.is_available():
        pytest.skip("requires NPU")
    torch.manual_seed(97)
    kernel = AscendDSAKernels()
    query = (torch.randn(8, total, 4, 576) * 0.1).to(dtype=torch.bfloat16, device="npu")
    key = (torch.randn(2, 128, 1, 576) * 0.1).to(dtype=torch.bfloat16, device="npu")
    indices = torch.full((total, 1, 2048), -1, dtype=torch.int32)
    indices[:, 0, :32] = torch.arange(32, dtype=torch.int32)
    indices = indices.npu()
    chunks = torch.ones(total, dtype=torch.int32, device="npu")
    ends = torch.arange(1, total + 1, dtype=torch.int32, device="npu")
    lengths = torch.full((total,), 32, dtype=torch.int32, device="npu")
    table = torch.ones((total, 1), dtype=torch.int32, device="npu")

    def invoke(start, end):
        return kernel.attention_partial(
            query[:, start:end].contiguous(),
            key,
            indices[start:end],
            chunks[start:end],
            ends[start:end] - start,
            lengths[start:end],
            table[start:end],
            576**-0.5,
            sparse_mode=3,
        )

    actual = invoke(0, total)
    parts = [invoke(start, min(start + 257, total)) for start in range(0, total, 257)]
    expected = tuple(torch.cat(values, dim=1) for values in zip(*parts))
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-6)
    assert actual[0].shape == (32, total, 512)
    assert actual[1].shape == actual[2].shape == (1, total, 32)
