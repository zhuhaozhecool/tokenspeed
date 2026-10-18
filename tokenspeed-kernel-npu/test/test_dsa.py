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

import pytest
import torch

pytest.importorskip("torch_npu")

from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


def test_full_scan_chunk_hints_are_cached_by_shape():
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    kernels._full_scan_chunks = {}
    indices = torch.zeros(3, 1, 2048, dtype=torch.int32)

    first = kernels._full_scan_chunk_hints(indices)
    second = kernels._full_scan_chunk_hints(indices)

    assert first.data_ptr() == second.data_ptr()
    assert first.tolist() == [16, 16, 16]


def test_optional_cp_primitives_fail_at_device_boundary():
    kernels = AscendDSAKernels.__new__(AscendDSAKernels)
    kernels._select_local = None
    kernels._merge_partials = None

    with pytest.raises(RuntimeError, match="SelectLocalTopkIndices"):
        kernels.select_local(torch.empty(0), torch.empty(0), 0)
    with pytest.raises(RuntimeError, match="KvpAttentionMerge"):
        kernels.merge_partials(torch.empty(0), torch.empty(0))


@pytest.mark.skipif(not torch.npu.is_available(), reason="NPU device not found")
@pytest.mark.parametrize("tokens", [1024, 1025, 1050])
def test_causal_query_batches_match_one_native_request(tokens):
    """A long causal span and per-query visible prefixes describe the same attention."""
    torch.npu.set_device(0)
    torch.manual_seed(341)
    kernels = AscendDSAKernels()
    pages = (tokens + 127) // 128
    query = torch.randn(8, tokens, 8, 576, dtype=torch.bfloat16, device="npu") * 0.1
    key = torch.randn(pages, 128, 1, 576, dtype=torch.bfloat16, device="npu") * 0.1
    before = key.clone()
    indices = torch.arange(2048, dtype=torch.int32, device="npu").view(1, 1, -1).expand(tokens, 1, -1).contiguous()
    chunks = torch.full((tokens,), pages, dtype=torch.int32, device="npu")
    table = torch.arange(pages, dtype=torch.int32, device="npu").view(1, -1)
    reference = kernels.attention_partial(
        query, key, indices, chunks,
        torch.tensor([tokens], dtype=torch.int32, device="npu"),
        torch.tensor([tokens], dtype=torch.int32, device="npu"),
        table, 576**-0.5, sparse_mode=3,
    )
    ends = torch.arange(1, tokens + 1, dtype=torch.int32, device="npu")
    actual = kernels.attention_partial(
        query, key, indices, chunks, ends, ends,
        table.expand(tokens, -1).contiguous(), 576**-0.5, sparse_mode=3,
    )
    torch.npu.synchronize()
    assert torch.equal(key, before)
    for output, expected in zip(actual, reference):
        assert bool(torch.isfinite(output).all())
        torch.testing.assert_close(output, expected, rtol=1e-4, atol=1e-5)
