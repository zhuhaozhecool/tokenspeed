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

"""Validate native ABI boundaries using only a replacement device operator."""

import pytest
import torch
from tokenspeed_kernel_npu.ops.dsa import AscendDSAKernels


def facade(index, attention):
    # Device operator loading is unavailable on a CPU host. The real facade
    # methods, validation, dtypes and arguments remain under test.
    kernel = object.__new__(AscendDSAKernels)
    kernel._index = index
    kernel._selection_windows = {}
    kernel._prefill_attention = attention
    return kernel


def test_native_index_uses_bf16_and_right_aligned_subspan():
    def index(q, k, weights, **kw):
        qlen = int(kw["actual_seq_lengths_query"][0])
        klen = int(kw["actual_seq_lengths_key"][0])
        assert qlen == 3 and klen == 135  # H129 + shard end6, not H+chunk end9.
        assert kw["layout_query"] == kw["layout_key"] == "TND"
        assert kw["block_table"] is None
        assert kw["local_tensor"].shape == (3,)
        assert kw["init_tensor"].shape == (1,)
        assert q.dtype == k.dtype == weights.dtype == torch.bfloat16
        # A replacement device op exposes causal last-index for every row.
        ids = torch.arange(klen - qlen, klen, dtype=torch.int32).view(3, 1, 1)
        return ids, (
            torch.ones_like(ids, dtype=torch.bfloat16) if kw["return_value"] else None
        )

    kernel = facade(index, None)
    q = torch.zeros((3, 16, 128), dtype=torch.bfloat16)
    k = torch.zeros((135, 1, 128), dtype=torch.bfloat16)
    weights = torch.zeros((3, 16), dtype=torch.bfloat16)
    ends = (
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([135], dtype=torch.int32),
    )
    actual, values = kernel.prefill_index(
        q, k, weights, *ends, topk=2048, initial=1, local=2, return_scores=False
    )
    debug, values_debug = kernel.prefill_index(
        q, k, weights, *ends, topk=2048, initial=1, local=2, return_scores=True
    )
    assert values is None and values_debug.dtype == torch.bfloat16
    torch.testing.assert_close(actual, debug)
    assert actual.flatten().tolist() == [132, 133, 134]
    with pytest.raises(ValueError, match="BF16 scoring weights"):
        kernel.prefill_index(
            q,
            k,
            weights.float(),
            *ends,
            topk=2048,
            initial=1,
            local=2,
            return_scores=False,
        )
    with pytest.raises(ValueError, match="INT32"):
        kernel.prefill_index(
            q,
            k,
            weights,
            ends[0].long(),
            ends[1],
            topk=2048,
            initial=1,
            local=2,
            return_scores=False,
        )


def test_sfa_receives_contiguous_separate_512_64():
    def attention(q, k, v, indices, scale, **kw):
        assert q.is_contiguous() and q.shape == (3, 64, 512)
        assert kw["query_rope"].is_contiguous()
        assert kw["query_rope"].shape == (3, 64, 64)
        assert k is v and k.shape == (135, 1, 512)
        assert kw["key_rope"].shape == (135, 1, 64)
        assert kw["layout_query"] == kw["layout_kv"] == "TND"
        assert kw["attention_mode"] == 2 and kw["sparse_mode"] == 3
        assert kw["block_table"] is None
        return q * scale, None, None

    kernel = facade(None, attention)
    q = torch.arange(3 * 64 * 576).reshape(3, 64, 576).to(torch.bfloat16)
    k = torch.zeros((135, 1, 512), dtype=torch.bfloat16)
    rope = torch.zeros((135, 1, 64), dtype=torch.bfloat16)
    indices = torch.zeros((3, 1, 2048), dtype=torch.int32)
    ends = (
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([135], dtype=torch.int32),
    )
    result = kernel.prefill_attention(q, k, rope, indices, *ends, scale=0.5)
    torch.testing.assert_close(result, q[..., :512] * 0.5)
    with pytest.raises(ValueError, match="512/64"):
        kernel.prefill_attention(
            q,
            torch.zeros((135, 1, 576), dtype=torch.bfloat16),
            rope,
            indices,
            *ends,
            scale=0.5,
        )
