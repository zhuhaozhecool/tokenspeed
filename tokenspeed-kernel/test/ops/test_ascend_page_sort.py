# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright 2023-2024 SGLang Team
#
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

"""Triton implementation of KVStore transfer kernels."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    _compact_dcp_pages_ascend,
)


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("with_prefix", [False, True])
def test_compaction_preserves_token_order(rank, with_prefix):
    table = torch.tensor([[0, 5, 2, 8, 1, 7, -1, 10], [4, 3, 0, 2, 9, 6, 1, 0]])
    lengths = torch.tensor([37, 61])
    out = torch.full_like(table, -99)
    local = torch.empty(2, dtype=torch.int64)
    prefix = torch.empty((2, 9), dtype=torch.int32) if with_prefix else None
    _compact_dcp_pages_ascend(
        table,
        lengths,
        page_size=8,
        subpages=1,
        virtual_block_count=10,
        degree=3,
        rank=rank,
        out=out,
        local_lengths=local,
        page_prefix=prefix,
    )
    expected = torch.zeros_like(table)
    expected_lengths = []
    for row in range(2):
        pages, tokens, counts = [], 0, [0]
        for col, block in enumerate(table[row].tolist()):
            owned = 0 < block < 10 and (block - 1) % 3 == rank
            counts.append(counts[-1] + int(owned))
            if owned and (with_prefix or col * 8 < lengths[row]):
                pages.append((block - 1) // 3 + 1)
                tokens += max(0, min(8, int(lengths[row]) - col * 8))
        expected[row, : len(pages)] = torch.tensor(pages, dtype=table.dtype)
        expected_lengths.append(tokens)
        if with_prefix:
            torch.testing.assert_close(
                prefix[row], torch.tensor(counts, dtype=prefix.dtype)
            )
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(local, torch.tensor(expected_lengths))
