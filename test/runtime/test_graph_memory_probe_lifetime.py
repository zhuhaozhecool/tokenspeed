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

"""Startup graph-probe storage ownership."""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ci_system.ci_register import register_cuda_ci  # noqa: E402
from runtime.cache_pool_test_utils import make_mha_pool_for_groups  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.execution.cudagraph_memory import GraphMemoryProbe  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)


@pytest.mark.parametrize("retain_alias", (False, True))
def test_closing_a_probe_releases_owned_storage_but_keeps_escaped_views_valid(
    monkeypatch, retain_alias: bool
) -> None:
    group = CacheGroupSpec(
        group_id="full_attention",
        attn_kind="full_history",
        rows_per_page=16,
        entry_stride_tokens=1,
        replayable=False,
    )
    pool = make_mha_pool_for_groups((group,), device="cpu")
    if retain_alias:
        # This is a distinct Tensor with the same storage, not a reference
        # to the exact Tensor object that the pool holds.
        alias = pool.get_key_buffer(0)[1:]
        alias.fill_(3)

    # No accelerator or captured graphs in this test. The pool and Tensor
    # storage still use the real ownership semantics.
    args = SimpleNamespace(
        device="cpu", speculative_algorithm=None, mapping=SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(torch.cpu, "synchronize", lambda _gpu_id: None)
    monkeypatch.setattr(torch.cpu, "empty_cache", lambda: None, raising=False)
    probe = GraphMemoryProbe(
        SimpleNamespace(token_to_kv_pool=pool, release_graphs=lambda: None), args, 0
    )
    del pool
    probe.close()
    probe.close()

    with pytest.raises(RuntimeError, match="probe is closed"):
        probe.measure(startup_resident_bytes=0)

    if retain_alias:
        with pytest.raises(RuntimeError, match="still has a live KV storage"):
            probe.require_storage_released()
        assert torch.all(alias == 3)
        alias.fill_(7)
        assert torch.all(alias == 7)
        del alias

    probe.require_storage_released()
