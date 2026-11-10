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

"""Dense MM selects compatible kernels after static weight preprocessing."""

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.ops.gemm import mm, prepare_dense_gemm_weight
from tokenspeed_kernel.registry import register_kernel
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import format_signatures
from tokenspeed_kernel.weights import get_weight_broker

pytestmark = pytest.mark.usefixtures("fresh_registry")


def _prepare(*, weight):
    # A different physical representation makes a layout-mismatched call fail
    # numerically, while retaining the checkpoint's logical [N,K] geometry.
    return {"weight": (2 * weight.T.contiguous()).T}


def _canonical_mm(A, B, A_scales, B_scales, out_dtype, *, alpha, block_size):
    return F.linear(A, B).to(out_dtype)


def _prepared_mm(A, B, A_scales, B_scales, out_dtype, *, alpha, block_size):
    return F.linear(A, B / 2).to(out_dtype)


def _register():
    canonical = format_signatures(("a", "b"), "dense", {torch.float32})
    prepared = format_signatures(("a", "b"), "dense", {torch.float32}, {_prepare})
    register_kernel(
        "gemm",
        "mm",
        name="test_canonical_mm",
        solution="test",
        signatures=canonical,
        priority=1,
    )(_canonical_mm)
    register_kernel(
        "gemm",
        "mm",
        name="test_prepared_mm",
        solution="prepared",
        signatures=prepared,
        traits={"weight_shape": frozenset({(6, 8)})},
        features={"weight_preparation"},
        weight_preprocessor=(("weight",), _prepare),
        priority=2,
    )(_prepared_mm)
    register_kernel(
        "gemm",
        "mm",
        name="test_small_prepared_mm",
        solution="small",
        signatures=prepared,
        traits={
            "weight_shape": frozenset({(6, 8)}),
            "m_max": frozenset({2}),
            "a_contiguous": frozenset({True}),
        },
        priority=3,
    )(_prepared_mm)


@torch.no_grad()
def test_mm_uses_storage_layout_across_row_counts_and_checkpoint_writes():
    _register()
    layer = torch.nn.Linear(8, 6, bias=False)
    canonical = layer.weight.clone()
    x = torch.randn(5, 8)
    torch.testing.assert_close(mm(x, layer.weight), F.linear(x, canonical))
    prepare_dense_gemm_weight(layer, batch_invariant=False)
    pointer = layer.weight.data_ptr()
    for rows in (1, 2, 3):
        for activation in (x[:rows], torch.randn(8, rows).T):
            torch.testing.assert_close(
                mm(activation, layer.weight), F.linear(activation, canonical)
            )
    torch.testing.assert_close(
        mm(x[:1], layer.weight, solution="small"), F.linear(x[:1], canonical)
    )
    with pytest.raises(NoKernelFoundError):
        mm(x, layer.weight, solution="small")
    with pytest.raises(NoKernelFoundError):
        mm(x, canonical, solution="prepared")

    canonical = torch.randn_like(canonical)
    # The layout's checkpoint update preserves the same storage; preparation
    # must not reapply a transform after the update or replace the allocation.
    layer.weight.copy_(2 * canonical)
    prepare_dense_gemm_weight(layer, batch_invariant=False)
    assert layer.weight.data_ptr() == pointer
    torch.testing.assert_close(mm(x, layer.weight), F.linear(x, canonical))


@pytest.mark.parametrize("strict,output_size", [(True, 6), (False, 7)])
def test_unsupported_preparation_retains_canonical_storage(strict, output_size):
    _register()
    layer = torch.nn.Linear(8, output_size, bias=False)
    pointer = layer.weight.data_ptr()
    prepare_dense_gemm_weight(layer, batch_invariant=strict)
    assert layer.weight.data_ptr() == pointer
    assert get_weight_broker().layout(layer.weight, allow_unknown=True) is None
    x = torch.randn(3, 8)
    torch.testing.assert_close(mm(x, layer.weight), F.linear(x, layer.weight))
