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

"""Unsupported block-scaled routes must fail before NPU execution."""

import subprocess
import sys
import textwrap

import pytest
import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.moe.block_scaled import block_scaled_moe_routes_plan
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(not current_platform().is_npu, reason="requires Ascend")


@pytest.mark.parametrize("solution", [None, "triton", "torch"])
def test_npu_routes_plan_rejects_unsupported_execution(solution):
    with pytest.raises(NotImplementedError, match="not supported on NPU"):
        block_scaled_moe_routes_plan(
            formats=("nvfp4", "mxfp8"),
            hidden=64,
            intermediate=32,
            device=torch.device("npu"),
            solution=solution,
        )


@pytest.mark.skipif(
    hasattr(triton, "constexpr_function"),
    reason="requires Triton without constexpr_function",
)
@pytest.mark.parametrize("vendor", ["nvidia", "amd"])
def test_missing_triton_api_is_not_ignored_on_gpu(vendor):
    probe = textwrap.dedent("""
        import sys
        from dataclasses import replace

        from tokenspeed_kernel._triton import triton
        from tokenspeed_kernel.ops.gemm import cute_dsl, triton_skinny
        import tokenspeed_kernel.ops.moe
        from tokenspeed_kernel.platform import Platform, current_platform

        # Only the hardware identity is simulated. The missing API comes from
        # the installed Ascend Triton, with all shared dependencies preloaded.
        Platform.override(replace(current_platform(), vendor=sys.argv[1]))
        try:
            import tokenspeed_kernel.ops.moe.block_scaled
        except AttributeError as exc:
            assert exc.name == "constexpr_function" and exc.obj is triton
        else:
            raise AssertionError("GPU model discovery must report the missing API")
        """)
    subprocess.run(
        [sys.executable, "-c", probe, vendor],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
