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

"""Capture-safe direct launch for Ascend owned-page metadata."""

import ctypes
import functools
import os

import torch


@functools.lru_cache(maxsize=None)
def _library(path):
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Native Ascend metadata requires TASK_QUEUE_ENABLE=0")
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Warm up Ascend metadata before Graph capture")
    library = ctypes.CDLL(path)
    library.KvpMetadataLaunch.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_int32),
    ]
    library.KvpMetadataLaunch.restype = ctypes.c_int
    return library


def compact_kvp_dsa_metadata(
    table,
    lengths,
    *,
    page_size,
    block_granularity,
    virtual_block_count,
    degree,
    rank,
    init_tokens,
    local_tokens,
    out,
    local_lengths,
    init_counts,
    local_counts,
):
    """Write ordered owned pages and DSA token windows to persistent outputs.

    Inputs and outputs use INT32 NPU tensors. Output page tables retain all
    allocated owned reserve pages; counts include only the visible endpoints.
    """
    if page_size <= 0 or block_granularity <= 0 or block_granularity % page_size:
        raise ValueError("ownership blocks must contain whole kernel pages")
    _launch(
        table,
        lengths,
        page_size=page_size,
        subpages=block_granularity // page_size,
        virtual_block_count=virtual_block_count,
        degree=degree,
        rank=rank,
        out=out,
        local_lengths=local_lengths,
        page_prefix=None,
        init_tokens=init_tokens,
        local_tokens=local_tokens,
        init_counts=init_counts,
        local_counts=local_counts,
        dsa=True,
    )


def compact_kvp_pages_native(
    table,
    lengths,
    *,
    page_size,
    subpages,
    virtual_block_count,
    degree,
    rank,
    out,
    local_lengths,
    page_prefix,
):
    _launch(
        table,
        lengths,
        page_size=page_size,
        subpages=subpages,
        virtual_block_count=virtual_block_count,
        degree=degree,
        rank=rank,
        out=out,
        local_lengths=local_lengths,
        page_prefix=page_prefix,
        init_tokens=0,
        local_tokens=0,
        init_counts=None,
        local_counts=None,
        dsa=False,
    )


def _launch(
    table,
    lengths,
    *,
    page_size,
    subpages,
    virtual_block_count,
    degree,
    rank,
    out,
    local_lengths,
    page_prefix,
    init_tokens,
    local_tokens,
    init_counts,
    local_counts,
    dsa,
):
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Native Ascend metadata requires TASK_QUEUE_ENABLE=0")
    if (
        page_size <= 0
        or subpages <= 0
        or virtual_block_count <= 1
        or degree < 1
        or not 0 <= rank < degree
        or init_tokens < 0
        or local_tokens < 0
    ):
        raise ValueError("invalid Ascend metadata geometry")
    if (
        table.device.type != "npu"
        or table.ndim != 2
        or table.stride(1) != 1
        or out.shape != table.shape
        or out.stride(1) != 1
        or lengths.shape != table.shape[:1]
        or local_lengths.shape != lengths.shape
    ):
        raise ValueError("Ascend metadata buffers disagree")
    if table.shape[1] == 0:
        raise ValueError("Ascend metadata requires a nonempty page dimension")
    for value in (lengths, local_lengths, init_counts, local_counts):
        if value is not None and (
            value.shape != lengths.shape or not value.is_contiguous()
        ):
            raise ValueError("Ascend metadata counts must be contiguous row vectors")
    if page_prefix is not None and (
        page_prefix.shape != (table.shape[0], table.shape[1] + 1)
        or not page_prefix.is_contiguous()
    ):
        raise ValueError("Ascend metadata prefix must match page-table geometry")
    if dsa and (init_counts is None or local_counts is None):
        raise ValueError("DSA metadata requires both selection-window outputs")
    if table.shape[0] == 0:
        return
    values = [
        table,
        lengths,
        out,
        local_lengths,
        page_prefix,
        init_counts,
        local_counts,
    ]
    for value in values:
        if value is not None and (
            value.dtype != torch.int32 or value.device != table.device
        ):
            raise ValueError(
                "Ascend metadata requires INT32 tensors on the same device"
            )
    geometry = (
        table.shape[0],
        table.shape[1],
        table.stride(0),
        out.stride(0),
        page_size,
        subpages,
        virtual_block_count,
        degree,
        rank,
        init_tokens,
        local_tokens,
        int(dsa),
    )
    if any(value < 0 or value >= 2**31 for value in geometry):
        raise ValueError("Ascend metadata geometry must fit INT32")
    rows, columns, in_stride, out_stride = geometry[:4]
    if (
        columns * page_size >= 2**31
        or (rows - 1) * in_stride + columns >= 2**31
        or (rows - 1) * out_stride + columns >= 2**31
        or (page_prefix is not None and rows * (columns + 1) >= 2**31)
    ):
        raise ValueError("Ascend metadata addresses and endpoints must fit INT32")
    pointers = (ctypes.c_uint64 * len(values))(
        *(0 if v is None else v.data_ptr() for v in values)
    )
    params = (ctypes.c_int32 * 12)(*geometry)
    library = _library(os.environ["TOKENSPEED_DSA_METADATA_LIBRARY"])
    status = library.KvpMetadataLaunch(
        torch.npu.current_stream().npu_stream, pointers, params
    )
    if status:
        raise RuntimeError(f"Ascend metadata launch failed: {status}")
