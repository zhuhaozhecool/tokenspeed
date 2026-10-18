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

"""Ascend cache-write slot conversion with explicit invalid writes."""

import ctypes
import functools
import os

import torch


@functools.lru_cache(maxsize=None)
def _library(path):
    if torch.npu.is_current_stream_capturing():
        raise RuntimeError("Warm up Ascend slot translation before Graph capture")
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Native Ascend slot translation requires TASK_QUEUE_ENABLE=0")
    library = ctypes.CDLL(path)
    library.KvpWriteSlotsLaunch.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 6
    library.KvpWriteSlotsLaunch.restype = ctypes.c_int
    return library


def ascend_write_slots(slots, *, rows_per_page, virtual_block_count, degree, rank, out):
    """Write local physical slots, with -1 for null, foreign or invalid slots.

    This is the Ascend scatter ABI. The common slot-placement API keeps its
    separate zero-slot and owner-mask contract. INT32 and INT64 are supported.
    """
    if os.environ.get("TASK_QUEUE_ENABLE") != "0":
        raise ValueError("Native Ascend slot translation requires TASK_QUEUE_ENABLE=0")
    if (
        rows_per_page < 1
        or virtual_block_count < 2
        or degree < 1
        or not 0 <= rank < degree
        or virtual_block_count * rows_per_page > 2**63 - 1
    ):
        raise ValueError("invalid Ascend write-slot geometry")
    if (
        slots.device.type != "npu"
        or slots.dtype not in (torch.int32, torch.int64)
        or out.shape != slots.shape
        or out.dtype != slots.dtype
        or out.device != slots.device
        or not out.is_contiguous()
    ):
        raise ValueError("Ascend write-slot buffers disagree")
    if slots.numel() == 0:
        return
    source = slots.contiguous()
    library = _library(os.environ["TOKENSPEED_DSA_SLOTS_LIBRARY"])
    status = library.KvpWriteSlotsLaunch(
        torch.npu.current_stream().npu_stream,
        source.data_ptr(),
        out.data_ptr(),
        slots.numel(),
        rows_per_page,
        virtual_block_count,
        degree,
        rank,
        int(slots.dtype == torch.int64),
    )
    if status:
        raise RuntimeError(f"Ascend slot translation failed: {status}")
