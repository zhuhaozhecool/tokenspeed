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

from __future__ import annotations

import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path

# Per-process files avoid interleaving across ranks; a lock orders local threads.
_lock = threading.Lock()
_files: dict[tuple[int, str], int] = {}
_sequence = 0


def pd_audit_enabled() -> bool:
    return bool(os.environ.get("TOKENSPEED_PD_AUDIT_DIR"))


def emit_pd_audit(
    event: str, *, role: str, global_rank: int, dp_rank: int, **fields
) -> None:
    """Record host observations only. No device reads or synchronization.

    Audit I/O failure propagates so a run cannot silently claim complete evidence.
    Monotonic timestamps are comparable within one host only.
    """
    directory = os.environ.get("TOKENSPEED_PD_AUDIT_DIR")
    if not directory:
        return
    global _sequence
    pid = os.getpid()
    with _lock:
        key = (pid, directory)
        if key not in _files:
            root = Path(directory)
            root.mkdir(parents=True, exist_ok=True)
            path = root / f"{socket.gethostname()}-{pid}.jsonl"
            _files[key] = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        record = dict(
            fields,
            schema_version=1,
            event=event,
            role=role,
            global_rank=int(global_rank),
            dp_rank=int(dp_rank),
            pid=pid,
            host=socket.gethostname(),
            monotonic_ns=time.monotonic_ns(),
            wall_time_ns=time.time_ns(),
            sequence=_sequence,
        )
        payload = (
            json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
        view = memoryview(payload)
        while view:
            written = os.write(_files[key], view)
            if written <= 0:
                raise OSError("PD audit write made no progress")
            view = view[written:]
        _sequence += 1


class TransferAudit:
    """Account descriptor payload for one peer transfer, not physical NIC bytes."""

    def __init__(
        self,
        *,
        room: int,
        global_rank: int,
        dp_rank: int,
        source_tp_rank: int,
        destination_tp_rank: int,
    ):
        self.context = dict(
            role="prefill",
            global_rank=global_rank,
            dp_rank=dp_rank,
            room=int(room),
            source_tp_rank=int(source_tp_rank),
            destination_tp_rank=int(destination_tp_rank),
            transfer_id=uuid.uuid4().hex,
        )
        self.payload_bytes = 0
        self.descriptor_count = 0
        self.started_ns = time.monotonic_ns()

    def field(
        self,
        *,
        group: str,
        field: str,
        source_owner: int,
        destination_owner: int,
        pages: int,
        bytes_per_page: int,
        descriptors_per_page: int,
    ) -> None:
        payload = int(pages) * int(bytes_per_page)
        descriptors = int(pages) * int(descriptors_per_page)
        self.payload_bytes += payload
        self.descriptor_count += descriptors
        emit_pd_audit(
            "transfer_plan",
            **self.context,
            group=group,
            field=field,
            source_owner=int(source_owner),
            destination_owner=int(destination_owner),
            pages=int(pages),
            planned_payload_bytes=payload,
            planned_descriptors=descriptors,
        )

    def done(self, return_code: int) -> None:
        # On failure, part of a lazy iterator may not have been visited. Do not
        # label its planned bytes as bytes successfully submitted or delivered.
        emit_pd_audit(
            "transfer_done",
            **self.context,
            return_code=int(return_code),
            success=return_code == 0,
            planned_payload_bytes=self.payload_bytes,
            submitted_bytes=self.payload_bytes if return_code == 0 else None,
            planned_descriptors=self.descriptor_count,
            application_retry_bytes=0,
            start_ns=self.started_ns,
            end_ns=time.monotonic_ns(),
        )
