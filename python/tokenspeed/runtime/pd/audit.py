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

"""Host observation call sites; plugins own serialization and storage."""
from tokenspeed.runtime.pd.extensions import TransferObservation, pd_observer


def pd_audit_enabled() -> bool:
    observer = pd_observer()
    return observer is not None and observer.enabled()


def emit_pd_audit(event: str, **fields) -> None:
    observer = pd_observer()
    if observer is not None and observer.enabled():
        observer.emit(event, **fields)


def start_transfer_audit(**context) -> TransferObservation:
    observer = pd_observer()
    if observer is None or not observer.enabled():
        raise RuntimeError("Transfer observation requires an enabled PD observer")
    return observer.start_transfer(**context)
