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

"""Optional PD policy and observation providers, registered before engine startup."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol


class TransferObservation(Protocol):
    def field(self, **fields) -> None: ...
    def done(self, return_code: int) -> None: ...


class PDObserver(Protocol):
    def enabled(self) -> bool: ...
    def emit(self, event: str, **fields) -> None: ...
    def start_transfer(self, **context) -> TransferObservation: ...


class PDPolicy(Protocol):
    def admit_remote_prefill(self, request_id, state, bootstrap) -> bool: ...
    def observe_committed(self, request_ids) -> None: ...
    def filter_transfer_events(self, events): ...


_PROVIDERS: dict[str, object] = {}


def _register(kind, provider, *, override):
    from tokenspeed.runtime.plugins.registry import _put

    _put(_PROVIDERS, kind, provider, kind="PD extension", override=override)


def register_pd_observer(observer: PDObserver, *, override: bool = False) -> None:
    """Register host-only observations; the observer must not synchronize devices."""
    _register("observer", observer, override=override)


def register_pd_policy(
    factory: Callable[..., PDPolicy | None], *, override: bool = False
) -> None:
    """Register one optional event admission/filter policy without changing cache ownership.

    Held events must remain leased by the executor. A policy must return the
    original success/failure events through the ordinary processing path and
    preserve the transfer completion fence before releasing destinations.
    """
    _register("policy", factory, override=override)


def register_pd_server_info(
    transform: Callable[[Mapping], dict], *, override: bool = False
) -> None:
    """Register discovery metadata normalization without mutating execution arguments."""
    _register("server_info", transform, override=override)


def create_pd_policy(loop, device) -> PDPolicy | None:
    factory = _PROVIDERS.get("policy")
    return None if factory is None else factory(loop, device)


def pd_observer() -> PDObserver | None:
    return _PROVIDERS.get("observer")


def pd_server_info(server_args: Mapping) -> dict:
    transform = _PROVIDERS.get("server_info")
    return dict(server_args) if transform is None else transform(server_args)
