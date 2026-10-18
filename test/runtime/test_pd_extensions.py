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


"""Behavior of optional PD registrations, including plugin-load rollback."""

import pytest

from tokenspeed.runtime.pd.audit import (
    emit_pd_audit,
    pd_audit_enabled,
    start_transfer_audit,
)
from tokenspeed.runtime.pd.extensions import create_pd_policy, pd_server_info
from tokenspeed.runtime.plugins.registry import (
    recording,
    register_pd_observer,
    register_pd_policy,
    register_pd_server_info,
)


class RegistrationFailed(Exception):
    pass


class Observer:
    def __init__(self):
        self.events = []

    def enabled(self):
        return True

    def emit(self, event, **fields):
        self.events.append((event, fields))

    def start_transfer(self, **context):
        return context


def test_unconfigured_pd_is_passthrough():
    source = {"attention_backend": "native", "extra": [1]}
    assert pd_server_info(source) == source
    assert pd_server_info(source) is not source
    assert create_pd_policy(None, None) is None
    assert not pd_audit_enabled()
    assert emit_pd_audit("idle") is None
    with pytest.raises(RuntimeError, match="enabled PD observer"):
        start_transfer_audit(room=12)


def test_registration_dispatch_and_failed_load_rollback():
    observer = Observer()
    policy = object()
    with pytest.raises(RegistrationFailed), recording():
        register_pd_observer(observer)
        register_pd_policy(lambda loop, device: policy)
        register_pd_server_info(lambda info: dict(info, resolved=True))
        assert pd_audit_enabled()
        emit_pd_audit("received", room=9)
        assert observer.events == [("received", {"room": 9})]
        assert start_transfer_audit(room=9) == {"room": 9}
        assert create_pd_policy(None, None) is policy
        assert pd_server_info({"x": 2}) == {"x": 2, "resolved": True}
        with pytest.raises(ValueError):
            register_pd_observer(Observer())
        raise RegistrationFailed()
    test_unconfigured_pd_is_passthrough()


def test_explicit_override_rolls_back_to_prior_provider():
    first, second = Observer(), Observer()
    with pytest.raises(RegistrationFailed), recording():
        register_pd_observer(first)
        register_pd_observer(second, override=True)
        emit_pd_audit("one")
        assert first.events == []
        assert second.events == [("one", {})]
        raise RegistrationFailed()
    assert not pd_audit_enabled()
