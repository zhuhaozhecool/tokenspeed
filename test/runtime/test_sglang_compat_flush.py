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

"""Flush HTTP responses report failures and each domain's actual result."""

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from tokenspeed.runtime.engine.io_struct import FlushCacheDomain, FlushCacheReqOutput
from tokenspeed.runtime.entrypoints.sglang_compat_http import build_sglang_compat_app


@dataclass
class ControlSettings:
    rl_control_api_key: str | None = None


class FlushEngine:
    def __init__(self, replies):
        self.replies = replies
        self.server_args = ControlSettings()

    async def flush_cache(self):
        return FlushCacheReqOutput(
            success=all(self.replies),
            domains=[
                FlushCacheDomain(dp_rank=i, success=value)
                for i, value in enumerate(self.replies)
            ],
        )


@pytest.mark.parametrize(
    "replies", [(True, True), (True, False), (False, True), (False, False)]
)
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_flush_reports_actual_domains(replies, method):
    app = build_sglang_compat_app(FlushEngine(replies))
    response = TestClient(app).request(method, "/flush_cache")
    assert response.status_code == (200 if all(replies) else 400)
    body = response.json()
    assert body["success"] == all(replies)
    assert body["domains"] == [
        dict(dp_rank=i, success=value) for i, value in enumerate(replies)
    ]
