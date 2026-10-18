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

"""Verify PD pin behavior at the control HTTP boundary (CPU only)."""

import asyncio
import json

import httpx
from fastapi.responses import JSONResponse

from tokenspeed.runtime.entrypoints import control_server


def native_request(monkeypatch, *, mapping, rank=1, supplied=None):
    async def run():
        async def upstream(request, *, body_override, headers_override):
            # Network-only substitute echoes what a real SMG gateway receives.
            return JSONResponse(
                {
                    "received": json.loads(body_override),
                    "target": headers_override["X-SMG-Target-Worker"],
                }
            )

        monkeypatch.setenv("TOKENSPEED_TOKEN_GATEWAY_DP_TARGETS", mapping)
        monkeypatch.setattr(control_server, "_proxy_request", upstream)
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=control_server.app),
            base_url="http://control",
        ) as client:
            return await client.post(
                "/generate",
                json={
                    "model": "cpu-model",
                    "input_ids": [7, 7, 2],
                    "sampling_params": {"max_new_tokens": 6},
                    "data_parallel_rank": rank,
                },
                headers={} if supplied is None else {"X-SMG-Target-Worker": supplied},
            )

    return asyncio.run(run())


def test_native_token_ids_and_pin_reach_gateway(monkeypatch):
    response = native_request(monkeypatch, mapping='{"0":0,"1":1}')
    assert response.status_code == 200
    assert response.json()["target"] == "1"
    assert response.json()["received"]["input_ids"] == [7, 7, 2]


def test_native_pin_fails_without_verified_mapping(monkeypatch):
    assert native_request(monkeypatch, mapping="{}").status_code == 503


def test_native_invalid_mapping_fails_cleanly(monkeypatch):
    for mapping in ("broken", "[]", '{"0":0,"1":0}'):
        assert native_request(monkeypatch, mapping=mapping).status_code == 503


def test_native_conflicting_target_is_rejected(monkeypatch):
    assert (
        native_request(monkeypatch, mapping='{"0":0,"1":1}', supplied="0").status_code
        == 400
    )
