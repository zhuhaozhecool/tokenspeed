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


"""Host-only HTTP control for the explicitly enabled benchmark cohort gate."""

import asyncio
import os

from fastapi import Request
from fastapi.responses import JSONResponse

from tokenspeed.runtime.pd.bench_cohort import CohortStore


async def handle_cohort_control(
    request: Request, action: str, *, decode_role: bool
) -> JSONResponse:
    directory = os.environ.get("TOKENSPEED_BENCH_COHORT_DIR")
    if not directory or not decode_role:
        return JSONResponse(
            {"success": False, "error": "D benchmark cohort control is disabled"},
            status_code=503,
        )
    try:
        store = CohortStore(directory)
        if request.method == "GET" and action == "status":
            result = await asyncio.to_thread(
                store.status,
                request.query_params["epoch"],
                request.query_params["cohort_id"],
            )
        elif request.method == "POST" and action in ("arm", "release", "abort"):
            body = await request.json()
            if action == "arm":
                result = await asyncio.to_thread(store.arm, body)
            elif action == "release":
                if not isinstance(body, dict) or set(body) != {"epoch", "cohort_id"}:
                    raise ValueError("release requires exactly epoch and cohort_id")
                result = await asyncio.to_thread(
                    store.release, body["epoch"], body["cohort_id"]
                )
            else:
                if not isinstance(body, dict) or set(body) != {
                    "epoch",
                    "cohort_id",
                    "reason",
                }:
                    raise ValueError(
                        "abort requires exactly epoch, cohort_id and reason"
                    )
                result = await asyncio.to_thread(
                    store.abort, body["epoch"], body["cohort_id"], body["reason"]
                )
        else:
            return JSONResponse(
                {"success": False, "error": "Unknown cohort action/method"},
                status_code=404,
            )
        return JSONResponse(result)
    except (ValueError, KeyError, TypeError, OSError) as error:
        return JSONResponse({"success": False, "error": str(error)}, status_code=409)
