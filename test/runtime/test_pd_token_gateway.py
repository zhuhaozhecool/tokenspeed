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

"""Host protocol tests using real HTTP, protobuf, and a small local tokenizer.

The HTTP upstream replaces only the missing model/PD network. It does not
prove Mooncake transfer or real device state handoff.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import aiohttp
from aiohttp import web
from fastapi import HTTPException
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import TokenizersBackend

SOURCE = Path(os.environ["NPU_TS_SOURCE"])
spec = importlib.util.spec_from_file_location(
    "token_completions",
    SOURCE / "python/tokenspeed/runtime/entrypoints/token_completions.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.model = self.temp.name
        tokenizer = Tokenizer(
            models.WordLevel(
                {"[UNK]": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6, "g": 7},
                unk_token="[UNK]",
            )
        )
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        TokenizersBackend(
            tokenizer_object=tokenizer, unk_token="[UNK]"
        ).save_pretrained(self.model)
        self.seen = []
        self.mode = "normal"
        self.cancelled = asyncio.Event()
        app = web.Application()
        app.router.add_post("/generate", self.generate)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.url = f"http://127.0.0.1:{self.site._server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        await self.runner.cleanup()
        module.tokenizer.cache_clear()
        self.temp.cleanup()

    async def generate(self, request):
        self.seen.append((await request.json(), dict(request.headers)))
        if self.mode == "error":
            return web.json_response({"error": "P unavailable"}, status=503)
        if self.mode == "headers_timeout":
            await asyncio.sleep(0.1)
        result = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await result.prepare(request)
        records = [
            ([1], 1, None),
            ([2, 3, 4, 5, 6], 6, None),
            ([6], 6, {"type": "length"}),
        ]
        try:
            for ids, count, reason in records:
                meta = dict(
                    id="real-gateway-id-0",
                    finish_reason=reason,
                    prompt_tokens=3,
                    cached_tokens=2,
                    completion_tokens=count,
                )
                if reason and self.mode != "missing_usage":
                    meta.update(spec_accepted_tokens=5, spec_draft_tokens=8)
                frame = dict(text="unused output", output_ids=ids, meta_info=meta)
                data = ("data: " + json.dumps(frame) + "\n\n").encode()
                await result.write(data[:9])
                await result.write(data[9:])
                if self.mode == "truncated":
                    return result
                if self.mode in ("slow", "timeout"):
                    await asyncio.sleep(0.1)
                    if self.mode == "timeout":
                        await asyncio.sleep(0.3)
            if self.mode == "truncated":
                return result
            await result.write(b"data: [DONE]\n\n")
        except (ConnectionResetError, asyncio.CancelledError):
            self.cancelled.set()
        return result

    def body(self, **kw):
        return dict(
            model=self.model,
            prompt=[7, 7, 2],
            max_tokens=6,
            min_tokens=6,
            ignore_eos=True,
            stream=True,
            data_parallel_rank=1,
            **kw,
        )

    async def response(self, body=None, **kw):
        return await module.gateway_token_completions(
            body or self.body(),
            gateway_url=self.url,
            served_model=self.model,
            dp_targets={"0": 0, "1": 1},
            timeout=kw.get("timeout", aiohttp.ClientTimeout(total=3)),
        )

    async def collect(self, response):
        return [part async for part in response.body_iterator]

    async def test_token_ids_pin_delta_frames_and_usage(self):
        body = self.body(stream_options={"include_usage": True})
        parts = await self.collect(await self.response(body))
        frames = [json.loads(x[6:]) for x in parts if x != "data: [DONE]\n\n"]
        self.assertEqual(self.seen[0][0]["input_ids"], [7, 7, 2])
        self.assertEqual(self.seen[0][1]["X-SMG-Target-Worker"], "1")
        self.assertNotIn("bootstrap_room", self.seen[0][0])
        self.assertEqual(
            [i for x in frames for i in x.get("token_ids", [])], [1, 2, 3, 4, 5, 6]
        )
        self.assertEqual(
            len(
                [
                    x
                    for x in frames
                    if x.get("choices") and x["choices"][0]["finish_reason"]
                ]
            ),
            1,
        )
        self.assertEqual(parts.count("data: [DONE]\n\n"), 1)
        self.assertEqual(frames[0]["gateway_request_id"], "real-gateway-id")
        self.assertEqual(
            frames[-1]["usage"]["completion_tokens_details"],
            dict(accepted_prediction_tokens=5, rejected_prediction_tokens=3),
        )
        self.assertEqual(
            frames[-1]["usage"]["prompt_tokens_details"]["cached_tokens"], 2
        )

    async def test_nonstream_uses_same_real_gateway(self):
        body = self.body()
        body["stream"] = False
        result = await self.response(body)
        data = json.loads(result.body)
        self.assertEqual(data["usage"]["completion_tokens"], 6)
        self.assertEqual(data["token_ids"], [1, 2, 3, 4, 5, 6])
        self.assertEqual(data["choices"][0]["finish_reason"], "length")
        self.assertTrue(self.seen[0][0]["stream"])

    async def test_close_cancels_upstream(self):
        self.mode = "slow"
        response = await self.response()
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        await asyncio.wait_for(self.cancelled.wait(), 2)

    async def test_inactivity_timeout_closes_upstream_and_reports_error(self):
        self.mode = "timeout"
        parts = await self.collect(
            await self.response(
                timeout=aiohttp.ClientTimeout(total=None, sock_read=0.03)
            )
        )
        self.assertTrue(any('"upstream_error"' in x for x in parts))
        self.assertNotIn("data: [DONE]\n\n", parts)
        await asyncio.wait_for(self.cancelled.wait(), 2)

    async def test_gateway_header_timeout_is_504(self):
        self.mode = "headers_timeout"
        with self.assertRaises(HTTPException) as caught:
            await self.response(timeout=aiohttp.ClientTimeout(total=0.03))
        self.assertEqual(caught.exception.status_code, 504)
        self.assertEqual(len(self.seen), 1)

    async def test_incomplete_stream_is_not_success(self):
        self.mode = "truncated"
        parts = await self.collect(await self.response())
        self.assertTrue(any('"upstream_error"' in x for x in parts))
        self.assertNotIn("data: [DONE]\n\n", parts)

    async def test_gateway_failure_is_not_retried(self):
        self.mode = "error"
        with self.assertRaises(HTTPException) as caught:
            await self.response()
        self.assertEqual(caught.exception.status_code, 503)
        self.assertEqual(len(self.seen), 1)

    async def test_missing_spec_usage_is_not_fabricated(self):
        self.mode = "missing_usage"
        parts = await self.collect(await self.response())
        self.assertTrue(any('"upstream_error"' in x for x in parts))
        self.assertNotIn("data: [DONE]\n\n", parts)

    def test_pin_requires_explicit_mapping(self):
        with self.assertRaises(HTTPException) as caught:
            module.gateway_request(self.body(), dp_targets={})
        self.assertEqual(caught.exception.status_code, 503)
        for value in (-1, True, 2):
            with self.assertRaises(HTTPException):
                body = self.body()
                body["data_parallel_rank"] = value
                module.gateway_request(body, dp_targets={"0": 0, "1": 1})

    def test_invalid_large_input_id_rejected_without_retokenization(self):
        body = self.body()
        body["prompt"] = [2**31]
        with self.assertRaises(HTTPException):
            module.gateway_request(body, dp_targets={"0": 0, "1": 1})


if __name__ == "__main__":
    unittest.main()
