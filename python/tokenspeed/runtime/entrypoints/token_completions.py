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

"""OpenAI completion transport for already-tokenized, single prompts.

Input IDs go unchanged to the engine. Tokenization is only used to decode
output IDs; it cannot alter prefix-cache identity. Text clients keep the
existing gateway route. This adapter intentionally rejects unsupported options.
"""

import asyncio
import itertools
import json
import os
import time
import uuid
from functools import lru_cache

import aiohttp
from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from smg_grpc_proto.generated import tokenspeed_scheduler_pb2 as pb
from tokenizers.decoders import DecodeStream
from transformers import AutoTokenizer

from tokenspeed.runtime.pd.bench_cohort import CohortError, tagged_request_id

_benchmark_rooms = itertools.count()


@lru_cache(maxsize=1)
def tokenizer(model):
    return AutoTokenizer.from_pretrained(model, local_files_only=True).backend_tokenizer


def build_request(body):
    allowed = {
        "model",
        "prompt",
        "max_tokens",
        "min_tokens",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
        "stop",
        "ignore_eos",
        "stream",
        "stream_options",
        "n",
        "seed",
        "data_parallel_rank",
        "bench_cohort",
    }
    unknown = set(body) - allowed
    if unknown:
        raise HTTPException(
            400, f"Unsupported token completion fields: {sorted(unknown)}"
        )
    ids = body["prompt"]
    if not ids or any(type(x) is not int or not 0 <= x < 2**32 for x in ids):
        raise HTTPException(400, "prompt must be one nonempty token-ID list")
    if body.get("n", 1) != 1:
        raise HTTPException(400, "Token completion adapter supports n=1")
    limit = body.get("max_tokens")
    if type(limit) is not int or limit < 1:
        raise HTTPException(400, "Positive max_tokens is required")
    minimum = body.get("min_tokens", 0)
    if type(minimum) is not int or not 0 <= minimum <= limit:
        raise HTTPException(400, "min_tokens must lie in 0..max_tokens")
    sampling = pb.SamplingParams(
        max_new_tokens=limit,
        min_new_tokens=minimum,
        n=1,
        ignore_eos=body.get("ignore_eos", False),
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    for key in (
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "repetition_penalty",
    ):
        if key in body:
            setattr(sampling, key, body[key])
    stop = body.get("stop")
    if stop:
        sampling.stop.extend([stop] if isinstance(stop, str) else stop)
    if "seed" in body:
        sampling.sampling_seed = body["seed"]
    req = pb.GenerateRequest(
        request_id=f"cmpl-{uuid.uuid4().hex}",
        tokenized=pb.TokenizedInput(input_ids=ids),
        sampling_params=sampling,
        stream=body.get("stream", False),
    )
    if "bench_cohort" in body:
        try:
            req.request_id = tagged_request_id(body["bench_cohort"])
        except CohortError as error:
            raise HTTPException(400, str(error)) from error
    if "data_parallel_rank" in body:
        req.data_parallel_rank = body["data_parallel_rank"]
    if os.environ.get("TOKENSPEED_STANDALONE_DECODE_BENCHMARK") == "reuse-cache":
        room = next(_benchmark_rooms)
        if "data_parallel_rank" in body:
            room = room * 2 + body["data_parallel_rank"]
        req.kv_bootstrap_info.CopyFrom(
            pb.KvBootstrapInfo(
                bootstrap_host="standalone-benchmark",
                bootstrap_port=1,
                bootstrap_room=room,
            )
        )
    return req


def usage(message):
    result = dict(
        prompt_tokens=message.prompt_tokens,
        completion_tokens=message.completion_tokens,
        total_tokens=message.prompt_tokens + message.completion_tokens,
        prompt_tokens_details={"cached_tokens": message.cached_tokens},
    )
    if isinstance(message, pb.GenerateComplete):
        result["completion_tokens_details"] = dict(
            accepted_prediction_tokens=message.spec_accepted_tokens,
            rejected_prediction_tokens=max(
                0, message.spec_draft_tokens - message.spec_accepted_tokens
            ),
        )
    return result


class StreamTokens:
    """Normalize engine delta/cumulative frames using cumulative token counts."""

    def __init__(self):
        self.ids = []

    def consume(self, ids, total, *, terminal):
        ids = list(ids)
        remaining = total - len(self.ids)
        if remaining < 0:
            raise RuntimeError("Engine completion count moved backwards")
        if len(ids) == total and ids[: len(self.ids)] == self.ids:
            delta = ids[len(self.ids) :]
        elif remaining == len(ids):
            delta = ids
        elif terminal and remaining == 0 and (not ids or self.ids[-len(ids) :] == ids):
            # gRPC repeats the last output frame as GenerateComplete.
            delta = []
        else:
            raise RuntimeError("Engine token IDs disagree with completion count")
        self.ids.extend(delta)
        return delta


async def token_completions(body, stub, *, served_model):
    if body.get("model") != served_model:
        raise HTTPException(404, "Requested model is not served by this engine")
    req = build_request(body)
    decoder = tokenizer(served_model)
    stream_decoder = DecodeStream(skip_special_tokens=True)
    common = dict(
        id=req.request_id,
        object="text_completion",
        created=int(time.time()),
        model=served_model,
    )
    call = stub.Generate(req)

    async def events():
        tokens = StreamTokens()
        complete = False
        try:
            async for response in call:
                kind = response.WhichOneof("response")
                if kind == "chunk":
                    chunk = response.chunk
                    ids = tokens.consume(
                        chunk.token_ids, chunk.completion_tokens, terminal=False
                    )
                    delta = stream_decoder.step(decoder, ids) or ""
                    # Preserve one event per engine chunk, even for special tokens.
                    payload = dict(
                        common,
                        choices=[
                            dict(index=0, text=delta, finish_reason=None, logprobs=None)
                        ],
                        token_ids=ids,
                        usage=None,
                    )
                    yield "data: " + json.dumps(payload) + "\n\n"
                elif kind == "complete":
                    message = response.complete
                    remaining = tokens.consume(
                        message.output_ids, message.completion_tokens, terminal=True
                    )
                    delta = stream_decoder.step(decoder, remaining) if remaining else ""
                    payload = dict(
                        common,
                        choices=[
                            dict(
                                index=0,
                                text=delta or "",
                                finish_reason=message.finish_reason,
                                logprobs=None,
                            )
                        ],
                        token_ids=remaining,
                        usage=usage(message),
                    )
                    yield "data: " + json.dumps(payload) + "\n\n"
                    if body.get("stream_options", {}).get("include_usage", False):
                        yield "data: " + json.dumps(
                            dict(common, choices=[], usage=usage(message))
                        ) + "\n\n"
                    complete = True
            if not complete:
                raise RuntimeError("Engine stream ended without completion")
            yield "data: [DONE]\n\n"
        finally:
            if not complete:
                call.cancel()

    if req.stream:
        return StreamingResponse(events(), media_type="text/event-stream")
    try:
        async for response in call:
            if response.WhichOneof("response") == "complete":
                message = response.complete
                return JSONResponse(
                    dict(
                        common,
                        choices=[
                            dict(
                                index=0,
                                text=decoder.decode(
                                    list(message.output_ids), skip_special_tokens=True
                                ),
                                finish_reason=message.finish_reason,
                                logprobs=None,
                            )
                        ],
                        usage=usage(message),
                    )
                )
        raise HTTPException(502, "Engine returned no completion")
    finally:
        call.cancel()


# Native SMG /generate accepts signed int32 input_ids. The protobuf permits
# uint32, so reject IDs that its HTTP boundary cannot represent unchanged.
def gateway_request(body, *, dp_targets):
    rank = body.get("data_parallel_rank")
    if "data_parallel_rank" in body and (
        type(rank) is not int or not 0 <= rank < 2**31
    ):
        raise HTTPException(400, "data_parallel_rank must be a nonnegative int32")
    options = body.get("stream_options", {})
    if (
        not isinstance(options, dict)
        or set(options) - {"include_usage"}
        or ("include_usage" in options and type(options["include_usage"]) is not bool)
    ):
        raise HTTPException(400, "stream_options supports only boolean include_usage")
    try:
        req = build_request(body)
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(400, f"Invalid token completion request: {exc}") from exc
    if req.HasField("kv_bootstrap_info"):
        raise HTTPException(
            400, "Synthetic bootstrap is forbidden on the PD gateway path"
        )
    if any(token >= 2**31 for token in req.tokenized.input_ids):
        raise HTTPException(400, "SMG input_ids must fit nonnegative int32")
    params = {
        key: body[key]
        for key in (
            "temperature",
            "top_p",
            "top_k",
            "frequency_penalty",
            "presence_penalty",
            "repetition_penalty",
            "stop",
            "ignore_eos",
            "seed",
        )
        if key in body
    }
    params.update(
        max_new_tokens=body["max_tokens"],
        min_new_tokens=body.get("min_tokens", 0),
        n=1,
        skip_special_tokens=True,
        spaces_between_special_tokens=True,
    )
    headers = gateway_pin_headers(body, dp_targets=dp_targets)
    return (
        req,
        dict(
            model=body["model"],
            input_ids=list(req.tokenized.input_ids),
            sampling_params=params,
            stream=True,
            rid=req.request_id,
        ),
        headers,
    )


def gateway_pin_headers(body, *, dp_targets):
    """Translate a validated domain pin to SMG's existing strict target hint."""
    if "data_parallel_rank" not in body:
        return {}
    rank = body["data_parallel_rank"]
    if type(rank) is not int or not 0 <= rank < 2**31:
        raise HTTPException(400, "data_parallel_rank must be a nonnegative int32")
    if not dp_targets:
        raise HTTPException(
            503, "Configure verified SMG DP target indices before pinning"
        )
    if str(rank) not in dp_targets:
        raise HTTPException(
            400, "data_parallel_rank is outside the configured gateway domains"
        )
    return {"X-SMG-Target-Worker": str(dp_targets[str(rank)])}


async def gateway_frames(content):
    """Read SSE records independently of HTTP packet and UTF-8 boundaries."""
    data = []
    async for raw in content:
        line = raw.decode("utf-8").rstrip("\r\n")
        if line.startswith("data:"):
            data.append(line[5:].lstrip(" "))
        elif not line and data:
            payload = "\n".join(data)
            data.clear()
            if payload == "[DONE]":
                return
            yield json.loads(payload)
    if data:
        raise RuntimeError("Gateway returned an incomplete SSE record")


def gateway_usage(meta):
    result = dict(
        prompt_tokens=meta["prompt_tokens"],
        completion_tokens=meta["completion_tokens"],
        total_tokens=meta["prompt_tokens"] + meta["completion_tokens"],
        prompt_tokens_details={"cached_tokens": meta["cached_tokens"]},
    )
    # Missing counters mean an older gateway. Do not silently report zero for
    # a speculative run: the matching SMG patch exports authoritative totals.
    accepted = meta["spec_accepted_tokens"]
    drafted = meta["spec_draft_tokens"]
    result["completion_tokens_details"] = dict(
        accepted_prediction_tokens=accepted,
        rejected_prediction_tokens=max(0, drafted - accepted),
    )
    return result


async def gateway_token_completions(
    body, *, gateway_url, served_model, dp_targets, timeout
):
    """Adapt OpenAI token prompts through SMG's real P/D orchestration.

    ``dp_targets`` explicitly maps DP ranks to verified SMG target indices.
    Configure both P/D policies as consistent_hashing. SMG owns rendezvous
    allocation, dispatch, and abort of both legs; this adapter never creates
    bootstrap metadata or contacts either engine directly.
    """
    if body.get("model") != served_model:
        raise HTTPException(404, "Requested model is not served by this gateway")
    req, native, headers = gateway_request(body, dp_targets=dp_targets)
    decoder = tokenizer(served_model)
    stream_decoder = DecodeStream(skip_special_tokens=True)
    common = dict(
        id=req.request_id,
        object="text_completion",
        created=int(time.time()),
        model=served_model,
    )
    session = aiohttp.ClientSession(timeout=timeout)
    try:
        upstream = await session.post(
            f"{gateway_url.rstrip('/')}/generate", json=native, headers=headers
        )
        if upstream.status != 200:
            detail = await upstream.text()
            raise HTTPException(upstream.status, detail)
    except asyncio.TimeoutError as exc:
        await session.close()
        raise HTTPException(504, "PD gateway timed out") from exc
    except aiohttp.ClientError as exc:
        await session.close()
        raise HTTPException(502, f"PD gateway unavailable: {exc}") from exc
    except BaseException:
        await session.close()
        raise

    async def events():
        tokens = StreamTokens()
        complete = False
        try:
            async for frame in gateway_frames(upstream.content):
                if "error" in frame:
                    raise RuntimeError(f"Gateway generation failed: {frame['error']}")
                if complete:
                    raise RuntimeError("Gateway emitted a frame after completion")
                meta = frame["meta_info"]
                reason = meta.get("finish_reason")
                terminal = reason is not None
                ids = tokens.consume(
                    frame["output_ids"], meta["completion_tokens"], terminal=terminal
                )
                delta = stream_decoder.step(decoder, ids) if ids else ""
                if isinstance(reason, dict):
                    reason = reason["type"]
                payload = dict(
                    common,
                    choices=[
                        dict(
                            index=0,
                            text=delta or "",
                            finish_reason=reason,
                            logprobs=None,
                        )
                    ],
                    token_ids=ids,
                    usage=gateway_usage(meta) if terminal else None,
                    gateway_request_id=meta["id"].removesuffix("-0"),
                    requested_data_parallel_rank=body.get("data_parallel_rank"),
                )
                yield payload
                complete = terminal
            if not complete:
                raise RuntimeError("Gateway stream ended without completion")
        finally:
            # Closing this HTTP stream triggers SMG's engine-stream abort guard
            # for both P/D legs, including prefill or transfer still in flight.
            upstream.close()
            await session.close()

    if req.stream:

        async def sse():
            records = events()
            try:
                async for payload in records:
                    yield "data: " + json.dumps(payload) + "\n\n"
                    if payload["usage"] is not None and body.get(
                        "stream_options", {}
                    ).get("include_usage", False):
                        yield "data: " + json.dumps(
                            dict(common, choices=[], usage=payload["usage"])
                        ) + "\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                yield "data: " + json.dumps(
                    {"error": {"message": str(exc), "type": "upstream_error"}}
                ) + "\n\n"
            finally:
                await records.aclose()

        return StreamingResponse(sse(), media_type="text/event-stream")
    text = []
    token_ids = []
    final = None
    try:
        async for payload in events():
            text.append(payload["choices"][0]["text"])
            token_ids.extend(payload["token_ids"])
            final = payload
    except Exception as exc:
        raise HTTPException(502, str(exc)) from exc
    final["choices"][0]["text"] = "".join(text)
    final["token_ids"] = token_ids
    return JSONResponse(final)
