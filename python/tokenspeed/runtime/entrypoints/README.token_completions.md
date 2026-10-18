# Token-ID completions for the isolated NPU benchmark

`POST /v1/completions` accepts one nonempty token-ID prompt and forwards IDs unchanged to the existing gRPC engine. Text prompts retain the gateway route. The endpoint uses the model tokenizer only to decode generated output.

Streaming normalizes both cumulative and delta engine frames against `completion_tokens`. The gRPC terminal message may repeat the last delta; it is not counted or decoded twice. One SSE choice frame is emitted per engine chunk, including an empty decoded text. The terminal choice carries usage, and `include_usage` also emits the standard usage-only frame. Client cancellation cancels the gRPC call.

The isolated benchmark supports single completion, explicit output limits, ignore-EOS, sampling parameters, and DP rank for prewarming each cache domain. Unsupported fields fail with HTTP 400. Request IDs, cache counters, and speculative accounting come from the engine. No prompt is decoded and re-tokenized.

Validation is recorded in the task artifact `scripts/test_token_stream.py` and the full Lazy graph HTTP probe. This adapter is an experiment on the pinned mainline worktree, not an upstream change.


## Real PD gateway

Without the explicit standalone benchmark environment, token prompts go through
SMG's native `/generate` HTTP endpoint. Input IDs remain unchanged; only generated
IDs are decoded. SMG allocates the real bootstrap room and dispatches the same
bootstrap metadata to both gRPC legs. The adapter never supplies a bootstrap
host/port/room or generates a bootstrap token.

For a single P engine and single D engine, both with attention DP2, configure
SMG `--pd-disaggregation --dp-aware --prefill-policy consistent_hashing
--decode-policy consistent_hashing --disable-retries`. Both TS controllers route
all PD requests by `bootstrap_room % dp_size`, including under load-budget and
single-worker policies. Use P TP8/KVP8/DCP1/QCP1/EP16/DP2 and D
TP8/KVP8/DCP8/EP16/DP2; the HTTP adapter does not change model topology.

A token request's `data_parallel_rank` maps to SMG's existing strict
`X-SMG-Target-Worker` index. Before pinned requests, verify the index-to-domain
mapping against both engine route logs and set
`TOKENSPEED_TOKEN_GATEWAY_DP_TARGETS='{"0":0,"1":1}'` in the control server's
process. This explicit mapping is scoped to that verified gateway membership;
recheck after restart or worker membership changes. Missing or invalid mappings
fail instead of dropping the pin. The control server also translates native `/generate` requests carrying
`data_parallel_rank` using this map. Callers contacting SMG directly must set
the same header. The wire's `data_parallel_rank` alone is insufficient in PD.

SMG `1.11.0.post20261007` accepts native token IDs but omits speculative counters
from native generate SSE terminal metadata. Apply the accompanying SMG patch to
export `spec_accepted_tokens` and `spec_draft_tokens`; the adapter rejects missing
terminal accounting rather than fabricating zero. Downstream non-streaming
requests use this same upstream streaming protocol and collect the final result.
The native terminal's repeated tail token is removed using cumulative counts.

SSE includes `gateway_request_id` for correlation with engine `rid` (SMG adds an
attempt suffix; the adapter removes the native single-choice `-0` suffix) and `requested_data_parallel_rank` as the requested domain, not
an observation of actual routing. The controller's debug route records include
request ID, room, role, and actual DP rank. Confirm both P/D roles in integration.

Client disconnect closes the gateway stream; SMG owns the two engine abort
guards and defers a decode abort during handoff until its first response.
Upstream failures and timeouts are not retried. An incomplete/error stream emits
an error frame without a success `[DONE]` marker. The inherited control-server
connection/inactivity timeouts remain in force.

Host verification: `test/runtime/test_pd_token_gateway.py` uses real HTTP and
protobuf messages with only the unavailable model/network upstream replaced.
The task's `smg_network_probe.py` also runs the fixed real SMG binary against
CPU gRPC model workers. These tests do not validate device cache transport,
NPU state handoff, or end-to-end 128K performance.
