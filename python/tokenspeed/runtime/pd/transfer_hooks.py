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

"""EventLoop-side PD transfer-event integration.

The KV transfer executors (prefill/decode) DECIDE — they own the senders /
receivers and surface progress as PD events. ``PdTransferHooks`` here ACTS on
those events with the event loop's collaborators (output processor, data
plane) and returns the enriched event list for the loop to advance the
scheduler with — feedback into the scheduler stays an explicit
``advance_scheduler`` call in the loop body.
"""

from __future__ import annotations

import hashlib
import os
import struct

from tokenspeed_scheduler import PD

from tokenspeed.runtime.pd.audit import emit_pd_audit
from tokenspeed.runtime.pd.bench_cohort import (
    CohortError,
    CohortGate,
    CohortStore,
    request_tag,
)
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor


class PdTransferHooks:
    """Integrate PD events and the explicitly enabled benchmark hold queue.

    A cheap no-op when PD is disabled (``kv_transfer=None``).
    """

    def __init__(self, loop, device) -> None:
        self._loop = loop
        # Injected, not reached through the loop — see PauseHooks.
        self._device = device
        self._cohort: CohortGate | None = None
        directory = os.environ.get("TOKENSPEED_BENCH_COHORT_DIR")
        if directory and isinstance(loop.kv_transfer, DisaggDecodeExecutor):
            import torch.distributed as dist

            mapping = loop.server_args.mapping
            if (
                mapping.world_size != 16
                or mapping.attn.tp_size != 8
                or mapping.attn.dp_size != 2
            ):
                raise CohortError(
                    "Cohort gate supports only D TP8 DP2 on one 16-rank host"
                )

            def sync(payload):
                values = [payload]
                dist.broadcast_object_list(
                    values,
                    src=mapping.attn.tp_group[0],
                    group=loop.kv_transfer.gloo_group,
                )
                return values[0]

            def emit(event, **fields):
                # dp_rank is already part of each actual request binding.
                fields.pop("dp_rank", None)
                emit_pd_audit(
                    event,
                    role="decode",
                    global_rank=loop.global_rank,
                    dp_rank=loop.dp_rank,
                    **fields,
                )

            self._cohort = CohortGate(
                CohortStore(directory),
                global_rank=loop.global_rank,
                dp_rank=loop.dp_rank,
                sync=sync,
                emit=emit,
            )

    def admit_cohort_request(self, rid, state, bootstrap) -> bool:
        """Bind explicit benchmark requests before any destination pages are leased."""
        try:
            if request_tag(rid) is None:
                return True
            if self._cohort is None:
                raise CohortError("D benchmark cohort gate is disabled")
            ids = state.prompt_input_ids
            input_hash = hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest()
            self._cohort.bind(
                rid,
                bootstrap.bootstrap_room,
                input_hash,
                input_tokens=state.input_length,
                output_tokens=state.sampling_params.max_new_tokens,
            )
            return True
        except CohortError as error:
            state.set_finish_with_abort(
                f"Benchmark cohort admission rejected: {error}", notify_client=True
            )
            self._loop.output_processor.publish_finished_at_admission(rid, state)
            return False

    def observe_committed(self, request_ids):
        if self._cohort is not None:
            self._cohort.committed(
                request_ids, self._loop.output_processor.rid_to_state
            )

    def record_prefill_usage(self, request_ids: list[str]) -> None:
        """Forward committed host-side usage before remote-decode dispatch.

        Runs on the commit path, so a final chunk's bootstrap token (and its
        logprob, when the request returns logprobs) is already on the state;
        the remote decode that ships them is planned no earlier than the next
        round.
        """
        loop = self._loop
        if not isinstance(loop.kv_transfer, DisaggPrefillExecutor):
            return
        for request_id in request_ids:
            state = loop.output_processor.rid_to_state.get(request_id)
            if state is None:
                continue
            loop.kv_transfer.record_cached_tokens(request_id, state.cached_tokens)
            if state.output_token_logprobs_val:
                loop.kv_transfer.record_bootstrap_logprob(
                    request_id, state.output_token_logprobs_val[0]
                )

    def poll_transfer_events(self) -> list:
        """Poll the KV transfer executor, act on its events, and return the
        (possibly enriched) event list for the scheduler advance. Empty when
        PD is disabled.
        """
        loop = self._loop
        if loop.kv_transfer is None:
            return []

        events = loop.kv_transfer.generate_events()
        gate = self._cohort
        if gate is not None:
            if not loop.output_processor.rid_to_state and not gate.holding:
                gate.idle()
            passed = []
            for event in events:
                if isinstance(event, PD.RemotePrefillDoneEvent):
                    if event.request_id in gate.bindings and gate.receive(
                        event.request_id,
                        event,
                        loop.kv_transfer.bench_completion_context(event.request_id),
                    ):
                        continue
                elif isinstance(event, PD.FailedEvent):
                    gate.fail(event.request_id, "Actual PD transfer failed")
                    loop.kv_transfer.discard_remote_completion(event.request_id)
                passed.append(event)
            cancelled = {
                rid
                for rid in gate.bindings
                if rid in loop.output_processor.rid_to_state
                and loop.output_processor.rid_to_state[rid].to_abort
            }
            for event, aborted in gate.tick(cancelled):
                if aborted:
                    loop.kv_transfer.discard_remote_completion(event.request_id)
                    event = PD.FailedEvent(event.request_id)
                passed.append(event)
            events = passed

        processed = []
        for event in events:
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                loop.kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(loop.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token
                cached_tokens = loop.kv_transfer.pop_remote_cached_tokens(req_id)
                bootstrap_logprob = loop.kv_transfer.pop_remote_bootstrap_logprob(
                    req_id
                )
                state = loop.output_processor.rid_to_state.get(req_id)
                if state is None or not state.to_abort:
                    loop.output_processor.on_remote_prefill_done(
                        req_id, bootstrap_token, cached_tokens, bootstrap_logprob
                    )
                processed.extend(
                    loop.output_processor.finish_remote_prefill_only_request(req_id)
                )
                if isinstance(loop.kv_transfer, DisaggDecodeExecutor):
                    remote_cache_slot = loop.kv_transfer.pop_remote_cache_slot(req_id)
                    candidate_info = loop.kv_transfer.pop_remote_spec_candidate_ids(
                        req_id
                    )
                    remaining_state = loop.output_processor.rid_to_state.get(req_id)
                    still_decoding = (
                        remaining_state is not None
                        and not remaining_state.to_abort
                        and not remaining_state.finished
                    )
                    # Both are device writes read by the request's first local
                    # decode, so they cross as one ordered submission rather
                    # than as two calls from this handler.
                    self._device.run_remote_prefill_landing(
                        candidate_info,
                        remote_cache_slot if still_decoding else None,
                    )
            elif isinstance(event, PD.FailedEvent):
                # A PD/EPD transfer failed: the decode KV receiver timed out (e.g. the
                # prefill aborted on embedding timeout so the KV never arrives), or a
                # transfer errored. Publish the client-visible failure here; the C++
                # FailedEvent handler atomically terminalizes and fences the leased
                # scheduler resources, so no Forward.Abort needs to follow.
                req_id = event.request_id
                state = loop.output_processor.rid_to_state.get(req_id)
                if state is not None:
                    if state.finished:
                        loop.output_processor.reap_finished_orphan(req_id, state)
                    else:
                        state.set_finish_with_abort(
                            "PD/EPD remote transfer failed or timed out"
                        )
                        loop.output_processor.publish_finished_at_admission(
                            req_id, state
                        )
        return processed
