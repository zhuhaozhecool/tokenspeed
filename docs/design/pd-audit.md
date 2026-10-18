# Optional PD transfer audit

Set `TOKENSPEED_PD_AUDIT_DIR` explicitly to emit per-process JSONL transfer
events. With no directory configured the writer is disabled. Sequence numbers
are serialized across threads; transfer completion counts each plan once.
This PR introduces the writer and tests; runtime call sites are separate.

Runtime events record routing, cache matches, selected owner payload and host
forward-call intervals. They are emitted only with the audit directory set.
Host call intervals are not device kernel timelines or HBM counters.
