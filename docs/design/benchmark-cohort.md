# Benchmark cohort admission at the decode receive boundary

This is an explicitly enabled benchmark scheduling option. It uses the ordinary
PD scheduler, transfers, model forward, speculation, landing and SSE output. A
benchmark curve using it must disclose that decode starts after the whole cohort
has received its real caches. TTFT includes this wait. This differs from the
reference benchmark's client-only fixed-batch gather.

`TOKENSPEED_BENCH_COHORT_DIR` enables the gate only on a decode node with two
contiguous TP8 domains (global ranks 0..15). Leave it unset on prefill. Use a fresh,
absolute, private directory for every decode service lifetime, shared by its 16
local worker processes. Persist the directory with the benchmark evidence. The
control API is `/bench/cohort/{arm,status,release,abort}` on the ordinary D control
endpoint. The token-ID completion gateway accepts an explicit `bench_cohort`
object containing `epoch` and `cohort_id`; ordinary requests, including output1
prefix seeds and cold requests, have no tag and follow the existing path.

## Boundary and ownership

The adapter encodes the tag in its original request-ID namespace
`cmpl-bench.<epoch>.<cohort_id>.<uuid>`. SMG still creates the real bootstrap room
and may append its existing opaque suffix. No bootstrap or token timestamp is
synthesized. Input IDs are hashed as little-endian uint32 without tokenization.
Before receiver registration, each TP leader claims the actual rid/room against
a service-lifetime registry; previous groups cannot reuse either identity.

`DisaggDecodeExecutor` retains the real destination manifest prefix length as
`d_hit_tokens`, separately from the prefill metadata's `p_cached_tokens`. After
ordinary transport polling and its existing TP success reduction,
`PdTransferHooks` holds `RemotePrefillDoneEvent`. It leaves the popped metadata in
the executor's pending dictionaries. The C++ scheduler remains in
`RemotePrefilling`, whose leased pages are pinned. No planned forward is removed
or discarded. Admission, real transfer submission and polling continue.

Only after explicit release does the hook pass the original success event to
its original handler. That handler consumes metadata, publishes the bootstrap
token, submits the ordinary ordered landing and advances the scheduler through
the existing tail call. Thus both first-token visibility and the first D forward
follow release. There is no new model math, shared loading of a batch's KV, or
preloading of a private suffix.

## Agreement and timing

A group declares B expected input hashes in each actual domain, B in
1/2/4/8/16/32, exact input131072/output1024 and expected P/D prefix117964. Each
actual worker writes its own atomic report with real PID, rank, host, boot ID,
rid, room, input hash, and local receive time. Release requires all 16 distinct
worker PIDs, exactly the same B rid/room/hash bindings across each TP8 domain,
unique rooms across both domains, real receive success and both exact hit counts.
Configured ranks or desired cache counts are insufficient.

Control operations use a local file lock and atomic JSON replacement. A release
certificate contains the actual 16 reports and a D-host
`release_decision_monotonic_ns`. Its SHA256 covers canonical JSON (sorted keys,
compact separators, `allow_nan=False`), excluding only `certificate_sha256`.
The certificate and each group's configuration are immutable across later arms.

Workers agree within TP8 through a leader decision broadcast at matching control
points: idle arm observation, explicit request binding and held-event polling.
They do not independently inspect release files to decide whether to participate
in a collective. After release there are no gate control reads or broadcasts
during decode; completion only writes terminal acknowledgments on the ordinary
commit path. Existing scheduler/PD collectives remain unchanged.

Audit uses the existing per-process JSONL emitter: `cohort_bind`,
`cohort_receive`, `cohort_release`, and `cohort_abort`. Receive/release use real
P and D hit fields; bind may carry null hits. Release includes certificate SHA
and decision time. Compare D-host monotonic clocks only: all receives precede
the decision, and each leader's local release and the decision precede its first
`d_step.start_ns`. Client clocks are not comparable to D-host monotonic clocks.

## Abort, cancellation and reuse

Before release, cancellation, deadline, cache mismatch or transport failure
aborts the entire group. Completed held success events become original
`PD.FailedEvent` only after their real transport success fence; pending WRITEs
remain pinned until the original receiver reports safe completion/failure.
The existing receiver's unconfirmed-WRITE exception still retires the worker;
the gate does not turn that fatal path into a page-recycling failure event.
Metadata for completed aborted receives is explicitly cleared, and a late
success after abort can never enter decode.

Release decision is the irreversible admission boundary. An abort marker never
changes an existing certificate. Already released ranks do not poll the control
directory during decode, so cancellation after decision also requires cancelling
all outstanding HTTP requests and draining the normal completion path. That
benchmark point fails. It cannot claim rollback of forwards already running in
another domain. A new arm is rejected until all actual workers report terminal
cleanup and no live bindings; epoch/cohort identities cannot be reused.

The token adapter and management routes are reviewed separately from cohort
release scheduling. Deployment must restrict benchmark management endpoints
to the benchmark control plane; their disabled defaults are retained.
