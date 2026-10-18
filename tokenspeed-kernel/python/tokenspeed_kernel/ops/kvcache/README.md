# KV cache kernels

## Range clearing

`kvcache.zero_byte_ranges` dispatches by physical placement. Device allocations
use the original Triton helper; pinned Host allocations use
`cute_dsl_zero_host_byte_ranges` on the execution device's current stream. CPU
arenas clear synchronously. Host callers retain backing storage until stream
completion, including before a CPU read. Offsets are relative to the tensor view;
the CuTe kernel uses int64 byte addressing and a bounded grid that strides each
range, preserving unaligned tails and neighboring bytes. Range counts, lengths,
and pointers are dynamic arguments and do not trigger new compilations.

This scheduler/lifecycle operation consumes a CPU range list outside CUDA graph
capture. Capturing it fails explicitly rather than retaining a temporary pinned
metadata pointer in a graph. Model forward graph replay is unchanged.

## Sparse KV offloading

On Ascend, page-field clearing bounds the flattened launch grid to 65535
programs and strides over additional page/field pairs. Long-context admission
therefore clears every requested page without exceeding the runtime launch
limit. Page/field counts remain runtime parameters.

`offload.py` is the runtime boundary. NVIDIA uses the registered
`cute_dsl_offload_materialize` implementation and CuTe DSL row helpers.
The private `_cute_dsl` Python package ships in both wheels and source archives.
NVIDIA tests check the GPU vendor before importing CuTe; ROCm also exposes
`torch.cuda`, so GPU availability alone does not identify this backend.
It accepts physical history IDs (positive int32; zero/negative are invalid),
request IDs (int32 or scheduler int64), persistent hot tags/LRU and preallocated
scratch. An invocation resolves the whole request's Q×K selection union.
Original order, duplicate references and invalid masks survive in the output.

One CTA per request constructs an open-addressed selection hash. Atomic minimum
of original indices makes duplicate ownership deterministic. After a CTA barrier,
resident scanning selects the minimum physical slot for duplicate tags, protects
selected/current ordinary rows and stably partitions the old LRU. Unique misses
use first-occurrence order and the oldest unprotected slots. Before publishing
tags, an unconditional device trap rejects capacity overflow; probes are bounded.
The new LRU is remaining old slots, installed misses, then protected old slots.
Reserved/ring rows participate in lookup but cannot be victims.

The hash uses two int32 arrays at a load factor ≤0.5. `hash_geometry` selects
shared tables up to 128 KiB, and budgeted global tables above that threshold.
Global `entry_dest` separates canonical destinations from the final output.
Miss pairs occupy their canonical input positions, with other entries masked;
the static copy grid therefore requires no CPU read of the miss count.

A separate row-parallel CuTe kernel copies mapped pinned Host bytes. Integer
32-bit transfers are used for aligned rows, with a byte path for arbitrary tails.
Current/seed/accepted/reset helpers also use CuTe. The caller records ready only
after both resolver and payload copy, and joins write-through before reusing
current slots. Shared tables are rebuilt on every replay; padding rows never
write persistent tags or LRU.

Compiled executors are cached by device, static geometry and pointer element
types. Batch size, pointers, table width/stride and sequence/accept counts remain
runtime values. Warmup precedes graph capture; an unwarmed capture fails explicitly.
The runtime also warms admission's int64 request reset before serving.
The residency path uses only CuTe DSL. The former resident/selection sorting,
binary-search lookup and adjacent-entry miss deduplication have been removed.
Hash ownership merges duplicate loads while preserving all attention references;
upstream top-k ordering remains part of the attention numerical contract.

### Ascend DSA metadata

`compact_kvp_dsa_metadata` refreshes owned kernel pages and the sequence,
initial-window and local-window token counts in one asynchronous AscendC
launch. It writes caller-owned INT32 buffers in place and retains allocated
reserve pages in source order. Every Graph replay reads the current virtual
page table and endpoints; no request-dependent state is cached on the host.

Build `cann-flash-ops/experimental/kvp_metadata_v1` from the Ascend kernel
package and set `TOKENSPEED_DSA_METADATA_LIBRARY` to the resulting
`libkvp_metadata.so` before warmup. This explicit library selection also
enables native compaction in `compact_kvp_pages`; without it the existing
Ascend implementation remains available. Warm up the selected library before
Graph capture.

`kvp_metadata_v2` also builds `libkvp_metadata_v2.so` with native cache-write
slot translation. Select it with `TOKENSPEED_DSA_SLOTS_LIBRARY`; the same
library can serve `TOKENSPEED_DSA_METADATA_LIBRARY`. A virtual slot maps to
its owning rank's physical slot, or -1 for null, out-of-range and foreign
slots. This sentinel is local to Ascend scatter; the common cache-placement
API still returns a separate owner mask. INT32 and INT64 slots are supported.

Both direct-launch helpers require `TASK_QUEUE_ENABLE=0` and warmup before
capture. The runtime memoizes outputs only within one forward; metadata
refresh and pool rebinding invalidate them. Replay recomputes current page
ownership, lengths and slots at stable output addresses.
