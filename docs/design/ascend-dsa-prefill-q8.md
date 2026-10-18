# Ascend DSA Prefill Q8 (candidate r3: r2 kernels, serving wiring fix)

Select explicitly with `--attention-backend ascend_dsa_prefill_q8`. Baseline
`dsa` and Decode keep their original leaves; model QCP stays 1, KVP/head-TP 8.
Only six DSA attention leaves change their temporary query layout. KDA/MoE,
persistent cache placement, cache publication and PD metadata remain shared.
The same registry name requires a frozen source/delta identity to distinguish
r1 (full-history TND LI) from r2 (original PA/KVP selection). r1 failure data
and source remain frozen.

## Selection and causal layout

r2 reuses the original PA `index_partial`, including BF16 exported scores,
per-query owner-local causal/initial/window counts, local candidate ordering,
and rank-major BF16 `torch.topk`. It exchanges candidate values and request-
logical INT32 ids to each Q8 owner. The existing owner-major history plan maps
compact local token ids to request token offsets; no physical-page arithmetic
or new inverse table is used. Padding is -inf/-1 and negative ids stay -1.
`topk(out=)` uses fixed storage; original-class device witnesses must still
verify tie ordering against ordinary baseline `topk`.

Each request shard [a,b) with history H gives native TND SFA Q=b-a and KV=H+b,
so its first query sees H+a+1. Separate requests are repacked independently.
All ranks gather the same request histories in the same order, including
owners with empty local Q. There is no full-history index-K gather or scratch.
Native SFA uses contiguous BF16 K512/RoPE64, original scale and INT32 cumulative
ends without leading zero. The original native adapter is retained.

The current host page plan is expanded into contiguous NumPy INT64 arrays and
exposed to PyTorch with `from_numpy` before the existing blocking workspace
copies. This avoids per-token Python integers and list-to-tensor conversion.
Every forward still rebuilds its actual page identities, owner counts and
stable owner-major order, including partial final pages and empty owners.
There is no cross-forward page-plan cache, new device Sort or extra collective;
the device workspace budget and copy completion/lifetime contract are unchanged.

A whole pure-extend batch with total query T<8 explicitly dispatches to the
complete original `super().forward_extend`. This is a documented dispatch
policy, not a tolerance change. T>=8 uses Q8 even when some ranks/requests are
empty. Decode still follows the inherited path. Mixed extend/decode and pooled
index keys are rejected for this candidate.

## Bounded storage

`AttnConfig.prefill_query_workspace_tokens` is candidate-only and must be a
positive value. The serving registry copies the actual `max_prefill_tokens`
**after** restoring the requested full-attention leaf beneath the Hybrid wrapper;
reading the temporary `hybrid_linear_attn` name would miss this reservation.
The ordinary and lazy Lite recipes both include this fixed workspace before
cache arena sizing. Use `--max-prefill-tokens 8192 --chunked-prefill-size 8192`
for the initial geometry; a 16K run requires both values explicitly set to 16384.
The history rows follow the physical attention context (logical model context
plus speculative padding), rounded to kernel pages. `max_bs` follows the
per-DP attention capacity, so global `max_num_seqs=64` with DP2 gives 32.
The byte examples below explicitly use max_bs64/context133120; they are not a
replacement for the actual service's plan/allocation receipt.

In addition,
standalone fixtures must set it explicitly (16K probes require >=16384).
Exceeding it raises rather than allocating an unbounded selection workspace.
Baseline None means no candidate workspace and no new allocation requirement.

The recipe accounts for packed BF16 history, padded gather staging, split
K/RoPE, four existing INT64 host-plan vectors copied to device, native ends,
and fixed PA selection exchange/order/topk/map buffers. One leaf reuses it
across layers. At context133120/max_bs64/maxQ8192 this is 1,082,532,352 bytes per
rank (1032.3833 MiB); maxQ16384 gives 1,430,659,584 bytes (1364.3833 MiB).
Native PA output indices/values are **additional activation storage**:
Q*2048*(4+2), 96/192 MiB at Q8192/16384. Q head/token reshuffle tensors,
per-forward final O/reshuffle tensors, native LI/SFA workspace and ordinary
activations are not all included in the fixed count. Allocation receipts and
real peak measurements must retain this distinction; a fixed count is not a
claim that the entire forward fits only that many bytes.

## Communication and validation

Old BF16 values A2A remains. New INT32 id A2A replaces the old global-position
AG; equal-sized uniform Q8 off-rank payload is 56 MiB/rank at Q8192/K2048.
Original packed-Q AG and FP32 partial-O/LSE exchanges are replaced by query
A2A, BF16 final-O A2A and real KV-history gather. No theoretical tensor payload
is presented as actual HCCL time or physical NIC bytes.

`_select_kvp_prefill_indices` returns the same formal indices, local PA
indices/values and local metadata consumed by attention. Optional debug
observer only reads this execution; it performs no extra LI invocation and
never substitutes indices. Its `dispatch` is `kvp_bf16_q8` or `baseline_tiny`.
The original PA path uses return_value=True by design; the r1 TND LI False/True
diagnostic is not an r2 algorithm.

Host tests replace only unavailable network/device calls and exercise actual
plans/tensors, mapping, padding, ties, fragmented pages and empty owners.
They do not establish NPU numerical correctness. Fresh8 must compare original
frozen-class selection witnesses and formal observer-off outputs at the
original rtol4e-3/atol1e-3. Same indices do not guarantee SFA/SFAD equality:
r1 tiny-Q3 exposed different BF16 probability rounding schedules. If that
occurs for a non-tiny candidate, preserve the strict failure and investigate;
do not change inputs, thresholds or silently choose another algorithm.
