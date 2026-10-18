# Ascend DSA context-parallel execution

Each KVP rank owns pages of the KV and indexer caches. The local indexer
returns candidate positions and scores; scores are exchanged, global TopK
positions are gathered, then each rank selects its own candidates. Query
AllGather is independent of this global selection. The backend joins it
after selection and before sparse attention consumes the gathered query.
Eager and capture warmup retain the same data dependencies.

The optional native libraries below are loaded during backend construction,
before graph capture. Omitting a library preserves its existing implementation.
Model geometry is checked before dispatch; other shapes use the original
operation.

| Environment variable | Operation and admitted geometry |
| --- | --- |
| `TOKENSPEED_DSA_TOPK_LIBRARY` | INT32 global TopK from contiguous BF16/FP32 scores `[8,16384]`, k=2048 |
| `TOKENSPEED_DSA_MERGE_LIBRARY` | Merge contiguous FP32 partial output `[8,R,512]` and LSE `[8,R]`, 1≤R≤4096, into BF16 `[R,512]` |
| `TOKENSPEED_DSA_RANKMAJOR_LIBRARY` | Global TopK directly from contiguous BF16 `[8,8,2048]` rank-major scores, sorted per rank/query, k=2048 |

The INT32 path uses CANN's TopKV2 primitive through a capture-safe executor.
It retains sorted output and the existing tie behavior, avoiding the public
TopK wrapper's INT64 conversion. Its descriptors use the actual two-dimensional
storage shape, including for contiguous views with a nonzero storage offset.

The optional rank-major path uses the descending-score contract of
LightningIndexer. Seven two-way merges retain the best 2048 score/index pairs,
including padding and forced initial/local candidates. It returns INT32 global
candidate IDs directly, avoiding both transpose and casts. The caller must
explicitly pass `sorted_per_rank=True`; unsupported inputs use transpose plus
the ordinary global TopK helper. This direct launch also requires
`TASK_QUEUE_ENABLE=0`.

Merge processes four rows per batch with two input buffers. LSE weights and
rank accumulation retain the registered operation's order, including zero
weights and empty-rank NaNs. The merge library uses a direct kernel launch
and requires `TASK_QUEUE_ENABLE=0`.

For native page metadata and write-slot conversion, see
`tokenspeed-kernel/python/tokenspeed_kernel/ops/kvcache/README.md`. Those
results are shared across layers only within the same forward, not cached
across changing requests.
