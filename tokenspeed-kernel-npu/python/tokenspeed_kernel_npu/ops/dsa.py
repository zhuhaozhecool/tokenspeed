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

"""Low-level Ascend BF16 DSA operator adapters."""

from __future__ import annotations

import os

import torch
import torch_npu

from tokenspeed_kernel_npu._triton import tl, triton


@triton.jit
def _scatter(
    X,
    Y,
    Loc,
    WIDTH: tl.constexpr,
    CAP: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.arange(0, BLOCK)
    loc = tl.load(Loc + row).to(tl.int64)
    x = tl.load(X + row * WIDTH + off, off < WIDTH, other=0)
    # Null page padding must not race on or corrupt live cache data.
    tl.store(Y + loc * WIDTH + off, x, (off < WIDTH) & (loc >= PAGE) & (loc < CAP))


def _composite_interleave_rope(tensor, positions, cos_sin_cache, *, rope_dim):
    """Run the one-sided interleaved RoPE used by FluentLLM's indexer."""
    if tensor.shape[0] == 0:
        return tensor
    rotary = tensor[..., :rope_dim]
    tail = tensor[..., rope_dim:]
    cos_sin = cos_sin_cache.index_select(0, positions.flatten())
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.repeat(1, 2).view(-1, 1, 1, rope_dim)
    sin = sin.repeat(1, 2).view(-1, 1, 1, rope_dim)
    rotated = torch_npu.npu_interleave_rope(
        rotary.reshape(rotary.shape[0], -1, 1, rope_dim),
        cos,
        sin,
    ).reshape_as(rotary)
    if tail.shape[-1] == 0:
        return rotated
    return torch.cat((rotated, tail), dim=-1)


@triton.jit
def _interleave_rope_fused(
    X, P, C, Y, heads, s0, s1, s2, cs0, cs1, ps0, BH: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1) * BH + tl.arange(0, BH)
    lane = tl.arange(0, 64)
    pos = tl.load(P + row * ps0).to(tl.int64)
    # Read each rotary row contiguously, then split adjacent pairs in UB.
    # Keeping the head axis avoids a per-lane gather from global memory.
    rot = tl.load(
        X + row * s0 + head[:, None] * s1 + lane[None, :] * s2,
        head[:, None] < heads,
        other=0,
    ).to(tl.float32)
    even, odd = tl.split(tl.reshape(rot, (BH, 32, 2)))
    pair = tl.arange(0, 32)
    c = tl.load(C + pos * cs0 + pair * cs1).to(tl.float32)
    s = tl.load(C + pos * cs0 + (pair + 32) * cs1).to(tl.float32)
    # CANN maps adjacent pairs into the two halves of the output.
    lo = even * c[None, :] - odd * s[None, :]
    hi = odd * c[None, :] + even * s[None, :]
    tail = tl.load(
        X + row * s0 + head[:, None] * s1 + (lane[None, :] + 64) * s2,
        head[:, None] < heads,
        other=0,
    )
    tl.store(
        Y + (row * heads + head[:, None]) * 128 + pair[None, :],
        lo,
        head[:, None] < heads,
    )
    tl.store(
        Y + (row * heads + head[:, None]) * 128 + pair[None, :] + 32,
        hi,
        head[:, None] < heads,
    )
    tl.store(
        Y + (row * heads + head[:, None]) * 128 + lane[None, :] + 64,
        tail,
        head[:, None] < heads,
    )


def _can_fuse_interleave_rope(x, positions, table, *, rope_dim):
    # Specialize the measured decode layouts. Other geometries/strides keep
    # the existing CANN path rather than relying on unsupported UB tiling.
    if (
        x.ndim not in (2, 3)
        or rope_dim != 64
        or x.shape[-1] != 128
        or x.dtype not in (torch.bfloat16, torch.float16)
        or positions.ndim != 1
        or positions.stride(0) != 1
        or positions.numel() != x.shape[0]
        or x.stride(-1) != 1
        or table.ndim != 2
        or table.shape[1] != rope_dim
        or table.stride(-1) != 1
        or (x.ndim == 3 and x.shape[1] not in (1, 4, 8, 16, 32, 64))
    ):
        return False
    heads = 1 if x.ndim == 2 else x.shape[1]
    # One program per token becomes costly for many rows with few heads.
    # Keep the ranges that improve both full-core and half-core measurements.
    max_tokens = 1024 if heads == 64 else 512 if heads >= 8 else 256
    return x.shape[0] <= max_tokens


def _launch_fused_interleave_rope(x, positions, table):
    rows = x.shape[0]
    heads = 1 if x.ndim == 2 else x.shape[1]
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if rows == 0:
        return out
    bh = min(64, triton.next_power_of_2(heads))
    _interleave_rope_fused[(rows, triton.cdiv(heads, bh))](
        x,
        positions,
        table,
        out,
        heads,
        x.stride(0),
        0 if x.ndim == 2 else x.stride(1),
        x.stride(-1),
        table.stride(0),
        table.stride(1),
        positions.stride(0),
        bh,
        enable_fp_fusion=False,
    )
    return out


def interleave_rope(x, positions, table, *, rope_dim):
    if _can_fuse_interleave_rope(x, positions, table, rope_dim=rope_dim):
        return _launch_fused_interleave_rope(x, positions, table)
    else:
        return _composite_interleave_rope(x, positions, table, rope_dim=rope_dim)


class AscendDSAKernels:
    def __init__(self):

        library = os.environ.get("TOKENSPEED_LONGCAT_DSA_LIBRARY")
        if library:
            torch.ops.load_library(library)
        self._index = getattr(torch.ops.custom, "npu_lightning_indexer", None)
        self._attention = getattr(
            torch.ops.custom, "npu_sparse_flash_attention_decode", None
        )
        self._select_local = getattr(
            torch.ops.custom, "npu_select_local_topk_indices", None
        )
        self._merge_partials = getattr(
            torch.ops.custom, "npu_kvp_attention_merge", None
        )
        if self._index is None or self._attention is None:
            raise RuntimeError(
                "LongCatDSA needs LightningIndexer and "
                "SparseFlashAttentionDecode operators"
            )
        self._prefill_attention = None
        merge_library = os.environ.get("TOKENSPEED_DSA_MERGE_LIBRARY")
        self._merge_partials_native = None
        if merge_library:
            if os.environ.get("TASK_QUEUE_ENABLE", "1") != "0":
                raise RuntimeError("Native DSA merge requires TASK_QUEUE_ENABLE=0")
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError(
                    "Load the DSA merge library before NPU Graph capture"
                )
            torch.ops.load_library(merge_library)
            self._merge_partials_native = torch.ops.kvp_attention_merge_native.merge
        topk_library = os.environ.get("TOKENSPEED_DSA_TOPK_LIBRARY")
        self._global_topk_native = None
        if topk_library:
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Load the DSA TopK library before NPU Graph capture")
            torch.ops.load_library(topk_library)
            self._global_topk_native = torch.ops.dsa_topk_int32_native.topk32
        rankmajor_library = os.environ.get("TOKENSPEED_DSA_RANKMAJOR_LIBRARY")
        self._global_topk_rankmajor_native = None
        if rankmajor_library:
            if os.environ.get("TASK_QUEUE_ENABLE", "1") != "0":
                raise RuntimeError(
                    "Native rank-major TopK requires TASK_QUEUE_ENABLE=0"
                )
            if torch.npu.is_current_stream_capturing():
                raise RuntimeError("Load rank-major TopK before NPU Graph capture")
            torch.ops.load_library(rankmajor_library)
            self._global_topk_rankmajor_native = (
                torch.ops.dsa_rankmajor_topk_native.topk
            )
        self._full_scan_chunks = {}
        self._selection_windows = {}

    interleave_rope = staticmethod(interleave_rope)

    def require_prefill(self):
        attention = getattr(torch.ops.custom, "npu_sparse_flash_attention", None)
        if attention is None:
            raise RuntimeError("Ascend DSA Prefill Q8 needs SparseFlashAttention")
        self._prefill_attention = attention

    def prefill_index(
        self,
        query,
        key,
        weights,
        q_ends,
        kv_ends,
        *,
        topk,
        initial,
        local,
        return_scores,
    ):
        """Original BF16 LightningIndexer ABI, contiguous TND history.

        The caller has cropped each request's KV to this query shard's end.
        End arrays are INT32 cumulative ends, without a leading zero.
        """
        if query.dtype != torch.bfloat16 or key.dtype != torch.bfloat16:
            raise ValueError("Prefill Q8 requires BF16 index query/key")
        if weights.dtype != torch.bfloat16:
            raise ValueError("Prefill Q8 must preserve BF16 scoring weights")
        if q_ends.dtype != torch.int32 or kv_ends.dtype != torch.int32:
            raise ValueError("native Prefill Indexer ends must be INT32")
        if query.ndim != 3 or key.ndim != 3 or key.shape[1] != 1:
            raise ValueError("Prefill Indexer requires TND query and single-head key")
        if initial < 0 or local < 0 or initial + local > topk:
            raise ValueError("Prefill initial/local windows exceed top-k")
        window_key = (query.device, q_ends.numel(), query.shape[0], initial, local)
        windows = self._selection_windows.get(window_key)
        if windows is None:
            windows = (
                torch.full_like(q_ends, initial, dtype=torch.int32),
                torch.full(
                    (query.shape[0],), local, dtype=torch.int32, device=query.device
                ),
            )
            self._selection_windows[window_key] = windows
        indices, values = self._index(
            query.contiguous(),
            key.contiguous(),
            weights.contiguous(),
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_key=kv_ends,
            block_table=None,
            init_tensor=windows[0],
            local_tensor=windows[1],
            layout_query="TND",
            layout_key="TND",
            sparse_count=topk,
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=return_scores,
        )
        return indices, values

    def prefill_attention(
        self, query, key, key_rope, indices, q_ends, kv_ends, *, scale
    ):
        """Sparse Prefill with separate contiguous latent512 and RoPE64."""
        if self._prefill_attention is None:
            raise RuntimeError("Prefill SFA capability was not validated")
        if query.ndim != 3 or query.shape[-1] != 576:
            raise ValueError("Prefill Q8 requires packed query width 512+64")
        if (
            key.ndim != 3
            or key.shape[1:] != (1, 512)
            or key_rope.shape != (key.shape[0], 1, 64)
            or not key.is_contiguous()
            or not key_rope.is_contiguous()
        ):
            raise ValueError("Prefill SFA key/RoPE must be contiguous TND512/64")
        if any(t.dtype != torch.bfloat16 for t in (query, key, key_rope)):
            raise ValueError("Prefill SFA requires BF16 inputs")
        if any(t.dtype != torch.int32 for t in (indices, q_ends, kv_ends)):
            raise ValueError("Prefill SFA requires INT32 indices and cumulative ends")
        result, _, _ = self._prefill_attention(
            query[..., :512].contiguous(),
            key,
            key,
            indices.contiguous(),
            scale,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_ends,
            query_rope=query[..., 512:].contiguous(),
            key_rope=key_rope,
            block_table=None,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=False,
        )
        if result.shape != query.shape[:-1] + (512,):
            raise RuntimeError("native Prefill SFA returned an unexpected output shape")
        return result

    def require_context_parallel(self):
        missing = []
        if self._select_local is None:
            missing.append("SelectLocalTopkIndices")
        if self._merge_partials is None:
            missing.append("KvpAttentionMerge")
        if missing:
            raise RuntimeError(
                "Ascend DSA context parallelism needs " + ", ".join(missing)
            )

    def _full_scan_chunk_hints(self, indices):
        """Stable scheduling hints for SFAD's full 2048-entry scan.

        LightningIndexer's second output is optional sparse values, not the
        ``valid_chunks`` emitted by FluentLLM's later KVP-local selection.
        This path has no KVP compaction and uses ``sparse_prep_mode=0``, so
        SFAD scans all entries and only needs a conservative scheduling cost.
        Cache the all-16 hint by graph shape to avoid adding a runtime kernel.
        """
        key = (indices.device, indices.shape[0])
        chunks = self._full_scan_chunks.get(key)
        if chunks is None:
            chunks = torch.full(
                (indices.shape[0],),
                indices.shape[-1] // 128,
                dtype=torch.int32,
                device=indices.device,
            )
            self._full_scan_chunks[key] = chunks
        return chunks

    def scatter(self, x, cache, locations):
        width = x.shape[-1]
        if x.dtype != torch.bfloat16 or cache.dtype != x.dtype:
            raise ValueError("LongCatDSA scatter requires BF16")
        if not x.is_contiguous() or not cache.is_contiguous():
            raise ValueError("LongCatDSA cache/input must be contiguous")
        if x.numel() // width != locations.numel():
            raise ValueError("LongCatDSA write locations must cover all tokens")
        _scatter[(locations.numel(),)](
            x,
            cache,
            locations,
            width,
            cache.numel() // width,
            cache.shape[1],
            triton.next_power_of_2(width),
        )

    def index(
        self, query, key, weights, q_ends, kv_lengths, table, topk, initial, local
    ):
        # Immutable model constants. Keep their tensors alive for every graph
        # that captured the pointers; lengths/page tables remain refreshed.
        window_key = (query.device, q_ends.numel(), query.shape[0], initial, local)
        windows = self._selection_windows.get(window_key)
        if windows is None:
            windows = (
                torch.full_like(q_ends, initial, dtype=torch.int32),
                torch.full(
                    (query.shape[0],), local, dtype=torch.int32, device=query.device
                ),
            )
            self._selection_windows[window_key] = windows
        initial_tensor, local_tensor = windows
        result, _ = self._index(
            query.contiguous(),
            key,
            weights.to(torch.bfloat16).contiguous(),
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_key=kv_lengths,
            block_table=table,
            init_tensor=initial_tensor,
            local_tensor=local_tensor,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=False,
        )
        return result, self._full_scan_chunk_hints(result)

    def index_partial(
        self,
        query,
        key,
        weights,
        q_ends,
        kv_lengths,
        table,
        topk,
        initial_counts,
        local_counts,
        *,
        sparse_mode,
    ):
        result, values = self._index(
            query.contiguous(),
            key,
            weights.to(torch.bfloat16).contiguous(),
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_key=kv_lengths,
            block_table=table,
            init_tensor=initial_counts,
            local_tensor=local_counts,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=topk,
            sparse_mode=sparse_mode,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            return_value=True,
        )
        return result, values

    def select_local(self, local_indices, global_positions, rank):
        if self._select_local is None:
            raise RuntimeError("LongCat DSA CP needs SelectLocalTopkIndices")
        return self._select_local(local_indices, global_positions, rank)

    def attention(
        self,
        q,
        qr,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
    ):
        output, _, _ = self._attention(
            q,
            key,
            key,
            indices,
            scale,
            valid_chunks=valid_chunks,
            block_table=table,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=qr,
            key_rope=None,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=False,
            sparse_prep_mode=0,
        )
        return output.to(q.dtype)

    def attention_packed(
        self,
        query,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
        *,
        head_major_output,
    ):
        """Consume packed latent/aux Q and cast only the live output heads.

        The existing rank-major ABI also represents a single rank. SFAD writes
        head-major FP32 output directly; no context-parallel collective is used.

        Args:
            query: Packed latent/rotary Q, [tokens, heads, 576].
            key: Paged packed KV cache in PA_BSND layout.
            indices: Selected sparse token positions for each query row.
            valid_chunks: Per-query scheduling hints for the native sparse scan.
            q_ends: Cumulative query ends, one per request.
            kv_lengths: Visible KV lengths, one per request.
            table: Request-to-cache block table.
            scale: Attention softmax scale.
            head_major_output: Whether to retain the native head-major layout.

        Returns:
            Attention in query dtype: [heads, tokens, 512] when head-major,
            otherwise contiguous [tokens, heads * 512].
        """
        heads = query.shape[1]
        if not 1 <= heads <= 64:
            raise ValueError("Ascend DSA supports 1 through 64 local heads")
        padded_heads = 32 if heads <= 32 else 64
        if heads != padded_heads:
            query = query.repeat(1, (padded_heads + heads - 1) // heads, 1)
            query = query[:, :padded_heads].contiguous()
        output, _, _ = self._attention(
            query.unsqueeze(0),
            key,
            key,
            indices,
            scale,
            valid_chunks=valid_chunks,
            block_table=table,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=None,
            key_rope=None,
            sparse_block_size=1,
            layout_query="RANK_MAJOR_TND_PACKED_KVP_COMM",
            layout_kv="PA_BSND",
            sparse_mode=3,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=False,
            sparse_prep_mode=0,
        )
        output = output[:heads].to(query.dtype)
        if head_major_output:
            return output
        return output.transpose(0, 1).contiguous().flatten(1)

    def attention_partial(
        self,
        packed_query,
        key,
        indices,
        valid_chunks,
        q_ends,
        kv_lengths,
        table,
        scale,
        *,
        sparse_mode,
    ):
        # KVP prefill represents every causal query as a one-token request.
        # The native SFAD ABI permits only 1024 requests per invocation.
        # Slice that query-batch axis after selection/cache writes, keeping
        # the same visible KV lengths, page rows, and rank-major head layout.
        batch = q_ends.numel()
        if batch > 1024:
            if packed_query.shape[1] != batch:
                raise ValueError(
                    "Ascend DSA batching above 1024 requests requires "
                    "one query token per request"
                )
            outputs = None
            for start in range(0, batch, 1024):
                end = min(start + 1024, batch)
                part = self.attention_partial(
                    packed_query[:, start:end].contiguous(),
                    key,
                    indices[start:end].contiguous(),
                    valid_chunks[start:end].contiguous(),
                    (q_ends[start:end] - start).contiguous(),
                    kv_lengths[start:end].contiguous(),
                    table[start:end].contiguous(),
                    scale,
                    sparse_mode=sparse_mode,
                )
                if outputs is None:
                    # O is [H,T,512]; both LSE components are [1,T,H].
                    outputs = tuple(
                        value.new_empty((value.shape[0], batch, *value.shape[2:]))
                        for value in part
                    )
                for output, value in zip(outputs, part):
                    output[:, start:end].copy_(value)
            return outputs
        output, softmax_max, softmax_sum = self._attention(
            packed_query,
            key,
            key,
            indices,
            scale,
            valid_chunks=valid_chunks,
            block_table=table,
            actual_seq_lengths_query=q_ends,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=None,
            key_rope=None,
            sparse_block_size=1,
            layout_query="RANK_MAJOR_TND_PACKED_KVP_COMM",
            layout_kv="PA_BSND",
            sparse_mode=sparse_mode,
            pre_tokens=9223372036854775807,
            next_tokens=9223372036854775807,
            attention_mode=2,
            return_softmax_lse=True,
            sparse_prep_mode=1,
        )
        return output, softmax_max, softmax_sum

    def global_topk(self, scores, k):
        if (
            self._global_topk_native is not None
            and scores.ndim == 2
            and scores.shape == (8, 16384)
            and scores.shape[-1] <= torch.iinfo(torch.int32).max
            and k == 2048
            and scores.dtype in (torch.bfloat16, torch.float32)
            and scores.device.type == "npu"
            and scores.is_contiguous()
        ):
            return self._global_topk_native(scores, k)
        return scores.topk(k, dim=-1).indices.to(torch.int32)

    def global_topk_rankmajor(self, scores, k, *, sorted_per_rank):
        if (
            self._global_topk_rankmajor_native is not None
            and sorted_per_rank is True
            and scores.ndim == 3
            and scores.shape == (8, 8, 2048)
            and scores.shape[0] * scores.shape[2] <= torch.iinfo(torch.int32).max
            and k == 2048
            and scores.dtype == torch.bfloat16
            and scores.device.type == "npu"
            and scores.is_contiguous()
        ):
            return self._global_topk_rankmajor_native(scores)
        degree, rows, candidates = scores.shape
        flat = scores.transpose(0, 1).contiguous().view(rows, degree * candidates)
        return self.global_topk(flat, k)

    def merge_partials(self, output, lse):
        if self._merge_partials is None:
            raise RuntimeError("LongCat DSA CP needs KvpAttentionMerge")
        if (
            self._merge_partials_native is not None
            and output.ndim == 3
            and output.shape[0] == 8
            and output.shape[2] == 512
            and 0 < output.shape[1] <= 4096
            and lse.ndim == 2
            and lse.shape == output.shape[:2]
            and output.dtype == torch.float32
            and lse.dtype == torch.float32
            and output.device.type == "npu"
            and lse.device == output.device
            and output.is_contiguous()
            and lse.is_contiguous()
        ):
            return self._merge_partials_native(output, lse)
        return self._merge_partials(output, lse)
