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

"""Explicit backend-local Q8 Prefill; persistent KVP and Decode are inherited."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
    AscendDSABackend,
    _AscendDCPKernelMetadata,
    ascend_kvp_index_partial,
    ascend_prefill_index_metadata,
    ascend_write_index_cache,
)
from tokenspeed.runtime.layers.attention.configs.dsa import (
    dsa_history_gather_workspace_rows,
)
from tokenspeed.runtime.layers.attention.dcp.ascend_selection import (
    PrefillKVPSelectionWorkspace,
)
from tokenspeed.runtime.layers.attention.dcp.cache import HistoryGatherPlan
from tokenspeed.runtime.layers.attention.dcp.prefill_query import (
    ASCEND_DSA_PREFILL_Q8,
    PrefillQueryPlan,
    history_owner_vectors,
    plan_prefill_history,
    plan_prefill_queries,
)
from tokenspeed.runtime.layers.attention.dcp.prefill_query_ops import (
    PrefillHistoryTensors,
    PrefillHistoryWorkspace,
    head_to_query,
    query_to_head,
)


class _PrefillHCCL:
    def __init__(self, dcp):
        self.dcp = dcp

    def exchange(self, output, source):
        self.dcp.all_to_all(output, source, auxiliary=False)

    def gather(self, output, source):
        dist.all_gather_into_tensor(
            output, source.contiguous(), group=self.dcp.primary_process_group
        )

    def broadcast(self, tensor, *, owner):
        dist.broadcast(
            tensor, src=self.dcp.ranks[owner], group=self.dcp.primary_process_group
        )


@dataclass(frozen=True)
class _PrefillMetadata:
    query: PrefillQueryPlan
    histories: tuple[PrefillHistoryTensors, ...]
    selection: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


@dataclass(frozen=True)
class _PrefillSelection:
    indices: torch.Tensor
    local_indices: torch.Tensor
    local_values: torch.Tensor
    local_metadata: _AscendDCPKernelMetadata


class AscendDSAPrefillQ8Backend(AscendDSABackend):
    """Pure Prefill changes only the attention leaf's temporary query layout.

    Select explicitly with ``--attention-backend ascend_dsa_prefill_q8``.
    Baseline ``dsa`` does not allocate this workspace. Model-wide QCP remains
    one; KDA/MoE, all persistent writes and Decode/verify keep their old paths.
    """

    prefill_requires_host_page_table = True

    def __init__(self, config, spec, *, kernel_page_size):
        if spec.backend_name != ASCEND_DSA_PREFILL_Q8:
            raise ValueError("Prefill Q8 requires explicit candidate backend selection")
        if config.kvp_size != 8 or spec.attn_tp_size != 8 or config.qcp_size != 1:
            raise ValueError("Prefill Q8 requires KVP8/head-TP8 and model QCP1")
        if (spec.kv_lora_rank, spec.qk_rope_head_dim, spec.index_head_dim) != (
            512,
            64,
            128,
        ):
            raise ValueError("Prefill Q8 native ABI requires latent512/RoPE64/index128")
        if spec.index_kpool is not None:
            raise ValueError("Prefill Q8 does not support pooled index keys")
        super().__init__(config, spec, kernel_page_size=kernel_page_size)
        self._indexer_kernels.require_prefill()
        if (
            config.prefill_query_workspace_tokens is None
            or config.prefill_query_workspace_tokens <= 0
        ):
            raise ValueError(
                "Prefill Q8 requires its explicit max Prefill token capacity"
            )
        self._q8_max_queries = config.prefill_query_workspace_tokens
        self._q8_workspace: PrefillHistoryWorkspace | None = None
        self._q8_selection: PrefillKVPSelectionWorkspace | None = None
        self._q8_metadata: _PrefillMetadata | None = None
        self._q8_max_requests = config.max_bs
        self._q8_collectives = _PrefillHCCL(self._dcp)

    def preallocate_history_gather_workspace(self, max_model_len):
        if self._q8_workspace is not None:
            raise RuntimeError("Prefill Q8 workspace was already allocated")
        spec = self._indexer_spec
        self._q8_workspace = PrefillHistoryWorkspace(
            rows=dsa_history_gather_workspace_rows(
                max_model_len, page_size=self.kernel_page_size
            ),
            degree=self._dcp.degree,
            page_size=self.kernel_page_size,
            kv_dim=spec.kv_cache_dim,
            rope_dim=spec.qk_rope_head_dim,
            max_requests=self._q8_max_requests,
            device=self.device,
        )
        self._q8_selection = PrefillKVPSelectionWorkspace(
            max_queries=self._q8_max_queries,
            degree=self._dcp.degree,
            topk=spec.index_topk,
            history_rows=self._q8_workspace.rows,
            device=self.device,
        )
        return self._q8_workspace.nbytes + self._q8_selection.nbytes

    def init_forward_metadata(
        self,
        bs,
        num_extends,
        seq_lens,
        page_table,
        forward_mode,
        *,
        extend_seq_lens,
        extend_seq_lens_cpu,
        extend_prefix_lens,
        extend_prefix_lens_cpu,
        extend_with_prefix,
        **kwargs,
    ):
        self._q8_metadata = None
        super().init_forward_metadata(
            bs,
            num_extends,
            seq_lens,
            page_table,
            forward_mode,
            extend_seq_lens=extend_seq_lens,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            extend_prefix_lens=extend_prefix_lens,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_with_prefix=extend_with_prefix,
            **kwargs,
        )
        if forward_mode.is_idle() or num_extends == 0:
            return
        if num_extends != bs or forward_mode.is_mixed():
            raise ValueError(
                "Prefill Q8 serves pure extend only; mixed batches are unsupported"
            )
        if kwargs.get("query_shard") is not None:
            raise ValueError(
                "Prefill Q8 cannot be combined with model-wide query sharding"
            )
        workspace = self._q8_workspace
        if workspace is None:
            raise RuntimeError(
                "Prefill Q8 workspace must be reserved before metadata init"
            )
        if bs > self._q8_max_requests:
            raise ValueError("Prefill Q8 request count exceeds its reserved metadata")
        table_cpu = kwargs.get("page_table_cpu")
        if table_cpu is None or table_cpu.device.type != "cpu":
            raise ValueError(
                "Prefill Q8 requires the router's actual host kernel-page table"
            )
        if (
            extend_seq_lens_cpu.device.type != "cpu"
            or extend_prefix_lens_cpu.device.type != "cpu"
        ):
            raise ValueError("Prefill Q8 length planning must use host metadata")
        lengths = tuple(int(v) for v in extend_seq_lens_cpu[:bs].tolist())
        prefixes = tuple(int(v) for v in extend_prefix_lens_cpu[:bs].tolist())
        query = plan_prefill_queries(
            lengths, prefixes, degree=self._dcp.degree, rank=self._dcp.rank
        )
        if query.total_queries > self._q8_max_queries:
            raise ValueError("Prefill Q8 exceeds its reserved query exchange capacity")
        if query.total_queries < self._dcp.degree:
            self._q8_metadata = _PrefillMetadata(query, (), None)
            return
        placement = self.cache_placement(None)
        if placement is None or tuple(placement.group) != self._dcp.ranks:
            raise ValueError("Prefill Q8 requires the actual KVP8 cache placement")
        histories = []
        for shard in query.requests:
            if shard.history_length > workspace.rows:
                raise ValueError("Prefill Q8 history exceeds its reserved model length")
            host_plan = plan_prefill_history(
                tuple(int(v) for v in table_cpu[shard.request].tolist()),
                rows=shard.history_length,
                page_size=self.kernel_page_size,
                block_granularity=placement.block_granularity,
                virtual_block_count=placement.virtual_block_count,
                degree=self._dcp.degree,
            )
            vectors = history_owner_vectors(host_plan, rank=self._dcp.rank)
            buffers = (
                workspace.virtual_slots,
                workspace.local_slots,
                workspace.order,
                workspace.fetch_slots,
            )
            views = []
            for values, buffer in zip(vectors, buffers):
                view = buffer[shard.request, : len(values)]
                view.copy_(torch.from_numpy(values))
                views.append(view)
            plan = HistoryGatherPlan(
                virtual_slots=views[0],
                local_slots=views[1],
                order=views[2],
                owned_rows_per_rank=host_plan.owned_rows,
                group=tuple(placement.group),
                rank=self._dcp.rank,
            )
            histories.append(PrefillHistoryTensors(plan=plan, fetch_slots=views[3]))
            workspace.q_ends[shard.request].fill_(shard.local_length)
            workspace.kv_ends[shard.request].fill_(shard.visible_key_length)
        selection = ascend_prefill_index_metadata(
            self.forward_prefill_metadata,
            device=self.device,
            context_parallel=True,
            total_queries=query.total_queries,
            num_requests=bs,
        )
        self._q8_metadata = _PrefillMetadata(query, tuple(histories), selection)

    def _select_kvp_prefill_indices(self, index_cache, metadata, kwargs):
        """Return the same formal selection consumed by SFA and debug receipts."""
        selection_workspace = self._q8_selection
        assert selection_workspace is not None and metadata.selection is not None
        q_ends, kv_lengths, table = metadata.selection
        (local_indices, local_values), local_metadata = ascend_kvp_index_partial(
            self, index_cache, q_ends, kv_lengths, table, kwargs
        )
        indices = selection_workspace.select(
            local_indices,
            local_values,
            metadata.query,
            metadata.histories,
            collectives=self._q8_collectives,
        )
        return _PrefillSelection(indices, local_indices, local_values, local_metadata)

    def forward_extend(
        self, q, k, v, layer, out_cache_loc, token_to_kv_pool, bs, **kwargs
    ):
        self._validate_logit_cap(layer.logit_cap)
        head_major_output = kwargs.pop("head_major_output", False)
        metadata = self._q8_metadata
        if metadata is None:
            if q.shape[0] != 0:
                raise RuntimeError("Prefill Q8 has no metadata for a nonempty extend")
            empty = q.new_empty(
                (self.num_local_heads, 0, self._indexer_spec.kv_lora_rank)
            )
            return empty if head_major_output else empty.transpose(0, 1).flatten(1)
        plan, histories = metadata.query, metadata.histories
        if bs != len(plan.requests) or q.shape != (
            plan.total_queries,
            self.num_local_heads,
            576,
        ):
            raise ValueError("Prefill Q8 query geometry does not match host metadata")
        observer = kwargs.get("prefill_debug_observer")
        if plan.total_queries < self._dcp.degree:
            if observer is not None:
                observer(
                    dict(
                        dispatch="baseline_tiny",
                        rank=self._dcp.rank,
                        total_queries=plan.total_queries,
                        threshold=self._dcp.degree,
                    )
                )
            return super().forward_extend(
                q,
                k,
                v,
                layer,
                out_cache_loc,
                token_to_kv_pool,
                bs,
                head_major_output=head_major_output,
                **kwargs,
            )
        if not self._dcp.ready:
            raise RuntimeError("Prefill Q8 HCCL process group was not configured")
        workspace = self._q8_workspace
        assert workspace is not None
        spec = self._indexer_spec
        index_cache = ascend_write_index_cache(
            self, layer, out_cache_loc, token_to_kv_pool, kwargs
        )
        packed = token_to_kv_pool.get_key_buffer(layer.layer_id)
        if (
            packed.dtype != torch.bfloat16
            or not packed.is_contiguous()
            or packed.shape[-1] != 576
        ):
            raise ValueError(
                "Prefill Q8 requires row-packed BF16 cache width576; no planar reinterpretation"
            )
        packed = packed.view(-1, 576)
        selection = self._select_kvp_prefill_indices(index_cache, metadata, kwargs)
        indices_q8 = selection.indices
        selection_workspace = self._q8_selection
        assert selection_workspace is not None
        query = head_to_query(q, plan, collectives=self._q8_collectives)
        output = q.new_zeros(
            (plan.padded_queries_per_rank, spec.num_attention_heads, spec.kv_lora_rank)
        )
        for shard, history in zip(plan.requests, histories):
            # Identical request order and gather calls on ALL ranks. An empty
            # query shard can still own cache pages needed by another rank.
            gathered = workspace.gather_history(
                packed, history, collectives=self._q8_collectives
            )
            if shard.local_length == 0:
                continue
            begin = shard.query_start + shard.local_start
            end = shard.query_start + shard.local_end
            local_begin = begin - plan.local_start
            q_ends = workspace.q_ends[shard.request]
            kv_ends = workspace.kv_ends[shard.request]
            indices = indices_q8[local_begin : local_begin + shard.local_length]
            if observer is not None:
                observer(
                    dict(
                        dispatch="kvp_bf16_q8",
                        rank=self._dcp.rank,
                        request=shard.request,
                        query_start=begin,
                        query_end=end,
                        visible_key_length=shard.visible_key_length,
                        indices=indices,
                        local_query_start=shard.query_start,
                        local_indices=selection.local_indices[
                            shard.query_start : shard.query_start + shard.query_length
                        ],
                        local_values=selection.local_values[
                            shard.query_start : shard.query_start + shard.query_length
                        ],
                        received_ids=selection_workspace.ordered_ids[
                            local_begin : local_begin + shard.local_length
                        ],
                        received_values=selection_workspace.ordered_values[
                            local_begin : local_begin + shard.local_length
                        ],
                        global_positions=selection_workspace.positions[
                            local_begin : local_begin + shard.local_length
                        ],
                    )
                )
            key, rope = workspace.split_keys(
                gathered, visible_rows=shard.visible_key_length
            )
            result = self._indexer_kernels.prefill_attention(
                query[local_begin : local_begin + shard.local_length],
                key,
                rope,
                indices,
                q_ends,
                kv_ends,
                scale=layer.scaling,
            )
            output[local_begin : local_begin + shard.local_length].copy_(result)
        result = query_to_head(output, plan, collectives=self._q8_collectives)
        return (
            result
            if head_major_output
            else result.transpose(0, 1).contiguous().flatten(1)
        )
