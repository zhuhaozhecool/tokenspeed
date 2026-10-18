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

"""Host-side cache90 routing and receive-failure ownership regressions."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from test.runtime.cache_pd_test_utils import group, layout, operation, segment
from types import SimpleNamespace

import numpy as np
import pytest

from tokenspeed.runtime.pd.base.status import TransferPoll
from tokenspeed.runtime.pd.cache_protocol import (
    CacheContractError,
    CachePDBlockManifest,
    build_cache_block_manifest,
    validate_cache_manifest,
)
from tokenspeed.runtime.pd.transfer_plan import (
    CacheTransferPlanner,
    local_transfer_pages,
)


def _dsa_layout(shards):
    return layout(
        group(
            "history",
            segment(
                "layer.0.latent_kv", shape=(128, 1, 8), dtype="bfloat16", stride=4096
            ),
            segment(
                "layer.0.index_k",
                shape=(128, 8),
                dtype="bfloat16",
                offset=1 << 20,
                stride=4096,
            ),
            q=128,
            shard_count=shards,
        ),
        group(
            "state",
            segment(
                "layer.1.recurrent_state",
                shape=(8, 128, 128),
                dtype="float32",
                offset=2 << 20,
            ),
            segment(
                "layer.1.conv_state", shape=(3, 3072), dtype="bfloat16", offset=1 << 28
            ),
            segment(
                "layer.1.checkpoint_len", shape=(1,), dtype="int64", offset=1 << 30
            ),
            attn_kind="state",
        ),
        block_size=128,
        capacity=140,
    )


@pytest.mark.parametrize("d_hit", [0, 117964])
@pytest.mark.parametrize("p_hit", [0, 117964])
@pytest.mark.parametrize("source_shards", [1, 8])
def test_cache90_all_dsa_fields_route_independent_virtual_owners(
    p_hit, d_hit, source_shards
):
    """P reuse affects computation; only D's hit trims transferred history."""
    source, destination = _dsa_layout(source_shards), _dsa_layout(8)
    # Distinct allocations after P reuse must still pair the same logical
    # position. Shift P's virtual pages so all KVP owners participate and
    # source rank cannot be inferred from the destination rank.
    source_offset = 11 if p_hit else 0
    source_ids = np.arange(1, 1025, dtype=np.int32) + source_offset
    destination_ids = np.arange(2, 1026, dtype=np.int32)
    if source_shards == 1:
        from dataclasses import replace

        source = replace(
            source,
            plan=replace(
                source.plan,
                num_lcm_blocks=1100,
                groups=tuple(replace(g, page_count=1101) for g in source.plan.groups),
            ),
        )
    planner = CacheTransferPlanner(
        prefill_tp_size=8,
        decode_tp_size=8,
        prefill_layout=source,
        decode_layout=destination,
        prefill_field_ids=None,
    )
    source_table = source_ids

    def manifest(contract, history, live):
        return build_cache_block_manifest(
            operation(
                {
                    "history": history[None, :],
                    "state": np.array([[77, live]], dtype=np.int32),
                }
            ),
            layout=contract,
            request_row=0,
            prefix_len=d_hit,
            prompt_len=131072,
            state_prefix_cached=False,
        )

    src = manifest(source, source_table, 3)
    dst = manifest(destination, destination_ids, 5)
    validate_cache_manifest(src, layout=source, peer="P")
    validate_cache_manifest(dst, layout=destination, peer="D")
    assert len(src.groups[0].block_ids) == (103 if d_hit else 1024)
    assert src.groups[1].block_ids == (3,)  # Different suffix always needs final state.
    assert dst.groups[1].block_ids == (5,)
    first = d_hit // 128
    observed = defaultdict(list)
    off_rank = 0
    for destination_rank in range(8):
        route = planner.plan_for_decode_rank(destination_rank)
        assert {
            f.field_id
            for fragments in route.fragments_by_prefill_rank.values()
            for f in fragments
            if f.group_id == "state"
        } == {
            "layer.1.recurrent_state",
            "layer.1.conv_state",
            "layer.1.checkpoint_len",
        }
        for source_rank in route.target_prefill_ranks:
            fragments = route.fragments_by_prefill_rank[source_rank]
            history_fields = {f.field_id for f in fragments if f.group_id == "history"}
            assert history_fields == {"layer.0.latent_kv", "layer.0.index_k"}
            a, b = local_transfer_pages(
                src.groups[0].block_ids,
                dst.groups[0].block_ids,
                group_id="history",
                source_layout=source,
                destination_layout=destination,
                source_owner_rank=source_rank % source_shards,
                destination_owner_rank=destination_rank,
            )
            for field in history_fields:
                for source_page, destination_page in zip(
                    a.tolist(), b.tolist(), strict=True
                ):
                    virtual_source = (
                        (source_page - 1) * source_shards
                        + source_rank % source_shards
                        + 1
                    )
                    virtual_destination = (
                        (destination_page - 1) * 8 + destination_rank + 1
                    )
                    token_page = virtual_source - source_offset - 1
                    assert virtual_destination == virtual_source - source_offset + 1
                    observed[field].append(token_page)
                    off_rank += source_rank != destination_rank
    if source_shards == 8:
        assert off_rank > 0
    for pages in observed.values():
        assert sorted(pages) == list(range(first, 1024))


def test_manifest_rejects_false_string_and_non_integer_blocks():
    contract = _dsa_layout(8)
    valid = build_cache_block_manifest(
        operation(
            {
                "history": np.array([[1]], dtype=np.int32),
                "state": np.array([[2, 3]], dtype=np.int32),
            }
        ),
        layout=contract,
        request_row=0,
        prefix_len=0,
        prompt_len=128,
        state_prefix_cached=False,
    )
    for field, value in [
        ("state_prefix_cached", "false"),
        ("prefix_len", False),
        ("prompt_len", 128.0),
    ]:
        wire = json.loads(valid.to_wire_bytes())
        wire[field] = value
        with pytest.raises(CacheContractError, match="invalid field types"):
            CachePDBlockManifest.from_wire_bytes(json.dumps(wire).encode())
    wire = json.loads(valid.to_wire_bytes())
    wire["groups"][0]["block_ids"] = [True]
    with pytest.raises(CacheContractError, match="invalid field types"):
        CachePDBlockManifest.from_wire_bytes(json.dumps(wire).encode())
    wire.pop("groups")
    with pytest.raises(CacheContractError, match="invalid Cache block manifest fields"):
        CachePDBlockManifest.from_wire_bytes(json.dumps(wire).encode())


def _manager():
    from tokenspeed.runtime.pd.mooncake.decode import MooncakeKVManagerDecode

    manager = object.__new__(MooncakeKVManagerDecode)
    manager.request_status = {9: TransferPoll.WaitingForInput}
    manager.expected_prefill_ranks_table = {9: frozenset({0, 1})}
    manager.prefill_response_tracker = defaultdict(set)
    manager.undrained_rooms = set()
    manager.failure_records = {}
    manager.failure_lock = threading.Lock()
    manager.bootstrap_token_table = {}
    manager.spec_candidate_ids_table = {}
    manager.cached_tokens_table = {}
    manager.bootstrap_logprob_table = {}
    manager._pending_bootstrap_token_table = {}
    manager._pending_spec_candidate_ids_table = {}
    manager._pending_bootstrap_logprob_table = {}
    manager.connection_lock = threading.Lock()
    manager.connection_pool = {}
    manager.prefill_parallel_info = {}
    manager.addr_to_rooms_tracker = defaultdict(set, {"bootstrap": {9}})
    manager.waiting_timeout = 0
    manager.kv_transfer_metrics = SimpleNamespace(
        record_kv_transfer_timeout=lambda _kind: None,
        record_kv_transfer_failure=lambda: None,
    )
    return manager


def _receiver(manager, *, published):
    from tokenspeed.runtime.pd.mooncake.receiver import MooncakeKVReceiver

    receiver = object.__new__(MooncakeKVReceiver)
    receiver.kv_mgr = manager
    receiver.bootstrap_room = 9
    receiver.bootstrap_addr = "bootstrap"
    receiver.conclude_state = None
    receiver.init_time = 0.0 if published else None
    return receiver


def test_completed_payload_is_published_only_after_every_owner():
    manager = _manager()
    manager._handle_prefill_status(9, TransferPoll.Success, 0, 42, [7, 8], 117964, -0.5)
    assert manager.request_status[9] == TransferPoll.WaitingForInput
    assert manager.bootstrap_token_table == {}
    manager._handle_prefill_status(9, TransferPoll.Success, 1, -1, None, 117964, None)
    assert manager.request_status[9] == TransferPoll.Success
    assert manager.pop_prefill_metadata(9) == (42, [7, 8], 117964, -0.5)


def test_failed_room_ignores_late_metadata_and_clear_removes_every_table():
    manager = _manager()
    manager.update_status(9, TransferPoll.Failed)
    manager._handle_prefill_status(9, TransferPoll.Success, 0, 42, [7], 117964, -0.5)
    assert manager._pending_bootstrap_token_table == {}
    assert manager.cached_tokens_table == {}
    manager.bootstrap_logprob_table[9] = -0.25
    manager._pending_bootstrap_logprob_table[9] = -0.5
    _receiver(manager, published=False).clear()
    assert (
        manager.request_status
        == manager.bootstrap_logprob_table
        == manager._pending_bootstrap_logprob_table
        == {}
    )
    manager._handle_prefill_status(9, TransferPoll.Success, 1, 99, None, 9, None)
    assert manager.request_status == {}


@pytest.mark.parametrize("failure", ["timeout", "heartbeat", "remote_failure"])
def test_unconfirmed_receive_raises_before_failed_event_or_page_reuse(failure):
    from tokenspeed.runtime.pd.mooncake.entities import KVTransferError

    manager = _manager()
    receiver = _receiver(manager, published=True)
    if failure == "heartbeat":
        manager._handle_node_failure("bootstrap")
    elif failure == "remote_failure":
        manager._handle_prefill_status(9, TransferPoll.Failed, 0, -1, None, 0, None)
    with pytest.raises(KVTransferError, match="unconfirmed"):
        receiver.poll()
    assert manager.undrained_rooms == ({9} if failure != "remote_failure" else set())
    assert receiver.conclude_state is None  # No reclaimable FailedEvent.
    assert (
        9 in manager.request_status
    )  # The scheduler's destination lease stays pinned.


def test_failure_before_destination_publication_remains_request_local():
    manager = _manager()
    manager.request_status[9] = TransferPoll.Bootstrapped
    manager._handle_node_failure("bootstrap")
    assert manager.undrained_rooms == set()
    assert _receiver(manager, published=False).poll() == TransferPoll.Failed
