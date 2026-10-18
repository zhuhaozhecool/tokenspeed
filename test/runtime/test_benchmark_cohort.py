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


"""Host lifecycle tests; only the TP object transport is replaced by CPU barriers."""

import copy
import hashlib
import multiprocessing as mp
import os
import tempfile
import time
import unittest
from pathlib import Path

from tokenspeed.runtime.pd import bench_cohort as cohort


def body(epoch="e1", group="c1", batch=1, timeout=100):
    return {
        "epoch": epoch,
        "cohort_id": group,
        "per_domain_batch": batch,
        "timeout_s": timeout,
        "input_tokens": 131072,
        "output_tokens": 1024,
        "expected_cached_tokens": 117964,
        "requests": [
            {
                "dp_rank": dp,
                "input_sha256": hashlib.sha256(f"{dp}/{i}".encode()).hexdigest(),
            }
            for dp in range(2)
            for i in range(batch)
        ],
    }


def room(cfg, dp, index):
    base = int(
        hashlib.sha256(f"{cfg['epoch']}/{cfg['cohort_id']}".encode()).hexdigest()[:10],
        16,
    )
    return base * 100000 + dp * 1000 + index


def worker(rank, directory, commands, results, barriers, values):
    events = []
    calls = 0
    dp = rank // 8

    def sync(payload):
        nonlocal calls
        calls += 1
        if rank % 8 == 0:
            values[dp] = payload
        barriers[dp].wait(timeout=20)
        result = copy.deepcopy(values[dp])
        barriers[dp].wait(timeout=20)
        return result

    def emit(event, **fields):
        events.append(dict(event=event, global_rank=rank, pid=os.getpid(), **fields))

    gate = cohort.CohortGate(
        cohort.CohortStore(directory),
        global_rank=rank,
        dp_rank=dp,
        sync=sync,
        emit=emit,
    )
    while True:
        command, args = commands.get()
        if command == "stop":
            return
        result = None
        try:
            if command == "idle":
                gate.idle()
            elif command == "bind":
                cfg, index = args
                row = [r for r in cfg["requests"] if r["dp_rank"] == dp][index]
                rid = f"cmpl-bench.{cfg['epoch']}.{cfg['cohort_id']}.{dp}{index:031x}-0"
                gate.bind(
                    rid,
                    room(cfg, dp, index),
                    row["input_sha256"],
                    input_tokens=cfg["input_tokens"],
                    output_tokens=cfg["output_tokens"],
                )
            elif command == "receive":
                cfg, index, targets, p_hit, d_hit = args
                if dp in targets:
                    rid = f"cmpl-bench.{cfg['epoch']}.{cfg['cohort_id']}.{dp}{index:031x}-0"
                    result = gate.receive(
                        rid,
                        rid,
                        {
                            "room": room(cfg, dp, index),
                            "p_cached_tokens": p_hit,
                            "d_hit_tokens": d_hit,
                        },
                    )
            elif command == "fail":
                cfg, index, targets = args
                if dp in targets:
                    gate.fail(
                        f"cmpl-bench.{cfg['epoch']}.{cfg['cohort_id']}.{dp}{index:031x}-0",
                        "actual fence failure",
                    )
            elif command == "tick":
                result = gate.tick(set(args))
            elif command == "commit":
                gate.committed(list(gate.bindings), {})
            elif command == "tagless":
                gate.bind(
                    "cmpl-normal", 17, "unused", input_tokens=117964, output_tokens=1
                )
            elif command == "stale":
                gate.bind(args, 90, "0" * 64, input_tokens=131072, output_tokens=1024)
            elif command != "inspect":
                raise AssertionError(command)
            results.put(
                {
                    "rank": rank,
                    "result": result,
                    "phase": gate.phase,
                    "calls": calls,
                    "pending": list(gate.pending),
                    "bindings": copy.deepcopy(gate.bindings),
                    "events": copy.deepcopy(events),
                }
            )
        except Exception as error:  # noqa: BLE001 - worker boundary
            results.put({"rank": rank, "error": f"{type(error).__name__}: {error}"})


class Fleet:
    def __init__(self, directory):
        self.ctx = mp.get_context("fork")
        self.manager = self.ctx.Manager()
        self.values = self.manager.dict()
        self.barriers = [self.ctx.Barrier(8), self.ctx.Barrier(8)]
        self.results = self.ctx.Queue()
        self.queues = [self.ctx.Queue() for _ in range(16)]
        self.processes = [
            self.ctx.Process(
                target=worker,
                args=(
                    r,
                    directory,
                    self.queues[r],
                    self.results,
                    self.barriers,
                    self.values,
                ),
            )
            for r in range(16)
        ]
        for proc in self.processes:
            proc.start()

    def run(self, command, args=None, errors=False):
        for channel in self.queues:
            channel.put((command, args))
        rows = [self.results.get(timeout=30) for _ in range(16)]
        rows.sort(key=lambda r: r["rank"])
        if not errors and any("error" in row for row in rows):
            raise AssertionError(rows)
        return rows

    def close(self):
        for channel in self.queues:
            channel.put(("stop", None))
        for proc in self.processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()
                proc.join()
        self.manager.shutdown()


class Lifecycle(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = cohort.CohortStore(self.temp.name)
        self.fleet = Fleet(self.temp.name)

    def tearDown(self):
        self.fleet.close()
        self.temp.cleanup()

    def arm_bind(self, cfg):
        self.store.arm(cfg)
        self.fleet.run("idle")
        for index in range(cfg["per_domain_batch"]):
            self.fleet.run("bind", (cfg, index))

    def receive_all(self, cfg):
        for index in range(cfg["per_domain_batch"]):
            self.fleet.run("receive", (cfg, index, [0, 1], 117964, 117964))

    def test_16_actual_workers_release_then_next_group(self):
        for group, batch in [("c1", 1), ("c2", 2), ("c3", 32)]:
            cfg = body(group=group, batch=batch)
            self.arm_bind(cfg)
            self.fleet.run("receive", (cfg, 0, [0], 117964, 117964))
            with self.assertRaises(cohort.CohortError):
                self.store.release("e1", group)
            held = self.fleet.run("tick", [])
            self.assertTrue(all(not r["result"] for r in held))
            self.fleet.run("receive", (cfg, 0, [1], 117964, 117964))
            for index in range(1, batch):
                self.fleet.run("receive", (cfg, index, [0, 1], 117964, 117964))
            status = self.store.status("e1", group)
            self.assertEqual(status["status"], "ready")
            self.assertEqual(len({r["pid"] for r in status["actual_rank_reports"]}), 16)
            released = self.store.release("e1", group)
            cert = released["ready_certificate"]
            unsigned = {k: v for k, v in cert.items() if k != "certificate_sha256"}
            self.assertEqual(cohort.digest(unsigned), cert["certificate_sha256"])
            decision = cert["release_decision_monotonic_ns"]
            self.assertTrue(
                all(
                    b["receive_monotonic_ns"] <= decision
                    for r in cert["actual_rank_reports"]
                    for b in r["bindings"]
                )
            )
            rows = self.fleet.run("tick", [])
            self.assertTrue(
                all(
                    len(r["result"]) == batch
                    and all(not abort for _, abort in r["result"])
                    for r in rows
                )
            )
            before = [r["calls"] for r in rows]
            self.assertEqual([r["calls"] for r in self.fleet.run("tick", [])], before)
            self.fleet.run("commit")
            self.assertEqual(self.store.status("e1", group)["status"], "complete")
            self.assertEqual(self.store.release("e1", group)["ready_certificate"], cert)

    def test_abort_keeps_unfinished_write_until_late_completion(self):
        cfg = body()
        self.arm_bind(cfg)
        self.fleet.run("receive", (cfg, 0, [0], 117964, 117964))
        self.store.abort("e1", "c1", "client cancellation")
        rows = self.fleet.run("tick", [])
        self.assertTrue(
            all(len(r["result"]) == (1 if r["rank"] < 8 else 0) for r in rows)
        )
        self.assertTrue(
            all(
                r["bindings"][next(iter(r["bindings"]))]["terminal"] is None
                for r in rows[8:]
            )
        )
        with self.assertRaises(cohort.CohortError):
            self.store.arm(body(group="next"))
        with self.assertRaises(cohort.CohortError):
            self.store.release("e1", "c1")
        self.fleet.run("receive", (cfg, 0, [1], 117964, 117964))
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(abort for r in rows for _, abort in r["result"]))
        self.assertEqual(self.store.status("e1", "c1")["status"], "aborted")
        self.store.arm(body(group="next"))
        self.fleet.run("idle")
        stale = self.fleet.run(
            "stale", "cmpl-bench.e1.c1.00000000000000000000000000000001", errors=True
        )
        self.assertTrue(all("Stale" in r["error"] for r in stale))

    def test_real_transfer_failure_aborts_other_pending_members(self):
        cfg = body()
        self.arm_bind(cfg)
        self.fleet.run("receive", (cfg, 0, [0], 117964, 117964))
        self.fleet.run("fail", (cfg, 0, [1]))
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(abort for r in rows for _, abort in r["result"]))
        self.assertEqual(self.store.status("e1", "c1")["status"], "aborted")

    def test_wrong_actual_d_hit_never_releases(self):
        cfg = body()
        self.arm_bind(cfg)
        self.fleet.run("receive", (cfg, 0, [0, 1], 117964, 0))
        with self.assertRaises(cohort.CohortError):
            self.store.release("e1", "c1")
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(abort for r in rows for _, abort in r["result"]))

    def test_timeout_and_tagless_seed_bypass(self):
        cfg = body(timeout=0.001)
        self.arm_bind(cfg)
        self.fleet.run("tagless")
        time.sleep(0.005)
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(not r["result"] for r in rows))
        self.assertEqual(self.store.status("e1", "c1")["status"], "aborting")
        self.receive_all(cfg)
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(abort for r in rows for _, abort in r["result"]))
        self.assertEqual(self.store.status("e1", "c1")["status"], "aborted")

    def test_tampered_rank_certificate_and_immutable_release(self):
        cfg = body()
        self.arm_bind(cfg)
        self.receive_all(cfg)
        reports = self.store.reports(self.store.active())
        reports[0]["pid"] = reports[1]["pid"]
        with self.assertRaises(cohort.CohortError):
            self.store._proof(self.store.active(), reports)
        good = self.store.reports(self.store.active())
        cases = []
        cases.append(good[:-1])
        wrong_hit = copy.deepcopy(good)
        wrong_hit[0]["bindings"][0]["d_hit_tokens"] = 0
        cases.append(wrong_hit)
        wrong_tp = copy.deepcopy(good)
        wrong_tp[1]["bindings"][0]["room"] += 1
        cases.append(wrong_tp)
        duplicate_room = copy.deepcopy(good)
        for row in duplicate_room[8:]:
            row["bindings"][0]["room"] = duplicate_room[0]["bindings"][0]["room"]
        cases.append(duplicate_room)
        for bad in cases:
            with self.assertRaises(cohort.CohortError):
                self.store._proof(self.store.active(), bad)
        cert = self.store.release("e1", "c1")["ready_certificate"]
        self.store.abort("e1", "c1", "cancel after decision before local release")
        rows = self.fleet.run("tick", [])
        self.assertTrue(all(abort for r in rows for _, abort in r["result"]))
        self.assertEqual(
            cohort.read(self.store.folder("e1", "c1") / "release.json"), cert
        )
        self.assertEqual(self.store.status("e1", "c1")["status"], "aborted")


class Validation(unittest.TestCase):
    def test_tags_and_arm_validation(self):
        rid = cohort.tagged_request_id({"epoch": "e_1", "cohort_id": "c-1"})
        self.assertEqual(cohort.request_tag(rid + "-0"), ("e_1", "c-1"))
        self.assertIsNone(cohort.request_tag("cmpl-normal"))
        for tag in [None, {"epoch": "../oops", "cohort_id": "x"}, {"epoch": "x"}]:
            with self.assertRaises(cohort.CohortError):
                cohort.tagged_request_id(tag)
        with tempfile.TemporaryDirectory() as directory:
            store = cohort.CohortStore(directory)
            for key, invalid in [
                ("timeout_s", float("nan")),
                ("per_domain_batch", True),
                ("output_tokens", 1),
                ("expected_cached_tokens", 117963),
            ]:
                cfg = body()
                cfg[key] = invalid
                with self.assertRaises(cohort.CohortError):
                    store.arm(cfg)

    def test_room_claim_cannot_be_reused_by_a_later_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            store = cohort.CohortStore(directory)
            config = store.arm(body())
            store.claim(
                config, "actual-rid", 91, config["requests"][0]["input_sha256"], 0
            )
            for rid, room in [("other-rid", 91), ("actual-rid", 92)]:
                with self.assertRaises(cohort.CohortError):
                    store.claim(
                        config, rid, room, config["requests"][0]["input_sha256"], 0
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
