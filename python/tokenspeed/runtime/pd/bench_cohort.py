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


"""Opt-in benchmark cohort control; real receive completion precedes decode release.

This module has no device dependencies. Runtime transport and audit emission are
injected by PdTransferHooks. The directory is private to one D service lifetime.
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import socket
import time
import uuid
from pathlib import Path


class CohortError(ValueError):
    pass


def canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise CohortError(
            "epoch/cohort_id must be 1..64 ASCII letters, digits, '_' or '-'"
        )
    return value


def request_tag(rid):
    if not rid.startswith("cmpl-bench."):
        return None
    parts = rid.split(".", 3)
    if len(parts) != 4:
        raise CohortError("Malformed benchmark request namespace")
    return name(parts[1]), name(parts[2])


def tagged_request_id(tag):
    if not isinstance(tag, dict) or set(tag) != {"epoch", "cohort_id"}:
        raise CohortError("bench_cohort must contain exactly epoch and cohort_id")
    return (
        f"cmpl-bench.{name(tag['epoch'])}.{name(tag['cohort_id'])}.{uuid.uuid4().hex}"
    )


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temp.open("xb") as out:
            out.write(canonical(value) + b"\n")
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


class CohortStore:
    """One-host control files: immutable configs/certificates, one writer per rank."""

    def __init__(self, root):
        self.root = Path(root)
        if not self.root.is_absolute():
            raise CohortError("Cohort directory must be absolute and service-private")
        self.root.mkdir(parents=True, exist_ok=True)

    def folder(self, epoch, cohort_id):
        return self.root / name(epoch) / name(cohort_id)

    @contextlib.contextmanager
    def lock(self):
        with (self.root / "control.lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            yield

    def active(self):
        path = self.root / "active.json"
        return read(path) if path.exists() else None

    def config(self, epoch, cohort_id):
        return read(self.folder(epoch, cohort_id) / "config.json")

    def reports(self, config):
        folder = self.folder(config["epoch"], config["cohort_id"])
        return [read(p) for p in sorted(folder.glob("rank-*.json"))]

    def arm(self, body):
        required = {
            "epoch",
            "cohort_id",
            "per_domain_batch",
            "timeout_s",
            "input_tokens",
            "output_tokens",
            "expected_cached_tokens",
            "requests",
        }
        optional = {"expected_prefill_cached_tokens"}
        if not isinstance(body, dict) or not required <= set(body) or set(body) - required - optional:
            raise CohortError(f"arm requires exactly {sorted(required)}")
        epoch, cohort_id = name(body["epoch"]), name(body["cohort_id"])
        batch = body["per_domain_batch"]
        if type(batch) is not int or batch not in (1, 2, 4, 8, 16, 32):
            raise CohortError("per_domain_batch must be 1/2/4/8/16/32")
        timeout = body["timeout_s"]
        if (
            type(timeout) not in (int, float)
            or not math.isfinite(timeout)
            or timeout <= 0
            or any(
                type(body[k]) is not int
                for k in ("input_tokens", "output_tokens", "expected_cached_tokens")
            )
            or tuple(
                body[k]
                for k in ("input_tokens", "output_tokens", "expected_cached_tokens")
            )
            not in ((131072, 1024, 117964), (131072, 1024, 0))
        ):
            raise CohortError(
                "Need finite positive timeout and exact 131072/1024 with D cache0 or117964"
            )
        p_hit = body.get("expected_prefill_cached_tokens", body["expected_cached_tokens"])
        if type(p_hit) is not int or (p_hit, body["expected_cached_tokens"]) not in (
            (117964, 117964), (131071, 0), (131071, 117964)
        ):
            raise CohortError("Unsupported exact P/D cache contract")
        requests = body["requests"]
        if not isinstance(requests, list) or len(requests) != batch * 2:
            raise CohortError("Need exactly 2B expected requests")
        for row in requests:
            if (
                not isinstance(row, dict)
                or set(row) != {"dp_rank", "input_sha256"}
                or type(row["dp_rank"]) is not int
                or row["dp_rank"] not in (0, 1)
                or not isinstance(row["input_sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", row["input_sha256"])
            ):
                raise CohortError("Invalid expected domain/hash")
        if len({r["input_sha256"] for r in requests}) != len(requests) or any(
            sum(r["dp_rank"] == dp for r in requests) != batch for dp in (0, 1)
        ):
            raise CohortError("Need B unique hashes in each actual domain")
        with self.lock():
            previous = self.active()
            if previous is not None:
                reports = self.reports(previous)
                if (
                    sorted(r["global_rank"] for r in reports) != list(range(16))
                    or len({r["pid"] for r in reports}) != 16
                    or any(
                        r["status"] not in ("complete", "aborted")
                        or r["config_sha256"] != previous["config_sha256"]
                        or r["epoch"] != previous["epoch"]
                        or r["cohort_id"] != previous["cohort_id"]
                        or r["dp_rank"] != r["global_rank"] // 8
                        or r["host"] != previous["host"]
                        or r["boot_id"] != previous["boot_id"]
                        for r in reports
                    )
                    or any(
                        any(b["terminal"] is None for b in r["bindings"])
                        for r in reports
                    )
                ):
                    raise CohortError(
                        "Previous cohort still active or missing actual rank cleanup acknowledgments"
                    )
            folder = self.folder(epoch, cohort_id)
            if folder.exists():
                raise CohortError("An epoch/cohort identity cannot be reused")
            config = dict(
                body,
                schema_version=1,
                host=socket.gethostname(),
                boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                arm_monotonic_ns=time.monotonic_ns(),
            )
            config["config_sha256"] = digest(config)
            atomic(folder / "config.json", config)
            atomic(self.root / "active.json", config)
            return dict(success=True, status="armed", **config)

    def claim(self, config, rid, room, input_sha256, dp_rank):
        """Reserve the actual room before receiver registration, for this service lifetime."""
        with self.lock():
            path = self.root / "claims.json"
            claims = read(path) if path.exists() else []
            if any(row["rid"] == rid or row["room"] == room for row in claims):
                raise CohortError(
                    "Actual request/room already used during this D service lifetime"
                )
            claims.append(
                {
                    "epoch": config["epoch"],
                    "cohort_id": config["cohort_id"],
                    "config_sha256": config["config_sha256"],
                    "rid": rid,
                    "room": room,
                    "input_sha256": input_sha256,
                    "dp_rank": dp_rank,
                }
            )
            atomic(path, claims)

    def _proof(self, config, reports):
        if sorted(r["global_rank"] for r in reports) != list(range(16)):
            raise CohortError("Need all sixteen actual rank reports")
        if len({r["pid"] for r in reports}) != 16:
            raise CohortError("Need sixteen distinct actual worker PIDs")
        domain_rows = {}
        batch = config["per_domain_batch"]
        for report in reports:
            rank = report["global_rank"]
            if (
                report["dp_rank"] != rank // 8
                or report["status"] != "ready"
                or report["config_sha256"] != config["config_sha256"]
                or report["host"] != config["host"]
                or report["boot_id"] != config["boot_id"]
                or report["epoch"] != config["epoch"]
                or report["cohort_id"] != config["cohort_id"]
                or not isinstance(report["pid"], int)
                or report["pid"] <= 0
            ):
                raise CohortError("Rank identity, readiness or topology mismatch")
            bindings = sorted(report["bindings"], key=lambda b: b["input_sha256"])
            expected = {
                r["input_sha256"]
                for r in config["requests"]
                if r["dp_rank"] == rank // 8
            }
            if (
                len(bindings) != batch
                or {b["input_sha256"] for b in bindings} != expected
            ):
                raise CohortError(
                    "Actual rank does not bind exactly its expected B hashes"
                )
            for binding in bindings:
                if (
                    binding["dp_rank"] != rank // 8
                    or binding["received"] is not True
                    or binding["terminal"] is not None
                    or binding["p_cached_tokens"] != config.get("expected_prefill_cached_tokens", config["expected_cached_tokens"])
                    or binding["d_hit_tokens"] != config["expected_cached_tokens"]
                    or binding["receive_monotonic_ns"] is None
                    or binding["receive_monotonic_ns"] < binding["bind_monotonic_ns"]
                    or not binding["rid"]
                    or type(binding["room"]) is not int
                ):
                    raise CohortError(
                        "Missing successful exact-cache receive or live binding"
                    )
            identity = [(b["rid"], b["room"], b["input_sha256"]) for b in bindings]
            prior = domain_rows.setdefault(rank // 8, identity)
            if identity != prior:
                raise CohortError("TP8 actual rid/room/hash bindings disagree")
        combined = domain_rows[0] + domain_rows[1]
        if (
            len({x[0] for x in combined}) != 2 * batch
            or len({x[1] for x in combined}) != 2 * batch
        ):
            raise CohortError("Request/room reused across the cohort")
        return {
            "schema_version": 1,
            "epoch": config["epoch"],
            "cohort_id": config["cohort_id"],
            "per_domain_batch": batch,
            "config_sha256": config["config_sha256"],
            "host": config["host"],
            "boot_id": config["boot_id"],
            "actual_rank_reports": sorted(reports, key=lambda r: r["global_rank"]),
        }

    def status(self, epoch, cohort_id):
        config = self.config(epoch, cohort_id)
        folder = self.folder(epoch, cohort_id)
        reports = self.reports(config)
        result = {
            "success": True,
            "epoch": epoch,
            "cohort_id": cohort_id,
            "config_sha256": config["config_sha256"],
            "actual_rank_reports": reports,
        }
        if (folder / "abort.json").exists():
            result.update(
                status=(
                    "aborted"
                    if len(reports) == 16
                    and all(r["status"] in ("aborted", "complete") for r in reports)
                    else "aborting"
                ),
                abort=read(folder / "abort.json"),
            )
        elif (folder / "release.json").exists():
            result.update(
                status=(
                    "complete"
                    if len(reports) == 16
                    and all(r["status"] == "complete" for r in reports)
                    else "released"
                ),
                ready_certificate=read(folder / "release.json"),
            )
        else:
            try:
                proof = self._proof(config, reports)
                result.update(status="ready", ready_certificate=proof)
            except CohortError as error:
                result.update(status="collecting", readiness_error=str(error))
        return result

    def release(self, epoch, cohort_id):
        with self.lock():
            config = self.config(epoch, cohort_id)
            folder = self.folder(epoch, cohort_id)
            if (folder / "abort.json").exists():
                raise CohortError("Aborted cohort cannot release")
            if (folder / "release.json").exists():
                certificate = read(folder / "release.json")
            else:
                if (
                    time.monotonic_ns()
                    > config["arm_monotonic_ns"] + config["timeout_s"] * 1e9
                ):
                    raise CohortError(
                        "Cohort deadline expired; abort and await transport-fenced cleanup"
                    )
                certificate = self._proof(config, self.reports(config))
                certificate["release_decision_monotonic_ns"] = time.monotonic_ns()
                certificate["certificate_sha256"] = digest(certificate)
                atomic(folder / "release.json", certificate)
            return {
                "success": True,
                "status": "released",
                "ready_certificate": certificate,
                "release_decision_monotonic_ns": certificate[
                    "release_decision_monotonic_ns"
                ],
                "certificate_sha256": certificate["certificate_sha256"],
            }

    def abort(self, epoch, cohort_id, reason):
        with self.lock():
            folder = self.folder(epoch, cohort_id)
            config = self.config(epoch, cohort_id)
            # Preserve an existing release certificate. Already released ranks do
            # not poll control files during decode; cancel their HTTP requests too.
            path = folder / "abort.json"
            if not path.exists():
                atomic(
                    path,
                    {
                        "epoch": epoch,
                        "cohort_id": cohort_id,
                        "reason": str(reason)[:2000],
                        "abort_decision_monotonic_ns": time.monotonic_ns(),
                        "config_sha256": config["config_sha256"],
                    },
                )
            return {"success": True, "status": "aborting", "abort": read(path)}


class CohortGate:
    """Rank-local held events. sync() broadcasts leader decisions within TP8.

    All calls are from the scheduler control thread. Bind comes from the same
    broadcast request stream, so peers enter sync at identical program points.
    No sync/file reads are performed while a released cohort is decoding.
    """

    def __init__(self, store, *, global_rank, dp_rank, sync, emit):
        if global_rank not in range(16) or dp_rank != global_rank // 8:
            raise CohortError("Benchmark gate requires one D node with TP8 DP2")
        self.store = store
        self.rank = global_rank
        self.dp = dp_rank
        self.sync = sync
        self.emit = emit
        self.config = None
        self.bindings = {}
        self.pending = {}
        self.phase = "idle"
        self.certificate = None

    @property
    def holding(self):
        return self.phase in ("armed", "holding", "aborting")

    def _leader(self, operation):
        payload = None
        if self.rank % 8 == 0:
            try:
                payload = {"value": operation()}
            except (OSError, ValueError, KeyError) as error:
                payload = {"error": f"{type(error).__name__}: {error}"}
        payload = self.sync(payload)
        if "error" in payload:
            raise CohortError(payload["error"])
        return payload["value"]

    def _load(self, config):
        if (
            self.config is not None
            and self.config["config_sha256"] == config["config_sha256"]
        ):
            return
        if any(b["terminal"] is None for b in self.bindings.values()):
            raise CohortError("Cannot replace a cohort with live rank-local bindings")
        self.config = config
        self.bindings = {}
        self.pending = {}
        self.certificate = None
        self.phase = "armed"
        self._snapshot()

    def idle(self):
        """Observe arm/abort with no active generation; never called in steady decode."""
        config = self._leader(self.store.active)
        if config is not None:
            self._load(config)

    def _snapshot(self):
        if self.config is None:
            return
        bindings = list(self.bindings.values())
        if (
            self.phase == "released"
            and bindings
            and all(b["terminal"] is not None for b in bindings)
        ):
            self.phase = "complete"
        if self.phase == "aborting" and all(
            b["terminal"] is not None for b in bindings
        ):
            self.phase = "aborted"
        status = self.phase
        if (
            status == "holding"
            and len(bindings) == self.config["per_domain_batch"]
            and all(b["received"] and b["terminal"] is None for b in bindings)
        ):
            status = "ready"
        atomic(
            self.store.folder(self.config["epoch"], self.config["cohort_id"])
            / f"rank-{self.rank:02d}.json",
            {
                "schema_version": 1,
                "epoch": self.config["epoch"],
                "cohort_id": self.config["cohort_id"],
                "config_sha256": self.config["config_sha256"],
                "global_rank": self.rank,
                "dp_rank": self.dp,
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                "status": status,
                "bindings": bindings,
                "report_monotonic_ns": time.monotonic_ns(),
            },
        )

    def _emit(self, event, binding, **fields):
        self.emit(
            event,
            epoch=self.config["epoch"],
            cohort_id=self.config["cohort_id"],
            per_domain_batch=self.config["per_domain_batch"],
            **binding,
            **fields,
        )

    def bind(self, rid, room, input_sha256, *, input_tokens, output_tokens):
        tag = request_tag(rid)
        if tag is None:
            return
        config = self._leader(lambda: self.store.config(*tag))
        active = self._leader(self.store.active)
        if active is None or active["config_sha256"] != config["config_sha256"]:
            raise CohortError("Stale or unknown cohort request")
        self._load(config)
        if self.phase not in ("armed", "holding"):
            raise CohortError("Cohort no longer accepts requests")
        if (input_tokens, output_tokens) != (131072, 1024):
            self._leader(
                lambda: self.store.abort(
                    *tag, "Tagged request has wrong input/output size"
                )
            )
            raise CohortError(
                "Only explicitly tagged 128K->1024 requests may join a cohort"
            )
        expected = {
            r["input_sha256"] for r in config["requests"] if r["dp_rank"] == self.dp
        }
        if (
            input_sha256 not in expected
            or rid in self.bindings
            or any(
                b["input_sha256"] == input_sha256 or b["room"] == room
                for b in self.bindings.values()
            )
        ):
            self._leader(
                lambda: self.store.abort(
                    *tag, "Unexpected/duplicate actual request binding"
                )
            )
            raise CohortError("Unexpected hash/domain or duplicate rid/room/hash")
        try:
            if type(room) is not int or room < 0:
                raise CohortError("Actual bootstrap room must be a nonnegative integer")
            self._leader(
                lambda: self.store.claim(config, rid, room, input_sha256, self.dp)
            )
        except CohortError:
            self._leader(
                lambda: self.store.abort(
                    *tag, "Actual request/room is invalid or reused"
                )
            )
            raise
        binding = {
            "rid": rid,
            "room": room,
            "input_sha256": input_sha256,
            "dp_rank": self.dp,
            "received": False,
            "terminal": None,
            "p_cached_tokens": None,
            "d_hit_tokens": None,
            "bind_monotonic_ns": time.monotonic_ns(),
            "receive_monotonic_ns": None,
        }
        self.bindings[rid] = binding
        self.phase = "holding"
        self._snapshot()
        self._emit("cohort_bind", binding)

    def receive(self, rid, event, context):
        if rid not in self.bindings:
            return False
        binding = self.bindings[rid]
        if binding["received"] or binding["terminal"] is not None:
            raise CohortError("Duplicate/terminal receive completion")
        binding.update(
            received=True,
            receive_monotonic_ns=time.monotonic_ns(),
            p_cached_tokens=context["p_cached_tokens"],
            d_hit_tokens=context["d_hit_tokens"],
        )
        self.pending[rid] = event
        self._emit("cohort_receive", binding)
        self._snapshot()
        if binding["room"] != context["room"] or (
            binding["p_cached_tokens"],
            binding["d_hit_tokens"],
        ) != (
            self.config.get("expected_prefill_cached_tokens", self.config["expected_cached_tokens"]),
            self.config["expected_cached_tokens"],
        ):
            self._leader(
                lambda: self.store.abort(
                    self.config["epoch"],
                    self.config["cohort_id"],
                    "Actual room/cache hits disagree with arm",
                )
            )
        return True

    def fail(self, rid, reason):
        if rid not in self.bindings:
            return
        self._leader(
            lambda: self.store.abort(
                self.config["epoch"], self.config["cohort_id"], reason
            )
        )
        self.bindings[rid]["terminal"] = "aborted"
        self._emit("cohort_abort", self.bindings[rid], reason=reason)
        self.phase = "aborting"
        self._snapshot()

    def tick(self, cancelled):
        if not self.holding:
            return []
        config = self.config

        def decision():
            folder = self.store.folder(config["epoch"], config["cohort_id"])
            expired = (
                time.monotonic_ns()
                > config["arm_monotonic_ns"] + config["timeout_s"] * 1e9
            )
            if cancelled or (expired and not (folder / "release.json").exists()):
                self.store.abort(
                    config["epoch"],
                    config["cohort_id"],
                    "Cancellation or cohort deadline",
                )
            if (folder / "abort.json").exists():
                return {"action": "abort", "detail": read(folder / "abort.json")}
            if (folder / "release.json").exists():
                return {
                    "action": "release",
                    "certificate": read(folder / "release.json"),
                }
            return {"action": "wait"}

        command = self._leader(decision)
        if command["action"] == "wait":
            return []
        abort = command["action"] == "abort"
        if abort:
            self.phase = "aborting"
        else:
            certificate = command["certificate"]
            unsigned = {
                k: v for k, v in certificate.items() if k != "certificate_sha256"
            }
            if (
                digest(unsigned) != certificate["certificate_sha256"]
                or certificate["config_sha256"] != config["config_sha256"]
                or len(self.pending) != config["per_domain_batch"]
            ):
                raise CohortError(
                    "Invalid release certificate or incomplete local held cohort"
                )
            own = [
                r
                for r in certificate["actual_rank_reports"]
                if r["global_rank"] == self.rank
            ]
            if len(own) != 1 or own[0]["bindings"] != list(self.bindings.values()):
                raise CohortError("Certificate differs from actual local bindings")
            self.certificate = certificate
            self.phase = "released"
        output = []
        for rid, event in self.pending.items():
            binding = self.bindings[rid]
            if abort:
                binding["terminal"] = "aborted"
                self._emit("cohort_abort", binding, reason=command["detail"]["reason"])
            else:
                self._emit(
                    "cohort_release",
                    binding,
                    certificate_sha256=self.certificate["certificate_sha256"],
                    release_decision_monotonic_ns=self.certificate[
                        "release_decision_monotonic_ns"
                    ],
                )
            output.append((event, abort))
        self.pending.clear()
        self._snapshot()
        return output

    def committed(self, request_ids, live_ids):
        if self.phase != "released":
            return
        changed = False
        for rid in request_ids:
            if rid in self.bindings and rid not in live_ids:
                self.bindings[rid]["terminal"] = "complete"
                changed = True
        if changed:
            self._snapshot()
