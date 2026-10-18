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

import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from tokenspeed.runtime.pd.audit import TransferAudit, emit_pd_audit


class AuditTests(unittest.TestCase):
    def read_events(self, directory):
        paths = list(Path(directory).glob("*.jsonl"))
        return [
            json.loads(line) for path in paths for line in path.read_text().splitlines()
        ]

    def test_disabled_creates_no_file(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"TOKENSPEED_PD_AUDIT_DIR": ""}),
        ):
            emit_pd_audit("route", role="decode", global_rank=8, dp_rank=1, room=7)
            self.assertEqual(self.read_events(directory), [])

    def test_threads_preserve_complete_events(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"TOKENSPEED_PD_AUDIT_DIR": directory}),
        ):

            def write(index):
                emit_pd_audit(
                    "route",
                    role="decode",
                    global_rank=8,
                    dp_rank=1,
                    room=index,
                    rid=f"request-{index}",
                )

            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(write, range(64)))
            events = self.read_events(directory)
            self.assertEqual(len(events), 64)
            self.assertEqual({e["room"] for e in events}, set(range(64)))
            self.assertEqual(
                [e["sequence"] for e in events], sorted(e["sequence"] for e in events)
            )
            self.assertTrue(
                all(e["global_rank"] == 8 and e["dp_rank"] == 1 for e in events)
            )
            self.assertTrue(all(e["monotonic_ns"] > 0 for e in events))

    def test_transfer_success_and_failure_do_not_double_count_plan(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(os.environ, {"TOKENSPEED_PD_AUDIT_DIR": directory}),
        ):
            for code in (0, -1):
                audit = TransferAudit(
                    room=11,
                    global_rank=2,
                    dp_rank=0,
                    source_tp_rank=2,
                    destination_tp_rank=5,
                )
                audit.field(
                    group="history",
                    field="latent",
                    source_owner=2,
                    destination_owner=5,
                    pages=3,
                    bytes_per_page=128,
                    descriptors_per_page=1,
                )
                audit.field(
                    group="state",
                    field="ssm",
                    source_owner=0,
                    destination_owner=0,
                    pages=1,
                    bytes_per_page=512,
                    descriptors_per_page=2,
                )
                audit.done(code)
            events = self.read_events(directory)
            done = [e for e in events if e["event"] == "transfer_done"]
            self.assertEqual([e["planned_payload_bytes"] for e in done], [896, 896])
            self.assertEqual([e["submitted_bytes"] for e in done], [896, None])
            self.assertEqual([e["success"] for e in done], [True, False])
            self.assertEqual([e["planned_descriptors"] for e in done], [5, 5])
            self.assertEqual(len({e["transfer_id"] for e in done}), 2)
            for event in done:
                fields = [
                    e
                    for e in events
                    if e["event"] == "transfer_plan"
                    and e["transfer_id"] == event["transfer_id"]
                ]
                self.assertEqual(
                    sum(e["planned_payload_bytes"] for e in fields),
                    event["planned_payload_bytes"],
                )


if __name__ == "__main__":
    unittest.main()
