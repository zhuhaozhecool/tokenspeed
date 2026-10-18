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

import pytest

from tokenspeed.runtime.execution.npu_graph_inputs import update_npu_graph_cpu_inputs


class Graph:
    def __init__(self, automatic, records):
        self.auto_dispatch_capture = automatic
        self.graph_dispatch_mode = Records(records)
        self.updates = []

    def update(self, **values):
        self.updates.append(values)


class Records:
    def __init__(self, records):
        self.graph_dispatch_records = records


class Lengths:
    def __init__(self):
        self.transfers = 0

    def to(self, device):
        assert device == "cpu"
        self.transfers += 1
        return self

    def tolist(self):
        return [127, 511]


@pytest.mark.parametrize(
    "automatic,records,skipped",
    [
        (True, [], True),
        (True, (), True),
        (False, [], False),
        (True, [1], False),
        (True, None, False),
    ],
)
def test_updates_only_skip_known_empty_dispatch(automatic, records, skipped):
    graph, lengths = Graph(automatic, records), Lengths()
    update_npu_graph_cpu_inputs(graph, lengths)
    assert lengths.transfers == int(not skipped)
    assert graph.updates == (
        []
        if skipped
        else [{"cpu_update_input": [{"actual_seq_lengths_kv": [127, 511]}]}]
    )
