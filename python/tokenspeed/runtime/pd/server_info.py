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

"""Cache-compatible identities for PD discovery, separate from execution."""

from collections.abc import Mapping


def canonical_pd_server_info(server_args: Mapping[str, object]) -> dict[str, object]:
    """Return discovery facts without mutating execution args or other facts.

    The explicit Q8 Prefill leaf inherits DSA's persistent KVP cache, index-K
    writes, transfer contract and Decode path. Only its temporary query/output
    layout and Prefill attention computation differ. Thus its PD cache identity
    is DSA. Keep its exact execution name visible as a separate response field.
    Unknown leaves and non-Prefill roles retain their original identities.
    """
    result = dict(server_args)
    backend = result.get("attention_backend")
    if (
        result.get("disaggregation_mode") == "prefill"
        and backend == "ascend_dsa_prefill_q8"
    ):
        existing = result.get("execution_attention_backend")
        if existing is not None and existing != backend:
            raise ValueError(
                "Conflicting execution attention backend in PD server info"
            )
        result["execution_attention_backend"] = backend
        result["attention_backend"] = "dsa"
    return result
