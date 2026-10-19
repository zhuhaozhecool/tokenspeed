# Ascend DSA kernel selection

`AscendDSABackend` shares indexed sparse attention and KVP communication between
prefill and decode. Both callers pass the phase explicitly to
`_attention_kernels(is_prefill=...)`. The default returns the existing kernel
provider; model plugins may override it to select a prefill or decode provider
without duplicating cache addressing, query exchange or result merging.

Providers retain the existing `attention` and `attention_partial` contracts.
