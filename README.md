# DSA RoPE performance figures

Supporting figures and exact values for [TokenSpeed PR #2369](https://github.com/lightseekorg/tokenspeed/pull/2369).

- BF16 on Ascend 910B2C, CANN 9.0.0, torch-npu 2.9.0.post2.
- Operator results: 100 alternating A/B pairs, 20 calls per captured graph, device-event microseconds per call; no samples trimmed.
- Expanded candidates and final dispatcher rechecks are separate runs. Candidate cases outside the enabled range are shown even when slower; final dispatch keeps CANN for these cases.
- Single-layer A/B: the complete isolated DSA graph, original two-stream overlap, 100 paired rounds, profiler off, maximum latency across eight ranks. Only RoPE fusion differs. Median and mean results are both retained in layer-ab.csv.
- Q and K operator savings overlap and should not be added to predict layer savings.

These assets do not modify the kernel PR's source diff.
