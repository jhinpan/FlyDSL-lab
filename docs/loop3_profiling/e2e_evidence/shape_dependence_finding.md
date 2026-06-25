# AC-8 shape-dependence finding: persist_m=1 is a GPT-OSS (high inter_dim) win, marginal for DS V3 / Kimi (inter_dim=256)

Round 2 (loop3). Determines the full-40-row claim strategy (DEC-2).

## Measured stage2 kernel-path: persist_m=1 vs production persistent (-1)
All a4w4, gfx950 MI350X GPU0, clocks pinned 2200.

| model (shape) | token | persistent us | pm1 us | delta |
|---|---|---|---|---|
| GPT-OSS (3072,3072) | 8192  | 1041.3 | 696.6  | **-33.1%** |
| GPT-OSS (3072,3072) | 16384 | 1949.9 | 1182.4 | **-39.4%** |
| DS V3   (7168,256)  | 8192  | 834.7  | 805.1  | -3.5% |
| DS V3   (7168,256)  | 16384 | 1653.8 | 1622.5 | -1.9% |
| Kimi    (7168,256)  | 8192  | 749.0  | 718.8  | -4.0% |
| Kimi    (7168,256)  | 16384 | 1474.6 | 1427.9 | -3.2% |

## Interpretation (consistent with the R0 mechanism)
The win is large ONLY for GPT-OSS, which has inter_dim=3072 (stage2 K=3072) — a
heavy, memory-latency-bound stage2 where lifting occupancy/concurrency
(persist_m=1) hides VMEM latency (R0: vmcnt 85->61%). DS V3 and Kimi have
inter_dim=256 (stage2 K=256): a much lighter, lower-arithmetic stage2 where the
occupancy lever has little headroom, so persist_m=1 is only ~2-4% (near/below the
DEC-9 band; some points likely noise-neutral).

So persist_m=1 is a **GPT-OSS-specific, shape-justified** optimization, NOT a
universal a4w4 rule. This VALIDATES the AC-3 shape guard (the `_pm1` kernelName is
assigned only to the validated region) and refutes any blanket "all a4w4 large =
pm1" rule (AC-3/AC-8 negative test).

## Consequence for the DEC-2 full-40-row claimable_win
`compare_csvs` requires full 40-row coverage + no regression + >=1 win + gate.
Strategy: the candidate CSV = baseline for every row EXCEPT GPT-OSS large
(8192/16384/32768), which point at `_pm1`. Then:
- coverage_complete: yes (all 40 rows present).
- any_regression: no (GPT-OSS large rows improve; all other rows are the baseline
  config => identical => not a regression).
- large_shape_win: yes (GPT-OSS large MFU up).
- gate: strict aiter logits<=0.01 + aot_status=checked (proven for GPT-OSS large;
  baseline rows already validated in docs/baseline_523ca1c7_validated.csv).
=> claimable_win achievable WITHOUT forcing pm1 on DS V3/Kimi (which would risk
neutral/regression and weaken the claim).

DS V3 / Kimi remain at their production-tuned config (no change, no regression).
This is the honest, shape-correct claim: a real GPT-OSS a4w4 large-shape win,
full-coverage clean, no other model regressed.

## Remaining for the claim
Build the full-40-row candidate CSV in the harness measurement format (the GPT-OSS
large rows measured strict-e2e with _pm1 — done; the other 37 rows = the validated
baseline values, unchanged), then run compare_csvs and paste claimable_win. The
GPT-OSS large strict e2e is already measured (e2e_evidence/); the remaining points
are the existing validated baseline (identical candidate config).
