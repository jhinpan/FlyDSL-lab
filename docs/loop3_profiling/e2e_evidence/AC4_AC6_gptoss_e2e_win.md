# AC-4 + AC-3(e2e) + AC-6(partial): GPT-OSS a4w4 persist_m=1 e2e win, strict + AOT-checked

Round 1 (loop3). The real, correctness-gated, AOT-checked end-to-end optimization
the prior loop never produced. All via the verified path (no hand-rolled tooling).

## Result (strict aiter e2e, AOT-checked, correctness-gated)
GPT-OSS a4w4 (model_dim=3072, inter_dim=3072, E=128, topk=4, Swiglu), gfx950
MI350X GPU0, clocks pinned 2200MHz, warmup5/iters20. Baseline = production
persistent dispatch (`_persist_m=-1`, what aiter auto picks for these m_blocks>256
rows). Candidate = the new `_pm1` dispatch variant (`persist_m=1`).

| token | baseline e2e us | candidate e2e us | e2e speedup | p95 speedup | logits_diff | aot |
|-------|-----------------|------------------|-------------|-------------|-------------|-----|
| 8192  | 1394.20 | 1174.76 | **-15.7%** | -17.5% | 6.17e-06 | checked |
| 16384 | 2229.29 | 1966.45 | **-11.8%** | -8.2%  | 6.17e-06 | checked |
| 32768 | 4065.98 | 3529.92 | **-13.2%** | -13.9% | 6.18e-06 | checked |

Stage2 kernel-path reference (production persistent vs pm1): 8192 1041->697us
(-33%), 16384 1950->1182us (-39%). (R0 had compared vs the test-default pm4; the
TRUE production baseline for these rows is persistent, against which the win is
even larger.)

## Why these rows (scope, m_blocks > 256)
With the canonical tuned block_m, GPT-OSS a4w4 tokens 8192/16384/32768 have
m_blocks {384,1152,2176} > 256, so aiter auto resolves persistent `_persist_m=-1`
and CANNOT select persist_m=1. Tokens <=4096 already have m_blocks<=256 and auto
already resolves persist_m=1 (no change, and R0 showed the win is small there).
So the `_pm1` dispatch variant targets exactly the rows production mis-dispatches.

## What was proven (gates)
- AC-4 (AOT shared cache): clean cache -> `python -m aiter.aot.flydsl.moe --csv
  <candidate>` compiled 16 kernels (incl. the 3 `_pm1` stage2) into
  FLYDSL_RUNTIME_CACHE_DIR; strict run on the SAME cache recorded
  `check_aot_cache=true` with NO AOT miss -> aot_status=checked. The `_pm1`
  cache key == runtime key (the aot/flydsl/moe.py persist_m_force mirror works).
- AC-3 (runtime resolution, no fallback): aiter `[fused_moe] using 2stage`
  resolved `kernelName2=..._pm1` for all 3 large tokens under the candidate CSV;
  baseline CSV resolved `..._atomic` (persistent). Negative-control passes.
- AC-6 (correctness gate): every row logits_diff=6.2e-06 <= 0.01,
  correctness_pass=true, strict_accuracy=true.

## Honest scope / what remains for the FULL claim (DEC-2)
This is a verified e2e WIN on the GPT-OSS a4w4 large rows, but it is NOT yet the
full `compare_csvs(...).claimable_win` over the 40-row baseline, which DEC-2
requires (DS V3 + Kimi K2 + GPT-OSS all covered, non-regressing, in one candidate
CSV). Remaining:
- Build the full-coverage candidate CSV across all 40 a4w4 points (the harness
  measurement format with kernel_path_us/e2e_us/mfu/logits/aot/provenance), not
  just these 3 strict points.
- AC-8: replicate the `_pm1` dispatch + e2e proof for Kimi K2 and DS V3 large a4w4
  rows (their large tokens are also m_blocks>256 -> same unreachability -> same
  fix applies; needs measurement).
- Run `compare_csvs` over the full 40-row baseline; paste claimable_win; repeat
  once for stability.

## Provenance / replay
- aiter change: commit 5a5a7196a (`_pm1` variant + persist_m_force).
- candidate CSV: /tmp/aiter_cand/tuned_fmoe_pm1.csv (large GPT-OSS k2 -> _pm1).
- caches: /tmp/aiter_cand/cache_pm1 (candidate), cache_base (baseline).
- strict cmd: `FLYDSL_RUNTIME_CACHE_DIR=<cache> AITER_CONFIG_FMOE=<csv>
  python3 scripts/aiter_strict_point.py --model-dim 3072 --inter-dim 3072 -e 128
  -k 4 -t <T> --aq fp4 --wq fp4 --act swiglu --gate separated --warmup 5 --iters 20`.
- AOT cmd: `FLYDSL_RUNTIME_CACHE_DIR=<cache> python3 -m aiter.aot.flydsl.moe
  --csv /tmp/aiter_cand/gptoss_pm1_only.csv`.
- Evidence JSON: gptoss_a4w4_e2e_pm1_vs_baseline.json (this dir).
