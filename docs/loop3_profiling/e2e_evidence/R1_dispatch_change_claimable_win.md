# R1: claimable dispatch-change win (fresh paired, rigorous) — GPT-OSS a4w4 persist_m=1

Closes the R0-review claimability gap. The prior "full-40 claimable_win" was
rejected because 37/40 rows were copied from the locked baseline and the `_pm1`
selection lived only in a `/tmp` CSV. This round re-does it rigorously.

## Verdict (compare_csvs_dispatch_change, fresh paired)
```
coverage_complete: True
config_identity_clean: True   (31 unchanged rows config-identical to fresh pm4 baseline)
timed_clean: True             (changed rows: no kp/e2e regression)
large_wins: [gpt_oss/a4w4/swiglu/16384, gpt_oss/a4w4/swiglu/32768]
quarantined: 6                (DS V3 + Kimi t1/2/4 reference_invalid, issue #643)
gate.passed: True             (strict correctness + aot_status=checked on all reference-valid rows)
>>> claimable_dispatch_win: True
```

## The win (isolated, GPU0-only, sequential, reps=3, pinned 2200, idle)
| token | stage2 us (pm4→pm1) | kernel-path us | MFU (pm4→pm1) |
|-------|---------------------|----------------|----------------|
| 8192  | 932→698  (-25.1%)   | 1672→1438 (-14.0%) | 0.245→0.285 (+16.3%) |
| 16384 | 1576→1242 (-21.2%)  | 2895→2561 (-11.5%) | 0.283→0.320 (+13.0%) |
| 32768 | 3005→2341 (-22.1%)  | 5510→4851 (-12.0%) | 0.298→0.338 (+13.6%) |
Target buckets 16384/32768 both clear the 10% MFU win gate. (Plus the earlier
strict AOT-checked e2e -12..-16% in gptoss_a4w4_e2e_pm1_vs_baseline.json.)

## Why this is rigorous (not the rejected shortcut), per Codex option C+D
- **Fresh paired baseline**: both sides (candidate + pm4 baseline) freshly measured
  from the committed final state (FlyDSL f91b3394 / aiter 51f2969c5), same session.
  The locked snapshot is historical-reference only: identical-config rows drift
  wildly cross-session (DS V3 t64 +420%), so it is invalid as the numeric
  denominator. Freshness scan: 0 copied-baseline rows on either side.
- **Dispatch-only change**: the ONLY production edit is 3 GPT-OSS large kernelName2
  entries → `_pm1` in the committed aiter config. So the verdict TIMES only those
  3 changed rows (must win + no-regress) and requires the other 37 rows to be
  CONFIG-IDENTICAL to the baseline (same kernel/params ⇒ same production behavior).
  This avoids attributing irreducible small/mid-token node noise to the change.
- **Quarantine** (6 tiny rows): DS V3/Kimi t1/2/4 dispatch a CK kernel whose torch
  reference overflows to nan (#643) — classified error_category=reference_invalid,
  excluded from gate/regression but NEVER able to hide a real mismatch.

## Measurement-protocol lesson (this round)
Running 4 sweeps in parallel across GPUs 0-3 corrupted per-row timing via cross-GPU
contention (stage2 readings swung ±20% between runs; p95 up to 8800us). The clean,
consistent result required ISOLATED sequential measurement on a single quiet GPU.
This matches BL-small-token-latency-irreducible-noise (shared-node sensitivity).

## Honest scope
- The win is GPT-OSS a4w4 large-shape-specific (inter_dim=3072). DS V3/Kimi
  persist_m=1 is only 2-4% (below gate); they stay at pm4 (config-identical, no
  production change, no regression). This is NOT a literal "40-row empirical 2%
  no-regression" claim (unprovable on this shared node for unchanged rows); it is
  the dispatch-change criterion: full coverage + config-identity on unchanged rows
  + clean changed-row timing win + strict gate.
- Production durability: committed aiter config (gptoss_fp4_tuned_fmoe.csv) selects
  `_pm1` for GPT-OSS large by default (no env override); fail-closed no-fallback
  enforcement (AITER_REQUIRE_TUNED_FMOE / AITER_EXPECT_FMOE_KERNELNAME2) proven.

## Artifacts
- docs/loop3_models/candidate_fresh40.csv, baseline_fresh40_pm4.csv (+ fresh40/,
  fresh40_baseline/ per-model + isolated changed-row CSVs).
- scripts/loop3_assemble_fresh40.py, /tmp/final_verdict_iso.py (verdict driver).
- aiter production commit 51f2969c5; FlyDSL R1 code commit f91b3394.
