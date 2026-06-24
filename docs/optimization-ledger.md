# MXFP4 MoE 2-Stage Tuning — Optimization Ledger (gfx950)

This ledger records every tuning attempt — win or loss — for the FlyDSL MXFP4
(per-1x32 microscale fp4) MoE 2-stage GEMM campaign on AMD gfx950 / MI350X.
Machine-readable attempt records live in [`attempts.jsonl`](attempts.jsonl); this
file is the human-facing running log.

## Reference

- Locked baseline ref: `upstream/main` @ `523ca1c7`, built in an isolated
  worktree and measured on a fixed idle MI350X (gfx950). The **validated** locked
  baseline is `docs/baseline_523ca1c7_validated.csv` — the 40 a4w4 points (DS V3,
  Kimi K2, GPT-OSS), measured via the strict/AOT/model-correct aiter guardrail
  (`scripts/aiter_strict_point.py`: `strict_accuracy=True`, AOT cache check, true
  per-model activation/gate, warmup=10/iters=100). It passes
  `validate_baseline_csv(expected_keys=validated_point_keys())` with all
  `correctness_pass=True`. **a8w4 (fp8×fp4) is correctness-BLOCKED** for all four
  models: under the strict path the non-fp4-activation e2e path fails the
  correctness gate (fp8 a8w4 AND bf16 a16w4 → `logits_diff ≈ 0.98`; only fp4
  activation passes). Root cause is an aiter-wrapper/layout contract mismatch for
  non-fp4 activation (NOT a FlyDSL kernel bug — this checkout's own
  `tests/kernels/test_moe_gemm.py --in_dtype a8w4` passes); fixing it is
  aiter-environment work outside the GEMM-tuning scope. a8w4 is quarantined;
  per the user-RESOLVED DEC-10 this campaign tunes a4w4 and DEFERS a8w4 (and the
  a8w4-only DeepSeek V4) with this reason; the aiter-wrapper fix is a stretch
  (DEC-10b) only if rounds remain after the a4w4 Pareto goal. No a8w4 win may be
  claimed until a8w4 e2e correctness is green.
- fp4 peak (MFU denominator): **4523 TFLOPS** (empirical ceiling on this node).
- Metric formula: `effective_tflops = token*model_dim*inter_dim*3*topk*2 / combined_us / 1e6`;
  `mfu = effective_tflops / 4523`. Combined kernel-path us = stage1 + stage2 + sorting.
- Win / no-regression gates (locked): see `kernels/moe_tuning_spec.py`.
  - Large (tokens >= 4096): `tuned_MFU >= baseline_MFU * 1.10` on tokens {16384, 32768}.
  - Small (tokens <= 64): `tuned_us <= baseline_us * 0.90` AND `(baseline_us - tuned_us) >= 2 us`.
  - Regression (DEC-9 regime-aware band): iff `tuned > baseline * 1.02` AND
    `(tuned - baseline) > abs_floor_us(token)`, per point, on kernel-path AND e2e,
    where `abs_floor_us = 8 us` for tokens <= 64 and `2 us` for tokens >= 128.
    The wider small-token floor absorbs the irreducible shared-node launch jitter
    (still << the 10% win margin, so win detection is unchanged).
- Protocol (identical for baseline and every candidate): warmup=10, iters=100,
  report median + p95, clocks pinned, graph-capture OFF, L2 flush per iter,
  idle-GPU verified.

## Hard gates (must hold for every selected candidate)

- aiter `op_tests/test_moe_2stage.py` with `strict_accuracy=True`,
  `logits_diff <= 0.01`, no FAIL/ERROR rows, for `QuantType.per_1x32` a4w4 / a8w4.
- AOT cache check (`fail_on_aot_cache_miss`) where the harness enforces it.
- Direct golden byte-layout comparison of preshuffled weight/scale vs aiter
  `ops/shuffle.py`.
- Unchanged output dtype and external kernel signature consumed by aiter `fused_moe`.

## Rules

- No win claimed from a single noisy near-threshold run; a win must hold across
  the full per-point table and a clean re-run within the noise band.
- One candidate change at a time unless coupling is technically necessary.
- Every entry names: candidate config, stage, model, dtype + act, GPU id + model,
  branch + commit, exact command, warmup/iters, CSV/profile path, and result.

## Attempts

<!-- Newest first.  Each entry mirrors an attempts.jsonl record. -->

### Candidate (PARTIAL DS-V3-subset small-token improvement) — DeepSeek V3 a4w4, stage1 `tile_n=128`

NOTE: this is a **partial** improvement, NOT a confirmed AC-4 win and NOT AC-3.
(Corrected after the Round-1 review caught an overclaim.)

- Lever: stage1 `tile_n` 256 → 128 (stage2 and stage1 tile_m/tile_k unchanged).
- Scope: a4w4 (per DEC-10). Protocol: warmup=10/iters=100, reps=3, clocks
  harness-verified pinned, regime-aware band (DEC-9). Two independent e2e sweeps
  via the candidate CLI; strict aiter e2e + AOT-cache ran (`aot_status=checked`),
  correctness pass (logits ≤ 0.0016 all 16 points).
- Small-token results (per the committed CSVs; latency % / MFU %):
  - token 1: −23.7% / +31.1%  ✓ clears the gate
  - token 16: −15.8% / +18.7% ✓ clears (tokens 1,2,4,8,16 all clear)
  - **token 32: −5.1% / +5.4%  ✗ FAILS the 10% gate**
  - **token 64: −3.9% / +4.1%  ✗ FAILS the 10% gate**
  AC-4 applies to the small-token set {1,2,4,8,16,32,64}; since 32 and 64 do NOT
  clear the 10% gate, this is **NOT a complete AC-4 win** — only a partial
  small-token (tokens 1–16) improvement.
- Large-MFU target buckets: **16384 MFU +9.75%** (BELOW the 10% margin, in both
  runs), **32768 MFU +5.80%** (below). `compare_csvs` reports no `large_wins`.
  → **NOT AC-3**.
- Pareto: `compare_csvs` over the **DS-V3-subset** baseline is coverage_complete +
  pareto_clean (0 regressions, kernel-path AND e2e), re-run stable on the win
  points. This is a DS-V3-subset statement only — the **full validated a4w4
  comparison is still missing 24 points** (Kimi K2 + GPT-OSS not yet swept), so it
  is NOT the plan's full a4w4 Pareto gate.
- Artifacts: `docs/candidate_dsv3_a4w4_stage1n128.csv` (run1, full per-point with
  e2e+correctness+aot), `docs/candidate_dsv3_a4w4_stage1n128_run2.csv` (stability
  re-run); candidates + the exact sweep command logged in `docs/attempts.jsonl`.

### Baseline — locked ref `523ca1c7` (strict path)

- Result: `baseline` (reference table; not a tuning attempt).
- Config: baseline default tiles per shape from `scripts/run_benchmark.sh`
  (stage1 64/256/256, or 32/128/256 for GPT-OSS; stage2 tile_n2/tile_k2 = 256/256).
- GPU: AMD Instinct MI350X (gfx950), `idle_gpu_verified=True`.
- Commit: `523ca1c7e224…` (isolated worktree build `flydsl-baseline-523ca1c7`).
- Protocol: warmup=10, iters=100, **true per-iteration timed-loop median + p95**
  (FlyDSL via `FLYDSL_PERF_DIST`; aiter e2e = rotated-average median +
  per-iteration p95), graph-capture OFF, clocks pinned. aiter e2e guardrail uses
  the strict/AOT/model-correct runner `scripts/aiter_strict_point.py`
  (`strict_accuracy=True`, true per-model activation/gate, AOT-cache check for
  a4w4) via `scripts/sync_aiter_flydsl_kernels.sh` (kernel overlay).
- CSVs:
  - `docs/baseline_523ca1c7_validated.csv` — the **40-point a4w4 baseline** (DS V3,
    Kimi K2, GPT-OSS a4w4), all `correctness_pass=True`, kernel-path + e2e
    median+p95; passes `validate_baseline_csv(validated_point_keys())` **valid=True,
    0 missing, 0 errors**. This is the validated reference for the in-scope a4w4 set.
  - `docs/baseline_523ca1c7_validated_run2.csv` +
    `docs/baseline_523ca1c7_repeatability.json` — two independent sweeps under the
    faithful L2-flush rotated protocol at reps=3 with clocks HARNESS-VERIFIED
    pinned (`setup_run_provenance` calls `pin_clocks` + `clocks_pinned_state`;
    `clocks_pinned=True` is now trustworthy, not a static default). Under the
    locked `max(2%, 2us)` band: kernel-path 9/40 unstable, e2e 7/40 unstable.
    CORRECTION (retracts an earlier "small-token-only" claim): the instability is
    NOT confined to tokens<=32 — kernel-path unstable tokens are {1,2,4,8,16,32,128}
    (incl. kimi_k2 token 128, 292.4->299.2us = 6.8us) and e2e unstable tokens are
    {1,2,4,32,64} (incl. a large kimi_k2 token-64 outlier 168.4->184.7us = 16.4us).
    With clocks harness-verified pinned, this is genuine run-to-run node variance
    across the low/mid token range, not just a tiny-token floor effect. In-protocol
    levers are EXHAUSTED (L2-flush rotation + reps=3 + verified clock pinning).
    Floor sensitivity: 2us->9/7, 3us->8/5, 5us->3/3, 6us->1/2, 10us->0/1, 20us->0/0.
    **RESOLVED by the user (DEC-9):** the no-regression/repeatability absolute band
    is now regime-aware — `max(2%, 8us)` for tokens<=64, `max(2%, 2us)` for
    tokens>=128. Under DEC-9 the residual reduces to kimi_k2/128 kernel-path
    (6.8us, mid-token watch — to re-measure under pinned clocks) and kimi_k2/64
    e2e (~16us, documented guardrail outlier; e2e is a guardrail not the tuning
    target).
  - `docs/baseline_523ca1c7.csv` — honest full 96-point record (40 a4w4 pass + 56
    a8w4 via the strict path, `correctness_pass=False`). Default
    `validate_baseline_csv` fails ONLY on the a8w4 correctness rows, 0 missing.
  - `docs/baseline_523ca1c7_a8w4_strict.csv` + `docs/a8w4_evidence.md` — the a8w4
    strict-path failure evidence with per-row `strict_error`, `error_category`,
    `aot_status`, and the FlyDSL command/tiles.
- **a8w4 correctness BLOCK (corrected; supersedes the earlier root cause):** under
  the strict, model-correct path the failing axis is the **non-fp4 activation**
  operand — fp8 (a8w4) AND bf16 (a16w4) both fail (`logits_diff ≈ 0.98`) with fp4
  weight; only fp4 activation (a4w4) passes (~1e-5). Root cause = an
  activation-dtype-dependent aiter weight/scale-prep + stage2 A2-scale CONTRACT
  mismatch (aiter uses `shuffle_weight_a16w4`/`shuffle_scale_a16w4` and
  `a2_scale=None` for non-fp4 activation; the FlyDSL mixed stage2 kernel expects a
  pre-scattered A2 E8M0 scale). It is **NOT a FlyDSL kernel math bug** — this
  checkout's own `tests/kernels/test_moe_gemm.py --in_dtype a8w4` passes with
  `--skip_ref false`. Fixing it is aiter-environment work outside the GEMM-tuning
  scope. All a8w4 are quarantined (`moe_tuning_spec.QUARANTINED_SHAPES`).
  **RESOLVED by the user (DEC-10):** this campaign tunes the a4w4 set; a8w4 (and
  DeepSeek V4, which is a8w4-only) are DEFERRED-with-reason, NOT abandoned. The
  out-of-scope aiter-wrapper fix is a stretch (DEC-10b) only if rounds remain
  after the a4w4 Pareto goal. No a8w4 win may be claimed until a8w4 e2e
  correctness is green.
- Status: the **a4w4 baseline is validated** (exit 0 over a4w4 keys) and a4w4
  tile-sweep tuning is UNDERWAY (DS V3 done; Kimi K2 / GPT-OSS next). The default
  full-96 baseline remains a8w4-correctness-blocked (DEC-10 deferred), with fully
  auditable per-row a8w4 failure evidence.
