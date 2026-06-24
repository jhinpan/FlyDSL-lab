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
  aiter-environment work outside the GEMM-tuning scope. a8w4 is quarantined
  pending a user scope decision — no a8w4 win may be claimed until it is green.
- fp4 peak (MFU denominator): **4523 TFLOPS** (empirical ceiling on this node).
- Metric formula: `effective_tflops = token*model_dim*inter_dim*3*topk*2 / combined_us / 1e6`;
  `mfu = effective_tflops / 4523`. Combined kernel-path us = stage1 + stage2 + sorting.
- Win / no-regression gates (locked): see `kernels/moe_tuning_spec.py`.
  - Large (tokens >= 4096): `tuned_MFU >= baseline_MFU * 1.10` on tokens {16384, 32768}.
  - Small (tokens <= 64): `tuned_us <= baseline_us * 0.90` AND `(baseline_us - tuned_us) >= 2 us`.
  - Regression iff `tuned > baseline * 1.02` AND `(tuned - baseline) > 2 us`, per point,
    on kernel-path AND e2e.
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
    `docs/baseline_523ca1c7_repeatability.json` — independent second sweep + DEC-2
    repeatability under the truthful timed-loop protocol. Kernel-path: 11/40
    points outside the band (worst ~4.6%, all small-token where absolute us is
    tiny); e2e (guardrail): 8/40 (worst ~7%). The true per-iteration timing is
    noisier than a profiler-rotated average; win-claims will need more reps or a
    tighter small-token band.
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
  scope. All a8w4 are quarantined (`moe_tuning_spec.QUARANTINED_SHAPES`); the a8w4
  scope question is OPEN for the user (a4w4-only tuning vs authorize aiter-wrapper
  work). No a8w4 win may be claimed until a8w4 e2e correctness is green.
- Status: the **a4w4 baseline is validated** (exit 0 over a4w4 keys). The default
  full-96 baseline remains a8w4-correctness-blocked, with fully auditable per-row
  a8w4 failure evidence. Tile-sweep tuning is NOT started; it awaits the user a8w4
  scope decision.
