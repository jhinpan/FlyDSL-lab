# MXFP4 MoE 2-Stage Tuning — Optimization Ledger (gfx950)

This ledger records every tuning attempt — win or loss — for the FlyDSL MXFP4
(per-1x32 microscale fp4) MoE 2-stage GEMM campaign on AMD gfx950 / MI350X.
Machine-readable attempt records live in [`attempts.jsonl`](attempts.jsonl); this
file is the human-facing running log.

## Reference

- Locked baseline ref: `upstream/main` @ `523ca1c7`, built in an isolated
  worktree and measured on a fixed idle MI350X (gfx950). Kernel-path metrics are
  recorded in `docs/baseline_523ca1c7_kernelpath.csv`. The full fused-MoE e2e
  guardrail column is pending an aiter harness env fix (see goal-tracker blocking
  issue); a win cannot be claimed until the e2e + strict-correctness columns are
  present and validated.
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

### Baseline — locked ref `523ca1c7` full (Round 2)

- Result: `baseline` (reference table; not a tuning attempt).
- Config: baseline default tiles per shape from `scripts/run_benchmark.sh`
  (stage1 64/256/256, or 32/128/256 for GPT-OSS; stage2 tile_n2/tile_k2 = 256/256).
- Scope: all 4 models × in-scope dtypes × full DEC-6 token grid = **96 points**.
- GPU: AMD Instinct MI350X (gfx950), `idle_gpu_verified=True`.
- Commit: `523ca1c7e224…` (isolated worktree build `flydsl-baseline-523ca1c7`).
- Protocol: warmup=10, iters=100, **median + p95** over reps=2, graph-capture OFF,
  L2 flush per iter (L2-rotation), clocks pinned.
- aiter e2e guardrail enabled via `scripts/sync_aiter_flydsl_kernels.sh` (overlays
  this checkout's MoE kernels onto aiter's stale 0.1.8-era vendored copies so the
  e2e path runs against the same kernels; strict correctness gated on
  `logits_diff <= 0.01` by the harness).
- CSVs:
  - `docs/baseline_523ca1c7.csv` — full 96-point sweep (kernel-path median+p95,
    e2e median+p95, logits_diff, correctness_pass).
  - `docs/baseline_523ca1c7_validated.csv` — the **56-point correctness-passing
    subset** (all a4w4 + DeepSeek V3 a8w4); passes
    `validate_baseline_csv(expected_keys=validated_point_keys())` with **valid=True,
    0 missing, 0 row errors**.
  - `docs/baseline_523ca1c7_run2.csv` + `docs/baseline_523ca1c7_repeatability.json`
    — independent second sweep + DEC-2 repeatability: **kernel-path is fully
    repeatable (0/96 unstable)**; e2e drifts up to ~10% at small tokens (tiny
    absolute us, host-dominated, reps=2).
- **Correctness quarantine (Round 2 finding):** a8w4 for **DeepSeek V4, Kimi K2,
  GPT-OSS** fails the aiter correctness gate (`logits_diff ≈ 0.99`; large GPT-OSS
  a8w4 also crashes/OOM). Root cause (confirmed against aiter source + Codex
  analyze): the aiter `test_moe_2stage.py` **legacy CLI path hardcodes
  ActivationType.Swiglu and GateMode.INTERLEAVE for the per_1x32 fp8×fp4 case**
  (`_iter_legacy_cases` ~L758, `_effective_gate_mode`), so Silu models are
  measured with a Swiglu+interleave kernel vs a Silu reference → near-total
  mismatch. This is a harness-path artifact, NOT a demonstrated FlyDSL kernel bug
  (a4w4 passes everywhere; DS V3 a8w4 passes through the same harness). These
  shapes are quarantined (`moe_tuning_spec.QUARANTINED_SHAPES`) and excluded from
  the validated baseline and from any win claim until validated via aiter's
  model-CSV mode.
- Status: **validated 56-point baseline is complete and passes its validator with
  exit 0.** Tile-sweep tuning may proceed on the validated subset; quarantined
  a8w4 shapes await the CSV-mode correctness fix.
