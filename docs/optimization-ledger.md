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

### Baseline — locked ref `523ca1c7` kernel-path (Round 1)

- Result: `baseline` (reference table; not a tuning attempt).
- Config: baseline default tiles per shape from `scripts/run_benchmark.sh`
  (stage1 64/256/256, or 32/128/256 for GPT-OSS; stage2 tile_n2/tile_k2 = 256/256).
- Scope: all 4 models × in-scope dtypes × full DEC-6 token grid = **96 points**.
- GPU: AMD Instinct MI350X (gfx950), `idle_gpu_verified=True`.
- Commit: `523ca1c7e224…` (isolated worktree build `flydsl-baseline-523ca1c7`).
- Protocol: warmup=10, iters=100, graph-capture OFF, L2 flush per iter, clocks pinned.
- CSV: `docs/baseline_523ca1c7_kernelpath.csv` (kernel-path us, effective TFLOPS,
  MFU present for every point).
- Status: kernel-path metrics complete and validated (`validate_baseline_csv`
  reports 0 missing points, all rows from the locked commit/idle/protocol). The
  full fused-MoE **e2e guardrail** and strict-correctness columns are still empty
  — the aiter `op_tests/test_moe_2stage.py` run fails under the current env with
  `AttributeError: 'Int32' object has no attribute 'type'` (flydsl/aiter version
  mismatch). No tuning win may be claimed until those columns are filled and
  validated.
