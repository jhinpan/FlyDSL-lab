# MXFP4 MoE 2-Stage Tuning — Optimization Ledger (gfx950)

This ledger records every tuning attempt — win or loss — for the FlyDSL MXFP4
(per-1x32 microscale fp4) MoE 2-stage GEMM campaign on AMD gfx950 / MI350X.
Machine-readable attempt records live in [`attempts.jsonl`](attempts.jsonl); this
file is the human-facing running log.

## Reference

- Locked baseline: `upstream/main` @ `523ca1c7`, measured on a fixed idle
  MI350X (gfx950), ROCm 7.2, AITER installed.
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

_No attempts recorded yet. The locked baseline measurement is the first entry._
