# aiter companion source state (exact, for strict-e2e replay)

The strict-e2e measurements in this loop run against the aiter checkout at
`/sgl-workspace/aiter`. This records the EXACT committed source state and confirms
the remaining untracked paths are generated/irrelevant (R3-review item).

## Committed source state (what the strict path actually uses)
- aiter HEAD: `d3cf8f89d` (and lineage 51f2969c5 → 43c873d69 → d3cf8f89d).
- The dispatch + production config + no-fallback + vendored FlyDSL kernel overlay
  are ALL committed:
  - `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv` (GPT-OSS large `_pm1`).
  - `aiter/ops/flydsl/moe_kernels.py` (`_pm1` variant + `persist_m_force`).
  - `aiter/fused_moe.py` (delegates to the no-fallback helper).
  - `aiter/ops/flydsl/no_fallback.py` (import-light enforce_no_fallback).
  - `aiter/ops/flydsl/kernels/*` (FlyDSL MoE overlay synced from FlyDSL a323f2a6).
  - `op_tests/flydsl_persist_pm1/*` (hermetic tests).

## Untracked paths — generated / pre-existing, NOT part of strict-replay source
Verified none is referenced by the fused-MoE / flydsl moe dispatch path:
- `3rdparty/HipKittens/` — unrelated 3rd-party checkout (HipKittens GEMM lib); not
  imported by `fused_moe.py` or `aiter/ops/flydsl/*` (grep: 0 refs). Pre-existing.
- `aiter/configs/profile_fmoe.csv` — dated 2026-06-03 (predates this work). Used
  ONLY under `AITER_ONLINE_TUNE` (fused_moe.py:910); the strict dispatch reads the
  MERGED tuned config, which globs `*tuned_fmoe*.csv` and excludes `profile_fmoe`.
  Not in the dispatch path.
- `aiter/jit/flydsl_cache/` — runtime JIT output directory (generated artifacts,
  not source). The strict runs use `FLYDSL_RUNTIME_CACHE_DIR=/tmp/loop3_committed_cache`.
- `aiter/ops/flydsl/kernels/.orig_bak/` — backups created by
  `scripts/sync_aiter_flydsl_kernels.sh` when overlaying FlyDSL sources; not
  imported (the live overlay files are imported, and they are committed).

## Replay
1. From FlyDSL: `bash scripts/sync_aiter_flydsl_kernels.sh /sgl-workspace/aiter`
   (idempotent; the overlay is already committed at d3cf8f89d).
2. AOT precompile from the committed merged config into a fresh cache:
   `FLYDSL_RUNTIME_CACHE_DIR=<dir> python3 -m aiter.aot.flydsl.moe --csv <a4w4 rows>`.
3. Strict run with the same cache + (for changed rows) `--expect-kernel-name2`.
The untracked paths above do not affect any of these steps.
