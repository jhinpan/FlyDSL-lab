# AC-3/AC-4 reachability finding: persist_m=1 is unreachable through aiter's stage2 dispatch/registry for a4w4 large shapes

Round 0 (loop3). Code-backed (read of `/sgl-workspace/aiter/aiter/ops/flydsl/moe_kernels.py`
and `aiter/aot/flydsl/moe.py`). This determines the FlyDSL-first-vs-aiter path
(DEC-1c) and what AC-3/AC-4 must actually change.

## The chain that picks persist_m in the strict aiter e2e + AOT path
1. Strict e2e: `op_tests/test_moe_2stage.py::test_fmoe` →
   `aiter/fused_moe.py` → `aiter/ops/flydsl/moe_kernels.py::flydsl_moe_stage2`.
2. AOT precompile: `aiter/aot/flydsl/moe.py::parse_csv` reads `kernelName2` from a
   tuned CSV → `get_flydsl_kernel_params(name)` → `_precompile_to_cache` →
   `flydsl_moe_stage2` (COMPILE_ONLY). Same wrapper, same cache key.

## Why persist_m=1 (the win) is unreachable for a4w4 large shapes
`flydsl_moe_stage2` (moe_kernels.py:1095-1103) maps the `persist` tri-state:
- `persist=True`  -> `_persist_m=-1` (persistent)
- `persist=False` -> `_persist_m = 4 if m_blocks>256 else 1`
- `persist=None`  -> `_persist_m = -1 if m_blocks>256 else 1`
- `a_dtype=="fp8"` override -> `_persist_m=1`
For a4w4 (`a_dtype!="fp8"`) with `m_blocks>256` (every GPT-OSS large-token point),
the THREE reachable values are {-1, 4, 1-only-if-m_blocks<=256}. The winning
**`_persist_m=1` at `m_blocks>256` is NOT producible by any `persist` value.**

The kernel NAME cannot express it either: `flydsl_kernel_name()` has no persist
field; `get_flydsl_stage2_kernels()` registers only `base_name` (persist absent ->
auto) and `base_name+"_persist"` (persist=True). There is NO registered stage2
kernel name whose params yield legacy `persist_m=1` for a large shape. So neither
a tuned-CSV `kernelName2` nor the runtime auto-path can select the candidate.

## Consequence for DEC-1 (FlyDSL-first, aiter fallback)
- The FlyDSL builder (`kernels/mixed_moe_gemm_2stage.py::compile_mixed_moe_gemm2`)
  DOES accept `persist_m=1` as a caller param and name-encodes it (`_pm1`); FlyDSL's
  own harness `tests/kernels/test_moe_gemm.py --persist_m2 1` already runs it (that
  is what R0 profiled). So FlyDSL-side the win is fully expressible and proven on
  the kernel path.
- BUT the aiter strict e2e + AOT gate (AC-4/AC-6 `aot_status=checked` +
  `claimable_win`) routes persist_m selection EXCLUSIVELY through the aiter
  `flydsl_moe_stage2` wrapper above, which lives in the AITER repo. There is no
  FlyDSL-only edit that makes aiter's strict gate execute `persist_m=1` for a4w4
  large shapes.
- Therefore, per DEC-1c, realizing the win in the gated path requires the AITER
  FALLBACK: a minimal, shape-guarded change to aiter `flydsl_moe_stage2` (and a
  registry name so the AOT CSV can select it) that lets a4w4 large shapes resolve
  `_persist_m=1`. This is the trigger condition the R0 goal-tracker queued.

## Proposed minimal aiter change (AC-3, shape-guarded, production-realizable)
Add a non-persistent legacy-`persist_m=1` stage2 variant reachable for a4w4 large
shapes, guarded to the saturated-grid region:
1. Registry: in `get_flydsl_stage2_kernels`, register a `base_name+"_pm1"` variant
   with params `{"persist": False, "persist_m_force": 1}` (or equivalent) so the
   AOT CSV `kernelName2` can name it and `get_flydsl_kernel_params` returns it.
2. Wrapper: in `flydsl_moe_stage2`, honor an explicit forced legacy persist_m
   (e.g. `persist_m_force`) -> `_persist_m=1` regardless of `m_blocks`, shape-
   guarded so only the validated a4w4 signature uses it; everything else unchanged
   (negative-control: default a4w4 large still resolves -1/4).
3. Dispatch (`fused_moe.py` / tuned CSV): point the GPT-OSS a4w4 large rows at the
   `_pm1` kernelName2 so production + AOT select it.

## Precise scope: which baseline rows actually need the change (m_blocks > 256)
Codex boundary nuance CONFIRMED + computed. The aiter wrapper threshold is strict
`m_blocks > 256`, and `m_blocks ≈ ceil((token*topk + E*block_m - topk)/block_m)`
capped at `token*topk`. Computed per a4w4 baseline row:
- ALL large-token rows (token >= 4096) for DS V3, Kimi K2, and GPT-OSS are
  `m_blocks > 256` → UNREACHABLE for persist_m=1 (need the change), across
  stage2 block_m ∈ {32,64,128}.
- The single corner `GPT-OSS token=4096 with block_m=128` gives m_blocks=256
  (NOT >256) → aiter-auto already resolves persist_m=1 there (no change needed,
  and it must be excluded from the negative-control's "before ∈ {-1,4}" claim).
- Small/mid tokens (<=2048) generally have m_blocks <= 256 → auto already picks
  persist_m=1 (and R0 showed the win is small there anyway). So the change is
  needed ONLY for the large-token (saturated-grid) rows — exactly the region R0's
  mechanism says the win lives. This is the AC-3 shape guard, now row-exact.

## Negative-control (AC-3) this makes possible
Assert: for the GPT-OSS a4w4 large shape, BEFORE the change the resolved
`_persist_m ∈ {-1,4}`; AFTER (with the `_pm1` variant selected) `_persist_m==1`.

## Status / honesty
This is a code-backed REACHABILITY finding, not a win. It converts the queued
side issue ("aiter decides the persist_m value") into the confirmed AC-3 work
item and confirms the DEC-1c fallback is REQUIRED for the gated win (FlyDSL-only
cannot reach it). Next: implement the minimal aiter variant + guard + negative-
control, then AC-4 (AOT precompile the new kernelName into the shared cache) and
AC-6 (strict e2e claimable_win). Kernel-path win remains FlyDSL-proven (R0).
