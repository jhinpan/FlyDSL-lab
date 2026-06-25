# Mechanism: why persist_m2=1 beats persist_m2=4 / aiter-auto on GPT-OSS a4w4 large

Round 0 (loop dir 2026-06-25_20-32-27). Profiler-backed (AC-1 / AC-2).
All evidence via the mandatory chain: `flyprof capture --with-pmc` (ATT+PMC) →
`flyprof bubbles` / `flyprof report` / `flyprof diff`; ROCmKernelWiki for prior
art. GPU: MI350X gfx950, GPU 0, clocks pinned 2200MHz (setperfdeterminism), idle.

## Configuration under test
- Shape: GPT-OSS a4w4, `-dim 3072,3072 -e 128 -k 4`, token=4096 (M_eff=16384).
- Stage2 GEMM `mfma_moe2_afp4_wfp4_f16_cshuffle_t32x256x256_vscale_fix3`.
- Baseline-auto: `persist_m2=4` (test default; aiter-auto resolves `_persist_m=-1`
  persistent for `m_blocks>256`, also a "few persistent WGs" regime).
- Candidate: `persist_m2=1` (one-tile-per-WG, non-persistent legacy).

## Kernel-path measurement (independent reproduction of the prior near-win)
- stage2 atomic: pm4 = 505.8us → pm1 = **384.6us** (~24% faster), 611→804 TFLOPS.
- stage2 reduce: pm4 = 545.2us → pm1 = 434.5us.
- Reproduces the prior loop's GPT-OSS persist_m2=1 kernel-path near-win.

## Artifacts (replayable)
- Baseline bundle: `docs/loop3_profiling/gptoss_t4096_auto/`
  (kernel `..._pm4`, ATT mapped 99.9%, 112 waves; report.json bound=memory).
- Candidate bundle: `docs/loop3_profiling/gptoss_t4096_pm1/`
  (kernel `..._pm1`, ATT mapped 99.9%, 432 waves; report.json bound=memory).
- `flyprof diff --before ...gptoss_t4096_auto --after ...gptoss_t4096_pm1`.

## Profile deltas (flyprof bubbles / diff)
| signal | pm4 (baseline) | pm1 (candidate) | delta |
|---|---|---|---|
| bound_type | memory | memory | — |
| rank-1 stall `vmcnt` (VMEM-wait bubble) | 85.08% | 61.08% | **-24.0 pts** |
| `vmem_load` (active load issue) | 3.23% | 27.2% | +23.97 pts |
| total stall | 93.2% | 94.7% | +1.5 |
| ATT waves traced | 112 | 432 | ~4x |
| waves/CU (occupancy) | 12 | 16 | +4 |
| arch_vgpr / thread | 167 | 99 | -68 |

## Mechanism (hypothesis, profile-delta-backed; corrected per Codex AC-2 cross-check)
The GPT-OSS a4w4 stage2 is **memory-latency-bound**: 93% stall, rank-1 bubble is
`vmcnt` (waiting on outstanding global `buffer_load`s) at
`kernels/mixed_moe_gemm_2stage.py:2719`, waiting on loads at lines 3609/3276
(per `flyprof report` baseline).

CORRECTED CAUSAL STATEMENT (do not over-identify "tiling alone" vs occupancy —
they are coupled in this kernel): **`persist_m2=1` changes the stage2 code shape
so the large GPT-OSS a4w4 kernel has lower VGPR pressure (arch_vgpr 167→99) and
many more concurrent waves/WGs (waves/CU 12→16, ATT waves 112→432); this improves
VMEM-latency hiding.** `persist_m2=4` (and aiter-auto persistent `_persist_m=-1`)
packs 4 M-tiles into each WG, carrying more per-WG loop/state; with arch_vgpr 167
that caps occupancy at 12 waves/CU and ~112 concurrent waves — too few to cover
the long VMEM latency, so the kernel sits in `vmcnt` 85% of the time. The simpler
`persist_m2=1` shape lifts concurrency, converting idle `vmcnt` waiting (85%→61%)
into active `vmem_load` issue (3%→27%) and cutting stage2 ~24%. Total stall barely
moves (93→95%) because the kernel is still memory-bound — it does the SAME memory
work in less wall-time by hiding latency across more waves (not by reducing
traffic).

Atomic-contention is NOT the driver: the same `accumulate=True` atomic epilogue
path is used for both pm4 and pm1, and the **reduce** variant (non-atomic) ALSO
wins (545.2→434.5us at token 4096), which a contention explanation would not
predict. This strengthens the generic stage2 latency-hiding explanation.

This matches ROCmKernelWiki prior art INVERTED on purpose: persistent kernels win
when there is a **tail** to absorb or launch cost to amortize; here `flyprof tile`
shows the grid is **saturated (tail_tiles=0)**, so persistence buys nothing on the
tail and instead costs concurrency. See `prior_art_persist_m.md`.

## Small-token provenance correction (Codex AC-2 cross-check)
At token=256 flyprof's "biggest kernel" heuristic captured **stage1**
(`mfma_moe1_silu_mul_afp4_wfp4_f16_t32x128x256_pm1_v32`), NOT stage2 — because at
small M stage1 dominates. Therefore the `gptoss_t256_*` BUBBLE numbers are stage1
and must NOT be presented as stage2 evidence. What IS valid at token=256 is the
stage2 TIMING delta from the test's own per-stage print (pm4 112.4us → pm1
102.6us atomic, ~9%) and the `flyprof tile` grid verdict (underfilled-25pct,
192 tiles < 256 ctas). To get small-token stage2 bubbles a separate stage2-pinned
capture (kernel filter) would be needed; deferred because the small-token regime
is not where the win lives (see shape bound) and AC-1's ">=1 small + >=1 large"
is met for the saturated-grid mechanism via the large point; the small point's
ROLE here is only to bound the shape region (timing + grid), which it does.

## Shape bound (informs the AC-3 dispatch guard)
The win holds where the stage2 grid is **saturated** (large M_eff = tokens×topk on
this small-expert-tile shape) AND the kernel is occupancy/latency bound (large
token: −24%, vmcnt 85→61, occ 12→16). It does NOT hold meaningfully where the grid
is underfilled (token=256: timing only −9%; grid 25%). Per the Codex cross-check,
the AC-3 dispatch guard should key on a STRUCTURAL RUNTIME predicate, not a
profiler-only signal: e.g. (exact GPT-OSS a4w4 stage2 tile/signature) AND
(total launched stage2 CTAs >= cu_num, i.e. saturated grid) AND (current auto
would choose persistent / pm4). Do NOT key dispatch on `vmcnt%` (needs profiling,
drifts with codegen); use occupancy/arch_vgpr only as a post-compile validation
invariant. `m_blocks > 256` matters because that is the exact condition under
which aiter auto currently picks persistent for fp4 (moe_kernels.py:1091,
aot/flydsl/moe.py:654), but "launched CTAs vs CU count" is the more direct guard.
The rule must be per-point measured-covered (AC-3 negative test forbids a blanket
"all GPT-OSS large = pm1" rule).

## Status
AC-1 MET: mandatory chain operationalized (flyprof capture --with-pmc → bubbles/
report/diff; ROCmKernelWiki prior art); both configs profiled at a large-token
point (stage2) with artifacts; small-token point bounds the shape region (timing +
grid; stage1-bubble caveat recorded). AC-2 MET: ONE mechanism hypothesis,
profile-delta-backed, independently cross-checked by Codex (PARTIALLY CONFIRMED →
over-identification + small-token-artifact wording corrected above; atomic-
contention ruled out via the reduce-variant control).
NOT a win yet — kernel-path only, no e2e/AOT/correctness gate. Codex explicitly
cautions: mechanism justifies a GUARDED dispatch experiment (AC-3), not landing a
dispatch change without strict e2e/AOT validation (AC-4+). Next: AC-4 cache path +
AC-3 shape-guarded dispatch.
