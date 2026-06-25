# Prior art (ROCmKernelWiki) — persist_m / persistent-kernel scheduling

Source: `external/ROCmKernelWiki/wiki/techniques/persistent-kernel.md`,
`wiki/patterns/tail-effect.md`, `wiki/patterns/low-occupancy.md` (gfx950).

## What the wiki says persistent mode is FOR
- `persistent` (grid = CUs × wgs_per_cu, each WG grid-strides over many tiles)
  helps via: (1) amortized launch/setup, (2) **tail-effect mitigation** (no short
  under-occupied final wave), (3) L2/XCD locality, (4) stable state across tiles.
- one-tile-per-WG (default) pays per-tile launch and leaves a tail.

## Why this is INVERTED for GPT-OSS a4w4 large (the near-win)
- Measured: `persist_m2=1` (NON-persistent legacy, one-tile-per-WG-ish) BEATS the
  default `persist_m2=4` and aiter-auto `_persist_m=-1` (persistent) — stage2
  atomic 384.6us vs 505.8us at token 4096 (M_eff=16384), ~24% faster.
- `flyprof tile moe_gemm --shape 16384,3072,3072 --dtype fp4`: grid
  `total_tiles=3072, target_ctas=256, full_waves=12, tail_tiles=0,
  tail_verdict="saturated"` → the grid is SATURATED with ZERO tail.
- So the persistent mode's headline benefit (tail mitigation) does NOT apply here:
  there is no tail to absorb. Meanwhile `persist_m>1` / persistent serializes
  multiple M-tiles into each WG, which on a saturated grid REDUCES the number of
  concurrently-resident WGs available for latency hiding.
- `flyprof tile` also reported the recommended tile is **LDS/occupancy-bound**
  (occupancy_pct≈25%, binding_limiter LDS). On a low-occupancy, saturated-grid,
  latency-bound stage2, maximizing concurrent WGs (persist_m=1) should hide more
  memory latency than packing tiles into fewer persistent WGs.

## Hypothesis direction (to confirm with PMC/ATT deltas, AC-2)
persist_m=1 wins because the GPT-OSS a4w4 stage2 grid is already saturated (no
tail for the persistent mode to fix), and at ~25% occupancy the kernel is
latency-bound; more concurrent one-tile WGs (persist_m=1) hide memory/MFMA-issue
latency better than fewer persistent WGs each serializing 4 M-tiles. Shape bound:
the win should hold where the grid is saturated (large tokens) and NOT where the
grid is small enough that the tail/launch amortization of persistence dominates.

To be VERIFIED against the captured profile deltas (occupancy/waves, kernel
duration distribution, VMEM/LDS wait, MFMA issue) — not asserted from prior art
alone (AC-2 negative test: narrative-only hypotheses are rejected).
