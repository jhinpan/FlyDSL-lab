# DS V3 a4w4 profiling pass — below-gate shapes

Profiled DeepSeek V3 a4w4 (model_dim=7168, inter_dim=256, E=257, topk=9) baseline
default tiles (stage1 64/256/256, stage2 .../256/256) on gfx950 / MI350X, clocks
pinned + idle verified, via rocprofv3 PMC counters (warmup=2, iters=5). Raw
counter CSVs: `docs/loop2_profiling/pmc_t{32,64,16384,32768}.csv`. Baseline
timing split from `docs/baseline_523ca1c7_validated.csv`.

Counters per stage (summed over dispatches): grid size, SQ_WAVES, SQ_BUSY_CYCLES,
GRBM_GUI_ACTIVE, SQ_INSTS_VALU/VMEM/LDS, SQ_WAIT_INST_LDS, TCC L2 hit.

## Timing split (baseline)

| token | kp us | stage1 | stage2 | sort | MFU | dominant |
|---|---|---|---|---|---|---|
| 32    | 179.8 | 102.9 (57%) | 76.9 (43%) | 0 | 0.0039 | stage1 |
| 64    | 203.0 | 113.4 (56%) | 89.6 (44%) | 0 | 0.0069 | stage1 |
| 16384 | 1902.7 | 669.1 (35%) | 1233.6 (65%) | 0 | 0.1886 | stage2 |
| 32768 | 3427.4 | 1048.9 (31%) | 2375.2 (69%) | 0 | 0.2095 | stage2 |

## Key counter signals

- **Grid size is enormous at tiny token counts.** At token 32, stage1 launches a
  grid of 67,072 and stage2 322,560 workgroups — for only 32 tokens. The MoE
  launches (roughly) the full expert-block range and the kernel early-exits blocks
  beyond `num_valid_ids`. So small-token latency is dominated by launching and
  retiring mostly-empty blocks, not by useful FLOPs (MFU ~0.4-0.7%).
- **stage2 is LDS-stall-bound at large tokens.** At 16384/32768, stage2
  `SQ_WAIT_INST_LDS` is ~23-26% of its busy cycles and LDS instruction count is
  ~half of VALU (LDS/VALU ≈ 0.52). VMEM/VALU stays low (~0.26) and L2 hit is only
  28-48%, so stage2 is not HBM-bandwidth bound — it is bound on the LDS path /
  compute pipeline. stage2 also uses 2x the grid of stage1 (more, smaller tiles).
- stage1 VMEM/VALU ≈ 0.27 across all sizes with low LDS wait — stage1 is
  comparatively compute/issue bound and scales cleanly with tokens.

## Bottleneck hypothesis + next lever, per shape

### token 32 (small-token latency) — hypothesis: grid/launch-overhead bound
The 67K/322K-block grids for 32 tokens mean nearly all cost is block
launch/early-exit overhead, not compute. **Next lever:** `persist_m` (persistent
stage workgroups: launch ~`cu_num` groups that loop over the valid blocks instead
of one group per padded block), which directly removes the empty-block launch
overhead at small M. Secondary: reduce stage2 grid via a larger stage2 `tile_m2`
at small tokens (fewer, fuller tiles) — but persist_m is the primary.

### token 64 (small-token latency) — same class as token 32
Identical signature (grid 68K/423K, MFU 0.7%, LDS wait small). **Next lever:**
same — `persist_m` first; it should help 32 and 64 together.

### token 16384 (large-bucket MFU) — hypothesis: stage2 LDS-stall bound
stage2 is 65% of time with LDS wait ~23% of busy and LDS/VALU ≈ 0.5. **Next
lever:** reduce stage2 LDS pressure / increase overlap — try stage2 `tile_n2` /
`tile_k2` retuning to cut LDS round-trips, and `xcd_swizzle` to improve L2 reuse
(L2 hit only 28%). persist_m is NOT indicated here (grid is genuinely large and
full). Primary candidate: stage2 tile retune to lower LDS-wait fraction.

### token 32768 (large-bucket MFU) — same class as 16384
stage2 69% of time, LDS wait ~26% of busy. **Next lever:** same as 16384 — stage2
tile_n2/tile_k2 + xcd_swizzle to attack the LDS-stall / L2-reuse bottleneck.

## Routing for the next round (one lever at a time, legality-checked)

1. Small tokens (32/64): try **persist_m** on the stage builders first.
2. Large buckets (16384/32768): try **stage2 tile_n2/tile_k2 retune + xcd_swizzle**
   to reduce the stage2 LDS-stall fraction and raise L2 reuse.

Each candidate goes through the legality filter, is measured pinned+idle reps=3,
compared via the official comparator (no Pareto regression), and recorded with
full provenance before any win claim.
