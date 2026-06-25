# DS V3 a4w4 profiling pass — below-gate shapes (corrected)

DeepSeek V3 a4w4 (model_dim=7168, inter_dim=256, E=257, topk=9), baseline default
tiles (stage1 64/256/256; stage2 tile_n2/tile_k2 = 256/256), gfx950 / MI350X,
clocks pinned (rocm-smi --setperfdeterminism 2200, harness-verified) + idle
verified (rocm-smi use 0%), rocprofv3 PMC (warmup=2, iters=5).

Replay: `bash docs/loop2_profiling/collect.sh 0` (counter list:
`pmc_input.txt`; summary builder: `summarize.py`). Compact per-(token,stage)
counters with explicit units + derived ratios: `dsv3_a4w4_pmc_summary.csv`.
Timing split from `docs/baseline_523ca1c7_validated.csv`.

NOTE on units: the rocprofv3 `Grid_Size` column is total work-ITEMS;
`workgroups = grid_workitems / block_threads` (block_threads = 256 here).
`busy_cyc` is summed across CUs, so it is NOT compared to single-counter
`active_cyc`; ratios below normalize by `busy_cyc`.

## Timing split (baseline)

| token | kp us | stage1 | stage2 | sort | MFU | dominant |
|---|---|---|---|---|---|---|
| 32    | 179.8 | 102.9 (57%) | 76.9 (43%) | 0 | 0.0039 | stage1 by time |
| 64    | 203.0 | 113.4 (56%) | 89.6 (44%) | 0 | 0.0069 | stage1 by time |
| 16384 | 1902.7 | 669.1 (35%) | 1233.6 (65%) | 0 | 0.1886 | stage2 |
| 32768 | 3427.4 | 1048.9 (31%) | 2375.2 (69%) | 0 | 0.2095 | stage2 |

## Corrected counters (workgroups + ratios; ratio = counter / busy_cyc unless noted)

| token | stage | workgroups | lds_wait/busy | lds/valu | vmem/valu | L2 hit% |
|---|---|---|---|---|---|---|
| 32    | stage1 | 262   | 0.013 | 0.146 | 0.274 | 3.1 |
| 32    | stage2 | 1232  | **0.644** | 0.389 | 0.119 | 3.8 |
| 64    | stage1 | 266   | 0.015 | 0.146 | 0.274 | 3.1 |
| 64    | stage2 | 1652  | **0.730** | 0.391 | 0.121 | 3.7 |
| 16384 | stage1 | 2561  | 0.024 | 0.146 | 0.271 | 14.8 |
| 16384 | stage2 | 17052 | **0.230** | 0.517 | 0.256 | 28.5 |
| 32768 | stage1 | 4865  | 0.031 | 0.146 | 0.271 | 47.9 |
| 32768 | stage2 | 33152 | **0.260** | 0.520 | 0.260 | 34.0 |

## Corrected key signals

- **stage2 is LDS-wait bound at EVERY size, and WORST at small tokens.** stage2
  `lds_wait/busy` is 0.64 (t32) and 0.73 (t64) — i.e. stage2 spends most of its
  busy cycles waiting on LDS — and is still 0.23–0.26 at the large buckets. (My R6
  report wrongly called the small-token stage2 LDS wait "small"; it is the
  largest.) stage1 LDS wait is negligible (0.013–0.031) at all sizes.
- **stage2 is not HBM-bandwidth bound.** `vmem/valu` ≈ 0.12 (small) / 0.26 (large);
  L2 hit is low at small tokens (~3.8%) and rises to 28–48% at large tokens. The
  limiter is the LDS path / pipeline, not DRAM bandwidth.
- **Small-token grids are modest in workgroups but mostly empty.** stage2 launches
  1232/1652 workgroups for 32/64 tokens (E=257, per-expert block padding), and the
  kernel early-exits blocks past `num_valid_ids`. So small-token cost = many
  near-empty stage2 workgroups, each paying the high LDS-wait fixed cost.
- stage1 scales cleanly (VALU/issue bound, tiny LDS wait); it is not the lever
  target for either regime.

## persist_m semantics (corrected, from the builders)

- stage1 `persist_m` default = 1; the source comment states `persist_m>1`
  *serializes* M blocks that could run in parallel (K=model_dim is large, each CTA
  is already compute-heavy). A *larger* persist_m reduces grid_y via
  `gy = ceil(size_expert_ids / persist_m)` but serializes work — not obviously a
  small-token win for stage1, and stage1 is not the bottleneck anyway.
- stage2 `persist_m` default = 4; `persist_m <= 0` is **persistent mode**
  (`grid_y = cu_num`, each CTA round-robins M tiles). At small token counts the
  valid-block count can be < cu_num, so persistent mode does NOT necessarily
  reduce launches there. A *larger positive* stage2 `persist_m` (each CTA handles
  more consecutive M tiles → fewer workgroups) is the more plausible small-token
  launch-overhead lever, but it must be proven by measurement, not assumed.

## Bottleneck hypothesis + next lever, per shape

### tokens 32 & 64 (small-token latency)
Hypothesis: stage2 dominated by per-workgroup fixed cost across many near-empty
blocks, each with a very high LDS-wait fraction (0.64/0.73). **Next levers to try
(one at a time, measured):** (1) larger positive stage2 `persist_m` (fewer stage2
workgroups, each doing more M tiles) to cut launch/tail overhead; (2) stage2
`tile_n2`/`tile_k2` retune to lower the LDS-wait fraction. stage1 is not the
target. Do NOT assume stage2 `persist_m<=0` helps here.

### tokens 16384 & 32768 (large-bucket MFU)
Hypothesis: stage2 LDS-stall bound (lds_wait/busy 0.23–0.26, lds/valu ≈ 0.52) with
mediocre L2 reuse (28–34%). **Next levers:** stage2 `tile_n2`/`tile_k2` retune to
reduce LDS round-trips, and `xcd_swizzle` to raise L2 reuse. Not HBM-bound, so
async-copy/split-K are lower priority unless a retune profile shifts the limiter.

## Routing for the next round (one lever at a time, legality + comparator gated)

1. Both regimes share a stage2-LDS-pressure root cause → start with stage2
   `tile_n2`/`tile_k2` retune (helps small and large), measured pinned+idle reps=3,
   strict-correctness checked, comparator-gated (no Pareto regression).
2. Small tokens: additionally try larger positive stage2 `persist_m`.
3. Large buckets: additionally try `xcd_swizzle` for L2 reuse.
