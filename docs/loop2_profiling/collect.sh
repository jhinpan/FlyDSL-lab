#!/usr/bin/env bash
# Replayable DS V3 a4w4 profiling collection for the below-gate shapes.
# Collects rocprofv3 PMC counters (see pmc_input.txt) for stage1+stage2 of the
# baseline default-tile MoE 2-stage kernel at tokens 32/64/16384/32768, then
# builds the compact provenance summary CSV.
#
# Pin clocks + verify idle BEFORE running (the harness does this in measurement
# mode; for a standalone profiling run, pin via rocm-smi --setperfdeterminism
# and check rocm-smi -d 0 --showuse == 0%).
#
# Usage:  bash docs/loop2_profiling/collect.sh [GPU]
set -euo pipefail
GPU="${1:-0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PMC="docs/loop2_profiling/pmc_input.txt"

for T in 32 64 16384 32768; do
  OUT="/tmp/pmc_t${T}"
  rm -rf "$OUT"; mkdir -p "$OUT"
  HIP_VISIBLE_DEVICES="$GPU" rocprofv3 -i "$PMC" -d "$OUT" --output-format csv -- \
    python3 tests/kernels/test_moe_gemm.py --in_dtype fp4 -dim 7168,256 -t "$T" -e 257 -k 9 \
    --tile_m 64 --tile_n 256 --tile_k 256 --tile_n2 256 --tile_k2 256 \
    --gemm2_mode atomic --skip_ref true --num_warmup 2 --num_iters 5
  cp "$OUT"/pmc_1/*/[0-9]*_counter_collection.csv "/tmp/pmc_raw_t${T}.csv"
done

python3 docs/loop2_profiling/summarize.py
echo "wrote docs/loop2_profiling/dsv3_a4w4_pmc_summary.csv"
