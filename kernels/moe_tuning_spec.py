# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Locked specification for the MXFP4 MoE 2-stage tuning campaign on gfx950.

This is the single source of truth for the campaign's fixed parameters: the
target model shapes, the token sweep grid, the measurement protocol, the
win/no-regression predicates, the MFU denominator, and the routing-distribution
set used in correctness checks.  The measurement harness and the (later)
shape->config dispatch both import from here so the numbers live in exactly one
place.

All values are fixed inputs locked by the user before the campaign began; do not
change them as part of tuning.  Tuning changes tile configs, not these gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# --- MFU denominator -------------------------------------------------------
# Empirically measured fp4 GEMM ceiling on the target MI350X (gfx950, 256 CU,
# sclk max 2200 MHz).  MFU = effective_TFLOPS / FP4_PEAK_TFLOPS.
FP4_PEAK_TFLOPS = 4523.0

# --- Win margins (DEC-1) ---------------------------------------------------
WIN_MARGIN = 0.10  # 10% relative improvement required to claim a win.
# Large-shape (tokens >= LARGE_TOKEN_MIN): tuned_MFU >= baseline_MFU * (1 + WIN_MARGIN).
# Small-token (tokens <= SMALL_TOKEN_MAX): tuned_us <= baseline_us * (1 - WIN_MARGIN)
#   AND (baseline_us - tuned_us) >= ABS_US_BAND.

# --- No-regression tolerance + protocol (DEC-2) ----------------------------
REGRESSION_REL = 0.02  # 2% relative.
ABS_US_BAND = 2.0  # microseconds; also the DEC-1 small-token absolute floor.

WARMUP_ITERS = 10
BENCH_ITERS = 100
# Reported statistics per point.
REPORT_STATS = ("median", "p95")
# Protocol flags (recorded with every measurement; runs under other settings are
# non-comparable).
GRAPH_CAPTURE = False
L2_FLUSH_PER_ITER = True
CLOCKS_PINNED = True

# --- Token regimes (DEC-1 / DEC-3) -----------------------------------------
LARGE_TOKEN_MIN = 4096  # MFU regime.
SMALL_TOKEN_MAX = 64  # latency regime.
# Predeclared MFU target buckets (DEC-3): the two largest in-sweep tokens.
MFU_TARGET_BUCKETS: Tuple[int, ...] = (16384, 32768)

# --- Token grids (DEC-6) ---------------------------------------------------
TOKEN_GRID_FULL: Tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
    32768,
)
TOKEN_GRID_GPTOSS: Tuple[int, ...] = (256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# --- Routing distributions for correctness (DEC-7) -------------------------
ROUTING_DISTRIBUTIONS: Tuple[str, ...] = (
    "default",
    "uniform",
    "expert_skewed",
    "few_active",
    "all_active",
    "sentinel_padding",
)

# --- Node environment (DEC-8) ----------------------------------------------
TARGET_ARCH = "gfx950"


@dataclass(frozen=True)
class ModelShape:
    """One target MoE model shape and its in-scope quant dtypes.

    ``dtypes`` are the activation x weight quant aliases in scope for this loop:
    ``"a4w4"`` (fp4 x fp4) and/or ``"a8w4"`` (fp8 x fp4).  ``i4`` is out of scope.
    ``token_grid`` is the sweep used for this model (DEC-6).
    """

    name: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    act: str  # "silu" or "swiglu"
    dtypes: Tuple[str, ...]
    token_grid: Tuple[int, ...]


# The four target models (DEC-8 + plan workload table).  DeepSeek V4 is a8w4
# only; i4 (Kimi a16wi4) is excluded from this loop.
MODELS: Tuple[ModelShape, ...] = (
    ModelShape("deepseek_v3", 7168, 256, 257, 9, "silu", ("a4w4", "a8w4"), TOKEN_GRID_FULL),
    ModelShape("deepseek_v4", 7168, 512, 385, 7, "silu", ("a8w4",), TOKEN_GRID_FULL),
    ModelShape("kimi_k2", 7168, 256, 384, 8, "silu", ("a4w4", "a8w4"), TOKEN_GRID_FULL),
    ModelShape("gpt_oss", 3072, 3072, 128, 4, "swiglu", ("a4w4", "a8w4"), TOKEN_GRID_GPTOSS),
)

# Map a quant alias to the activation operand dtype passed to the kernel builder
# (the weight operand is fp4 in both in-scope cases).
DTYPE_ALIAS_TO_A_DTYPE = {"a4w4": "fp4", "a8w4": "fp8"}


def is_large_token(token: int) -> bool:
    """True if ``token`` is in the large-shape MFU regime (tokens >= 4096)."""
    return token >= LARGE_TOKEN_MIN


def is_small_token(token: int) -> bool:
    """True if ``token`` is in the small-token latency regime (tokens <= 64)."""
    return token <= SMALL_TOKEN_MAX


def is_regression(baseline_us: float, tuned_us: float) -> bool:
    """No-regression gate (DEC-2): regression iff BOTH the relative AND absolute
    bands are exceeded — ``tuned > baseline*1.02`` AND ``tuned-baseline > 2us``.

    Applied per point on BOTH the kernel-path and e2e metrics; a point is a
    regression if either metric regresses.
    """
    return (tuned_us > baseline_us * (1.0 + REGRESSION_REL)) and ((tuned_us - baseline_us) > ABS_US_BAND)


def is_large_shape_win(baseline_mfu: float, tuned_mfu: float) -> bool:
    """Large-shape win gate (DEC-1): ``tuned_MFU >= baseline_MFU * 1.10``."""
    return tuned_mfu >= baseline_mfu * (1.0 + WIN_MARGIN)


def is_small_token_win(baseline_us: float, tuned_us: float) -> bool:
    """Small-token win gate (DEC-1): both a relative and an absolute floor —
    ``tuned_us <= baseline_us*0.90`` AND ``(baseline_us - tuned_us) >= 2us``.

    The absolute floor rejects sub-microsecond percentage-only claims.
    """
    return (tuned_us <= baseline_us * (1.0 - WIN_MARGIN)) and ((baseline_us - tuned_us) >= ABS_US_BAND)


def effective_tflops(token: int, model_dim: int, inter_dim: int, topk: int, combined_us: float) -> float:
    """Combined effective TFLOPS per the aiter test_moe_2stage formula:
    ``token*model_dim*inter_dim*3*topk*2 / us`` (us in microseconds).
    """
    return token * model_dim * inter_dim * 3 * topk * 2 / combined_us / 1e6


def mfu(effective_tflops_value: float) -> float:
    """MFU = effective TFLOPS / fp4 peak (4523 TFLOPS)."""
    return effective_tflops_value / FP4_PEAK_TFLOPS
