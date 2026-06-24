#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measurement harness for the MXFP4 MoE 2-stage tuning campaign on gfx950.

The harness emits a per-point CSV that is the single reference table every
candidate is compared against.  Two measurement paths feed it:

* **Per-stage kernel-path us** comes from the FlyDSL ``tests/kernels/test_moe_gemm.py``
  benchmark, which prints ``FlyDSL MoE stage1[..]`` / ``FlyDSL MoE stage2 [..]``
  lines with per-stage us.  Combined kernel-path us = stage1 + stage2 + sorting.
* **Strict correctness + full fused-MoE e2e us** comes from the aiter
  ``op_tests/test_moe_2stage.py`` harness (``strict_accuracy``,
  ``logits_diff <= 0.01``, ``fail_on_aot_cache_miss``).  That harness times the
  whole ``fused_moe`` call as the e2e guardrail.

Every row records full provenance (GPU id+model, branch+commit, exact command,
shape, dtype+act, warmup/iters, idle-GPU check) and the resolved metric formula,
under the locked protocol in :mod:`kernels.moe_tuning_spec`.

This module keeps the parsing / metric / provenance / CSV logic as pure
functions so they are unit-testable without a GPU.  The live sweep driver
(:func:`run_point`) shells out to the two harnesses and is intended to run on the
fixed idle gfx950 node.
"""

from __future__ import annotations

import csv
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels import moe_tuning_spec as spec  # noqa: E402

# CSV columns: provenance first, then shape/config, then metrics.
CSV_COLUMNS = [
    # provenance
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
    "idle_gpu_verified",
    "graph_capture",
    "l2_flush_per_iter",
    "clocks_pinned",
    "metric_formula",
    # shape / config
    "model",
    "model_dim",
    "inter_dim",
    "experts",
    "topk",
    "dtype",
    "act",
    "token",
    "tile_m1",
    "tile_n1",
    "tile_k1",
    "tile_m2",
    "tile_n2",
    "tile_k2",
    # metrics (median + p95 over iters)
    "stage1_us",
    "stage2_us",
    "sorting_us",
    "kernel_path_us",
    "kernel_path_us_p95",
    "effective_tflops",
    "mfu",
    "e2e_us",
    "e2e_us_p95",
    "logits_diff",
    "correctness_pass",
]

METRIC_FORMULA = (
    "effective_tflops = token*model_dim*inter_dim*3*topk*2 / combined_us / 1e6; mfu = effective_tflops / 4523"
)

# Print formats from tests/kernels/test_moe_gemm.py:
#   "FlyDSL MoE stage1[fp4]: 1163.2 us, 1654.24 TFLOPS(logical, M=4608), 0.377 TB/s (...)"
#   "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | 7168x2048, ... | 1163.2 us, 1654.24 TFLOPS, 0.377 TB/s"
_STAGE1_RE = re.compile(r"FlyDSL MoE stage1\[[^\]]+\]:\s*([0-9.]+)\s*us")
_STAGE2_RE = re.compile(r"FlyDSL MoE stage2 \[[^\]]+\]\s+\S+\s+(atomic|reduce)\b.*?([0-9.]+)\s*us")


@dataclass
class Provenance:
    """Run provenance recorded with every measured point."""

    gpu_id: str = ""
    gpu_model: str = ""
    branch: str = ""
    commit: str = ""
    warmup: int = spec.WARMUP_ITERS
    iters: int = spec.BENCH_ITERS
    idle_gpu_verified: bool = False
    graph_capture: bool = spec.GRAPH_CAPTURE
    l2_flush_per_iter: bool = spec.L2_FLUSH_PER_ITER
    clocks_pinned: bool = spec.CLOCKS_PINNED
    metric_formula: str = METRIC_FORMULA

    REQUIRED_FIELDS = ("gpu_id", "gpu_model", "branch", "commit", "warmup", "iters")

    def missing_fields(self) -> List[str]:
        """Required provenance fields that are empty/unset (AC-1 negative gate)."""
        missing = []
        for f in self.REQUIRED_FIELDS:
            v = getattr(self, f)
            if v in ("", None):
                missing.append(f)
        return missing

    def is_complete(self) -> bool:
        return not self.missing_fields()


@dataclass
class PointRow:
    """One per-point measurement row (provenance + shape/config + metrics)."""

    provenance: Provenance
    command: str
    model: str
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    dtype: str
    act: str
    token: int
    tile_m1: int = 0
    tile_n1: int = 0
    tile_k1: int = 0
    tile_m2: int = 0
    tile_n2: int = 0
    tile_k2: int = 0
    stage1_us: Optional[float] = None
    stage2_us: Optional[float] = None
    sorting_us: Optional[float] = None
    kernel_path_us: Optional[float] = None
    kernel_path_us_p95: Optional[float] = None
    effective_tflops: Optional[float] = None
    mfu: Optional[float] = None
    e2e_us: Optional[float] = None
    e2e_us_p95: Optional[float] = None
    logits_diff: Optional[float] = None
    correctness_pass: Optional[bool] = None

    def to_csv_dict(self) -> dict:
        p = self.provenance
        row = {
            "gpu_id": p.gpu_id,
            "gpu_model": p.gpu_model,
            "branch": p.branch,
            "commit": p.commit,
            "command": self.command,
            "warmup": p.warmup,
            "iters": p.iters,
            "idle_gpu_verified": p.idle_gpu_verified,
            "graph_capture": p.graph_capture,
            "l2_flush_per_iter": p.l2_flush_per_iter,
            "clocks_pinned": p.clocks_pinned,
            "metric_formula": p.metric_formula,
        }
        for k in (
            "model",
            "model_dim",
            "inter_dim",
            "experts",
            "topk",
            "dtype",
            "act",
            "token",
            "tile_m1",
            "tile_n1",
            "tile_k1",
            "tile_m2",
            "tile_n2",
            "tile_k2",
            "stage1_us",
            "stage2_us",
            "sorting_us",
            "kernel_path_us",
            "kernel_path_us_p95",
            "effective_tflops",
            "mfu",
            "e2e_us",
            "e2e_us_p95",
            "logits_diff",
            "correctness_pass",
        ):
            row[k] = getattr(self, k)
        return row


# --- pure parsing / metric helpers (unit-testable, no GPU) -----------------


def parse_flydsl_stage_us(stdout: str) -> dict:
    """Extract stage1 / stage2 us from FlyDSL test_moe_gemm.py stdout.

    Returns ``{"stage1_us": float|None, "stage2_us": float|None}`` using the last
    matching line for each stage (the benchmarked, post-warmup print).
    """
    s1 = _STAGE1_RE.findall(stdout)
    s2 = _STAGE2_RE.findall(stdout)
    return {
        "stage1_us": float(s1[-1]) if s1 else None,
        "stage2_us": float(s2[-1][1]) if s2 else None,
    }


def combined_kernel_path_us(stage1_us: float, stage2_us: float, sorting_us: float = 0.0) -> float:
    """Combined kernel-path latency = stage1 + stage2 + sorting (microseconds)."""
    return float(stage1_us) + float(stage2_us) + float(sorting_us)


def summarize(samples: List[float]) -> dict:
    """Median + p95 over a list of per-iter latencies (the locked statistics)."""
    if not samples:
        return {"median": None, "p95": None}
    ordered = sorted(samples)
    median = statistics.median(ordered)
    # Nearest-rank p95.
    idx = max(0, min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1)))))
    return {"median": median, "p95": ordered[idx]}


def compute_metrics(*, token: int, model_dim: int, inter_dim: int, topk: int, combined_us: float) -> dict:
    """Effective TFLOPS + MFU for a combined kernel-path us, via the spec formula."""
    tflops = spec.effective_tflops(token, model_dim, inter_dim, topk, combined_us)
    return {"effective_tflops": tflops, "mfu": spec.mfu(tflops)}


# --- provenance collection (uses the host; safe no-ops when tools absent) ---


def _run(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return ""


def git_provenance(repo_root: str = _REPO_ROOT) -> dict:
    """Current branch + commit SHA of ``repo_root`` (empty strings on failure)."""
    branch = _run(["git", "-C", repo_root, "rev-parse", "--abbrev-ref", "HEAD"])
    commit = _run(["git", "-C", repo_root, "rev-parse", "HEAD"])
    return {"branch": branch, "commit": commit}


def gpu_provenance(gpu_id: str) -> dict:
    """GPU model name from rocm-smi for ``gpu_id`` (empty string on failure)."""
    out = _run(["rocm-smi", "--showproductname"])
    model = ""
    for line in out.splitlines():
        if "Card Series" in line:
            model = line.split(":")[-1].strip()
            break
    return {"gpu_id": str(gpu_id), "gpu_model": model}


def write_csv(rows: List[PointRow], path: str) -> None:
    """Write per-point rows to ``path`` using the fixed CSV schema."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow(r.to_csv_dict())


__all__ = [
    "CSV_COLUMNS",
    "METRIC_FORMULA",
    "Provenance",
    "PointRow",
    "parse_flydsl_stage_us",
    "combined_kernel_path_us",
    "summarize",
    "compute_metrics",
    "git_provenance",
    "gpu_provenance",
    "write_csv",
]
