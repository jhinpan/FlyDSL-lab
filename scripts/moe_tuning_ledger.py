#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attempt ledger + Pareto comparison for the MXFP4 MoE tuning campaign.

Every candidate attempt — win or loss — is appended to ``docs/attempts.jsonl``
with full provenance (config, stage, model, dtype, act, GPU id+model,
branch+commit, command, warmup/iters, CSV/profile path, result).  A human-facing
running log lives in ``docs/optimization-ledger.md``.

The Pareto comparison takes a baseline per-point CSV and a candidate per-point
CSV (both emitted by ``scripts/moe_tuning_harness.py``) and reports, per point,
whether the candidate is a win / regression / neutral under the locked the win-margin policy /
the no-regression policy predicates.  A win is only claimable when no point regresses on either the
kernel-path or e2e metric (no Pareto regression) and the re-run-stability rule
holds.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from kernels import moe_tuning_spec as spec  # noqa: E402

ATTEMPTS_JSONL = os.path.join(_REPO_ROOT, "docs", "attempts.jsonl")
LEDGER_MD = os.path.join(_REPO_ROOT, "docs", "optimization-ledger.md")

# Required provenance keys for any ledger attempt (the ledger contract).
REQUIRED_ATTEMPT_FIELDS = (
    "config",
    "stage",
    "model",
    "dtype",
    "act",
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
    "result",
)


@dataclass
class Attempt:
    """One tuning attempt record (win or loss)."""

    config: dict
    stage: int
    model: str
    dtype: str
    act: str
    gpu_id: str
    gpu_model: str
    branch: str
    commit: str
    command: str
    warmup: int
    iters: int
    result: str  # "win" | "loss" | "rejected" | "neutral"
    csv_path: str = ""
    profile_path: str = ""
    note: str = ""
    timestamp: Optional[float] = None

    def missing_fields(self) -> List[str]:
        return [f for f in REQUIRED_ATTEMPT_FIELDS if getattr(self, f, None) in ("", None)]


def append_attempt(attempt: Attempt, path: str = ATTEMPTS_JSONL, now: Optional[float] = None) -> dict:
    """Append an attempt to the JSONL ledger.

    Raises ``ValueError`` if any required provenance field is missing, so a win
    can never be recorded without complete provenance (the ledger contract negative gate).
    """
    missing = attempt.missing_fields()
    if missing:
        raise ValueError(f"attempt missing required provenance fields: {missing}")
    rec = asdict(attempt)
    rec["timestamp"] = now if now is not None else time.time()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def read_point_csv(path: str) -> Dict[Tuple, dict]:
    """Read a per-point harness CSV keyed by (model, dtype, token, stage tiles).

    The key is (model, dtype, act, token) — the comparison axis between baseline
    and candidate at one shape/token point.
    """
    table: Dict[Tuple, dict] = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("model"), row.get("dtype"), row.get("act"), row.get("token"))
            table[key] = row
    return table


def _f(row: dict, col: str) -> Optional[float]:
    v = row.get(col)
    if v in (None, "", "None"):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


@dataclass
class PointVerdict:
    key: Tuple
    token: int
    kernel_path_regression: bool = False
    e2e_regression: bool = False
    large_shape_win: bool = False
    small_token_win: bool = False
    note: str = ""


def compare_point(baseline: dict, candidate: dict) -> PointVerdict:
    """Apply the win-margin policy / the no-regression policy predicates to one (baseline, candidate) point pair."""
    token = int(float(candidate.get("token") or baseline.get("token") or 0))
    key = (candidate.get("model"), candidate.get("dtype"), candidate.get("act"), candidate.get("token"))
    v = PointVerdict(key=key, token=token)

    b_kp, c_kp = _f(baseline, "kernel_path_us"), _f(candidate, "kernel_path_us")
    b_e2e, c_e2e = _f(baseline, "e2e_us"), _f(candidate, "e2e_us")
    b_mfu, c_mfu = _f(baseline, "mfu"), _f(candidate, "mfu")

    if b_kp is not None and c_kp is not None:
        v.kernel_path_regression = spec.is_regression(b_kp, c_kp)
    if b_e2e is not None and c_e2e is not None:
        v.e2e_regression = spec.is_regression(b_e2e, c_e2e)

    if spec.is_large_token(token) and token in spec.MFU_TARGET_BUCKETS:
        if b_mfu is not None and c_mfu is not None:
            v.large_shape_win = spec.is_large_shape_win(b_mfu, c_mfu)
    if spec.is_small_token(token):
        if b_kp is not None and c_kp is not None:
            v.small_token_win = spec.is_small_token_win(b_kp, c_kp)
    return v


def _required_fields_for_point(token: int) -> Tuple[str, ...]:
    """Comparison fields a candidate row must carry for its token regime.

    Every point needs both latency metrics; large target buckets additionally
    need ``mfu`` (the large-shape win/regression axis).
    """
    fields = ["kernel_path_us", "e2e_us"]
    if spec.is_large_token(token) and token in spec.MFU_TARGET_BUCKETS:
        fields.append("mfu")
    return tuple(fields)


def _row_missing_fields(row: dict, fields: Tuple[str, ...]) -> List[str]:
    return [f for f in fields if _f(row, f) is None]


@dataclass
class CampaignVerdict:
    points: List[PointVerdict] = field(default_factory=list)
    any_regression: bool = False
    large_wins: List[Tuple] = field(default_factory=list)
    small_wins: List[Tuple] = field(default_factory=list)
    missing_candidate_points: List[Tuple] = field(default_factory=list)
    incomplete_points: List[Tuple] = field(default_factory=list)

    @property
    def coverage_complete(self) -> bool:
        """True only if every baseline point has a candidate row with all the
        regime-required comparison fields present (no cherry-picking)."""
        return not self.missing_candidate_points and not self.incomplete_points

    @property
    def pareto_clean(self) -> bool:
        """True only if coverage is complete AND no point regressed on kernel-path
        or e2e.  Incomplete/cherry-picked candidate CSVs can never be clean."""
        return self.coverage_complete and not self.any_regression


def compare_csvs(baseline_csv: str, candidate_csv: str) -> CampaignVerdict:
    """Full per-point Pareto comparison of a candidate vs the locked baseline.

    Iterates the COMPLETE baseline key set so a candidate cannot pass by omitting
    a regressing/uncovered point.  A point with a missing candidate row, or whose
    candidate row lacks a regime-required field (kernel_path_us/e2e_us for every
    point; mfu for large target buckets), makes ``coverage_complete`` False, which
    forces ``pareto_clean`` False.

    A win is only claimable when ``pareto_clean`` holds (the no-regression policy + full coverage)
    AND at least one target-bucket / small-token win is present (the win-margin policy).
    Re-run-stability is enforced separately by re-running and re-comparing.
    """
    base = read_point_csv(baseline_csv)
    cand = read_point_csv(candidate_csv)
    cv = CampaignVerdict()
    for key, b_row in base.items():
        token = int(float(b_row.get("token") or 0))
        c_row = cand.get(key)
        if c_row is None:
            cv.missing_candidate_points.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="missing_candidate_point"))
            continue
        missing = _row_missing_fields(c_row, _required_fields_for_point(token))
        if missing:
            cv.incomplete_points.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="missing_fields:" + ",".join(missing)))
            continue
        pv = compare_point(b_row, c_row)
        cv.points.append(pv)
        if pv.kernel_path_regression or pv.e2e_regression:
            cv.any_regression = True
        if pv.large_shape_win:
            cv.large_wins.append(key)
        if pv.small_token_win:
            cv.small_wins.append(key)
    return cv


def repeatability_check(csv_a: str, csv_b: str) -> dict:
    """Compare two independent sweeps of the SAME config under the no-regression policy.

    For each shared (model, dtype, act, token) point, a metric is "stable" if the
    two runs agree within the the no-regression policy noise band (NOT a regression in either
    direction): ``|b - a| <= max(a*REGRESSION_REL, ABS_US_BAND)``.  Returns the
    set of unstable points per metric; an empty unstable set demonstrates the
    harness is repeatable (the measurement protocol).
    """
    a = read_point_csv(csv_a)
    b = read_point_csv(csv_b)
    shared = sorted(set(a) & set(b))
    unstable = {"kernel_path_us": [], "e2e_us": []}
    band = lambda x: max(abs(x) * spec.REGRESSION_REL, spec.ABS_US_BAND)  # noqa: E731
    for key in shared:
        for metric in ("kernel_path_us", "e2e_us"):
            va, vb = _f(a[key], metric), _f(b[key], metric)
            if va is None or vb is None:
                unstable[metric].append((key, "missing"))
            elif abs(vb - va) > band(va):
                unstable[metric].append((key, va, vb))
    return {
        "n_shared": len(shared),
        "unstable": unstable,
        "stable": not unstable["kernel_path_us"] and not unstable["e2e_us"],
    }


__all__ = [
    "ATTEMPTS_JSONL",
    "LEDGER_MD",
    "REQUIRED_ATTEMPT_FIELDS",
    "Attempt",
    "append_attempt",
    "read_point_csv",
    "compare_point",
    "compare_csvs",
    "repeatability_check",
    "PointVerdict",
    "CampaignVerdict",
]
