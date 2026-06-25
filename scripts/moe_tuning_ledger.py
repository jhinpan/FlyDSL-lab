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
from typing import Dict, List, Optional, Set, Tuple

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

# A rejected search candidate never reaches compile/GPU, so it has no measured
# metrics (csv_path/profile_path stay empty), but it MUST still carry the same
# identity + run-provenance class as a measured attempt so the rejection is
# auditable (the rejected-candidate ledger contract).  ``stage`` is 0 when the
# rejection is at the candidate-tile level spanning both stages; the reason
# string still names the offending stage.  ``selection`` records the run's
# model/dtype/tokens filter so the rejection is reproducible.
REQUIRED_REJECTED_FIELDS = (
    "model",
    "dtype",
    "act",
    "token",
    "stage",
    "config",
    "reason",
    "selection",
    "gpu_id",
    "gpu_model",
    "branch",
    "commit",
    "command",
    "warmup",
    "iters",
)

# Keys that must be PRESENT on a rejected record but may legitimately be empty
# strings: a pre-compile rejection produces no measured CSV/profile artifact, yet
# the keys must exist so the record schema matches a measured attempt.
REQUIRED_REJECTED_PRESENT_KEYS = (
    "csv_path",
    "profile_path",
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


def append_rejected_candidate(record: dict, path: str = ATTEMPTS_JSONL, now: float = None) -> dict:
    """Append a machine-readable rejected-candidate record to the JSONL ledger.

    ``record`` must carry the full provenance class (``REQUIRED_REJECTED_FIELDS``)
    so a rejected search candidate is as auditable as a measured attempt — even
    though it never reached compile/GPU.  The measured-artifact keys
    (``REQUIRED_REJECTED_PRESENT_KEYS``: ``csv_path``/``profile_path``) must be
    present but may be empty strings (no artifact exists pre-compile).  Raises
    ``ValueError`` if any required field is missing, so an incomplete rejection can
    never be recorded (the rejected-candidate contract negative gate).
    """
    # Treat only None / "" as missing — integer 0 (stage, warmup, iters) is valid.
    missing = [k for k in REQUIRED_REJECTED_FIELDS if record.get(k) in (None, "")]
    # Artifact keys must EXIST (empty string allowed); only a truly absent key fails.
    missing += [k for k in REQUIRED_REJECTED_PRESENT_KEYS if k not in record]
    if missing:
        raise ValueError(f"rejected-candidate record missing fields: {missing}")
    # selection must be a non-empty dict so the rejection's run filter is recorded.
    sel = record.get("selection")
    if not isinstance(sel, dict) or not sel:
        raise ValueError("rejected-candidate record 'selection' must be a non-empty dict")
    rec = {"result": "rejected_candidate", **record}
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
        v.kernel_path_regression = spec.is_regression(b_kp, c_kp, token=token)
    if b_e2e is not None and c_e2e is not None:
        v.e2e_regression = spec.is_regression(b_e2e, c_e2e, token=token)

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
    # Documented reference_invalid rows (issue #643) excluded from coverage/gate.
    quarantined: List[Tuple] = field(default_factory=list)
    # Strict correctness + AOT-cache hard gate over the candidate CSV
    # (``selected_candidate_gate`` output).  Populated by ``compare_csvs``; a
    # candidate that fails this gate (e.g. ``aot_status=no_aot``) can never be a
    # claimable win even if its metrics look winning.
    gate: dict = field(default_factory=lambda: {"passed": False, "n_rows": 0, "violations": []})

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

    @property
    def claimable_win(self) -> bool:
        """The SINGLE source of truth for whether a candidate may be promoted to a
        win.  True only when ALL hold:
        - ``pareto_clean`` (full coverage + no kernel-path/e2e regression),
        - at least one target-bucket or small-token win is present, and
        - the strict correctness + AOT-cache hard gate passed
          (``aot_status=checked`` + correctness + ``logits_diff<=0.01`` on every
          row) -- so a ``no_aot`` / failed-correctness candidate is never claimable
          regardless of how good its metrics look.
        Re-run stability is enforced separately by re-running and re-comparing."""
        return self.pareto_clean and bool(self.large_wins or self.small_wins) and bool(self.gate.get("passed"))


CONFIG_IDENTITY_FIELDS: Tuple[str, ...] = (
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
    "persist_m1",
    "persist_m2",
    "xcd_swizzle1",
    "xcd_swizzle2",
    "stage2_lds_load_bytes",
    "stage2_a_prefetch_schedule",
    "stage2_a_prefetch_scope",
    "k_batch1",
    "waves_per_eu2",
)


@dataclass
class DispatchChangeVerdict:
    """Verdict for a production dispatch-table-only change.

    ``changed_keys`` are compared with timing metrics.  Every other baseline key
    must still be present, but its no-regression proof is config identity rather
    than timing because production dispatches the exact same kernel/config there.
    """

    changed_keys: Set[Tuple] = field(default_factory=set)
    timed_points: List[PointVerdict] = field(default_factory=list)
    unchanged_config_checked: List[Tuple] = field(default_factory=list)
    missing_candidate_points: List[Tuple] = field(default_factory=list)
    incomplete_changed_points: List[Tuple] = field(default_factory=list)
    incomplete_config_points: List[Tuple] = field(default_factory=list)
    unchanged_config_mismatches: List[Tuple] = field(default_factory=list)
    unknown_changed_keys: List[Tuple] = field(default_factory=list)
    any_changed_regression: bool = False
    large_wins: List[Tuple] = field(default_factory=list)
    small_wins: List[Tuple] = field(default_factory=list)
    quarantined: List[Tuple] = field(default_factory=list)
    gate: dict = field(default_factory=lambda: {"passed": False, "n_rows": 0, "violations": []})

    @property
    def coverage_complete(self) -> bool:
        return (
            not self.missing_candidate_points
            and not self.incomplete_changed_points
            and not self.incomplete_config_points
            and not self.unknown_changed_keys
        )

    @property
    def config_identity_clean(self) -> bool:
        return not self.incomplete_config_points and not self.unchanged_config_mismatches

    @property
    def timed_clean(self) -> bool:
        return not self.any_changed_regression and not self.incomplete_changed_points

    @property
    def claimable_dispatch_win(self) -> bool:
        return (
            self.coverage_complete
            and self.config_identity_clean
            and self.timed_clean
            and bool(self.large_wins or self.small_wins)
            and bool(self.gate.get("passed"))
        )


def _norm_cell(row: dict, field_name: str) -> str:
    return (row.get(field_name) or "").strip()


def _missing_config_fields(row: dict, fields: Tuple[str, ...]) -> List[str]:
    return [f for f in fields if f not in row or _norm_cell(row, f) == ""]


def compare_csvs(baseline_csv: str, candidate_csv: str, quarantine_keys: Optional[set] = None) -> CampaignVerdict:
    """Full per-point Pareto comparison of a candidate vs a baseline.

    Iterates the COMPLETE baseline key set so a candidate cannot pass by omitting
    a regressing/uncovered point.  A point with a missing candidate row, or whose
    candidate row lacks a regime-required field (kernel_path_us/e2e_us for every
    point; mfu for large target buckets), makes ``coverage_complete`` False, which
    forces ``pareto_clean`` False.

    The candidate is run through ``selected_candidate_gate`` and the result is
    stored on the verdict.  ``CampaignVerdict.claimable_win`` is the single source
    of truth for promotability: it requires ``pareto_clean`` + at least one win +
    the gate (``aot_status=checked`` + correctness + ``logits_diff<=0.01``).  Do
    NOT promote a candidate from ``pareto_clean`` + win lists alone -- a ``no_aot``
    candidate can be pareto_clean with wins yet must not be claimable.

    ``quarantine_keys`` (opt-in) excludes documented ``reference_invalid`` rows
    (issue #643 tiny-token reference-nonfinite harness artifact) from BOTH the gate
    and the regression/coverage scan, recording them on ``cv.quarantined``.  A
    quarantined key is only honored if its candidate row's ``error_category`` is
    ``reference_invalid`` (enforced in ``selected_candidate_gate``); the comparison
    side additionally requires the row's metrics be non-comparable (it skips them
    from regression so a nan-latency artifact cannot count as a regression OR a
    win).  Baseline MUST be a fresh paired baseline (same code state/session), not
    the stale locked snapshot, when cross-session drift is present.
    """
    base = read_point_csv(baseline_csv)
    cand = read_point_csv(candidate_csv)
    quarantine_keys = quarantine_keys or set()
    cv = CampaignVerdict()
    cv.gate = selected_candidate_gate(candidate_csv, quarantine_keys=quarantine_keys)
    for key, b_row in base.items():
        token = int(float(b_row.get("token") or 0))
        c_row = cand.get(key)
        if c_row is None:
            cv.missing_candidate_points.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="missing_candidate_point"))
            continue
        # Quarantined reference-invalid rows are excluded from coverage/regression.
        if key in quarantine_keys and (c_row.get("error_category") or "").strip() == "reference_invalid":
            cv.quarantined.append(key)
            cv.points.append(PointVerdict(key=key, token=token, note="quarantined:reference_invalid"))
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


def compare_csvs_dispatch_change(
    baseline_csv: str,
    candidate_csv: str,
    changed_keys: Set[Tuple],
    quarantine_keys: Optional[set] = None,
    config_fields: Tuple[str, ...] = CONFIG_IDENTITY_FIELDS,
) -> DispatchChangeVerdict:
    """Compare a fresh paired baseline and candidate for a dispatch-table-only change.

    This is intentionally narrower than ``compare_csvs``.  It is for the case
    where the production edit changes dispatch/config selection only for an
    explicit allow-list of keys.  The allow-listed keys still use the locked
    timing no-regression/win gates.  All other rows must prove that production
    selects the same kernel/config by matching every required config-identity
    column; their noisy fresh timing deltas are not used for the claim.
    """
    base = read_point_csv(baseline_csv)
    cand = read_point_csv(candidate_csv)
    changed_keys = set(changed_keys)
    quarantine_keys = quarantine_keys or set()
    cv = DispatchChangeVerdict(changed_keys=changed_keys)
    cv.gate = selected_candidate_gate(candidate_csv, quarantine_keys=quarantine_keys)
    cv.unknown_changed_keys = sorted(changed_keys - set(base))

    for key, b_row in base.items():
        token = int(float(b_row.get("token") or 0))
        c_row = cand.get(key)
        if c_row is None:
            cv.missing_candidate_points.append(key)
            continue
        if key in quarantine_keys and (c_row.get("error_category") or "").strip() == "reference_invalid":
            cv.quarantined.append(key)
            continue

        if key in changed_keys:
            missing = _row_missing_fields(c_row, _required_fields_for_point(token))
            if missing:
                cv.incomplete_changed_points.append((key, tuple(missing)))
                continue
            pv = compare_point(b_row, c_row)
            cv.timed_points.append(pv)
            if pv.kernel_path_regression or pv.e2e_regression:
                cv.any_changed_regression = True
            if pv.large_shape_win:
                cv.large_wins.append(key)
            if pv.small_token_win:
                cv.small_wins.append(key)
            continue

        missing_base = _missing_config_fields(b_row, config_fields)
        missing_cand = _missing_config_fields(c_row, config_fields)
        if missing_base or missing_cand:
            cv.incomplete_config_points.append((key, tuple(missing_base), tuple(missing_cand)))
            continue
        mismatches = [
            (field_name, _norm_cell(b_row, field_name), _norm_cell(c_row, field_name))
            for field_name in config_fields
            if _norm_cell(b_row, field_name) != _norm_cell(c_row, field_name)
        ]
        if mismatches:
            cv.unchanged_config_mismatches.append((key, tuple(mismatches)))
        else:
            cv.unchanged_config_checked.append(key)
    return cv


def selected_candidate_gate(
    candidate_csv: str, max_logits_diff: float = 0.01, quarantine_keys: Optional[set] = None
) -> dict:
    """Hard gate a candidate CSV before it can be promoted to a win.

    A selected candidate must clear the strict correctness + AOT-cache hard gate on
    EVERY row: ``aot_status == "checked"`` (the strict aiter run required a
    pre-populated AOT cache, not the ``no_aot`` repeatability/diagnostic bypass),
    ``correctness_pass`` is true, and ``logits_diff <= max_logits_diff``.  Rows
    measured with ``--no-aot-check`` (``aot_status == "no_aot"``) are valid for
    NEUTRAL repeatability/diagnostic artifacts but can never be promoted to a win,
    so they fail this gate.

    ``quarantine_keys`` is an OPT-IN allow-list of ``(model, dtype, act, token)``
    keys that are excluded from the gate ONLY when their ``error_category`` is
    ``reference_invalid`` (the torch reference itself was non-finite -- a known
    tiny-token CK-path harness artifact, issue #643, unrelated to the tuned lever).
    A quarantined key whose category is anything else (a real correctness failure)
    is still a violation -- quarantine can never hide a genuine kernel mismatch.

    Returns ``{"passed", "n_rows", "violations", "quarantined"}``.
    """
    rows = read_point_csv(candidate_csv)
    quarantine_keys = quarantine_keys or set()
    violations: List[Tuple] = []
    quarantined: List[Tuple] = []
    for key, row in rows.items():
        cat = (row.get("error_category") or "").strip()
        if key in quarantine_keys and cat == "reference_invalid":
            quarantined.append((key, "reference_invalid (issue #643 tiny-token reference nonfinite; quarantined)"))
            continue
        aot = (row.get("aot_status") or "").strip()
        if aot != "checked":
            violations.append((key, f"aot_status={aot or 'missing'} (need 'checked')"))
        cp = (row.get("correctness_pass") or "").strip().lower()
        if cp not in ("true", "1"):
            violations.append((key, f"correctness_pass={row.get('correctness_pass')!r} (need True)"))
        ld = _f(row, "logits_diff")
        if ld is None:
            violations.append((key, "logits_diff missing"))
        elif ld > max_logits_diff:
            violations.append((key, f"logits_diff={ld} > {max_logits_diff}"))
    return {
        "passed": bool(rows) and not violations,
        "n_rows": len(rows),
        "violations": violations,
        "quarantined": quarantined,
    }


def repeatability_check(csv_a: str, csv_b: str) -> dict:
    """Compare two independent sweeps of the SAME config under the no-regression policy.

    For each shared (model, dtype, act, token) point, a metric is "stable" if the
    two runs agree within the no-regression noise band (NOT a regression in either
    direction): ``|b - a| <= max(a*REGRESSION_REL, abs_floor_us(token))``, where
    the absolute floor is regime-aware (8 us for tokens <= SMALL_TOKEN_MAX, 2 us
    otherwise).  Returns the set of unstable points per metric; an empty unstable
    set demonstrates the harness is repeatable (the measurement protocol).
    """
    a = read_point_csv(csv_a)
    b = read_point_csv(csv_b)
    shared = sorted(set(a) & set(b))
    unstable = {"kernel_path_us": [], "e2e_us": []}

    def band(x, token):
        return max(abs(x) * spec.REGRESSION_REL, spec.abs_floor_us(token))

    for key in shared:
        token = int(float(a[key].get("token") or 0))
        for metric in ("kernel_path_us", "e2e_us"):
            va, vb = _f(a[key], metric), _f(b[key], metric)
            if va is None or vb is None:
                unstable[metric].append((key, "missing"))
            elif abs(vb - va) > band(va, token):
                unstable[metric].append((key, va, vb))
    return {
        "n_shared": len(shared),
        "unstable": unstable,
        "stable": not unstable["kernel_path_us"] and not unstable["e2e_us"],
    }


def scan_replay_consistency(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find committed attempts whose ``csv_path`` lists files the ``command`` cannot replay.

    A multi-file attempt (``csv_path`` = ``a.csv;b.csv``) must name EVERY listed
    file in its ``command`` string, so the attempt is replayable end-to-end from
    the ledger alone (no brace shorthand like ``run{1,2}.csv``, no required step
    hidden behind a ``#`` comment).  Superseded records are skipped.  Returns a
    list of ``(timestamp, [missing files])`` for offending records (empty == clean).
    """
    if not os.path.exists(path):
        return []
    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if "superseded_by" in rec:
                continue
            csv_path = rec.get("csv_path") or ""
            files = [p for p in csv_path.split(";") if p.strip()]
            if len(files) < 2:
                continue  # single/no file: nothing multi-file to reconcile
            command = rec.get("command") or ""
            # Strip anything after a '#' on each segment: a required step hidden in
            # a comment is not actually replayed by a shell.
            replayable = " ".join(seg.split("#", 1)[0] for seg in command.splitlines())
            missing = [fp for fp in files if fp not in replayable]
            if missing:
                offenders.append((rec.get("timestamp"), missing))
    return offenders


def _rejected_key(rec: dict) -> Tuple:
    """Identity of a rejected probe: model/dtype/token/act + the tile config.
    Used to detect duplicate non-superseded rejection records for the same probe."""
    cfg = rec.get("config") or {}
    cfg_key = tuple(sorted((str(k), str(v)) for k, v in cfg.items()))
    return (rec.get("model"), rec.get("dtype"), rec.get("act"), rec.get("token"), cfg_key)


def scan_duplicate_rejected_candidates(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find probes with more than one ACTIVE (non-superseded) rejected record.

    Two ledger entries that reject the same (model,dtype,act,token,config) probe
    are a provenance defect -- there must be exactly one active reason per probe
    (older duplicates must be marked ``superseded_by``).  Returns a list of
    ``(key, [timestamps])`` for probes with >1 active record (empty == clean).
    """
    if not os.path.exists(path):
        return []
    seen: Dict[Tuple, List] = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "rejected_candidate" or "superseded_by" in rec:
                continue
            seen.setdefault(_rejected_key(rec), []).append(rec.get("timestamp"))
    return [(k, ts) for k, ts in seen.items() if len(ts) > 1]


def _measured_supersede_sig(rec: dict) -> Tuple:
    """Identity for matching a superseded rejection to an active MEASURED successor.

    A measured attempt (loss/neutral/win) covers its whole token list in one CSV
    and carries no top-level ``token``, so it cannot share a rejected_candidate's
    ``token``-bearing key.  Match instead on (model,dtype,act,stage,config) -- the
    tuning identity that makes the rejection and the measurement the same probe."""
    cfg = rec.get("config") or {}
    cfg_key = tuple(sorted((str(k), str(v)) for k, v in cfg.items()))
    return (rec.get("model"), rec.get("dtype"), rec.get("act"), rec.get("stage"), cfg_key)


def scan_superseded_rejected_candidates(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find superseded rejected records that do NOT link to a matching successor.

    Every ``rejected_candidate`` carrying ``superseded_by`` must point at an
    EXISTING active (non-superseded) successor that is the SAME probe.  Two valid
    successor kinds:

    1. An active same-key ``rejected_candidate`` (same
       ``(model,dtype,act,token,config)`` -- the original rejection->rejection
       chain, e.g. a regenerated provenance record).
    2. An active MEASURED result (``loss``/``neutral``/``win``) with the same
       ``_measured_supersede_sig`` (model/dtype/act/stage/config).  This is the
       "broken rejection -> fixed-and-measured" case: when a (d)-bucket
       compiles-but-incorrect kernel is later REPAIRED and measured, the original
       correctness rejection is superseded by the measured-loss/win attempt, not
       by a fake same-key rejection marker.

    A link to neither (or to no record) is an evidence-integrity defect.  Returns
    ``(timestamp, reason)`` per offender (empty == clean).
    """
    if not os.path.exists(path):
        return []
    active_rej_ts_by_key: Dict[Tuple, set] = {}
    active_measured_ts_by_sig: Dict[Tuple, set] = {}
    superseded: List[dict] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            result = rec.get("result")
            if result == "rejected_candidate":
                if "superseded_by" in rec:
                    superseded.append(rec)
                else:
                    active_rej_ts_by_key.setdefault(_rejected_key(rec), set()).add(rec.get("timestamp"))
            elif result in _MEASURED_RESULTS and "superseded_by" not in rec:
                active_measured_ts_by_sig.setdefault(_measured_supersede_sig(rec), set()).add(rec.get("timestamp"))
    offenders: List[Tuple] = []
    for rec in superseded:
        target = rec.get("superseded_by")
        ok = target in active_rej_ts_by_key.get(_rejected_key(rec), set()) or target in active_measured_ts_by_sig.get(
            _measured_supersede_sig(rec), set()
        )
        if not ok:
            offenders.append(
                (
                    rec.get("timestamp"),
                    f"superseded_by={target} is not an active same-key rejection or matching measured result",
                )
            )
    return offenders


import re as _re  # noqa: E402
import subprocess as _subprocess  # noqa: E402

# Repo-relative EXECUTABLE/INPUT paths an attempt command depends on (script, CLI,
# or counter-list).  Deliberately EXCLUDES output artifacts (.csv/.json/.md): an
# attempt's output files are produced by the run and committed in the SAME or a
# LATER commit, so they legitimately do not exist at the runtime commit -- only the
# command's inputs must.  Used by scan_attempt_command_paths.
_REPO_PATH_RE = _re.compile(r"(?:docs|scripts|kernels|tests)/[\w./-]+\.(?:sh|py|txt)")


def scan_attempt_command_paths(path: str = ATTEMPTS_JSONL, repo_root: str = _REPO_ROOT) -> List[Tuple]:
    """Find attempts whose ``command`` names a repo path absent at the recorded ``commit``.

    Replayable provenance requires branch+commit+exact-command to be internally
    consistent: an attempt that says "run docs/.../collect.sh" at commit X is only
    replayable if that path exists in commit X.  For every non-superseded attempt
    with a ``commit``, this extracts repo-relative artifact/script paths from the
    ``command`` and checks ``git cat-file -e <commit>:<path>``.  Paths that are not
    tracked at HEAD either (i.e. never committed, e.g. /tmp helpers) are ignored —
    only a path that EXISTS now but is MISSING at the recorded commit is flagged.
    Returns ``(timestamp, [missing paths])`` per offender (empty == clean).
    """
    if not os.path.exists(path):
        return []

    def _exists_at(commit: str, p: str) -> bool:
        try:
            return (
                _subprocess.run(
                    ["git", "-C", repo_root, "cat-file", "-e", f"{commit}:{p}"],
                    capture_output=True,
                ).returncode
                == 0
            )
        except Exception:
            return True  # cannot check (no git) -> do not flag

    def _tracked_at_head(p: str) -> bool:
        return _exists_at("HEAD", p)

    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if "superseded_by" in rec:
                continue
            commit = rec.get("commit") or ""
            command = rec.get("command") or ""
            if not commit:
                continue
            missing = []
            for p in dict.fromkeys(_REPO_PATH_RE.findall(command)):  # dedupe, keep order
                # Only meaningful for paths that are real tracked artifacts (exist at HEAD);
                # output CSVs written by the run are also tracked, so this catches both
                # "script missing at commit" and "output not yet committed at commit".
                if _tracked_at_head(p) and not _exists_at(commit, p):
                    missing.append(p)
            if missing:
                offenders.append((rec.get("timestamp"), missing))
    return offenders


# Legality-filter reason tokens that name a specific guard in kernels/moe_tuning.py.
# An active rejection record citing one of these must record a commit whose
# kernels/moe_tuning.py actually contains the token, or the rejection is not
# reproducible by replaying the command at that commit.  Used by
# scan_rejection_reason_present_at_commit.  Add new pre-compile legality reason
# strings here as guards are introduced.
_LEGALITY_REASON_TOKENS = (
    "stage2_tile_n_not_div_64",
    "stage1_tile_k_unsupported",
    "inter_dim_not_div_tile_n",
)
_LEGALITY_FILTER_FILE = "kernels/moe_tuning.py"


def _git_file_at_commit(commit: str, repo_path: str, repo_root: str) -> Optional[str]:
    """Return the text of ``repo_path`` at ``commit`` (None if it cannot be read).

    None means "cannot resolve" (no git / path absent at the commit) and callers
    treat it as not-an-offender so the scan never false-flags on a missing toolchain.
    """
    try:
        r = _subprocess.run(
            ["git", "-C", repo_root, "show", f"{commit}:{repo_path}"],
            capture_output=True,
            text=True,
        )
        return r.stdout if r.returncode == 0 else None
    except Exception:
        return None


def scan_rejection_reason_present_at_commit(path: str = ATTEMPTS_JSONL, repo_root: str = _REPO_ROOT) -> List[Tuple]:
    """Find active legality rejections whose recorded commit lacks the cited guard.

    A pre-compile legality rejection claims the filter rejected the candidate for a
    named reason (e.g. ``stage2_tile_n_not_div_64``).  That is only replayable if
    ``kernels/moe_tuning.py`` AT THE RECORDED COMMIT actually contains the reason
    token -- otherwise replaying the exact command at that commit cannot produce the
    recorded reason (the guard did not exist yet).  ``scan_replay_consistency`` and
    ``scan_attempt_command_paths`` do not catch this: the command paths exist and
    the CSVs are absent-but-legal, yet the recorded behavior is not reproducible.
    For every active (non-superseded) rejected-candidate record whose ``reason``
    names a known legality token, this checks ``git show <commit>:<filter file>``
    for that token.  Returns ``(timestamp, reason_token)`` per offender (empty ==
    clean).
    """
    if not os.path.exists(path):
        return []

    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "rejected_candidate" or "superseded_by" in rec:
                continue
            commit = rec.get("commit") or ""
            reason = rec.get("reason") or ""
            tokens = [t for t in _LEGALITY_REASON_TOKENS if t in reason]
            if not commit or not tokens:
                continue
            src = _git_file_at_commit(commit, _LEGALITY_FILTER_FILE, repo_root)
            if src is None:
                continue  # cannot resolve the file at that commit -> do not flag
            for tok in tokens:
                if tok not in src:
                    offenders.append((rec.get("timestamp"), tok))
    return offenders


# Tunable-knob CLI/source tokens that a measured attempt's command may exercise,
# each mapped to the repo file(s) whose content at the recorded commit must
# contain the token.  A measured attempt whose command cites one of these must
# record a commit where the harness AND the kernel test actually implement it --
# otherwise the measurement was run from uncommitted code and is not replayable.
# Used by scan_measured_attempt_tokens_present_at_commit.  Add new tunable knobs
# here as they are threaded.
_MEASURED_KNOB_TOKENS = {
    "persist-m1": ("scripts/moe_tuning_harness.py",),
    "persist-m2": ("scripts/moe_tuning_harness.py",),
    "xcd-swizzle1": ("scripts/moe_tuning_harness.py",),
    "xcd-swizzle2": ("scripts/moe_tuning_harness.py",),
    "persist_m1": ("tests/kernels/test_moe_gemm.py",),
    "persist_m2": ("tests/kernels/test_moe_gemm.py",),
    "xcd_swizzle1": ("tests/kernels/test_moe_gemm.py",),
    "xcd_swizzle2": ("tests/kernels/test_moe_gemm.py",),
    "stage2-lds-load-bytes": ("scripts/moe_tuning_harness.py",),
    "stage2_lds_load_bytes": ("tests/kernels/test_moe_gemm.py",),
    "stage2-a-prefetch-schedule": ("scripts/moe_tuning_harness.py",),
    "stage2_a_prefetch_schedule": ("tests/kernels/test_moe_gemm.py",),
    "stage2-a-prefetch-scope": ("scripts/moe_tuning_harness.py",),
    "stage2_a_prefetch_scope": ("tests/kernels/test_moe_gemm.py",),
    "waves-per-eu2": ("scripts/moe_tuning_harness.py",),
    "waves_per_eu2": ("tests/kernels/test_moe_gemm.py",),
}
_MEASURED_RESULTS = ("win", "loss", "neutral")


def scan_measured_attempt_tokens_present_at_commit(
    path: str = ATTEMPTS_JSONL, repo_root: str = _REPO_ROOT
) -> List[Tuple]:
    """Find active measured attempts run from code absent at their recorded commit.

    A measured attempt (``win``/``loss``/``neutral``) whose ``command`` exercises a
    tunable knob (e.g. ``--persist-m2``) is only replayable if the file that
    implements that knob CONTAINS the token at the recorded commit.  The R14 defect
    class: the sweeps were run from an uncommitted working tree while provenance
    still pointed at the prior commit, so the recorded commit lacked the
    knob-threading code.  ``scan_attempt_command_paths`` does not catch this (the
    script path exists at the commit; only its *content* is stale).  For every
    active measured attempt, this checks each knob token present in the ``command``
    (or in the ``config``) against the mapped file at the recorded commit.  Returns
    ``(timestamp, token)`` per offender (empty == clean).
    """
    if not os.path.exists(path):
        return []

    src_cache: Dict[Tuple[str, str], Optional[str]] = {}

    def _src(commit: str, repo_path: str) -> Optional[str]:
        key = (commit, repo_path)
        if key not in src_cache:
            src_cache[key] = _git_file_at_commit(commit, repo_path, repo_root)
        return src_cache[key]

    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") not in _MEASURED_RESULTS or "superseded_by" in rec:
                continue
            commit = rec.get("commit") or ""
            if not commit:
                continue
            command = rec.get("command") or ""
            cfg_keys = set((rec.get("config") or {}).keys())
            for tok, files in _MEASURED_KNOB_TOKENS.items():
                cited = (f"--{tok}" in command) or (tok in command) or (tok in cfg_keys)
                if not cited:
                    continue
                for repo_path in files:
                    src = _src(commit, repo_path)
                    if src is None:
                        continue  # cannot resolve -> do not flag
                    if tok not in src:
                        offenders.append((rec.get("timestamp"), tok))
                        break  # one offense per (record, token) is enough
    return offenders


# Wording that must never appear in an ACTIVE rejected_candidate row: a rejection
# that says the underlying defect was fixed / the path now passes / the lever is a
# measured loss is self-contradictory -- such a record must be SUPERSEDED (by the
# fixed-and-measured successor), not left active.  Used by
# scan_no_self_contradictory_active_rejection.
_CONTRADICTORY_REJECTION_MARKERS = (
    "SUPERSEDED-STATUS",
    "is now CORRECT",
    "strict-ref passes",
    "strict-ref now passes",
    "now a measured loss",
    "measured loss, not a correctness",
    "no longer broken",
    "was FIXED",
)


def scan_no_self_contradictory_active_rejection(path: str = ATTEMPTS_JSONL) -> List[Tuple]:
    """Find ACTIVE rejected_candidate rows whose reason contradicts the rejection.

    A `result:"rejected_candidate"` that is still active (no `superseded_by`) but
    whose `reason` says the defect was fixed / the path now passes / the lever is a
    measured loss is an AC-7 truthfulness defect: if the kernel is correct now, the
    rejection must be SUPERSEDED by the fixed-and-measured successor, not kept as an
    active "rejection".  Returns `(timestamp, marker)` per offender (empty == clean).
    """
    if not os.path.exists(path):
        return []
    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "rejected_candidate" or "superseded_by" in rec:
                continue
            reason = rec.get("reason") or ""
            for marker in _CONTRADICTORY_REJECTION_MARKERS:
                if marker in reason:
                    offenders.append((rec.get("timestamp"), marker))
                    break
    return offenders


def scan_win_label_backed_by_claimable(
    path: str = ATTEMPTS_JSONL, baseline_csv: str = None, repo_root: str = _REPO_ROOT
) -> List[Tuple]:
    """Find active ``result:"win"`` rows not backed by a real claimable-win verdict.

    ``result:"win"`` is the strongest label and must mean a promotable win that
    passed the full comparator (``compare_csvs(...).claimable_win``): full-grid
    Pareto coverage, no regression, AND the strict correctness + AOT-checked
    ``selected_candidate_gate``.  A subset-clean candidate (e.g. a single
    large-bucket MFU improvement measured with ``--no-e2e``) is NOT a win and must
    be recorded as ``result:"neutral"`` with a "subset candidate, not claimable"
    note.

    The scan FAILS CLOSED: an active ``result:"win"`` row is clean ONLY when ALL
    of the following hold, and ANY failure (including an unloadable artifact) is an
    offender -- a hand-written marker can never bypass the official comparator by
    pointing at a missing/empty/unreadable CSV:

    1. ``config.claimable is True`` (the promotion-path marker), AND
    2. ``csv_path`` is present and non-empty, AND
    3. both the candidate CSV and the locked baseline CSV exist on disk, AND
    4. ``compare_csvs(baseline, csv_path).claimable_win`` recomputes to True
       (a comparator exception on a malformed/unreadable CSV is an offender).

    ``baseline_csv`` defaults to the committed locked baseline.  Returns
    ``(timestamp, model, reason)`` per offender (empty == clean).
    """
    if not os.path.exists(path):
        return []
    if baseline_csv is None:
        baseline_csv = os.path.join(repo_root, "docs", "baseline_523ca1c7_validated.csv")
    offenders: List[Tuple] = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rec = json.loads(ln)
            if rec.get("result") != "win" or "superseded_by" in rec:
                continue
            ts, model = rec.get("timestamp"), rec.get("model")
            if (rec.get("config") or {}).get("claimable") is not True:
                offenders.append((ts, model, "win row missing config.claimable=True backing"))
                continue
            # Marker is set -> the official comparator MUST be recomputable and True.
            cand = rec.get("csv_path") or ""
            if not cand:
                offenders.append((ts, model, "config.claimable=True but csv_path is missing/empty"))
                continue
            cand_abs = cand if os.path.isabs(cand) else os.path.join(repo_root, cand)
            if not os.path.exists(cand_abs):
                offenders.append((ts, model, f"config.claimable=True but candidate csv_path does not exist: {cand}"))
                continue
            # An active result:"win" is ALWAYS validated against the locked-baseline
            # comparator (the immutable win definition).  There is NO dispatch_change
            # bypass: compare_csvs_dispatch_change is diagnostic-only and must never
            # back a result:"win" row (R1 review: redefining the win in mutable text
            # is an integrity violation).  A dispatch-only result must be recorded as
            # result:"neutral" dispatch-evidence, not "win".
            if not os.path.exists(baseline_csv):
                offenders.append((ts, model, f"config.claimable=True but baseline CSV does not exist: {baseline_csv}"))
                continue
            try:
                claimable = compare_csvs(baseline_csv, cand_abs).claimable_win
            except Exception as e:
                offenders.append((ts, model, f"config.claimable=True but compare_csvs raised on csv_path: {e!r}"))
                continue
            if not claimable:
                offenders.append(
                    (ts, model, "config.claimable=True but compare_csvs(...).claimable_win is False for csv_path")
                )
    return offenders


def scan_changed_rows_fail_closed(candidate_csv: str, changed_keys: set) -> List[Tuple]:
    """Require every CHANGED candidate row to carry expected-kernel enforcement.

    A dispatch-only change is only trustworthy if the changed rows were measured
    FAIL-CLOSED: the strict run must have required the intended stage2 kernelName2
    (``expected_kernel_name2`` non-empty, and ending in the tuned suffix), so a
    silent fallback to the default/heuristic kernel could not have produced a
    passing row.  Returns ``(model,dtype,act,token,reason)`` offenders; empty == OK.
    """
    offenders: List[Tuple] = []
    if not os.path.exists(candidate_csv):
        return [("", "", "", "", f"candidate_csv missing: {candidate_csv}")]
    rows = read_point_csv(candidate_csv)
    for key in changed_keys:
        row = rows.get(tuple(key))
        if row is None:
            offenders.append((*key, "changed row missing from candidate CSV"))
            continue
        exp = (row.get("expected_kernel_name2") or "").strip()
        if not exp:
            offenders.append((*key, "changed row has no expected_kernel_name2 (not measured fail-closed)"))
    return offenders


LOCKED_BASELINE_COMMIT = "523ca1c7e224ee62d5e3a4c0f52a18b9cec5e727"


def scan_candidate_csv_freshness(
    candidate_csv: str,
    baseline_commit: str = LOCKED_BASELINE_COMMIT,
    win_tokens: Optional[List[str]] = None,
) -> List[Tuple]:
    """Reject a candidate CSV that reuses locked-baseline rows as candidate rows.

    A claimable-win candidate must be FRESHLY measured from the final branch state
    (AC-5): every row's ``commit`` must NOT be the locked baseline commit.  The
    previous loop built a "full-40" candidate by copying 37 byte-for-byte rows from
    ``docs/baseline_523ca1c7_validated.csv`` (commit ``523ca1c7...``) and measuring
    only the GPT-OSS large rows; ``compare_csvs`` happily returned ``claimable_win``
    because it does not inspect provenance.  This scan closes that hole: any
    candidate row still carrying the baseline commit is an offender.

    Returns a list of ``(model, dtype, act, token, reason)`` offenders; empty == OK.
    ``win_tokens`` is unused by the freshness check itself but documents that the
    win rows in particular must be fresh.
    """
    offenders: List[Tuple] = []
    short_base = (baseline_commit or "")[:12]
    if not os.path.exists(candidate_csv):
        return [("", "", "", "", f"candidate_csv missing: {candidate_csv}")]
    with open(candidate_csv, newline="") as f:
        for row in csv.DictReader(f):
            commit = (row.get("commit") or "").strip()
            key = (row.get("model"), row.get("dtype"), row.get("act"), row.get("token"))
            if commit == baseline_commit or (short_base and commit[:12] == short_base):
                offenders.append(
                    (
                        *key,
                        f"row carries locked-baseline commit {short_base} (copied baseline row, not a fresh measurement)",
                    )
                )
            elif not commit:
                offenders.append((*key, "row has empty commit (no provenance)"))
    return offenders


__all__ = [
    "ATTEMPTS_JSONL",
    "LEDGER_MD",
    "LOCKED_BASELINE_COMMIT",
    "REQUIRED_ATTEMPT_FIELDS",
    "Attempt",
    "append_attempt",
    "read_point_csv",
    "CONFIG_IDENTITY_FIELDS",
    "compare_point",
    "compare_csvs",
    "compare_csvs_dispatch_change",
    "selected_candidate_gate",
    "scan_replay_consistency",
    "scan_duplicate_rejected_candidates",
    "scan_superseded_rejected_candidates",
    "scan_attempt_command_paths",
    "scan_rejection_reason_present_at_commit",
    "scan_measured_attempt_tokens_present_at_commit",
    "scan_win_label_backed_by_claimable",
    "scan_no_self_contradictory_active_rejection",
    "scan_candidate_csv_freshness",
    "scan_changed_rows_fail_closed",
    "repeatability_check",
    "PointVerdict",
    "CampaignVerdict",
    "DispatchChangeVerdict",
]
