"""Unit tests for the loop3-R1 ledger hardening:
- scan_candidate_csv_freshness rejects copied-baseline rows (commit == locked base).
- selected_candidate_gate / compare_csvs quarantine honors reference_invalid ONLY.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.moe_tuning_ledger import (  # noqa: E402
    LOCKED_BASELINE_COMMIT,
    compare_csvs,
    compare_csvs_dispatch_change,
    scan_candidate_csv_freshness,
    scan_changed_rows_fail_closed,
    scan_win_label_backed_by_claimable,
    selected_candidate_gate,
)

_COLS = [
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
    "flydsl_command",
    "strict_error",
    "error_category",
    "aot_status",
    "expected_kernel_name2",
]


def _row(**over):
    r = {c: "" for c in _COLS}
    r.update(
        commit="8e09bfab79f5deadbeef",
        model="gpt_oss",
        model_dim="3072",
        inter_dim="3072",
        experts="128",
        topk="4",
        dtype="a4w4",
        act="swiglu",
        token="16384",
        tile_m1="32",
        tile_n1="128",
        tile_k1="256",
        tile_m2="32",
        tile_n2="256",
        tile_k2="256",
        persist_m1="1",
        persist_m2="4",
        xcd_swizzle1="0",
        xcd_swizzle2="0",
        stage2_lds_load_bytes="16",
        stage2_a_prefetch_schedule="baseline",
        stage2_a_prefetch_scope="front",
        k_batch1="1",
        waves_per_eu2="0",
        kernel_path_us="2554.0",
        kernel_path_us_p95="2600.0",
        mfu="0.317",
        e2e_us="1881.0",
        e2e_us_p95="2000.0",
        logits_diff="6.2e-06",
        correctness_pass="True",
        aot_status="checked",
        error_category="",
    )
    r.update(over)
    return r


def _write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_COLS)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_freshness_flags_locked_baseline_commit(tmp_path):
    p = _write(tmp_path / "c.csv", [_row(commit=LOCKED_BASELINE_COMMIT)])
    off = scan_candidate_csv_freshness(p)
    assert len(off) == 1 and "copied baseline row" in off[0][-1]


def test_freshness_flags_empty_commit(tmp_path):
    p = _write(tmp_path / "c.csv", [_row(commit="")])
    off = scan_candidate_csv_freshness(p)
    assert len(off) == 1 and "empty commit" in off[0][-1]


def test_freshness_passes_fresh_commit(tmp_path):
    p = _write(tmp_path / "c.csv", [_row(commit="8e09bfab79f5")])
    assert scan_candidate_csv_freshness(p) == []


def test_gate_quarantine_honors_reference_invalid_only(tmp_path):
    key = ("deepseek_v3", "a4w4", "silu", "1")
    # reference_invalid row, quarantined -> not a violation
    rinv = _row(
        model="deepseek_v3",
        model_dim="7168",
        inter_dim="256",
        experts="257",
        topk="9",
        act="silu",
        token="1",
        correctness_pass="False",
        logits_diff="nan",
        error_category="reference_invalid",
    )
    p = _write(tmp_path / "q.csv", [rinv])
    g = selected_candidate_gate(p, quarantine_keys={key})
    assert g["passed"] is True
    assert g["quarantined"] and g["quarantined"][0][0] == key


def test_gate_quarantine_does_not_hide_real_correctness_fail(tmp_path):
    key = ("deepseek_v3", "a4w4", "silu", "1")
    # genuine correctness failure (finite logits > 0.01) must NOT be quarantined
    bad = _row(
        model="deepseek_v3",
        model_dim="7168",
        inter_dim="256",
        experts="257",
        topk="9",
        act="silu",
        token="1",
        correctness_pass="False",
        logits_diff="0.5",
        error_category="correctness",
    )
    p = _write(tmp_path / "b.csv", [bad])
    g = selected_candidate_gate(p, quarantine_keys={key})
    assert g["passed"] is False
    assert any("correctness_pass" in v[1] for v in g["violations"])


def test_compare_csvs_quarantine_excludes_from_regression(tmp_path):
    key = ("deepseek_v3", "a4w4", "silu", "1")
    base = _write(
        tmp_path / "base.csv",
        [
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="1",
                kernel_path_us="30.0",
                e2e_us="34.0",
                mfu="0.001",
                logits_diff="0.002",
            )
        ],
    )
    # candidate row is reference_invalid (nan) -> would be a "regression" if compared
    cand = _write(
        tmp_path / "cand.csv",
        [
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="1",
                kernel_path_us="9999.0",
                e2e_us="9999.0",
                mfu="0.0001",
                correctness_pass="False",
                logits_diff="nan",
                error_category="reference_invalid",
            )
        ],
    )
    v = compare_csvs(base, cand, quarantine_keys={key})
    assert key in v.quarantined
    assert v.any_regression is False  # quarantined row excluded from regression


def test_dispatch_change_uses_config_identity_for_unchanged_noisy_rows(tmp_path):
    changed = ("gpt_oss", "a4w4", "swiglu", "16384")
    unchanged = ("deepseek_v3", "a4w4", "silu", "512")
    base = _write(
        tmp_path / "base.csv",
        [
            _row(token="16384", kernel_path_us="2500.0", e2e_us="1900.0", mfu="0.30"),
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="512",
                tile_m1="64",
                tile_n1="256",
                tile_m2="64",
                kernel_path_us="200.0",
                e2e_us="240.0",
                mfu="0.01",
            ),
        ],
    )
    cand = _write(
        tmp_path / "cand.csv",
        [
            _row(token="16384", persist_m2="1", kernel_path_us="2200.0", e2e_us="1700.0", mfu="0.34"),
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="512",
                tile_m1="64",
                tile_n1="256",
                tile_m2="64",
                kernel_path_us="800.0",
                e2e_us="900.0",
                mfu="0.002",
            ),
        ],
    )
    assert compare_csvs(base, cand).any_regression is True
    v = compare_csvs_dispatch_change(base, cand, changed_keys={changed})
    assert unchanged in v.unchanged_config_checked
    assert v.any_changed_regression is False
    assert v.large_wins == [changed]
    assert v.claimable_dispatch_win is True


def test_dispatch_change_fails_on_unchanged_config_mismatch(tmp_path):
    changed = ("gpt_oss", "a4w4", "swiglu", "16384")
    unchanged = ("deepseek_v3", "a4w4", "silu", "512")
    base = _write(
        tmp_path / "base.csv",
        [
            _row(token="16384", kernel_path_us="2500.0", e2e_us="1900.0", mfu="0.30"),
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="512",
                tile_m1="64",
                tile_m2="64",
                kernel_path_us="200.0",
                e2e_us="240.0",
                mfu="0.01",
            ),
        ],
    )
    cand = _write(
        tmp_path / "cand.csv",
        [
            _row(token="16384", persist_m2="1", kernel_path_us="2200.0", e2e_us="1700.0", mfu="0.34"),
            _row(
                model="deepseek_v3",
                model_dim="7168",
                inter_dim="256",
                experts="257",
                topk="9",
                act="silu",
                token="512",
                tile_m1="64",
                tile_m2="32",
                kernel_path_us="200.0",
                e2e_us="240.0",
                mfu="0.01",
            ),
        ],
    )
    v = compare_csvs_dispatch_change(base, cand, changed_keys={changed})
    assert v.claimable_dispatch_win is False
    assert v.unchanged_config_mismatches
    assert v.unchanged_config_mismatches[0][0] == unchanged
    assert any(m[0] == "tile_m2" for m in v.unchanged_config_mismatches[0][1])


def test_dispatch_change_requires_config_identity_columns(tmp_path):
    cols = [c for c in _COLS if c not in ("persist_m1", "persist_m2")]
    row = _row()
    base = tmp_path / "base.csv"
    cand = tmp_path / "cand.csv"
    for path in (base, cand):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerow({c: row[c] for c in cols})
    v = compare_csvs_dispatch_change(str(base), str(cand), changed_keys=set())
    assert v.claimable_dispatch_win is False
    assert v.incomplete_config_points
    assert "persist_m1" in v.incomplete_config_points[0][1]
    assert v.large_wins == []


def test_reference_invalid_never_passes_gate_without_quarantine(tmp_path):
    # A reference_invalid row must FAIL the selected-candidate gate when it is NOT
    # in the quarantine allow-list -- quarantine is opt-in, never automatic.
    bad = _row(
        model="deepseek_v3",
        model_dim="7168",
        inter_dim="256",
        experts="257",
        topk="9",
        act="silu",
        token="1",
        correctness_pass="False",
        logits_diff="nan",
        error_category="reference_invalid",
    )
    p = _write(tmp_path / "r.csv", [bad])
    g = selected_candidate_gate(p)  # no quarantine_keys
    assert g["passed"] is False
    assert any("correctness_pass" in v[1] for v in g["violations"])


def test_changed_rows_fail_closed_requires_expected_kernel(tmp_path):
    changed = {("gpt_oss", "a4w4", "swiglu", "16384")}
    # row WITHOUT expected_kernel_name2 -> offender
    no_enf = _row(model="gpt_oss", token="16384", expected_kernel_name2="")
    p = _write(tmp_path / "noenf.csv", [no_enf])
    off = scan_changed_rows_fail_closed(p, changed)
    assert len(off) == 1 and "expected_kernel_name2" in off[0][-1]
    # row WITH expected_kernel_name2 -> clean
    enf = _row(
        model="gpt_oss",
        token="16384",
        expected_kernel_name2="flydsl_moe2_afp4_wfp4_bf16_t64x128x256_atomic_pm1",
    )
    p2 = _write(tmp_path / "enf.csv", [enf])
    assert scan_changed_rows_fail_closed(p2, changed) == []


def test_dec2b_unauthorized_dispatch_change_win_is_flagged(tmp_path):
    # A result:win with verdict=dispatch_change but WITHOUT dec2b_authorized must
    # still be validated against the locked comparator (and flagged if it can't be
    # recomputed True). The integrity guard must not be bypassable without the
    # explicit user authorization flag.
    import json

    rec = {
        "result": "win",
        "model": "gpt_oss",
        "timestamp": "t",
        "config": {"claimable": True, "verdict": "dispatch_change"},  # no dec2b_authorized
        "csv_path": "docs/loop3_models/candidate_fresh40.csv",
    }
    p = tmp_path / "a.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    off = scan_win_label_backed_by_claimable(str(p))
    assert off, "unauthorized dispatch_change win must be flagged (no silent bypass)"


def test_dec2b_requires_baseline_and_changed_keys(tmp_path):
    # An authorized dispatch_change win missing the fresh paired baseline / changed
    # keys evidence is flagged (cannot recompute).
    import json

    rec = {
        "result": "win",
        "model": "gpt_oss",
        "timestamp": "t",
        "config": {"claimable": True, "verdict": "dispatch_change", "dec2b_authorized": True},
        "csv_path": "docs/loop3_models/candidate_fresh40.csv",
        "evidence": {},  # no baseline_csv / changed_keys
    }
    p = tmp_path / "b.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    off = scan_win_label_backed_by_claimable(str(p))
    assert off, "dec2b win without baseline/changed_keys evidence must be flagged"


if __name__ == "__main__":
    import pytest as _pt

    raise SystemExit(_pt.main([__file__, "-q"]))
