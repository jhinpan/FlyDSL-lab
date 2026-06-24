# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend-agnostic tests for the MXFP4 MoE tuning harness, spec, and ledger.

These exercise the pure host-side logic (decision predicates, stage-us parsing,
metric computation, provenance gating, attempt-ledger validation, and per-point
Pareto comparison) with no GPU and no compile.
"""

import os
import sys

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPTS = os.path.join(_REPO_ROOT, "scripts")
for p in (_REPO_ROOT, _SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

import moe_tuning_harness as harness  # noqa: E402
import moe_tuning_ledger as ledger  # noqa: E402

from kernels import moe_tuning_spec as spec  # noqa: E402

pytestmark = pytest.mark.l0_backend_agnostic


# --- spec: locked values + predicates --------------------------------------


def test_locked_constants():
    assert spec.FP4_PEAK_TFLOPS == 4523.0
    assert spec.WIN_MARGIN == 0.10
    assert spec.REGRESSION_REL == 0.02
    assert spec.ABS_US_BAND == 2.0
    assert spec.WARMUP_ITERS == 10
    assert spec.BENCH_ITERS == 100
    assert spec.MFU_TARGET_BUCKETS == (16384, 32768)
    assert spec.LARGE_TOKEN_MIN == 4096
    assert spec.SMALL_TOKEN_MAX == 64
    assert spec.TARGET_ARCH == "gfx950"


def test_token_grids():
    assert spec.TOKEN_GRID_FULL[0] == 1 and spec.TOKEN_GRID_FULL[-1] == 32768
    assert len(spec.TOKEN_GRID_FULL) == 16
    assert spec.TOKEN_GRID_GPTOSS[0] == 256 and spec.TOKEN_GRID_GPTOSS[-1] == 32768


def test_models_in_scope_dtypes():
    by_name = {m.name: m for m in spec.MODELS}
    assert set(by_name) == {"deepseek_v3", "deepseek_v4", "kimi_k2", "gpt_oss"}
    # DeepSeek V4 is a8w4-only; i4 excluded everywhere.
    assert by_name["deepseek_v4"].dtypes == ("a8w4",)
    assert by_name["kimi_k2"].dtypes == ("a4w4", "a8w4")
    assert all("i4" not in m.dtypes for m in spec.MODELS)
    assert by_name["gpt_oss"].act == "swiglu"
    assert by_name["deepseek_v4"].model_dim == 7168 and by_name["deepseek_v4"].inter_dim == 512


def test_regression_predicate_requires_both_bands():
    # 1.5% over but only +1.5us: relative under 2%? 1.5% < 2% -> not a regression.
    assert not spec.is_regression(100.0, 101.5)
    # 3% over but only +0.3us absolute (small base): abs band not exceeded -> not a regression.
    assert not spec.is_regression(10.0, 10.3)
    # 5% over AND +5us: both bands exceeded -> regression.
    assert spec.is_regression(100.0, 105.0)
    # exactly at boundaries (strict >): 102.0 and +2.0 -> not a regression.
    assert not spec.is_regression(100.0, 102.0)


def test_large_shape_win_predicate():
    assert spec.is_large_shape_win(0.50, 0.55)  # exactly +10%
    assert not spec.is_large_shape_win(0.50, 0.549)


def test_small_token_win_predicate():
    # 12% faster AND >= 2us absolute -> win.
    assert spec.is_small_token_win(100.0, 88.0)
    # 12% faster but only 0.6us absolute (tiny base) -> rejected (abs floor).
    assert not spec.is_small_token_win(5.0, 4.4)
    # 8% faster -> rejected (under 10%).
    assert not spec.is_small_token_win(100.0, 92.0)


def test_effective_tflops_and_mfu_formula():
    # token*model_dim*inter_dim*3*topk*2 / us / 1e6
    tflops = spec.effective_tflops(4096, 7168, 256, 9, combined_us=1000.0)
    expected = 4096 * 7168 * 256 * 3 * 9 * 2 / 1000.0 / 1e6
    assert abs(tflops - expected) < 1e-9
    assert abs(spec.mfu(tflops) - tflops / 4523.0) < 1e-12


# --- harness: parsing / metrics / provenance -------------------------------


def test_parse_flydsl_stage_us():
    stdout = (
        "noise\n"
        "FlyDSL MoE stage1[fp4]: 1163.2 us, 1654.24 TFLOPS(logical, M=4608), 0.377 TB/s (doweight_stage1=False)\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 atomic | 7168x2048, E=32, K=8, M_eff=4608 | 845.5 us, 1200.00 TFLOPS, 0.300 TB/s\n"
        "FlyDSL MoE stage2 [moe_gemm2] fp4 reduce | 7168x2048, E=32, K=8, M_eff=4608 | 900.1 us, 1100.00 TFLOPS, 0.280 TB/s\n"
    )
    got = harness.parse_flydsl_stage_us(stdout)
    assert got["stage1_us"] == 1163.2
    # last matching stage2 line wins
    assert got["stage2_us"] == 900.1


def test_parse_flydsl_stage_us_missing():
    got = harness.parse_flydsl_stage_us("nothing here")
    assert got["stage1_us"] is None and got["stage2_us"] is None


def test_combined_and_metrics():
    combined = harness.combined_kernel_path_us(1000.0, 800.0, 50.0)
    assert combined == 1850.0
    m = harness.compute_metrics(token=4096, model_dim=7168, inter_dim=256, topk=9, combined_us=combined)
    assert m["effective_tflops"] > 0 and 0 < m["mfu"] < 10


def test_summarize_median_p95():
    s = harness.summarize([10, 11, 12, 13, 100])
    assert s["median"] == 12
    assert s["p95"] == 100


def test_provenance_missing_fields_gate():
    p = harness.Provenance()  # gpu_id/gpu_model/branch/commit unset
    missing = p.missing_fields()
    assert "gpu_id" in missing and "commit" in missing
    assert not p.is_complete()
    p2 = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="rlcr/mxfp4-moe", commit="deadbeef")
    assert p2.is_complete()


def test_pointrow_csv_dict_has_all_columns():
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    row = harness.PointRow(
        provenance=p,
        command="cmd",
        model="kimi_k2",
        model_dim=7168,
        inter_dim=256,
        experts=384,
        topk=8,
        dtype="a4w4",
        act="silu",
        token=4096,
    )
    d = row.to_csv_dict()
    assert set(d.keys()) == set(harness.CSV_COLUMNS)
    assert d["metric_formula"] == harness.METRIC_FORMULA


def test_write_csv_roundtrip(tmp_path):
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="c")
    rows = [
        harness.PointRow(
            provenance=p,
            command="cmd",
            model="kimi_k2",
            model_dim=7168,
            inter_dim=256,
            experts=384,
            topk=8,
            dtype="a4w4",
            act="silu",
            token=4096,
            kernel_path_us=1850.0,
            e2e_us=2000.0,
            mfu=0.5,
        )
    ]
    out = tmp_path / "baseline.csv"
    harness.write_csv(rows, str(out))
    text = out.read_text()
    assert "kernel_path_us" in text.splitlines()[0]
    assert "kimi_k2" in text


# --- ledger: attempt validation + comparison -------------------------------


def _complete_attempt(**over):
    base = dict(
        config={"tile_m": 64},
        stage=1,
        model="kimi_k2",
        dtype="a4w4",
        act="silu",
        gpu_id="0",
        gpu_model="MI350X",
        branch="b",
        commit="c",
        command="cmd",
        warmup=10,
        iters=100,
        result="loss",
    )
    base.update(over)
    return ledger.Attempt(**base)


def test_attempt_missing_provenance_rejected(tmp_path):
    bad = _complete_attempt(commit="")  # missing required field
    assert "commit" in bad.missing_fields()
    with pytest.raises(ValueError):
        ledger.append_attempt(bad, path=str(tmp_path / "attempts.jsonl"))


def test_attempt_append_roundtrip(tmp_path):
    path = str(tmp_path / "attempts.jsonl")
    rec = ledger.append_attempt(_complete_attempt(result="win"), path=path, now=123.0)
    assert rec["timestamp"] == 123.0
    lines = open(path).read().strip().splitlines()
    assert len(lines) == 1 and '"result": "win"' in lines[0]


def _csv(path, rows):
    import csv as _c

    with open(path, "w", newline="") as f:
        w = _c.DictWriter(f, fieldnames=["model", "dtype", "act", "token", "kernel_path_us", "e2e_us", "mfu"])
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_compare_csvs_detects_regression_and_wins(tmp_path):
    base = str(tmp_path / "base.csv")
    cand = str(tmp_path / "cand.csv")
    _csv(
        base,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.50,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 100,
                "e2e_us": 150,
                "mfu": 0.05,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 500,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    _csv(
        cand,
        [
            # large bucket: +10% MFU win, no kernel-path regression
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 950,
                "e2e_us": 1180,
                "mfu": 0.56,
            },
            # small token: 20% faster and >=2us -> win
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 80,
                "e2e_us": 150,
                "mfu": 0.05,
            },
            # mid token: regression on kernel-path (+10% and +50us)
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 550,
                "e2e_us": 600,
                "mfu": 0.30,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert cv.any_regression is True  # the 128-token point regressed
    assert not cv.pareto_clean
    assert ("kimi_k2", "a4w4", "silu", "16384") in cv.large_wins
    assert ("kimi_k2", "a4w4", "silu", "16") in cv.small_wins
