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


def test_parse_aiter_output_pass_warning_line():
    out = (
        "calling test_fmoe(...)\n"
        "ck_moe_2stages:  234.56 us,   654.00 tflops......(quant:fp4x2)[checkAllclose passed~]\n"
        "logits_diff: 0.0008\n"
    )
    res = harness.parse_aiter_output(out)
    assert res["e2e_us"] == 234.56
    assert res["logits_diff"] == 0.0008
    assert res["correctness_pass"] is True


def test_parse_aiter_output_pass_markdown_row():
    # logits_diff below 1e-3 prints no warning line; it only appears in the
    # summary markdown row.  The loose "checkAllclose ... failed!" line is the
    # EXPECTED fp4 elementwise warning and must NOT fail correctness.
    out = (
        "ck_moe_2stages:   84.32 us,  18.80 tflops......(quant:fp4x2)[checkAllclose atol=0.01 rtol=0.01 failed!]\n"
        "moe_2stage summary (markdown):\n"
        "| dtype | token | ... |      us |   logits_diff | model   |\n"
        "|:------|------:| ... |--------:|--------------:|:--------|\n"
        "| torch.bfloat16 | 16 | ... | 87.195 |    9.6236e-06 | legacy  |\n"
    )
    res = harness.parse_aiter_output(out)
    assert res["e2e_us"] == 84.32
    assert res["logits_diff"] == 9.6236e-06
    assert res["correctness_pass"] is True


def test_parse_aiter_output_fail_cases():
    # logits over 0.01 (markdown row) -> fail.
    out_logits = "ck_moe_2stages:  100.00 us, 100.00 tflops\n" "| torch.bfloat16 | 16 | ... | 100.0 | 0.05 | legacy |\n"
    assert harness.parse_aiter_output(out_logits)["correctness_pass"] is False
    # hard assertion text -> fail even if a number was produced.
    out_assert = "ck_moe_2stages:  100.00 us\naccuracy check failed: err=1, logits_diff=0.2\n"
    assert harness.parse_aiter_output(out_assert)["correctness_pass"] is False
    # no logits at all -> fail (cannot confirm correctness).
    out_no_logits = "ck_moe_2stages:  100.00 us, 100.00 tflops\n"
    assert harness.parse_aiter_output(out_no_logits)["correctness_pass"] is False
    # no e2e number at all -> fail.
    assert harness.parse_aiter_output("nothing")["correctness_pass"] is False


def test_aiter_cmd_is_strict_aot_model_correct():
    # Round 3: the aiter guardrail must use the strict/AOT/model-correct runner
    # (scripts/aiter_strict_point.py), NOT the non-strict legacy CLI, and must
    # carry the model's true act/gate, locked warmup/iters, and AOT enabled.
    rp = harness.RunPoint("kimi_k2", 7168, 256, 384, 8, "silu", "a4w4", 16)
    cmd = harness._aiter_cmd(rp)
    joined = " ".join(cmd)
    assert "aiter_strict_point.py" in joined
    # Must NOT be the legacy CLI path.
    assert "test_moe_2stage.py" not in joined
    assert "--no-flydsl-csv" not in cmd
    assert cmd[cmd.index("--aq") + 1] == "fp4"  # a4w4 -> fp4 activation
    assert cmd[cmd.index("--act") + 1] == "silu"
    assert cmd[cmd.index("--gate") + 1] == "separated"
    assert cmd[cmd.index("--warmup") + 1] == "10"
    assert cmd[cmd.index("--iters") + 1] == "100"
    assert "--no-aot" not in cmd  # AOT cache check ON by default
    assert cmd[cmd.index("-t") + 1] == "16"
    # a8w4 -> fp8 activation; swiglu model carries swiglu act.
    rpg = harness.RunPoint("gpt_oss", 3072, 3072, 128, 4, "swiglu", "a8w4", 512)
    cmdg = harness._aiter_cmd(rpg)
    assert cmdg[cmdg.index("--aq") + 1] == "fp8"
    assert cmdg[cmdg.index("--act") + 1] == "swiglu"
    # --no-aot toggle is honored.
    assert "--no-aot" in harness._aiter_cmd(rp, check_aot=False)


def test_parse_strict_aiter_output():
    ok = 'noise\nSTRICT_RESULT {"e2e_us": 80.7, "logits_diff": 1.0e-05, "correctness_pass": true}\n'
    r = harness.parse_strict_aiter_output(ok)
    assert r["e2e_us"] == 80.7 and r["logits_diff"] == 1.0e-05 and r["correctness_pass"] is True
    fail = 'STRICT_RESULT {"error": "AssertionError: accuracy check failed", "correctness_pass": false}\n'
    rf = harness.parse_strict_aiter_output(fail)
    assert rf["correctness_pass"] is False and "AssertionError" in rf["error"]
    miss = harness.parse_strict_aiter_output("no result here")
    assert miss["correctness_pass"] is False and miss["error"] == "no_strict_result"


# --- run-list coverage (full DEC-6 grid from spec) -------------------------


def test_run_list_covers_full_dec6_grid():
    rl = harness.build_run_list()
    # DS V3 (16 tok x 2 dtype) + DS V4 (16 x 1) + Kimi (16 x 2) + GPT-OSS (8 x 2)
    assert len(rl) == 16 * 2 + 16 * 1 + 16 * 2 + 8 * 2 == 96
    keys = harness.expected_point_keys()
    # DeepSeek V4 is a8w4-only.
    assert ("deepseek_v4", "a8w4", "silu", "1") in keys
    assert ("deepseek_v4", "a4w4", "silu", "1") not in keys
    # GPT-OSS has no tiny-token regime; starts at 256.
    assert ("gpt_oss", "a4w4", "swiglu", "256") in keys
    assert ("gpt_oss", "a4w4", "swiglu", "1") not in keys
    # full small + large coverage for a skinny model.
    for tok in (1, 16, 64, 4096, 16384, 32768):
        assert ("kimi_k2", "a4w4", "silu", str(tok)) in keys


# --- baseline validation gate (AC-1 negative tests) ------------------------


def _good_baseline_row(**over):
    row = {
        "gpu_id": "0",
        "gpu_model": "MI350X",
        "branch": "rlcr/mxfp4-moe",
        "commit": "523ca1c7deadbeef",
        "command": "python3 test_moe_gemm.py ... ; python3 test_moe_2stage.py ...",
        "warmup": "10",
        "iters": "100",
        "idle_gpu_verified": "True",
        "graph_capture": "False",
        "l2_flush_per_iter": "True",
        "clocks_pinned": "True",
        "model": "kimi_k2",
        "dtype": "a4w4",
        "act": "silu",
        "token": "16",
        # All AC-1/DEC-2 metric fields present and numeric.
        "stage1_us": "55.3",
        "stage2_us": "21.8",
        "sorting_us": "0.0",
        "kernel_path_us": "77.1",
        "kernel_path_us_p95": "79.0",
        "effective_tflops": "12.3",
        "mfu": "0.0027",
        "e2e_us": "150.0",
        "e2e_us_p95": "155.0",
        "logits_diff": "0.0008",
        "correctness_pass": "True",
    }
    row.update(over)
    return row


def test_validate_baseline_row_accepts_good_row():
    assert harness.validate_baseline_row(_good_baseline_row()) == []


@pytest.mark.parametrize(
    "over,expect",
    [
        ({"commit": "abc123"}, "commit_not_523ca1c7"),
        ({"commit": ""}, "missing_commit"),
        ({"idle_gpu_verified": "False"}, "idle_gpu_not_verified"),
        ({"command": ""}, "missing_command"),
        ({"dtype": ""}, "missing_dtype"),
        ({"act": ""}, "missing_act"),
        ({"e2e_us": ""}, "missing_e2e_us"),
        ({"logits_diff": ""}, "missing_logits_diff"),
        # Hardened metric-field requirements (Codex blocking #2).
        ({"stage1_us": ""}, "missing_stage1_us"),
        ({"stage2_us": ""}, "missing_stage2_us"),
        ({"sorting_us": ""}, "missing_sorting_us"),
        ({"kernel_path_us": ""}, "missing_kernel_path_us"),
        ({"kernel_path_us_p95": ""}, "missing_kernel_path_us_p95"),
        ({"effective_tflops": ""}, "missing_effective_tflops"),
        ({"mfu": ""}, "missing_mfu"),
        ({"e2e_us_p95": ""}, "missing_e2e_us_p95"),
        ({"kernel_path_us": "not-a-number"}, "missing_kernel_path_us"),
        ({"correctness_pass": "False"}, "correctness_not_passed"),
        ({"correctness_pass": ""}, "correctness_not_passed"),
        ({"warmup": "2"}, "warmup_mismatch"),
        ({"iters": "5"}, "iters_mismatch"),
        ({"graph_capture": "True"}, "graph_capture_must_be_off"),
        ({"l2_flush_per_iter": "False"}, "l2_flush_must_be_on"),
        ({"clocks_pinned": "False"}, "clocks_must_be_pinned"),
    ],
)
def test_validate_baseline_row_rejections(over, expect):
    reasons = harness.validate_baseline_row(_good_baseline_row(**over))
    assert expect in reasons


def test_validate_baseline_csv_missing_coverage(tmp_path):
    # A single fully-valid row is not enough; the full workload must be covered.
    out = tmp_path / "baseline.csv"
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True)
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
        token=16,
        stage1_us=55.3,
        stage2_us=21.8,
        sorting_us=0.0,
        kernel_path_us=77.1,
        kernel_path_us_p95=79.0,
        effective_tflops=12.3,
        mfu=0.0027,
        e2e_us=150.0,
        e2e_us_p95=155.0,
        logits_diff=0.0008,
        correctness_pass=True,
    )
    harness.write_csv([row], str(out))
    res = harness.validate_baseline_csv(str(out))
    assert res["valid"] is False
    assert res["missing_points"]  # almost all points missing
    assert res["row_errors"] == {}  # the one present row is itself fully valid


def test_validate_baseline_csv_rejects_missing_kernel_metrics(tmp_path):
    # Codex blocking #2 regression: a full-coverage CSV with e2e/logits present
    # but kernel metrics empty must NOT validate.
    out = tmp_path / "baseline.csv"
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True)
    rows = []
    for rp in harness.build_run_list():
        rows.append(
            harness.PointRow(
                provenance=p,
                command="cmd",
                model=rp.model,
                model_dim=rp.model_dim,
                inter_dim=rp.inter_dim,
                experts=rp.experts,
                topk=rp.topk,
                dtype=rp.dtype,
                act=rp.act,
                token=rp.token,
                # kernel metrics deliberately omitted
                e2e_us=150.0,
                e2e_us_p95=155.0,
                logits_diff=0.0008,
                correctness_pass=True,
            )
        )
    harness.write_csv(rows, str(out))
    res = harness.validate_baseline_csv(str(out))
    assert res["valid"] is False
    assert not res["missing_points"]  # coverage is complete...
    assert res["row_errors"]  # ...but rows fail on missing kernel metrics
    some = next(iter(res["row_errors"].values()))
    assert "missing_kernel_path_us" in some and "missing_mfu" in some


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
    assert cv.coverage_complete  # candidate covers all 3 baseline points
    assert not cv.pareto_clean
    assert ("kimi_k2", "a4w4", "silu", "16384") in cv.large_wins
    assert ("kimi_k2", "a4w4", "silu", "16") in cv.small_wins


def test_compare_csvs_rejects_cherry_picked_candidate(tmp_path):
    # Baseline has 3 points; candidate reports only the single winning large
    # point and omits the others.  Coverage must be incomplete and the verdict
    # must NOT be pareto_clean -- a cherry-picked win cannot pass.
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
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 900,
                "e2e_us": 1100,
                "mfu": 0.56,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert not cv.coverage_complete
    assert ("kimi_k2", "a4w4", "silu", "16") in cv.missing_candidate_points
    assert ("kimi_k2", "a4w4", "silu", "128") in cv.missing_candidate_points
    assert not cv.pareto_clean  # forced False by incomplete coverage


def test_compare_csvs_rejects_missing_regime_fields(tmp_path):
    # Candidate covers every point but the large target bucket lacks mfu, and a
    # point lacks e2e.  Those points are incomplete -> not pareto_clean.
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
            # large bucket missing mfu
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 900,
                "e2e_us": 1100,
                "mfu": "",
            },
            # mid point missing e2e
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 128,
                "kernel_path_us": 480,
                "e2e_us": "",
                "mfu": 0.30,
            },
        ],
    )
    cv = ledger.compare_csvs(base, cand)
    assert not cv.coverage_complete
    assert ("kimi_k2", "a4w4", "silu", "16384") in cv.incomplete_points
    assert ("kimi_k2", "a4w4", "silu", "128") in cv.incomplete_points
    assert not cv.pareto_clean


def test_repeatability_check(tmp_path):
    a = str(tmp_path / "a.csv")
    b = str(tmp_path / "b.csv")
    _csv(
        a,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1000,
                "e2e_us": 1200,
                "mfu": 0.5,
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
        ],
    )
    # b: first point within band (1.5% < 2% and +15us... wait 15us>2us, so need <=max(2%*1000=20us,2us)=20us -> 1015 ok),
    # second point unstable (+10us on a 100us base -> band=max(2us,2us)=2us, 10>2 -> unstable).
    _csv(
        b,
        [
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16384,
                "kernel_path_us": 1015,
                "e2e_us": 1210,
                "mfu": 0.5,
            },
            {
                "model": "kimi_k2",
                "dtype": "a4w4",
                "act": "silu",
                "token": 16,
                "kernel_path_us": 110,
                "e2e_us": 150,
                "mfu": 0.05,
            },
        ],
    )
    res = ledger.repeatability_check(a, b)
    assert res["n_shared"] == 2
    assert not res["stable"]  # the 16-token kernel_path drifted > band
    assert any(u[0] == ("kimi_k2", "a4w4", "silu", "16") for u in res["unstable"]["kernel_path_us"])
    # 16384 kernel_path within band, e2e within band -> not flagged.
    assert all(u[0] != ("kimi_k2", "a4w4", "silu", "16384") for u in res["unstable"]["kernel_path_us"])


def test_quarantine_and_validated_keys():
    from kernels import moe_tuning_spec as spec

    # Round 3: ALL a8w4 shapes are correctness-quarantined (the non-fp4-activation
    # e2e path fails the aiter correctness gate for fp8 AND bf16 activation; only
    # fp4 activation passes).  DS V3 a8w4 is included (its Round 2 "pass" was the
    # legacy-Swiglu artifact, not a real Silu a8w4 pass).
    assert spec.is_quarantined("deepseek_v3", "a8w4")
    assert spec.is_quarantined("deepseek_v4", "a8w4")
    assert spec.is_quarantined("kimi_k2", "a8w4")
    assert spec.is_quarantined("gpt_oss", "a8w4")
    # a4w4 is NOT quarantined for any model.
    assert not spec.is_quarantined("deepseek_v3", "a4w4")
    assert not spec.is_quarantined("kimi_k2", "a4w4")

    vkeys = spec.validated_point_keys()
    # Validated = all a4w4: DS V3 (16) + Kimi (16) + GPT-OSS (8) = 40.
    assert len(vkeys) == 40
    assert ("deepseek_v3", "a4w4", "silu", "1") in vkeys
    assert ("deepseek_v3", "a8w4", "silu", "1") not in vkeys  # quarantined
    assert ("kimi_k2", "a8w4", "silu", "1") not in vkeys  # quarantined
    assert ("gpt_oss", "a8w4", "swiglu", "256") not in vkeys  # quarantined
    # validated subset is a strict subset of the full workload.
    assert vkeys < harness.expected_point_keys()


def test_validate_baseline_csv_subset_keys(tmp_path):
    # A CSV covering only the validated subset validates against validated keys,
    # but fails against the full workload (missing the quarantined points).
    from kernels import moe_tuning_spec as spec

    out = tmp_path / "sub.csv"
    p = harness.Provenance(gpu_id="0", gpu_model="MI350X", branch="b", commit="523ca1c7", idle_gpu_verified=True)
    rows = []
    for key in spec.validated_point_keys():
        model, dtype, act, token = key
        rows.append(
            harness.PointRow(
                provenance=p,
                command="cmd",
                model=model,
                model_dim=7168,
                inter_dim=256,
                experts=257,
                topk=9,
                dtype=dtype,
                act=act,
                token=int(token),
                stage1_us=10.0,
                stage2_us=5.0,
                sorting_us=0.0,
                kernel_path_us=15.0,
                kernel_path_us_p95=15.5,
                effective_tflops=1.0,
                mfu=0.01,
                e2e_us=12.0,
                e2e_us_p95=12.5,
                logits_diff=0.0001,
                correctness_pass=True,
            )
        )
    harness.write_csv(rows, str(out))
    assert harness.validate_baseline_csv(str(out), expected_keys=spec.validated_point_keys())["valid"] is True
    assert harness.validate_baseline_csv(str(out))["valid"] is False  # full workload not covered
