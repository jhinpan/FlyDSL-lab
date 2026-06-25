# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Backend-agnostic tests for the MoE tile-config legality filter.

These tests exercise pure host-side math in ``kernels/moe_tuning.py`` and do not
require a GPU, the FlyROCDL bindings, or a compile.  They lock in two properties:

1. Every tile config currently used by ``scripts/run_benchmark.sh`` for the
   in-scope MXFP4 / A8W4 MoE shapes is accepted.
2. Each named illegal case is rejected with the expected machine-readable reason.
"""

import pytest

from kernels.moe_tuning import (
    LDS_LIMIT_BYTES,
    check_tile_config,
    enumerate_legal_configs,
    stage2_block_count,
)

pytestmark = pytest.mark.l0_backend_agnostic


def test_stage2_block_count_covers_valid_rows():
    # stage2_block_count returns the number of stage2 M-tiles needed to COVER the
    # valid padded rows = ceil(num_valid_rows / tile_m).  It is NOT tied to the
    # stage1 routing block count; smaller stage2 tiles just need proportionally
    # more tiles to span the same rows.
    num_valid_rows = 1280
    assert stage2_block_count(num_valid_rows, 256) == 5  # ceil(1280/256)
    assert stage2_block_count(num_valid_rows, 64) == 20  # ceil(1280/64)
    # coverage holds: tiles * tile_m >= num_valid_rows for every case.
    for tm in (32, 64, 128, 256):
        n = stage2_block_count(num_valid_rows, tm)
        assert n * tm >= num_valid_rows and (n - 1) * tm < num_valid_rows
    # ceil behavior on a non-multiple extent.
    assert stage2_block_count(130, 64) == 3
    assert stage2_block_count(0, 64) == 0
    with pytest.raises(ValueError):
        stage2_block_count(128, 0)
    with pytest.raises(ValueError):
        stage2_block_count(-1, 64)


# (stage, model_dim, inter_dim, tile_m, tile_n, tile_k, a_dtype)
# Derived from run_benchmark.sh MOE_FP4_SHAPES / MOE_A8W4_SHAPES.  Stage1 uses
# (tile_m, tile_n, tile_k); stage2 uses (tile_m, tile_n2, tile_k2).  In the
# benchmark tables tile_n2 == tile_k2 == 256 for all in-scope MoE rows.
_RUN_BENCHMARK_CONFIGS = [
    # MOE_FP4_SHAPES group A: 7168/256/257/9, tile 64/256/256, n2/k2 256/256
    (1, 7168, 256, 64, 256, 256, "fp4"),
    (2, 7168, 256, 64, 256, 256, "fp4"),
    # MOE_FP4_SHAPES group B: 7168/2048/32/8, tile 64/256/256
    (1, 7168, 2048, 64, 256, 256, "fp4"),
    (2, 7168, 2048, 64, 256, 256, "fp4"),
    # MOE_A8W4_SHAPES GPT-OSS: 3072/3072/128/4, stage1 tile 32/128/256
    (1, 3072, 3072, 32, 128, 256, "fp8"),
    # stage2 tile_n2=256, tile_k2=256
    (2, 3072, 3072, 32, 256, 256, "fp8"),
]


@pytest.mark.parametrize("stage,model_dim,inter_dim,tile_m,tile_n,tile_k,a_dtype", _RUN_BENCHMARK_CONFIGS)
def test_accepts_run_benchmark_configs(stage, model_dim, inter_dim, tile_m, tile_n, tile_k, a_dtype):
    res = check_tile_config(
        stage=stage,
        model_dim=model_dim,
        inter_dim=inter_dim,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype=a_dtype,
        gpu_arch="gfx950",
    )
    assert res.legal, f"expected legal, got reason={res.reason!r} ({res.detail})"
    assert res.lds_bytes is not None and res.lds_bytes <= LDS_LIMIT_BYTES["gfx950"]


def test_rejects_stage1_nonstandard_tile_k():
    # tile_k=288 is rejected at stage1: the stage1 mixed-fp4 path only supports
    # tile_k=256, so the stage1_tile_k_unsupported guard fires (it precedes the
    # legacy tile_k_bytes%64 check, which is now effectively stage2-only).
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=288, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "stage1_tile_k_unsupported"


def test_rejects_splitk_k_per_batch_not_div_tile_k():
    # model_dim=7168, k_batch=56 -> k_per_batch=128; 128 % 256 != 0.
    res = check_tile_config(
        stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4", k_batch=56
    )
    assert not res.legal
    assert res.reason == "k_per_batch_not_div_tile_k"


def test_rejects_splitk_model_dim_not_div_k_batch():
    res = check_tile_config(
        stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4", k_batch=3
    )
    assert not res.legal
    assert res.reason == "model_dim_not_div_k_batch"


def test_rejects_stage2_model_dim_not_div_tile_n():
    # 7168 % 384 != 0
    res = check_tile_config(stage=2, model_dim=7168, inter_dim=256, tile_m=64, tile_n=384, tile_k=256, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "model_dim_not_div_tile_n"


def test_rejects_stage2_inter_dim_not_div_tile_k():
    # inter_dim=2048, tile_k=768 -> 2048 % 768 != 0 (and 768 % 64 == 0, tile_k>=256)
    res = check_tile_config(stage=2, model_dim=7168, inter_dim=2048, tile_m=64, tile_n=256, tile_k=768, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "inter_dim_not_div_tile_k"


def test_rejects_lds_over_limit():
    # A very large tile pushes stage1 LDS past the gfx950 163840-byte limit.
    # tile_n=256 divides inter_dim=256 so it reaches the LDS check (not the
    # inter_dim%tile_n gate); tile_m=512 overflows LDS.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=512, tile_n=256, tile_k=256, a_dtype="fp8")
    assert not res.legal
    assert res.reason == "lds_over_limit"
    assert res.lds_bytes is not None and res.lds_bytes > LDS_LIMIT_BYTES["gfx950"]


def test_stage1_fp4_lds_mirrors_builder_no_vec_pack_halving():
    # Regression: stage1 sizes _single_x_bytes from the FULL lds_stride for fp4
    # (no a_elem_vec_pack division), matching compile_mixed_moe_gemm1.  The mirror
    # math must still report the source-faithful footprints 230400 / 197632 bytes
    # for these large-tile_k fp4 configs.  (check_tile_config now rejects tile_k!=256
    # earlier via stage1_tile_k_unsupported, so we assert the LDS mirror directly.)
    from kernels.moe_tuning import stage1_lds_bytes

    r1 = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=32, tile_n=32, tile_k=3584, a_dtype="fp4")
    assert not r1.legal and r1.reason == "stage1_tile_k_unsupported"
    assert stage1_lds_bytes(tile_m=32, tile_n=32, tile_k=3584, a_dtype="fp4") == 230400

    r2 = check_tile_config(stage=1, model_dim=3072, inter_dim=3072, tile_m=32, tile_n=32, tile_k=3072, a_dtype="fp4")
    assert not r2.legal and r2.reason == "stage1_tile_k_unsupported"
    assert stage1_lds_bytes(tile_m=32, tile_n=32, tile_k=3072, a_dtype="fp4") == 197632

    # fp4 and fp8 share the same single_x sizing at stage1 (a_elem_bytes==1, no
    # vec-pack division), so equal tiles give equal LDS.
    assert stage1_lds_bytes(tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4") == stage1_lds_bytes(
        tile_m=64, tile_n=256, tile_k=256, a_dtype="fp8"
    )


def test_rejects_fp4_tile_m_too_small():
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=16, tile_n=256, tile_k=256, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "tile_m_lt_32"


def test_rejects_fp4_tile_k_too_small():
    # tile_k=128 is < 256; still tile_k_bytes % 64 == 0, so the MX-FP4 floor must catch it.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=128, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "tile_k_lt_256"


def test_rejects_stage1_tile_n_not_dividing_inter_dim():
    # Stage1 GEMM1 tiles the inter_dim output by tile_n; the kernel asserts
    # inter_dim % tile_n == 0.  tile_n=512 does not divide inter_dim=256 -> reject
    # pre-compile (previously slipped through and crashed stage1 at runtime).
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=512, tile_k=256, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "inter_dim_not_div_tile_n"
    # tile_n that divides inter_dim stays legal at the filter level.
    ok = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4")
    assert ok.legal
    # large inter_dim shapes (e.g. GPT-OSS 3072) are unaffected.
    ok2 = check_tile_config(stage=1, model_dim=3072, inter_dim=3072, tile_m=32, tile_n=128, tile_k=256, a_dtype="fp4")
    assert ok2.legal


def test_rejects_stage1_fp4_tile_k_not_256():
    # tile_k=512 passes divisibility (model_dim % 512 == 0) and fits LDS, but the
    # stage1 mixed-fp4/fp8 compile path only supports tile_k=256 (tile_k>256 hits a
    # compute_tile IndexError at compile).  Must be rejected pre-compile.
    res = check_tile_config(stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=512, a_dtype="fp4")
    assert not res.legal
    assert res.reason == "stage1_tile_k_unsupported"
    # tile_k=256 stays legal.
    assert check_tile_config(
        stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4"
    ).legal
    # stage2 tile_k is governed by inter_dim divisibility, not this stage1 guard.
    assert check_tile_config(
        stage=2, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4", sort_block_m=256
    ).legal


def test_rejects_bad_stage_and_dtype():
    assert (
        check_tile_config(
            stage=3, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="fp4"
        ).reason
        == "bad_stage"
    )
    assert (
        check_tile_config(
            stage=1, model_dim=7168, inter_dim=256, tile_m=64, tile_n=256, tile_k=256, a_dtype="bogus"
        ).reason
        == "bad_a_dtype"
    )


def test_enumerate_logs_rejections_with_reasons():
    rejected = []
    legal = enumerate_legal_configs(
        stage=1,
        model_dim=7168,
        inter_dim=256,
        a_dtype="fp4",
        tile_m_choices=(16, 32, 64),  # 16 is illegal (tile_m_lt_32)
        tile_n_choices=(256,),
        tile_k_choices=(128, 256),  # 128 is illegal (tile_k_lt_256)
        rejected_log=rejected,
    )
    # At least one legal config (e.g. tile_m in {32,64}, tile_k=256).
    assert legal, "expected some legal configs"
    assert all(r.legal for r in legal)
    # Every rejection carries a machine-readable reason.
    assert rejected, "expected some rejected configs"
    assert all(r["reason"] for r in rejected)
    reasons = {r["reason"] for r in rejected}
    assert "tile_m_lt_32" in reasons
    assert "tile_k_lt_256" in reasons
