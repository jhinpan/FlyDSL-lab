#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""E3: what does traced order cost at run time?

E1 shows that ``gemm_a16w16_gfx950`` keeps its traced instruction order when
LLVM's machine schedulers are off and the ``sched_barrier`` fences are removed
(variant D), and where the shipped kernel (variant A) differs.  This times the
variants on a GPU and checks that they compute the same result.

    A  fences on,  schedulers on    (as shipped)
    C  fences off, schedulers on    (the scheduler alone decides)
    D  fences off, schedulers off   (traced order)

Each worker process compiles one variant of one kernel configuration, checks
the output against ``torch.mm`` with the tolerance of
``tests/kernels/test_gemm_a16w16_gfx950.py``, then times the compiled launcher.
The driver rotates the variant order round by round, so clock drift lands on all.

Checks made by the driver:
  * every worker's output passes the tolerance;
  * all workers of a shape read identical inputs;
  * the variants give bit-identical outputs, or the result says they do not;
  * the launch loop returns well before the GPU finishes (``enqueue_share``), so
    the time is GPU time and not the cost of the Python launch path.

Run it on one idle GPU, with ``HIP_VISIBLE_DEVICES`` set:

    python3 experiments/agent_first/e3_runtime_ab.py --out /tmp/e3
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "A": {"fences": True, "sched": True},
    "C": {"fences": False, "sched": True},
    "D": {"fences": False, "sched": False},
}
BASELINE = "A"

SCHED_OFF = {"enable-misched": False, "enable-post-misched": False}

# (M, N, K): one shape dominated by the K loop, one by the per-tile prologue and epilogue.
DEFAULT_SHAPES = ((4096, 4096, 4096), (16384, 16384, 256))
DEFAULT_CONFIGS = ('{"use_half_tile_interleaved": true}', '{"use_half_tile_interleaved": false}')


# ---------------------------------------------------------------------------
# Worker: one variant of one kernel configuration, all shapes.
# ---------------------------------------------------------------------------


def _digest(tensor) -> str:
    import torch

    return hashlib.sha256(tensor.contiguous().view(torch.int16).cpu().numpy().tobytes()).hexdigest()[:16]


def _time_launches(launch, sync, warmup_s: float, batch_s: float, batches: int) -> dict:
    """Seconds per launch for each batch, after a timed warm-up."""
    start, count = time.perf_counter(), 0
    while time.perf_counter() - start < warmup_s:
        for _ in range(16):
            launch()
        sync()
        count += 16
    per_launch = (time.perf_counter() - start) / count
    iters = max(16, int(batch_s / per_launch))

    samples, enqueue = [], []
    for _ in range(batches):
        sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            launch()
        t1 = time.perf_counter()
        sync()
        t2 = time.perf_counter()
        samples.append((t2 - t0) / iters)
        enqueue.append((t1 - t0) / (t2 - t0))
    # enqueue_share near 1 would mean the host launch path, not the GPU, sets the time.
    return {"iters_per_batch": iters, "seconds_per_launch": samples, "enqueue_share": enqueue}


def run_worker(args) -> None:
    cfg = VARIANTS[args.worker]
    user_kwargs = json.loads(args.config)
    os.environ["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
    if args.compile_only:
        os.environ.update(COMPILE_ONLY="1", FLYDSL_GPU_ARCH="gfx950")
    sys.path.insert(0, str(REPO_ROOT))

    import torch

    import flydsl
    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import rocdl

    if not cfg["fences"]:
        # The kernel calls rocdl.sched_barrier(...) through the module attribute.
        rocdl.sched_barrier = lambda *a, **k: None

    import kernels.gemm.gemm_a16w16_gfx950 as gemm

    captured = {}
    run_cached = gemm.run_cached

    def capture(jit_func, *compile_args, constexpr_param, compiler, dispatch_args):
        compiled = run_cached(
            jit_func, *compile_args, constexpr_param=constexpr_param, compiler=compiler, dispatch_args=dispatch_args
        )
        captured.update(compiled=compiled, dispatch_args=dispatch_args, param=constexpr_param)
        return compiled

    gemm.run_cached = capture

    def hints():
        if cfg["sched"]:
            return nullcontext()
        return CompilationContext.compile_hints({"llvm_options": dict(SCHED_OFF)})

    device = "cpu" if args.compile_only else "cuda"
    record = {
        "variant": args.worker,
        "config": user_kwargs,
        "versions": {"flydsl": flydsl.__version__, "torch": torch.__version__},
        "device": None if args.compile_only else torch.cuda.get_device_name(0),
        "shapes": [],
    }
    for m, n, k in args.shapes:
        torch.manual_seed(1234)
        a = torch.empty((m, k), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        b = torch.empty((k, n), dtype=torch.bfloat16, device=device).uniform_(-1, 1)
        out = torch.zeros((m, n), dtype=torch.bfloat16, device=device)
        stream = fx.Stream(None) if args.compile_only else None
        with hints():
            gemm.gemm_a16w16(a, b, out=out, user_kwargs=dict(user_kwargs), stream=stream, layout="nn")
        entry = {"shape": [m, n, k], "kernel": gemm.make_gemm_a16w16_gfx950_kernel_name(captured["param"])}
        if not args.compile_only:
            torch.cuda.synchronize()
            ref = torch.mm(a, b)
            atol = 2e-1 * (k / 8192) ** 0.5
            torch.testing.assert_close(out, ref, atol=atol, rtol=2e-1, check_dtype=True)
            entry.update(
                max_abs_diff=(out.float() - ref.float()).abs().max().item(),
                input_digest=_digest(a) + _digest(b),
                output_digest=_digest(out),
            )
            compiled, dispatch_args = captured["compiled"], captured["dispatch_args"]
            entry.update(
                _time_launches(
                    lambda: compiled(*dispatch_args),
                    torch.cuda.synchronize,
                    args.warmup_s,
                    args.batch_s,
                    args.batches,
                )
            )
        record["shapes"].append(entry)
    print("E3_RECORD " + json.dumps(record))


# ---------------------------------------------------------------------------
# Driver: alternate the variants and compare.
# ---------------------------------------------------------------------------


def _worker_record(cmd: list) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    lines = [line for line in proc.stdout.splitlines() if line.startswith("E3_RECORD ")]
    if proc.returncode != 0 or len(lines) != 1:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-6000:])
        raise SystemExit(f"worker failed: {' '.join(cmd)}")
    return json.loads(lines[0][len("E3_RECORD ") :])


def run_driver(args) -> int:
    args.out.mkdir(parents=True, exist_ok=True)
    variants = list(args.variants)
    if BASELINE not in variants or len(variants) < 2:
        raise SystemExit(f"--variants needs {BASELINE} and at least one other variant")
    shape_args = [str(v) for shape in args.shapes for v in shape]
    base = [args.python, __file__, "--shapes", *shape_args]
    base += ["--warmup-s", str(args.warmup_s), "--batch-s", str(args.batch_s), "--batches", str(args.batches)]
    if args.compile_only:
        base.append("--compile-only")

    records = []
    for config in args.configs:
        for rnd in range(args.rounds):
            shift = rnd % len(variants)
            for variant in variants[shift:] + variants[:shift]:
                record = _worker_record([*base, "--worker", variant, "--config", config])
                record["round"] = rnd
                records.append(record)
                print(f"  done: config={config} round={rnd} variant={variant}", flush=True)

    others = [v for v in variants if v != BASELINE]
    summary, ok = [], True
    for config in args.configs:
        cfg = json.loads(config)
        for index, shape in enumerate(args.shapes):
            rows = [(r["round"], r["variant"], r["shapes"][index]) for r in records if r["config"] == cfg]
            item = {"config": cfg, "shape": list(shape), "kernel": rows[0][2]["kernel"]}
            if not args.compile_only:
                # One number per worker: the median of its timed batches.
                by_round = {v: {} for v in variants}
                for rnd, variant, entry in rows:
                    by_round[variant][rnd] = statistics.median(entry["seconds_per_launch"])
                item["variants"] = {}
                for v in variants:
                    mid = statistics.median(by_round[v].values())
                    item["variants"][v] = {
                        "ms": mid * 1e3,
                        "tflops": 2.0 * shape[0] * shape[1] * shape[2] / mid / 1e12,
                        "round_spread_pct": 100 * (max(by_round[v].values()) - min(by_round[v].values())) / mid,
                    }
                item["over_baseline"] = {}
                for v in others:
                    ratios = [by_round[v][rnd] / by_round[BASELINE][rnd] for rnd in sorted(by_round[BASELINE])]
                    item["over_baseline"][v] = {
                        "median": statistics.median(ratios),
                        "min": min(ratios),
                        "max": max(ratios),
                        "by_round": ratios,
                    }
                item.update(
                    same_inputs=len({e["input_digest"] for _, _, e in rows}) == 1,
                    bit_identical_outputs=len({e["output_digest"] for _, _, e in rows}) == 1,
                    max_abs_diff_vs_torch=max(e["max_abs_diff"] for _, _, e in rows),
                    max_enqueue_share=max(max(e["enqueue_share"]) for _, _, e in rows),
                )
                ok &= item["same_inputs"]
            summary.append(item)

    print()
    for item in summary:
        head = f"{item['kernel']}  shape={tuple(item['shape'])}"
        if args.compile_only:
            print(f"{head}  compiled")
            continue
        print(head)
        for v, t in item["variants"].items():
            print(f"  {v}  {t['ms']:.4f} ms  {t['tflops']:7.1f} TFLOPS  round spread {t['round_spread_pct']:.2f}%")
        for v, r in item["over_baseline"].items():
            print(f"  {v}/{BASELINE}  median {r['median']:.4f}  range [{r['min']:.4f}, {r['max']:.4f}]")
        print(
            f"  bit-identical outputs={item['bit_identical_outputs']}  same inputs={item['same_inputs']}"
            f"  max |out - torch.mm|={item['max_abs_diff_vs_torch']:.4f}"
            f"  max enqueue share={item['max_enqueue_share']:.3f}"
        )

    result = {
        "variants": variants,
        "rounds": args.rounds,
        "batches": args.batches,
        "batch_s": args.batch_s,
        "warmup_s": args.warmup_s,
        "versions": records[0]["versions"],
        "device": records[0]["device"],
        "summary": summary,
        "records": records,
    }
    (args.out / "e3_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out / 'e3_result.json'}")
    return 0 if ok else 1


def _shapes(values: list) -> tuple:
    if len(values) % 3:
        raise argparse.ArgumentTypeError("--shapes takes M N K triples")
    return tuple(tuple(values[i : i + 3]) for i in range(0, len(values), 3))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="directory for e3_result.json")
    parser.add_argument("--shapes", type=int, nargs="+", help="M N K triples; default: two shapes")
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS), help="JSON user_kwargs, one per kernel")
    parser.add_argument("--variants", nargs="+", choices=sorted(VARIANTS), default=["A", "D"], help="variants to time")
    parser.add_argument("--rounds", type=int, default=5, help="passes over the variants per kernel configuration")
    parser.add_argument("--batches", type=int, default=7, help="timed batches per shape in each worker")
    parser.add_argument("--batch-s", type=float, default=0.3, help="target seconds per timed batch")
    parser.add_argument("--warmup-s", type=float, default=1.0, help="seconds of launches before timing")
    parser.add_argument(
        "--compile-only", action="store_true", help="compile every variant on the CPU; no GPU, no timing"
    )
    parser.add_argument("--python", default=sys.executable, help="interpreter with flydsl and torch installed")
    parser.add_argument("--worker", choices=sorted(VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument("--config", default="{}", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.shapes = _shapes(args.shapes) if args.shapes else DEFAULT_SHAPES

    if args.worker:
        run_worker(args)
        return 0
    if args.out is None:
        parser.error("--out is required")
    return run_driver(args)


if __name__ == "__main__":
    sys.exit(main())
