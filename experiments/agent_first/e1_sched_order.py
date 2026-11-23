#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""E1: is the hand-fenced instruction order just source order?

Kernel authors place ``rocdl.sched_barrier(0)`` fences so that LLVM's machine
scheduler keeps the order they wrote.  If that is the only job of the fences,
turning the pre-RA and post-RA machine schedulers off must give the same order
without any fence.

Each variant compiles ``gemm_a16w16_gfx950`` in its own process (LLVM cl::opts
are process-global), compile-only, with no GPU:

    A  fences on,  schedulers on    (as shipped)
    B  fences on,  schedulers off
    C  fences off, schedulers on
    D  fences off, schedulers off

The order is compared on anchor instructions only: MFMA, LDS reads and writes,
VMEM loads and stores, and s_barrier.  Wait counters, nops and ALU are left out
because late passes insert or move them.  Each variant's emitted order is
compared with the other variants and with the traced order, read from the LLVM
IR that FlyDSL hands to LLVM.

Controls, all checked by the driver:
  * fences-off variants trace zero ``rocdl.sched.barrier`` ops, fences-on more;
  * the schedulers-off flag changes the ISA of the unfenced kernel (C != D),
    so the flag is not a dead knob;
  * every variant keeps the same number of anchors of each class;
  * the LLVM IR holds the same anchors as the ISA, class by class, and the
    traced order is the same in all four variants.

Whether the dumped ISA is the code that runs is checked by e2_asm_roundtrip.py.

    python3 experiments/agent_first/e1_sched_order.py --out /tmp/e1
"""

import argparse
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter
from contextlib import nullcontext
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

VARIANTS = {
    "A": {"fences": True, "sched": True},
    "B": {"fences": True, "sched": False},
    "C": {"fences": False, "sched": True},
    "D": {"fences": False, "sched": False},
}

SCHED_OFF = {"enable-misched": False, "enable-post-misched": False}

ANCHOR_CLASSES = (
    ("mfma", re.compile(r"^v_mfma_")),
    ("ds_read", re.compile(r"^ds_read")),
    ("ds_write", re.compile(r"^ds_write")),
    ("vmem_load", re.compile(r"^(buffer|global)_load")),
    ("vmem_store", re.compile(r"^(buffer|global)_(store|atomic)")),
    ("barrier", re.compile(r"^s_barrier")),
)

_INSTR = re.compile(r"^\s+([a-z][a-z0-9_]*)\b\s*(.*?)\s*(?:;.*)?$")
_FUNC_LABEL = re.compile(r"^([A-Za-z_][\w.$]*):")
_FUNC_END = re.compile(r"^\.Lfunc_end\d+:")
_DIRECTIVE_VALUE = re.compile(
    r"^\s+\.amdhsa_(next_free_vgpr|next_free_sgpr|group_segment_fixed_size|private_segment_fixed_size)\s+(\d+)"
)


# ---------------------------------------------------------------------------
# Worker: compile one variant and leave the IR dump behind.
# ---------------------------------------------------------------------------


def run_worker(variant: str, dump_dir: str, shape: tuple, user_kwargs: dict) -> None:
    cfg = VARIANTS[variant]
    os.environ.update(
        FLYDSL_GPU_ARCH="gfx950",
        COMPILE_ONLY="1",
        FLYDSL_RUNTIME_ENABLE_CACHE="0",
        FLYDSL_DUMP_IR="1",
        FLYDSL_DUMP_DIR=dump_dir,
    )
    sys.path.insert(0, str(REPO_ROOT))

    import torch

    import flydsl.expr as fx
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl.expr import rocdl

    if not cfg["fences"]:
        # The kernel calls rocdl.sched_barrier(...) through the module attribute.
        rocdl.sched_barrier = lambda *args, **kwargs: None

    import kernels.gemm.gemm_a16w16_gfx950 as gemm

    m, n, k = shape
    a = torch.empty((m, k), dtype=torch.bfloat16)
    b = torch.empty((k, n), dtype=torch.bfloat16)
    hints = nullcontext() if cfg["sched"] else CompilationContext.compile_hints({"llvm_options": dict(SCHED_OFF)})
    with hints:
        gemm.gemm_a16w16(a, b, user_kwargs=user_kwargs, stream=fx.Stream(None), layout="nn")

    import flydsl

    versions = {"flydsl": flydsl.__version__, "torch": torch.__version__}
    Path(dump_dir, "versions.json").write_text(json.dumps(versions))


# ---------------------------------------------------------------------------
# Driver: parse the dumps and compare.
# ---------------------------------------------------------------------------


def _single(dump_dir: Path, suffix: str) -> Path:
    hits = sorted(dump_dir.rglob(f"*{suffix}"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one *{suffix} under {dump_dir}, found {len(hits)}")
    return hits[0]


def parse_isa(path: Path) -> dict:
    """Instructions of every function in the ISA dump, in emission order."""
    funcs, current, meta = {}, None, {}
    for line in path.read_text().splitlines():
        m = _DIRECTIVE_VALUE.match(line)
        if m:
            meta[m.group(1)] = int(m.group(2))
            continue
        if _FUNC_END.match(line):
            current = None
            continue
        m = _FUNC_LABEL.match(line)
        if m and not line.startswith("."):
            current = funcs.setdefault(m.group(1), [])
            continue
        if current is None:
            continue
        m = _INSTR.match(line)
        if m:
            current.append((m.group(1), m.group(2)))
    return {"funcs": funcs, "meta": meta}


def anchor_class(mnemonic: str):
    for name, pattern in ANCHOR_CLASSES:
        if pattern.match(mnemonic):
            return name
    return None


# The same anchor classes on the LLVM IR that FlyDSL hands to LLVM, which is still in traced order.
_IR_ANCHORS = (
    ("mfma", re.compile(r"call .*@llvm\.amdgcn\.mfma\.")),
    ("ds_read", re.compile(r"call .*@llvm\.amdgcn\.ds\.read\.|= load .*, ptr addrspace\(3\)")),
    ("ds_write", re.compile(r"^\s+store .*, ptr addrspace\(3\)")),
    (
        "vmem_load",
        re.compile(r"call .*@llvm\.amdgcn\.(raw\.(ptr\.)?buffer\.load|global\.load)|= load .*, ptr addrspace\(1\)"),
    ),
    (
        "vmem_store",
        re.compile(r"call .*@llvm\.amdgcn\.raw\.(ptr\.)?buffer\.(store|atomic)|^\s+store .*, ptr addrspace\(1\)"),
    ),
    ("barrier", re.compile(r"call .*@llvm\.amdgcn\.s\.barrier")),
)


def parse_ir_anchors(path: Path, kernel: str) -> list:
    """Anchor classes of the kernel's LLVM IR, in textual (traced) order."""
    anchors, inside = [], False
    for line in path.read_text().splitlines():
        if line.startswith("define "):
            inside = f"@{kernel}(" in line
            continue
        if line.startswith("}"):
            inside = False
            continue
        if not inside or not line.startswith(" "):
            continue
        for name, pattern in _IR_ANCHORS:
            if pattern.search(line):
                anchors.append(name)
                break
    return anchors


def summarize(variant: str, dump_dir: Path) -> dict:
    isa = parse_isa(_single(dump_dir, "_final_isa.s"))
    origin = _single(dump_dir, "00_origin.mlir").read_text()
    ir_path = _single(dump_dir, "_llvm_ir.ll")
    # The kernel is the function with the most instructions; host code is not in the ISA dump.
    name, instrs = max(isa["funcs"].items(), key=lambda kv: len(kv[1]))
    mnemonics = [mn for mn, _ in instrs]
    anchors = [c for c in map(anchor_class, mnemonics) if c]
    return {
        "variant": variant,
        **VARIANTS[variant],
        "kernel": name,
        "versions": json.loads((dump_dir / "versions.json").read_text()),
        "traced_sched_barriers": len(re.findall(r"\brocdl\.sched\.barrier\b", origin)),
        "instructions": len(instrs),
        "anchor_counts": dict(Counter(anchors)),
        "authored_s_waitcnt": len(re.findall(r"call .*@llvm\.amdgcn\.s\.waitcnt", ir_path.read_text())),
        "s_waitcnt": sum(mn.startswith("s_waitcnt") for mn in mnemonics),
        "s_nop": sum(mn == "s_nop" for mn in mnemonics),
        "vgpr": isa["meta"].get("next_free_vgpr"),
        "sgpr": isa["meta"].get("next_free_sgpr"),
        "lds_bytes": isa["meta"].get("group_segment_fixed_size"),
        # Spills: the private segment the kernel reserves and its scratch loads and stores.
        "scratch_bytes": isa["meta"].get("private_segment_fixed_size"),
        "scratch_accesses": sum(mn.startswith("scratch_") for mn in mnemonics),
        "anchors": anchors,
        "ir_anchors": parse_ir_anchors(ir_path, name),
        "mnemonics": mnemonics,
        "text": [f"{mn} {ops}" for mn, ops in instrs],
    }


def _digest(items) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()[:12]


def _runs(seq) -> str:
    """Run-length form of an anchor slice, e.g. '6xmfma 5xds_write mfma'."""
    out = []
    for item in seq:
        if out and out[-1][0] == item:
            out[-1][1] += 1
        else:
            out.append([item, 1])
    return " ".join(f"{n}x{c}" if n > 1 else c for c, n in out)


def order_distance(label: str, ax: list, ay: list) -> dict:
    """How far anchor order ay is from ax; the edits list where they part."""
    matcher = difflib.SequenceMatcher(a=ax, b=ay, autojunk=False)
    edits = [
        {"op": tag, "at": i1, "left": _runs(ax[i1:i2]), "right": _runs(ay[j1:j2])}
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]
    return {
        "pair": label,
        "order_equal": ax == ay,
        "in_common_order": sum(block.size for block in matcher.get_matching_blocks()),
        "total": len(ax),
        "first_divergence": next((i for i, (p, q) in enumerate(zip(ax, ay)) if p != q), None),
        "edits": edits,
    }


def _print_distance(d: dict) -> None:
    where = "-" if d["first_divergence"] is None else d["first_divergence"]
    print(
        f"  {d['pair']:22} equal={str(d['order_equal']):5}  in common order "
        f"{d['in_common_order']}/{d['total']}  first divergence at anchor {where}"
    )


def run_driver(out: Path, shape: tuple, user_kwargs: str, python: str, reuse: bool) -> int:
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for variant in VARIANTS:
        dump_dir = out / variant
        if not reuse:
            if dump_dir.exists():
                raise SystemExit(f"{dump_dir} exists; pass a fresh --out, or --reuse to re-read its dumps")
            cmd = [python, __file__, "--worker", variant, "--dump-dir", str(dump_dir), "--shape", *map(str, shape)]
            cmd += ["--user-kwargs", user_kwargs]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
                raise SystemExit(f"variant {variant} failed to compile")
        summaries[variant] = summarize(variant, dump_dir)

    between = [
        order_distance(f"{x} vs {y}", summaries[x]["anchors"], summaries[y]["anchors"])
        for x, y in (("A", "D"), ("B", "D"), ("A", "B"), ("A", "C"), ("C", "D"))
    ]
    to_source = [order_distance(f"traced IR vs {v}", s["ir_anchors"], s["anchors"]) for v, s in summaries.items()]

    controls = {
        "fences_removed": summaries["C"]["traced_sched_barriers"] == 0
        and summaries["D"]["traced_sched_barriers"] == 0
        and summaries["A"]["traced_sched_barriers"] > 0
        and summaries["B"]["traced_sched_barriers"] == summaries["A"]["traced_sched_barriers"],
        "sched_flag_is_live": summaries["C"]["text"] != summaries["D"]["text"],
        "same_anchor_counts": len({json.dumps(s["anchor_counts"], sort_keys=True) for s in summaries.values()}) == 1,
        "ir_classifier_complete": all(Counter(s["ir_anchors"]) == Counter(s["anchors"]) for s in summaries.values()),
        "same_traced_order": len({_digest(s["ir_anchors"]) for s in summaries.values()}) == 1,
    }

    ref = summaries["A"]
    print(
        f"kernel: {ref['kernel']}  shape (M, N, K) = {shape}  user_kwargs: {user_kwargs}  versions: {ref['versions']}"
    )
    print(f"anchor counts: {ref['anchor_counts']}  authored s_waitcnt: {ref['authored_s_waitcnt']}")
    print(
        f"{'var':3} {'fences':6} {'sched':5} {'traced fences':>13} {'instr':>6} {'waitcnt':>7} {'nop':>4} "
        f"{'vgpr':>5} {'scratch bytes':>13} {'scratch ops':>11}"
    )
    for v, s in summaries.items():
        print(
            f"{v:3} {'on' if s['fences'] else 'off':6} {'on' if s['sched'] else 'off':5} "
            f"{s['traced_sched_barriers']:13d} {s['instructions']:6d} {s['s_waitcnt']:7d} {s['s_nop']:4d} "
            f"{s['vgpr']:5d} {s['scratch_bytes']:13d} {s['scratch_accesses']:11d}"
        )
    print("\nanchor order, variant against variant:")
    for d in between:
        _print_distance(d)
    print("\nanchor order, traced LLVM IR against emitted ISA:")
    for d in to_source:
        _print_distance(d)
    print("\nwhere the shipped order (A) leaves the no-fence, no-scheduler order (D):")
    for edit in between[0]["edits"]:
        print(f"  {edit['op']:8} at A[{edit['at']}]: A has [{edit['left']}]  D has [{edit['right']}]")
    print()
    for name, ok in controls.items():
        print(f"control {name}: {'ok' if ok else 'FAILED'}")

    hidden = ("anchors", "ir_anchors", "mnemonics", "text")
    record = {
        "shape": shape,
        "user_kwargs": json.loads(user_kwargs),
        "variants": {
            v: {k: s[k] for k in s if k not in hidden}
            | {"anchor_order_sha": _digest(s["anchors"]), "isa_text_sha": _digest(s["text"])}
            for v, s in summaries.items()
        },
        "variant_vs_variant": between,
        "traced_ir_vs_isa": to_source,
        "controls": controls,
    }
    (out / "e1_result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out / 'e1_result.json'}")
    return 0 if all(controls.values()) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, help="fresh directory for the four IR dumps and e1_result.json")
    parser.add_argument("--shape", type=int, nargs=3, default=(4096, 4096, 4096), metavar=("M", "N", "K"))
    parser.add_argument("--user-kwargs", default="{}", help="JSON dict passed to gemm_a16w16 as user_kwargs")
    parser.add_argument("--python", default=sys.executable, help="interpreter with flydsl and torch installed")
    parser.add_argument("--reuse", action="store_true", help="re-read the dumps already under --out; no compile")
    parser.add_argument("--worker", choices=sorted(VARIANTS), help=argparse.SUPPRESS)
    parser.add_argument("--dump-dir", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        run_worker(args.worker, args.dump_dir, tuple(args.shape), json.loads(args.user_kwargs))
        return 0
    if args.out is None:
        parser.error("--out is required")
    return run_driver(args.out, tuple(args.shape), args.user_kwargs, args.python, args.reuse)


if __name__ == "__main__":
    sys.exit(main())
