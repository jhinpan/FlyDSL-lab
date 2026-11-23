#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""E2: is GCN assembly text a sufficient interface to the code object?

A backend that owns instruction selection, ordering and register allocation
still needs instruction encoding and the HSA code object.  If the LLVM
assembler and LLD rebuild FlyDSL's code object from the dumped ``.s`` byte for
byte, those two tools can be the whole remaining LLVM dependency.

For every IR dump under ``--dumps`` (the E1 output works as is), this assembles
``*_final_isa.s`` with ``tools/inject_asm`` and compares the result with the
code object embedded in ``*_fly_emit_gpu_binary.mlir``.

Control: the same ``.s`` with one extra ``s_nop`` after the first MFMA must
give a different code object, so an equal result is not an artifact of the
comparison.

    python3 experiments/agent_first/e1_sched_order.py --out /tmp/e1
    python3 experiments/agent_first/e2_asm_roundtrip.py --dumps /tmp/e1
"""

import argparse
import importlib.util
import json
import re
import struct
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_inject_asm():
    spec = importlib.util.spec_from_file_location("inject_asm", REPO_ROOT / "tools" / "inject_asm" / "inject_asm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def elf_sections(blob: bytes) -> dict:
    """Section name -> bytes of a little-endian ELF64 object."""
    shoff, shentsize, shnum, shstrndx = struct.unpack_from("<Q", blob, 0x28) + struct.unpack_from("<HHH", blob, 0x3A)
    headers = [struct.unpack_from("<IIQQQQ", blob, shoff + i * shentsize) for i in range(shnum)]
    strtab_off, strtab_size = headers[shstrndx][4], headers[shstrndx][5]
    names = blob[strtab_off : strtab_off + strtab_size]
    out = {}
    for name_off, sh_type, _flags, _addr, offset, size in headers:
        name = names[name_off : names.index(b"\0", name_off)].decode()
        out[name] = b"" if sh_type == 8 else blob[offset : offset + size]  # SHT_NOBITS has no bytes
    return out


def roundtrip(inject_asm, dump_dir: Path, clang: Path = None) -> dict:
    isa = next(iter(sorted(dump_dir.rglob("*_final_isa.s"))))
    binary_ir = next(iter(sorted(dump_dir.rglob("*_fly_emit_gpu_binary.mlir"))))
    ir_text = binary_ir.read_text()

    _, rebuilt, original = inject_asm.replace_binary(ir_text, isa, clang)

    # Control: one extra s_nop after the first MFMA must change the object.
    asm = isa.read_text()
    mutated, count = re.subn(r"^(\s+v_mfma_\S+ .*)$", r"\1\n\ts_nop 0", asm, count=1, flags=re.M)
    if count != 1:
        raise RuntimeError(f"no MFMA to mutate in {isa}")
    with tempfile.TemporaryDirectory() as tmp:
        mutated_path = Path(tmp) / "mutated.s"
        mutated_path.write_text(mutated)
        _, rebuilt_mutated, _ = inject_asm.replace_binary(ir_text, mutated_path, clang)

    old, new = elf_sections(original), elf_sections(rebuilt)
    differing = sorted(name for name in old.keys() | new.keys() if old.get(name) != new.get(name))
    return {
        "dump": dump_dir.name,
        "code_object_bytes": len(original),
        "text_bytes": len(old.get(".text", b"")),
        "code_object_equal": rebuilt == original,
        "text_section_equal": old.get(".text") == new.get(".text"),
        "differing_sections": differing,
        "control_mutation_differs": rebuilt_mutated != original,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dumps", type=Path, required=True, help="directory whose subdirectories hold IR dumps")
    parser.add_argument("--clang", type=Path, help="assembler; defaults to the ROCm clang, as in tools/inject_asm")
    args = parser.parse_args()

    inject_asm = _load_inject_asm()
    dump_dirs = sorted(d for d in args.dumps.iterdir() if d.is_dir() and any(d.rglob("*_final_isa.s")))
    if not dump_dirs:
        parser.error(f"no IR dumps under {args.dumps}")

    results = [roundtrip(inject_asm, d, args.clang) for d in dump_dirs]
    clang, lld = inject_asm._toolchain(args.clang)
    print(f"assembler: {clang}   linker: {lld or 'in-process LLD of the installed flydsl'}")
    print(
        f"{'dump':6} {'object bytes':>12} {'.text bytes':>11} {'object equal':>12} {'.text equal':>11} {'control':>8}  differing sections"
    )
    for r in results:
        print(
            f"{r['dump']:6} {r['code_object_bytes']:12d} {r['text_bytes']:11d} {str(r['code_object_equal']):>12} "
            f"{str(r['text_section_equal']):>11} {'ok' if r['control_mutation_differs'] else 'FAILED':>8}  "
            f"{', '.join(r['differing_sections']) or '-'}"
        )
    (args.dumps / "e2_result.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {args.dumps / 'e2_result.json'}")
    ok = all(r["text_section_equal"] and r["control_mutation_differs"] for r in results)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
