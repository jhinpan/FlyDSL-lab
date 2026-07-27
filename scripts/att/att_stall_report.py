#!/usr/bin/env python3
"""Join ATT per-PC stall counts with a disassembly listing.

  stallmap.py <stalls.csv> <disasm.s> <code_object_id> [top-N]

Prints the highest-stall instructions and a per-mnemonic roll-up.
"""
import csv
import re
import sys
from collections import Counter, defaultdict

ADDR = re.compile(r"^\s*(\S.*?)\s*//\s*([0-9A-F]{8,16}):")
CATS = ["NONE", "SMEM", "SALU", "VMEM", "FLAT", "LDS", "VALU", "JUMP", "NEXT",
        "IMMED", "CONTEXT", "MESSAGE", "BVH"]


def main():
    stalls_csv, disasm, co_id = sys.argv[1], sys.argv[2], int(sys.argv[3])
    topn = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    insn = {}
    for line in open(disasm):
        m = ADDR.match(line)
        if m:
            insn[int(m.group(2), 16)] = m.group(1).strip()

    rows = []
    tot_stall = tot_dur = tot_exec = 0
    for r in csv.DictReader(open(stalls_csv)):
        if int(r["code_object_id"]) != co_id:
            continue
        a = int(r["address"])
        rows.append((a, int(r["category"]), int(r["executions"]),
                     int(r["stall_cycles"]), int(r["total_cycles"])))
        tot_stall += int(r["stall_cycles"])
        tot_dur += int(r["total_cycles"])
        tot_exec += int(r["executions"])

    print(f"code object {co_id}: {len(rows)} PCs, {tot_exec:,} executions, "
          f"{tot_stall:,} stall cycles, {tot_dur:,} total cycles")
    print(f"matched {sum(1 for a, *_ in rows if a in insn)}/{len(rows)} PCs to disassembly\n")

    by_mnem = defaultdict(lambda: [0, 0, 0])  # count, stall, dur
    by_cat = defaultdict(lambda: [0, 0])
    for a, cat, n, st, du in rows:
        text = insn.get(a, "<unknown>")
        mn = text.split()[0] if text != "<unknown>" else "<unknown>"
        e = by_mnem[mn]
        e[0] += n
        e[1] += st
        e[2] += du
        c = by_cat[CATS[cat] if cat < len(CATS) else str(cat)]
        c[0] += n
        c[1] += st

    print("=== stall cycles by instruction category ===")
    for k, (n, st) in sorted(by_cat.items(), key=lambda x: -x[1][1]):
        print(f"  {k:10s} exec={n:10,}  stall={st:12,}  ({100*st/tot_stall:5.1f}%)  "
              f"stall/exec={st/max(n,1):6.1f}")

    print("\n=== stall cycles by mnemonic (top 20) ===")
    for mn, (n, st, du) in sorted(by_mnem.items(), key=lambda x: -x[1][1])[:20]:
        print(f"  {mn:32s} exec={n:10,}  stall={st:12,} ({100*st/tot_stall:5.1f}%)  "
              f"stall/exec={st/max(n,1):6.1f}")

    print(f"\n=== top {topn} individual PCs by stall ===")
    for a, cat, n, st, du in sorted(rows, key=lambda r: -r[3])[:topn]:
        text = insn.get(a, "<unknown>")
        print(f"  0x{a:06x} {CATS[cat] if cat < len(CATS) else cat:6s} exec={n:8,} "
              f"stall={st:11,} ({100*st/tot_stall:4.1f}%) per={st/max(n,1):7.1f}  {text[:88]}")


if __name__ == "__main__":
    main()
