"""Assemble fresh full-40 a4w4 candidate + fresh paired baseline, then verify the
GPT-OSS persist_m=1 claim with compare_csvs. Loop3 R1.

Per Codex (option A/C): the locked snapshot baseline cannot be the numeric
denominator because identical-config rows drift cross-session. So we measure BOTH
sides fresh in the same session/state:
  candidate = pm4 everywhere EXCEPT GPT-OSS large (8192/16384/32768) = _pm1
  baseline  = pm4 everywhere (the production-default/test-default config)
Only the GPT-OSS large rows differ -> isolates the persist_m=1 lever. The strict
dispatch-change verdict times only those changed rows and proves every other row
keeps the same production dispatch config by comparing config-identity columns.

Tiny-token DS V3/Kimi t1/t2/t4 dispatch a CK kernel whose torch REFERENCE overflows
to nan (issue #643); those are quarantined (error_category=reference_invalid),
excluded from the gate/regression but recorded. Quarantine is honored only when the
row's category is actually reference_invalid (a real mismatch is never hidden).

Usage: python3 scripts/loop3_assemble_fresh40.py
"""

import csv

from scripts.moe_tuning_ledger import compare_csvs, compare_csvs_dispatch_change, scan_candidate_csv_freshness

FRESH = "docs/loop3_models/fresh40"
FRESH_BASE = "docs/loop3_models/fresh40_baseline"
LOCKED = "docs/baseline_523ca1c7_validated.csv"
CAND_OUT = "docs/loop3_models/candidate_fresh40.csv"
BASE_OUT = "docs/loop3_models/baseline_fresh40_pm4.csv"
WIN_TOKENS = {"8192", "16384", "32768"}
CHANGED_KEYS = {("gpt_oss", "a4w4", "swiglu", token) for token in WIN_TOKENS}
QUARANTINE = {
    ("deepseek_v3", "a4w4", "silu", "1"),
    ("deepseek_v3", "a4w4", "silu", "2"),
    ("deepseek_v3", "a4w4", "silu", "4"),
    ("kimi_k2", "a4w4", "silu", "1"),
    ("kimi_k2", "a4w4", "silu", "2"),
    ("kimi_k2", "a4w4", "silu", "4"),
}


def _key(r):
    return (r["model"], r["dtype"], r["act"], r["token"])


def load(path):
    with open(path, newline="") as f:
        return {_key(r): r for r in csv.DictReader(f)}


def _hdr():
    # The old locked baseline predates the persist/swizzle/config-identity
    # columns.  Use the fresh measured files for schema so dispatch identity can
    # be audited, while still using the locked file only for row ordering.
    paths = [
        f"{FRESH}/dsv3_a4w4_fresh.csv",
        f"{FRESH}/kimi_a4w4_fresh.csv",
        f"{FRESH}/gptoss_a4w4_fresh.csv",
        f"{FRESH}/gptoss_a4w4_large_pm1.csv",
        f"{FRESH_BASE}/dsv3_pm4.csv",
        f"{FRESH_BASE}/kimi_pm4.csv",
        f"{FRESH_BASE}/gptoss_pm4.csv",
    ]
    hdr = []
    for path in paths:
        with open(path, newline="") as f:
            for name in csv.DictReader(f).fieldnames or []:
                if name not in hdr:
                    hdr.append(name)
    return hdr


def _base_keys():
    with open(LOCKED, newline="") as f:
        return [_key(r) for r in csv.DictReader(f)]


def assemble(out, parts, overrides=None):
    hdr = _hdr()
    merged = {}
    for p in parts:
        merged.update(load(p))
    if overrides:
        for p in overrides:
            for k, r in load(p).items():
                if k[3] in WIN_TOKENS:
                    merged[k] = r
    rows, missing = [], []
    for k in _base_keys():
        if k in merged:
            rows.append({c: merged[k].get(c, "") for c in hdr})
        else:
            missing.append(k)
    if missing:
        print(f"ERROR {out}: {len(missing)} missing points: {missing}")
        return False
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} rows")
    return True


def main():
    # candidate: default fresh rows + GPT-OSS large pm1 override
    ok_c = assemble(
        CAND_OUT,
        [f"{FRESH}/dsv3_a4w4_fresh.csv", f"{FRESH}/kimi_a4w4_fresh.csv", f"{FRESH}/gptoss_a4w4_fresh.csv"],
        overrides=[f"{FRESH}/gptoss_a4w4_large_pm1.csv"],
    )
    # fresh paired baseline: pm4 everywhere
    ok_b = assemble(
        BASE_OUT,
        [f"{FRESH_BASE}/dsv3_pm4.csv", f"{FRESH_BASE}/kimi_pm4.csv", f"{FRESH_BASE}/gptoss_pm4.csv"],
    )
    if not (ok_c and ok_b):
        return
    print("\ncandidate freshness offenders:", len(scan_candidate_csv_freshness(CAND_OUT)))
    print("baseline  freshness offenders:", len(scan_candidate_csv_freshness(BASE_OUT)))
    v = compare_csvs(BASE_OUT, CAND_OUT, quarantine_keys=QUARANTINE)
    print("\n=== compare_csvs(fresh paired baseline, fresh candidate) ===")
    print("coverage_complete:", v.coverage_complete)
    print("any_regression:", v.any_regression)
    print("large_wins:", v.large_wins)
    print("small_wins:", v.small_wins)
    print("quarantined:", v.quarantined)
    print("gate.passed:", v.gate["passed"], "| violations:", v.gate["violations"][:6])
    print(">>> claimable_win:", v.claimable_win)

    d = compare_csvs_dispatch_change(BASE_OUT, CAND_OUT, changed_keys=CHANGED_KEYS, quarantine_keys=QUARANTINE)
    print("\n=== compare_csvs_dispatch_change(fresh paired baseline, fresh candidate) ===")
    print("coverage_complete:", d.coverage_complete)
    print("config_identity_clean:", d.config_identity_clean)
    print("timed_clean:", d.timed_clean)
    print("any_changed_regression:", d.any_changed_regression)
    print("large_wins:", d.large_wins)
    print("small_wins:", d.small_wins)
    print("unchanged_config_checked:", len(d.unchanged_config_checked))
    print("incomplete_config_points:", d.incomplete_config_points[:3])
    print("unchanged_config_mismatches:", d.unchanged_config_mismatches[:3])
    print("quarantined:", d.quarantined)
    print("gate.passed:", d.gate["passed"], "| violations:", d.gate["violations"][:6])
    print(">>> claimable_dispatch_win:", d.claimable_dispatch_win)


if __name__ == "__main__":
    main()
