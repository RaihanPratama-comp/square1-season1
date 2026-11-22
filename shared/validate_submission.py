#!/usr/bin/env python3
"""Validate a Square1 submission.csv against its contest spec + sample_submission.csv.

Checks (mirrors official submission format on each contest page):
- file exists, UTF-8, <= 10 MB
- header exactly `id,<target>` (column order per official spec; extra cols ignored by grader but we require exact)
- row count == test rows == sample_submission rows
- every test id present exactly once (missing id = official rejection)
- predictions numeric, no NaN/inf, no thousands separators

Usage: python validate_submission.py <contest> [--data data] [--sub submission.csv]
Contest keys: solar | breathe | emissions | nba
Exit 0 = PASS, 1 = FAIL (reasons printed).
"""
import argparse, sys, os
import numpy as np, pandas as pd

SPEC = {
    "solar":     ("generation_mw", 17520),
    "breathe":   ("pm25", 66699),
    "emissions": ("annual_co2e_t", 12560),
    "nba":       ("home_margin", 3944),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("contest", choices=sorted(SPEC))
    ap.add_argument("--data", default="data")
    ap.add_argument("--sub", default="submission.csv")
    a = ap.parse_args()
    target, expected_rows = SPEC[a.contest]
    fails = []

    if not os.path.exists(a.sub):
        print(f"[FAIL] {a.sub} not found"); sys.exit(1)
    size = os.path.getsize(a.sub)
    if size > 10 * 1024 * 1024:
        fails.append(f"file {size} bytes > 10 MB limit")

    try:
        sub = pd.read_csv(a.sub)
    except Exception as e:
        print(f"[FAIL] cannot parse CSV: {e}"); sys.exit(1)

    if list(sub.columns) != ["id", target]:
        fails.append(f"header {list(sub.columns)} != ['id', '{target}']")

    test = pd.read_csv(f"{a.data}/test.csv", usecols=["id"])
    sample_path = f"{a.data}/sample_submission.csv"
    if os.path.exists(sample_path):
        sample = pd.read_csv(sample_path)
        if list(sample.columns) != list(sub.columns):
            fails.append(f"header differs from sample_submission.csv: {list(sample.columns)}")
        if len(sample) != len(sub):
            fails.append(f"rows {len(sub)} != sample_submission rows {len(sample)}")

    if len(sub) != len(test):
        fails.append(f"rows {len(sub)} != test rows {len(test)}")
    if len(test) != expected_rows:
        print(f"[warn] test rows {len(test)} != official spec {expected_rows} (synthetic data?)")

    if "id" in sub.columns:
        missing = set(test["id"]) - set(sub["id"])
        extra = set(sub["id"]) - set(test["id"])
        dup = sub["id"].duplicated().sum()
        if missing: fails.append(f"{len(missing)} test ids missing (official rejection)")
        if extra: fails.append(f"{len(extra)} ids not in test")
        if dup: fails.append(f"{dup} duplicate ids")

    if target in sub.columns:
        vals = pd.to_numeric(sub[target], errors="coerce")
        if vals.isna().any(): fails.append(f"{int(vals.isna().sum())} non-numeric/NaN predictions")
        if np.isinf(vals.fillna(0)).any(): fails.append("infinite predictions present")
        if a.contest in ("solar", "breathe", "emissions") and (vals < 0).any():
            fails.append(f"{int((vals < 0).sum())} negative predictions (must be >= 0)")
        print(f"[info] {target}: min={vals.min():.3f} mean={vals.mean():.3f} max={vals.max():.3f} rows={len(sub)} size={size}B")

    if fails:
        for f in fails: print(f"[FAIL] {f}")
        sys.exit(1)
    print(f"[PASS] {a.contest}: {a.sub} matches official spec (header id,{target}; {len(sub)} rows)")
    sys.exit(0)

if __name__ == "__main__":
    main()
