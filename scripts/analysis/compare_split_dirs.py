#!/usr/bin/env python3
"""
Compare two split directories file by file.

This script checks whether one split directory matches another after sorting
rows by a stable set of key columns such as `smiles` or `smiles, solvent`.
We used it to verify that our generated split files matched the original
Greenman outputs, but it is written as a general split-comparison utility.

Example
-------
python scripts/analysis/compare_split_dirs.py \
    --ours data/uvvisml_test/abs/deep4chem/splits/scaffold \
    --theirs third_party/greenman/results/split_type/scaffold/chemprop/morgan_fingerprint/fp/deep4chem \
    --keys smiles \
    --rtol 1e-10 \
    --atol 1e-12

If one side does not include a `solvent` column, using `smiles` alone as the
sort key is fine.
"""


import os
import argparse
import hashlib
import pandas as pd
import numpy as np

EXPECTED = [
    "smiles_target_train.csv",
    "smiles_target_val.csv",
    "smiles_target_test.csv",
    "features_train.csv",
    "features_val.csv",
    "features_test.csv",
]

def read_csv_safe(path: str) -> pd.DataFrame:
    """Load a CSV and let pandas infer types before normalization."""
    return pd.read_csv(path)

def coerce_missing_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def select_and_order_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return a view with columns ordered consistently across both inputs."""
    existing = [c for c in cols if c in df.columns]
    return df[existing].copy()

def normalize_df(df: pd.DataFrame, sort_keys: list[str], float_cols: list[str] | None,
                 rtol: float, atol: float) -> pd.DataFrame:
    out = df.copy()

    # Sort deterministically on any provided keys that are present in the file.
    keys = [k for k in sort_keys if k in out.columns]
    if keys:
        out[keys] = out[keys].fillna("")
        out = out.sort_values(by=keys, kind="mergesort")
    out = out.reset_index(drop=True)

    if float_cols is None:
        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in float_cols if c in out.columns]

    # A tight fixed rounding makes comparisons stable across CSV serialization.
    decimals = 12
    for c in num_cols:
        out[c] = out[c].astype(float).round(decimals)

    return out

def df_hash(df: pd.DataFrame) -> str:
    """Hash a normalized DataFrame via a stable CSV serialization."""
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()

def compare_files(ours_path: str, theirs_path: str, sort_keys: list[str],
                  rtol: float, atol: float) -> bool:
    ours = read_csv_safe(ours_path)
    theirs = read_csv_safe(theirs_path)

    # Compare against the union of both column sets so missing columns are explicit.
    all_cols = list(dict.fromkeys(list(ours.columns) + list(theirs.columns)))
    ours = coerce_missing_cols(ours, all_cols)
    theirs = coerce_missing_cols(theirs, all_cols)
    ours = select_and_order_cols(ours, all_cols)
    theirs = select_and_order_cols(theirs, all_cols)

    ours_n  = normalize_df(ours, sort_keys, float_cols=None, rtol=rtol, atol=atol)
    theirs_n = normalize_df(theirs, sort_keys, float_cols=None, rtol=rtol, atol=atol)

    if list(ours_n.columns) != list(theirs_n.columns):
        print(f"  [DIFF] Column sets differ.")
        print(f"    ours:   {list(ours_n.columns)}")
        print(f"    theirs: {list(theirs_n.columns)}")
        return False
    if len(ours_n) != len(theirs_n):
        print(f"  [DIFF] Row counts differ: ours={len(ours_n)} vs theirs={len(theirs_n)}")
        return False

    # The hash gives a quick exact-equivalence check after normalization.
    if df_hash(ours_n) == df_hash(theirs_n):
        print("  [OK] Files are identical after normalization.")
        return True

    neq_mask = (ours_n != theirs_n) & ~(ours_n.isna() & theirs_n.isna())

    # For numeric columns, allow approximate equality within the requested tolerance.
    num_cols = ours_n.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        a = ours_n[c].to_numpy()
        b = theirs_n[c].to_numpy()
        both_notna = ~np.isnan(a) & ~np.isnan(b)
        approx_eq = np.isclose(a[both_notna], b[both_notna], rtol=rtol, atol=atol)
        mismatch_idx = np.where(both_notna)[0][~approx_eq]
        if mismatch_idx.size == 0:
            pass
        else:
            full_idx = np.where(both_notna)[0][~approx_eq]
            neq_mask.loc[full_idx, c] = True

    if not neq_mask.to_numpy().any():
        print("  [OK] Files are numerically equal within tolerances.")
        return True

    # Report a few mismatches
    mismatch_rows = np.where(neq_mask.any(axis=1))[0][:10]
    print(f"  [DIFF] Found {neq_mask.any(axis=1).sum()} mismatched rows (showing up to 10):")
    show_cols = list(ours_n.columns[:10]) 
    for i in mismatch_rows:
        print(f"    Row {i}:")
        print("      ours:   ", ours_n.loc[i, show_cols].to_dict())
        print("      theirs: ", theirs_n.loc[i, show_cols].to_dict())
    return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True, help="First split directory to compare")
    ap.add_argument("--theirs", required=True, help="Second split directory to compare")
    ap.add_argument("--keys", nargs="*", default=["smiles"], help="Columns to sort by for stable comparison")
    ap.add_argument("--rtol", type=float, default=1e-10, help="Relative tolerance for numeric comparison")
    ap.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for numeric comparison")
    args = ap.parse_args()

    ok = True
    print(f"Comparing our dir: {args.ours}")
    print(f"      to theirs:   {args.theirs}")
    print(f"Sort keys: {args.keys} | rtol={args.rtol} | atol={args.atol}\n")

    for fname in EXPECTED:
        ours_path = os.path.join(args.ours, fname)
        theirs_path = os.path.join(args.theirs, fname)
        if not os.path.exists(ours_path):
            print(f"[MISS] Ours missing:   {ours_path}")
            ok = False
            continue
        if not os.path.exists(theirs_path):
            print(f"[MISS] Theirs missing: {theirs_path}")
            ok = False
            continue
        print(f"File: {fname}")
        same = compare_files(ours_path, theirs_path, sort_keys=args.keys,
                             rtol=args.rtol, atol=args.atol)
        ok = ok and same
        print("")

    if ok:
        print("All expected files match (after normalization).")
    else:
        print("Differences found (see logs above).")

if __name__ == "__main__":
    main()


"""
for target in em; do
  for dataset in dsscdb chemfluor deep4chem; do
    for split in scaffold; do
      echo "----------------------------------------"
      echo "Checking $target / $dataset / $split"

      python scripts/analysis/compare_split_dirs.py \
        --ours data/uvvisml_test/$target/$dataset/splits/$split \
        --theirs data/uvvisml/$target/$dataset/splits/$split \
        --keys smiles solvent

    done
  done
done
"""
