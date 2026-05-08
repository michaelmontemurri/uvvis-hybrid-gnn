#!/usr/bin/env python3
"""Aggregate saved tabular-baseline predictions across random seeds.

This helper reads `preds_test.csv` files from a directory of per-seed baseline
runs, stacks the seed-level predictions into one table, and writes mean/std
columns for downstream plotting or summary analysis.
"""

import argparse
from pathlib import Path
import pandas as pd


def collect_seed_preds(xgb_dir: Path, n_seeds: int):
    """Collect one prediction column per seed and add aggregate statistics."""
    base = None
    seed_cols = []

    for s in range(1, n_seeds):
        seed_name = f"seed_{s:04d}"
        csv_path = xgb_dir / seed_name / "preds_test.csv"

        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")

        df = pd.read_csv(csv_path)

        pred_col = "pred_mean"
        out_col = f"pred_{seed_name}"
        seed_cols.append(out_col)

        if base is None:
            base = df[["smiles", "y_true", pred_col]].copy()
            base = base.rename(columns={pred_col: out_col})
        else:
            # Keep strict alignment so downstream mean/std values are meaningful.
            if not (df["smiles"].equals(base["smiles"]) and df["y_true"].equals(base["y_true"])):
                raise ValueError(
                    f"Row mismatch between {csv_path} and base DataFrame. "
                    "Smiles/y_true order or content differ."
                )
            base[out_col] = df[pred_col].values

    if base is None:
        raise RuntimeError(f"No seed predictions were aggregated from {xgb_dir}")

    base["pred_mean"] = base[seed_cols].mean(axis=1)
    base["pred_std"] = base[seed_cols].std(axis=1, ddof=0)

    cols = ["smiles", "y_true", "pred_mean", "pred_std"] + seed_cols
    base = base[cols]

    return base


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate per-seed preds_test.csv into a single file "
                    "with per-seed columns + mean/std."
    )
    ap.add_argument(
        "--xgb-dir",
        required=True,
        help="Path to the xgb/ directory "
             "(e.g. results/phase1C/deep4chem/scaffold/fp_baseline_N00250_s42/xgb)"
    )
    ap.add_argument(
        "--n-seeds",
        type=int,
        default=30,
        help="Number of seeds to aggregate (default: 30)"
    )
    ap.add_argument(
        "--out-name",
        default="preds_test_allseeds.csv",
        help="Output CSV name inside xgb-dir (default: preds_test_allseeds.csv)"
    )
    args = ap.parse_args()

    xgb_dir = Path(args.xgb_dir).resolve()
    df = collect_seed_preds(xgb_dir, args.n_seeds)

    out_path = xgb_dir / args.out_name
    df.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path}  (n_rows={len(df)})")


if __name__ == "__main__":
    main()
