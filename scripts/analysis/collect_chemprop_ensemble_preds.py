#!/usr/bin/env python3
"""Combine per-member Chemprop predictions into a single test-set CSV.

This helper reads `preds_test_model_*.csv` files from an ensemble run, aligns
 them with the canonical test split, and writes one table with per-member
 predictions plus ensemble mean and standard deviation columns.
"""

import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser(
        description="Collect per-model chemprop predictions into a single CSV "
                    "with per-model columns + mean/std."
    )
    ap.add_argument("--data-dir", required=True,
                    help="Subset dir with smiles_target_test.csv "
                         "(e.g. data/uvvisml/deep4chem/subsets/scaffold/N11816_s42)")
    ap.add_argument("--pred-dir", required=True,
                    help="Dir with preds_test_model_*.csv "
                         "(e.g. results/phase1C/deep4chem/scaffold/fromscratch_N11816_s42)")
    ap.add_argument("--n-members", type=int, default=5)
    ap.add_argument("--out", required=True,
                    help="Output CSV path")
    ap.add_argument("--target-col", default="peakwavs_max",
                    help="Name of the target column in smiles_target_test.csv ")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    target_col = args.target_col

    # Use the canonical test CSV as the row ordering anchor.
    base = pd.read_csv(data_dir / "smiles_target_test.csv")
    if "smiles" not in base.columns or target_col not in base.columns:
        raise ValueError(f"Expected columns 'smiles' and '{target_col}' in smiles_target_test.csv")
    base = base[["smiles", target_col]].rename(columns={target_col: "y_true"})

    pred_cols = []
    for i in range(args.n_members):
        path = pred_dir / f"preds_test_model_{i}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        df = pd.read_csv(path)

        # Chemprop prediction files can vary slightly by version, so fall back to
        # the first numeric column if there is no obvious single prediction column.
        if df.shape[1] == 1:
            pred_series = df.iloc[:, 0]
        else:
            numeric_cols = df.select_dtypes(include="number").columns
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric prediction column found in {path}")
            pred_series = df[numeric_cols[0]]

        col_name = f"pred_model_{i}"
        base[col_name] = pred_series.to_numpy()
        pred_cols.append(col_name)

    base["pred_mean"] = base[pred_cols].mean(axis=1)
    base["pred_std"] = base[pred_cols].std(axis=1, ddof=0)

    cols = ["smiles", "y_true", "pred_mean", "pred_std"] + pred_cols
    base = base[cols]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} (n_rows={len(base)})")


if __name__ == "__main__":
    main()
