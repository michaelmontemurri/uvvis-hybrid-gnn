#!/usr/bin/env python3
"""Collect ensemble predictions from transfer-learning runs.

This helper expects the tracked member layout:

- `results_dir/member_model_i/<stage-name>/preds_test.csv`

and it requires each member prediction file to use the tabular prediction
schema written by the public artifact scripts:

- `smiles`
- `y_true`
- `pred_head0`
- `pred_mean`
- `pred_std`
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

REQUIRED_PRED_COLUMNS = ("smiles", "y_true", "pred_mean")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ensemble predictions from transfer learning runs.")
    parser.add_argument("--results-dir", required=True, help="Base directory for the ensemble run (e.g. .../N03072_s42)")
    parser.add_argument("--data-dir", required=True, help="Directory containing the ground truth data (smiles_target_test.csv)")
    parser.add_argument("--n-members", type=int, default=5, help="Number of ensemble members")
    parser.add_argument("--stage-name", default="stageB_unfreeze", help="Name of the stage folder containing preds_test.csv")
    parser.add_argument("--target-col", default="peakwavs_max", help="Name of the target column")
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output CSV path. Defaults to results_dir/ensemble_mean_over_pretrained/preds_test_ensemble_mean.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data_dir = Path(args.data_dir)
    target_col = args.target_col

    gt_path = data_dir / "smiles_target_test.csv"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found at {gt_path}")

    df_gt = pd.read_csv(gt_path)
    if target_col not in df_gt.columns:
        raise ValueError(f"Target column '{target_col}' not found in ground truth CSV")

    df_truth = df_gt[["smiles", target_col]].rename(columns={target_col: "y_true"}).copy()

    df_base = None
    pred_cols = []
    for i in range(args.n_members):
        member_dir = results_dir / f"member_model_{i}" / args.stage_name
        pred_path = member_dir / "preds_test.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Prediction file missing for member {i}: {pred_path}")

        df_pred = pd.read_csv(pred_path)
        missing = [col for col in REQUIRED_PRED_COLUMNS if col not in df_pred.columns]
        if missing:
            raise ValueError(
                f"Missing required columns {missing} in {pred_path}; "
                f"found columns={list(df_pred.columns)}"
            )

        col_name = f"pred_model_{i}"
        df_pred = df_pred[["smiles", "y_true", "pred_mean"]].rename(columns={"pred_mean": col_name})

        if df_base is None:
            df_base = df_pred.copy()
            if len(df_base) != len(df_truth):
                raise ValueError(
                    f"Row-count mismatch between {pred_path} and {gt_path}: "
                    f"{len(df_base)} vs {len(df_truth)}"
                )
            if not df_base["smiles"].equals(df_truth["smiles"]):
                raise ValueError(
                    f"SMILES order/content mismatch between {pred_path} and {gt_path}."
                )
            if not np.allclose(df_base["y_true"].to_numpy(float), df_truth["y_true"].to_numpy(float), rtol=0, atol=1e-8):
                raise ValueError(
                    f"y_true mismatch between {pred_path} and {gt_path}."
                )
        else:
            if not df_pred["smiles"].equals(df_base["smiles"]):
                raise ValueError(
                    f"SMILES order/content mismatch between {pred_path} and the first member prediction file."
                )
            if not np.allclose(df_pred["y_true"].to_numpy(float), df_base["y_true"].to_numpy(float), rtol=0, atol=1e-8):
                raise ValueError(
                    f"y_true mismatch between {pred_path} and the first member prediction file."
                )
            df_base[col_name] = df_pred[col_name].to_numpy(float)

        pred_cols.append(col_name)

    if df_base is None:
        raise RuntimeError(f"No member predictions were aggregated from {results_dir}")

    df_base["pred_mean"] = df_base[pred_cols].mean(axis=1)
    df_base["pred_std"] = df_base[pred_cols].std(axis=1, ddof=0)

    y_true = df_base["y_true"]
    y_pred = df_base["pred_mean"]
    mask = ~y_true.isna() & ~y_pred.isna()
    rmse = float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])))
    mae = float(mean_absolute_error(y_true[mask], y_pred[mask]))
    r2 = float(r2_score(y_true[mask], y_pred[mask]))

    print(f"Ensemble Metrics (N={len(pred_cols)}): RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    if args.out_file:
        out_path = Path(args.out_file)
    else:
        out_dir = results_dir / "ensemble_mean_over_pretrained"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "preds_test_ensemble_mean.csv"
        metrics_path = out_dir / "metrics.txt"
        with open(metrics_path, "w") as f:
            f.write(f"rmse = {rmse}\n")
            f.write(f"mae = {mae}\n")
            f.write(f"r2 = {r2}\n")
        print(f"[INFO] Saved metrics to {metrics_path}")

    final_cols = ["smiles", "y_true", "pred_mean", "pred_std"] + pred_cols
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_base[final_cols].to_csv(out_path, index=False)
    print(f"[INFO] Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
