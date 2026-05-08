#!/usr/bin/env python3
"""
Create train/validation/test splits for the long-format Jeffries dataset.

This script loads the processed Jeffries long-format table, drops rows with
missing values in the required molecule, solvent, and target columns,
resolves duplicate `(smiles, solvent)` measurements with the project’s
duplicate-handling rule, and writes Chemprop-ready split files using
solvent fingerprint features only.

The prediction target is selected with `--target`, for example
`peakwavs_max` or `empeakwavs_max`.
"""

import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from uvvis_hybrid.utils.greenman_utils import handle_duplicates, data_split_and_write, _write_split_metadata

DATASET   = "jeffries"
DATASOURCE = "custom"
DEFAULT_TARGET = "empeakwavs_max" 


def parse_args():
    """Parse CLI arguments for Jeffries split generation."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-type",
        choices=["random", "scaffold", "holdout"],
        default="random",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--target",
        type=str,
        default=DEFAULT_TARGET,
        help="Name of target"
             "(e.g. 'em' or 'abs').",
    )
    ap.add_argument(
        "--sizes",
        type=str,
        default="0.8,0.1,0.1",
        help="Train,Val,Test fractions (e.g., '0.8,0.1,0.1').",
    )
    return ap.parse_args()


def _write_split_files(
    out_dir: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: list[str],
    target_names: list[str],
) -> None:
    """Write Chemprop-style split CSVs and the matching metadata files."""
    old_cwd = os.getcwd()
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir)
    try:
        train_df[["smiles", "solvent"] + target_names].to_csv("smiles_target_train.csv", index=False)
        val_df[["smiles", "solvent"] + target_names].to_csv("smiles_target_val.csv", index=False)
        test_df[["smiles", "solvent"] + target_names].to_csv("smiles_target_test.csv", index=False)

        if feature_names:
            train_df[feature_names].to_csv("features_train.csv", index=False)
            val_df[feature_names].to_csv("features_val.csv", index=False)
            test_df[feature_names].to_csv("features_test.csv", index=False)

        _write_split_metadata("train", train_df, target_names)
        _write_split_metadata("val", val_df, target_names)
        _write_split_metadata("test", test_df, target_names)
    finally:
        os.chdir(old_cwd)


def _split_with_holdout(
    trainval_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    sizes: tuple[float, float, float],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Use the fixed holdout table as test and split the remaining rows into train/val."""
    test_df = holdout_df.copy()

    if test_df.empty:
        raise ValueError("Holdout split requested, but the holdout table is empty.")
    if trainval_df.empty:
        raise ValueError("Holdout split requested, but no rows remain for train/val.")

    train_frac, val_frac, _ = sizes
    if train_frac <= 0 or val_frac < 0:
        raise ValueError("Holdout mode requires non-negative train/val fractions.")

    total_trainval = train_frac + val_frac
    if total_trainval <= 0:
        raise ValueError("Holdout mode requires a non-zero train/val allocation.")

    if val_frac == 0:
        train_df = trainval_df.copy()
        val_df = trainval_df.iloc[0:0].copy()
        return train_df, val_df, test_df

    train_share = train_frac / total_trainval
    gss = GroupShuffleSplit(n_splits=1, train_size=train_share, random_state=seed)
    train_idx, val_idx = next(gss.split(trainval_df, groups=trainval_df["smiles"]))
    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()
    return train_df, val_df, test_df


def main():

    args = parse_args()

    # Parse the train/val/test fractions once up front.
    sizes_tuple = tuple(float(x.strip()) for x in args.sizes.split(","))
    target = args.target 
    if target =="em":
        target_col = "empeakwavs_max"
    elif target == "abs":
        target_col = "peakwavs_max"

    # The target family determines both the input CSV location and output root.
    in_csv = ROOT / "data" / DATASOURCE / target / DATASET / "jeffries.csv"
    holdout_csv = ROOT / "data" / DATASOURCE / target / DATASET / "jeffries_holdout.csv"
    out_root = ROOT / "data" / DATASOURCE / target / DATASET / "splits"

    in_csv = str(in_csv)
    out_root = str(out_root)

    if not os.path.exists(in_csv):
        raise FileNotFoundError(f"Missing input CSV: {in_csv}")
    if args.split_type == "holdout" and not os.path.exists(holdout_csv):
        raise FileNotFoundError(f"Missing holdout CSV: {holdout_csv}")

    df = pd.read_csv(in_csv)

    # Normalize column names so the Greenman helper utilities can be reused.
    rename_map = {
        "SMILES": "smiles",
        "peakwavs_max": "peakwavs_max",
        "em_peakwavs_max": "empeakwavs_max",
        "solvent": "solvent",
        "HOMO": "homo",
        "LUMO": "lumo",
    }
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)
    print(df.columns)
    required = {"smiles", "solvent", target_col}
    missing_req = required - set(df.columns)
    if missing_req:
        raise ValueError(f"Missing required columns in {in_csv}: {missing_req}")

    df = df.dropna(subset=["smiles", "solvent", target_col])

    # The downstream split writer expects a source column.
    if "source" not in df.columns:
        df["source"] = DATASET

    df = handle_duplicates(df, cutoff=5)

    # Jeffries splits keep only solvent-side features, not HOMO/LUMO values.
    for col in ["homo", "lumo"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    core_cols = {"smiles", "solvent", "empeakwavs_max", "peakwavs_max","source"}
    feature_names = [c for c in df.columns if c not in core_cols]

    if not feature_names:
        raise ValueError(
            "No solvent fingerprint feature columns found. "
            "Expected jeffries.csv to contain additional numeric columns beyond "
            f"{sorted(core_cols)}."
        )

    target_names = [target_col]

    data = df[["smiles", "solvent"] + feature_names + target_names].copy()

    out_dir = os.path.join(out_root, args.split_type)
    if args.split_type == "holdout":
        holdout_df = pd.read_csv(holdout_csv)
        holdout_df = holdout_df.rename(columns={"SMILES": "smiles", "em_peakwavs_max": "empeakwavs_max"})
        holdout_df = holdout_df[["smiles", "solvent"] + feature_names + target_names].copy()
        holdout_smiles = set(holdout_df["smiles"])
        data = data[~data["smiles"].isin(holdout_smiles)].copy()
        X_train, X_val, X_test = _split_with_holdout(data, holdout_df, sizes_tuple, args.seed)
        _write_split_files(out_dir, X_train, X_val, X_test, feature_names, target_names)
    else:
        os.makedirs(out_dir, exist_ok=True)
        old_cwd = os.getcwd()
        os.chdir(out_dir)
        try:
            X_train, X_val, X_test = data_split_and_write(
                data,
                feature_names=feature_names,
                target_names=target_names,
                solvation=True,
                sizes=sizes_tuple,
                split_type=args.split_type,
                scale_targets=False,
                write_files=True,
                random_seed=args.seed,
                write_split_metadata=True,
            )
        finally:
            os.chdir(old_cwd)

    print(f"Wrote splits to: {out_dir}")


if __name__ == "__main__":
    main()

# example usage:
# python scripts/get_jeffries_dataset_splits.py --target em --split-type scaffold --seed 42 --sizes 1.0,0.0,0.0
