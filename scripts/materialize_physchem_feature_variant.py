#!/usr/bin/env python3
"""Build non-destructive physchem-augmented feature CSVs for an existing split.

This script reads an existing split directory containing:

- smiles_target_train.csv / val / test
- features_train.csv / val / test

It computes RDKit physchem descriptors from the SMILES in each split file and
writes new feature CSVs under a deeper subdirectory, leaving the original CSVs
untouched.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uvvis_hybrid.features.rdkit_feats import PHYSCHM_COLS, physchem_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize physchem-augmented feature CSVs without overwriting an existing split."
    )
    parser.add_argument(
        "--split-dir",
        type=Path,
        required=True,
        help="Directory containing smiles_target_{train,val,test}.csv and features_{train,val,test}.csv.",
    )
    parser.add_argument(
        "--out-subdir",
        default="feature_variants/physchem_included",
        help="Deeper output subdirectory under --split-dir.",
    )
    parser.add_argument(
        "--smiles-col",
        default="smiles",
        help="SMILES column in smiles_target_{split}.csv.",
    )
    parser.add_argument(
        "--splits",
        nargs="*",
        default=["train", "val", "test"],
        help="Split names to process.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files in the output subdirectory.",
    )
    return parser.parse_args()


def load_split_tables(split_dir: Path, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_path = split_dir / f"smiles_target_{split_name}.csv"
    feature_path = split_dir / f"features_{split_name}.csv"

    if not target_path.is_file():
        raise FileNotFoundError(f"Missing target CSV: {target_path}")
    if not feature_path.is_file():
        raise FileNotFoundError(f"Missing feature CSV: {feature_path}")

    target_df = pd.read_csv(target_path)
    feature_df = pd.read_csv(feature_path)
    return target_df, feature_df


def build_physchem_features(target_df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    if smiles_col not in target_df.columns:
        raise ValueError(f"Missing smiles column '{smiles_col}' in target CSV.")

    physchem_df = physchem_dataframe(target_df, smiles_col=smiles_col, key_col=smiles_col)
    if physchem_df[PHYSCHM_COLS].isnull().any().any():
        bad_rows = physchem_df.loc[physchem_df[PHYSCHM_COLS].isnull().any(axis=1), [smiles_col]].head(3)
        raise ValueError(f"Physchem computation produced NaNs. Example rows:\n{bad_rows}")
    return physchem_df.drop(columns=[smiles_col])


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    out_dir = split_dir / args.out_subdir

    if not split_dir.is_dir():
        raise SystemExit(f"Split directory does not exist: {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "source_split_dir": str(split_dir),
        "output_dir": str(out_dir),
        "smiles_col": args.smiles_col,
        "splits": [],
        "physchem_columns": PHYSCHM_COLS,
    }

    for split_name in args.splits:
        target_df, feature_df = load_split_tables(split_dir, split_name)
        if len(target_df) != len(feature_df):
            raise ValueError(
                f"Row mismatch for split '{split_name}': "
                f"{len(target_df)} target rows vs {len(feature_df)} feature rows"
            )

        existing_pc_cols = [col for col in feature_df.columns if col.startswith("pc_")]
        if existing_pc_cols:
            raise ValueError(
                f"Existing features_{split_name}.csv already contains physchem columns: {existing_pc_cols[:5]}"
            )

        physchem_df = build_physchem_features(target_df, args.smiles_col)
        merged_df = pd.concat([feature_df.reset_index(drop=True), physchem_df.reset_index(drop=True)], axis=1)

        out_path = out_dir / f"features_{split_name}.csv"
        if out_path.exists() and not args.force:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {out_path}\n"
            )
        merged_df.to_csv(out_path, index=False)

        manifest["splits"].append(
            {
                "split": split_name,
                "rows": int(len(merged_df)),
                "source_target_csv": str(split_dir / f"smiles_target_{split_name}.csv"),
                "source_feature_csv": str(split_dir / f"features_{split_name}.csv"),
                "output_feature_csv": str(out_path),
            }
        )
        print(f"[write] {out_path}")

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {manifest_path}\n"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[write] {manifest_path}")


if __name__ == "__main__":
    main()
