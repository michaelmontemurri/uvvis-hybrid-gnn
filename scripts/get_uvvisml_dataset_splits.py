#!/usr/bin/env python3
"""
Create Chemprop-ready UVVisML dataset splits from the processed master data.

This script loads a processed UVVisML master CSV from `data/processed/`,
filters to a single source dataset, drops rows with missing values, resolves
duplicate `(smiles, solvent)` measurements, selects the requested solvent
feature block from `data/metadata/feature_names.json`, and writes
train/validation/test target and feature CSVs under
`data/uvvisml/{target}/{dataset}/splits/{split_type}/`.

Example
-------
python scripts/get_uvvisml_dataset_splits.py \
    --target em \
    --dataset dsscdb \
    --split-type random \
    --solvent-rep morgan_fingerprint \
    --out-root data/uvvisml_test
"""


from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is importable when the script is run directly.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uvvis_hybrid.utils.greenman_utils import (
    data_split_and_write,
    get_morgan_fingerprints,
    handle_duplicates,
)


DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

FEATURE_JSON = METADATA_DIR / "feature_names.json"

TARGET_TO_MASTER = {
    "em": PROCESSED_DIR / "uvvis_em_master_full.csv",
    "abs": PROCESSED_DIR / "uvvis_abs_master_full.csv",
}

TARGET_TO_COLUMN = {
    "em": "empeakwavs_max",
    "abs": "peakwavs_max",
}

# Solvent representation names map onto feature blocks listed in metadata.
SOLVENT_MAP = {
    "morgan_fingerprint": "sfp",
}


def parse_sizes(sizes_str: str) -> tuple[float, float, float]:
    """Parse and validate train/val/test fractions."""
    parts = tuple(float(x.strip()) for x in sizes_str.split(","))
    if len(parts) != 3 or abs(sum(parts) - 1.0) > 1e-6:
        raise ValueError("--sizes must contain three fractions summing to 1.0, e.g. 0.8,0.1,0.1")
    return parts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Chemprop train/val/test splits from processed UVVisML artifact data."
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_TO_MASTER.keys()),
        default="em",
        help="Prediction target family.",
    )
    parser.add_argument(
        "--dataset",
        default="chemfluor",
        help="Value of the source column to filter to, e.g. chemfluor.",
    )
    parser.add_argument(
        "--split-type",
        choices=["random", "group_by_smiles", "scaffold"],
        default="random",
        help="Dataset splitting strategy.",
    )
    parser.add_argument(
        "--solvent-rep",
        choices=sorted(SOLVENT_MAP.keys()),
        default="morgan_fingerprint",
        help="Solvent feature representation to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for splitting.",
    )
    parser.add_argument(
        "--sizes",
        type=str,
        default="0.8,0.1,0.1",
        help="Train/val/test fractions, e.g. 0.8,0.1,0.1.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DATA_DIR / "uvvisml",
        help="Base directory where split files will be written."
    )
    return parser.parse_args()


def load_feature_block(feature_json_path: Path, solvent_rep: str) -> list[str]:
    if not feature_json_path.exists():
        raise FileNotFoundError(f"Missing feature metadata file: {feature_json_path}")

    with feature_json_path.open("r") as f:
        feature_blocks = json.load(f)

    block_name = SOLVENT_MAP[solvent_rep]
    if block_name not in feature_blocks:
        raise KeyError(f"Feature block '{block_name}' not found in {feature_json_path}")

    return list(feature_blocks[block_name])


def main() -> None:
    args = parse_args()
    sizes = parse_sizes(args.sizes)

    master_csv = TARGET_TO_MASTER[args.target]
    target_col = TARGET_TO_COLUMN[args.target]

    if not master_csv.exists():
        raise FileNotFoundError(f"Missing processed master CSV: {master_csv}")

    df = pd.read_csv(master_csv)

    if "source" not in df.columns:
        raise KeyError(f"'source' column not found in {master_csv}")

    dataset_df = df[df["source"].astype(str).str.lower() == args.dataset.lower()].copy()
    if dataset_df.empty:
        raise ValueError(f"No rows found for dataset/source='{args.dataset}' in {master_csv}")

    dataset_df.dropna(inplace=True)
    dataset_df = handle_duplicates(dataset_df, target_col=target_col, cutoff=5)
    feature_names = load_feature_block(FEATURE_JSON, args.solvent_rep)

    # Public artifact masters keep only compact base columns. If precomputed
    # solvent fingerprints are absent, rebuild them deterministically here.
    missing_feature_names = [col for col in feature_names if col not in dataset_df.columns]
    if missing_feature_names:
        if args.solvent_rep != "morgan_fingerprint":
            raise ValueError(
                f"{len(missing_feature_names)} feature columns are missing from {master_csv}. "
                f"First few: {missing_feature_names[:3]}"
            )
        dataset_df, feature_names = get_morgan_fingerprints(dataset_df, mol_or_solv="solvents")

    required_columns = ["smiles", "solvent", target_col, "source"] + feature_names
    missing_columns = [col for col in required_columns if col not in dataset_df.columns]
    if missing_columns:
        raise ValueError(
            f"{len(missing_columns)} required columns are missing from {master_csv}. "
            f"First few: {missing_columns[:3]}"
        )

    data = dataset_df[["smiles", "solvent", target_col] + feature_names].copy()

    out_dir = args.out_root / args.target / args.dataset / "splits" / args.split_type
    out_dir.mkdir(parents=True, exist_ok=True)

    old_cwd = Path.cwd()
    try:
        os.chdir(out_dir)
        data_split_and_write(
            data,
            feature_names=feature_names,
            target_names=[target_col],
            solvation=True,
            sizes=sizes,
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
