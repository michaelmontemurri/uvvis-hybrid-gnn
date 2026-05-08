#!/usr/bin/env python3
"""
Materialize canonical UVVisML split CSVs from tracked split metadata.

Unlike `get_uvvisml_dataset_splits.py`, this script does not generate a new
train/val/test partition. It reconstructs the tracked canonical split payloads
from:

- the frozen processed master CSV
- the tracked `*_idx.txt` membership files
- the tracked split manifests used for validation

Example:
    python scripts/materialize_uvvisml_splits.py \
      --target abs \
      --dataset deep4chem \
      --split-type scaffold
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from uvvis_hybrid.utils.greenman_utils import get_morgan_fingerprints, handle_duplicates


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

SOLVENT_MAP = {
    "morgan_fingerprint": "sfp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize canonical UVVisML split CSVs from tracked split metadata "
            "without generating a new partition."
        )
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGET_TO_MASTER.keys()),
        required=True,
        help="Prediction target family.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Value of the source column to filter to, e.g. deep4chem.",
    )
    parser.add_argument(
        "--split-type",
        choices=["random", "group_by_smiles", "scaffold"],
        required=True,
        help="Tracked split family to materialize.",
    )
    parser.add_argument(
        "--solvent-rep",
        choices=sorted(SOLVENT_MAP.keys()),
        default="morgan_fingerprint",
        help="Solvent feature representation to materialize.",
    )
    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=DATA_DIR / "uvvisml",
        help="Root that contains tracked split metadata.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DATA_DIR / "uvvisml",
        help="Root where split CSV payloads will be written.",
    )
    return parser.parse_args()


def load_feature_block(feature_json_path: Path, solvent_rep: str) -> list[str]:
    with feature_json_path.open("r") as f:
        feature_blocks = json.load(f)
    block_name = SOLVENT_MAP[solvent_rep]
    if block_name not in feature_blocks:
        raise KeyError(f"Feature block '{block_name}' not found in {feature_json_path}")
    return list(feature_blocks[block_name])


def load_processed_dataset(target: str, dataset: str, solvent_rep: str) -> tuple[pd.DataFrame, list[str], str]:
    master_csv = TARGET_TO_MASTER[target]
    target_col = TARGET_TO_COLUMN[target]

    if not master_csv.exists():
        raise FileNotFoundError(f"Missing processed master CSV: {master_csv}")

    df = pd.read_csv(master_csv)
    dataset_df = df[df["source"].astype(str).str.lower() == dataset.lower()].copy()
    if dataset_df.empty:
        raise ValueError(f"No rows found for dataset/source='{dataset}' in {master_csv}")

    dataset_df.dropna(inplace=True)
    dataset_df = handle_duplicates(dataset_df, target_col=target_col, cutoff=5)
    feature_names = load_feature_block(FEATURE_JSON, solvent_rep)

    missing_feature_names = [col for col in feature_names if col not in dataset_df.columns]
    if missing_feature_names:
        if solvent_rep != "morgan_fingerprint":
            raise ValueError(
                f"{len(missing_feature_names)} feature columns are missing from {master_csv}. "
                f"First few: {missing_feature_names[:3]}"
            )
        dataset_df, feature_names = get_morgan_fingerprints(dataset_df, mol_or_solv="solvents")

    required = ["smiles", "solvent", target_col] + feature_names
    missing = [col for col in required if col not in dataset_df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required columns are missing from {master_csv}. "
            f"First few: {missing[:3]}"
        )

    return dataset_df.reset_index(drop=True), feature_names, target_col


def load_indices(path: Path) -> list[int]:
    values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [int(v) for v in values]


def reconstruct_from_manifest(
    split_name: str,
    dataset_df: pd.DataFrame,
    manifest_path: Path,
    idx_path: Path,
    feature_names: list[str],
    target_col: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    idx = load_indices(idx_path)
    if len(manifest) != len(idx):
        raise ValueError(
            f"{split_name} metadata length mismatch: manifest has {len(manifest)} rows "
            f"but idx file has {len(idx)} entries."
        )

    required_manifest_cols = ["row_idx", "smiles", "solvent", target_col]
    if list(manifest.columns) != required_manifest_cols:
        raise ValueError(
            f"{split_name} manifest columns differ: {manifest.columns.tolist()} vs {required_manifest_cols}"
        )

    if manifest["row_idx"].tolist() != idx:
        raise ValueError(
            f"{split_name} manifest row_idx values differ from {idx_path.name}."
        )

    join_cols = ["smiles", "solvent"]
    lookup_cols = join_cols + [target_col] + feature_names
    lookup = dataset_df.loc[:, lookup_cols].copy()
    if lookup.duplicated(subset=join_cols).any():
        dupes = lookup.loc[lookup.duplicated(subset=join_cols, keep=False), join_cols].head(3)
        raise ValueError(
            f"{split_name} feature lookup is not unique on {join_cols}. Example rows:\n{dupes}"
        )

    manifest = manifest.reset_index().rename(columns={"index": "_manifest_order"})
    merged = manifest.merge(lookup, on=join_cols, how="left", validate="one_to_one")
    if merged[feature_names].isnull().any().any():
        missing = merged.loc[merged[feature_names].isnull().any(axis=1), join_cols + [target_col]].head(3)
        raise ValueError(
            f"Could not recover feature rows for {split_name} manifest entries. Example rows:\n{missing}"
        )
    target_delta = (merged[target_col + "_x"] - merged[target_col + "_y"]).abs()
    if (target_delta > 1e-5).any():
        bad = merged.loc[target_delta > 1e-5, join_cols + [target_col + "_x", target_col + "_y"]].head(3)
        raise ValueError(
            f"Recovered {split_name} rows disagree with manifest target values. Example rows:\n{bad}"
        )

    merged = merged.sort_values("_manifest_order").reset_index(drop=True)
    merged[target_col] = merged[target_col + "_x"]
    return merged.loc[:, ["smiles", "solvent", target_col] + feature_names]


def write_split_payload(
    split_name: str,
    split_df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    out_dir: Path,
) -> None:
    target_df = split_df.loc[:, ["smiles", "solvent", target_col]].copy()
    feature_df = split_df.loc[:, feature_names].copy()
    target_df.to_csv(out_dir / f"smiles_target_{split_name}.csv", index=False)
    feature_df.to_csv(out_dir / f"features_{split_name}.csv", index=False)


def main() -> None:
    args = parse_args()
    metadata_dir = args.metadata_root / args.target / args.dataset / "splits" / args.split_type
    out_dir = args.out_root / args.target / args.dataset / "splits" / args.split_type

    required_meta = [
        metadata_dir / "train_idx.txt",
        metadata_dir / "val_idx.txt",
        metadata_dir / "test_idx.txt",
        metadata_dir / "train_manifest.csv",
        metadata_dir / "val_manifest.csv",
        metadata_dir / "test_manifest.csv",
    ]
    missing_meta = [str(p) for p in required_meta if not p.exists()]
    if missing_meta:
        raise FileNotFoundError(
            "Missing tracked split metadata files:\n- " + "\n- ".join(missing_meta)
        )

    dataset_df, feature_names, target_col = load_processed_dataset(
        args.target, args.dataset, args.solvent_rep
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_df = reconstruct_from_manifest(
            split_name=split_name,
            dataset_df=dataset_df,
            manifest_path=metadata_dir / f"{split_name}_manifest.csv",
            idx_path=metadata_dir / f"{split_name}_idx.txt",
            feature_names=feature_names,
            target_col=target_col,
        )
        write_split_payload(
            split_name=split_name,
            split_df=split_df,
            feature_names=feature_names,
            target_col=target_col,
            out_dir=out_dir,
        )

    print(f"Materialized canonical split payloads in: {out_dir}")


if __name__ == "__main__":
    main()
