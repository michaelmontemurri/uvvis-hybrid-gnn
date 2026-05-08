#!/usr/bin/env python3
"""Create fixed-size training subsets from an existing canonical split directory.

The script subsamples only the training split, writes a stable subset manifest,
and either copies or links the shared validation/test files into each subset
directory so downstream workflows can treat them as ordinary split folders.

Example:
split_type=scaffold
dataset=chemfluor
data_source=uvvisml
target=em

# Fixed subsampled train sets (N = 50, 100, 250, 1000, 3072), shared val/test
python scripts/make_fixed_subsets.py \
  --base data/${data_source}/${target}/${dataset}/splits/${split_type} \
  --out-root data/${data_source}/${target}/${dataset}/subsets/${split_type} \
  --n 50, 100, 250, 1000 \
  --copy-valtest \
  --seed 42
"""


import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd


def symlink_or_copy(src: Path, dst: Path):
    """Create a symlink from `dst` to `src`, falling back to a copy if needed."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        os.symlink(os.path.abspath(src), dst)
    except OSError:
        shutil.copy2(src, dst)


def write_meta(dest_dir: Path, meta: dict):
    """Write a small metadata record describing one generated subset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    with open(dest_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


def normalize_sizes(vals):
    """Allow either space-separated or comma-separated subset sizes."""
    out = []
    for v in vals:
        if isinstance(v, str) and ("," in v):
            out.extend(int(x) for x in v.split(",") if x.strip())
        else:
            out.append(int(v))
    return out


def write_subset_manifest(df_sub: pd.DataFrame, idx: np.ndarray, dest: Path) -> None:
    """Write stable subset membership metadata aligned to the canonical train split."""
    required = ["smiles"]
    optional = ["solvent", "peakwavs_max", "empeakwavs_max"]
    missing = [col for col in required if col not in df_sub.columns]
    if missing:
        raise ValueError(f"Subset manifest requires columns: {missing}")

    cols = required + [col for col in optional if col in df_sub.columns]
    manifest = df_sub.loc[:, cols].copy()
    manifest.insert(0, "row_idx_in_train", idx)
    manifest.to_csv(dest, index=False)


def main():
    ap = argparse.ArgumentParser(
        description="Create subsampled (train-only) variants from canonical splits."
    )
    ap.add_argument(
        "--base",
        required=True,
        help="Canonical split dir, e.g. data/uvvisml/deep4chem/splits/scaffold",
    )
    ap.add_argument(
        "--out-root",
        required=True,
        help="Root for variants, e.g. data/uvvisml/deep4chem/subsets/scaffold",
    )
    ap.add_argument(
        "--n",
        nargs="+",
        required=True,
        help="Target train sizes (e.g. 250 1000 4000 or '250,1000,4000')",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Sampling seed",
    )
    ap.add_argument(
        "--copy-valtest",
        action="store_true",
        help="Materialize val/test CSVs by copying them instead of inheriting them from the canonical split.",
    )
    ap.add_argument(
        "--symlink-valtest",
        action="store_true",
        help="Materialize val/test CSVs as symlinks to the canonical split.",
    )
    args = ap.parse_args()
    sizes = normalize_sizes(args.n)

    base = Path(args.base)
    out_root = Path(args.out_root)

    st_train = base / "smiles_target_train.csv"
    st_val = base / "smiles_target_val.csv"
    st_test = base / "smiles_target_test.csv"
    ft_train = base / "features_train.csv"
    ft_val = base / "features_val.csv"
    ft_test = base / "features_test.csv"

    assert st_train.exists() and st_val.exists() and st_test.exists(), "Missing smiles_target_* CSVs"

    have_features = ft_train.exists() and ft_val.exists() and ft_test.exists()

    # Load the canonical training split once and reuse it for every subset size.
    df_train = pd.read_csv(st_train)
    if "smiles" not in df_train.columns:
        raise ValueError("Expected a 'smiles' column in smiles_target_train.csv")

    # Feature CSVs are optional but, if present, should stay aligned to the sampled rows.
    ftr_train = pd.read_csv(ft_train) if have_features else None
    has_smiles_in_features = have_features and ("smiles" in ftr_train.columns)

    for n in sizes:
        if n > len(df_train):
            raise ValueError(f"Requested N={n} exceeds train size {len(df_train)}")

        # Subsample rows from canonical train, keep original indices for alignment
        df_sub = df_train.sample(n=n, random_state=args.seed, replace=False).reset_index(drop=False)
        idx = df_sub["index"].to_numpy()  # positions in canonical train
        df_sub = df_sub.drop(columns=["index"])

        out_dir = out_root / f"N{n:05d}_s{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Write subsampled smiles/targets
        df_sub.to_csv(out_dir / "smiles_target_train.csv", index=False)

        # Filter features_train if present
        if have_features:
            if has_smiles_in_features:
                # Merge on smiles (strict)
                ftr_sub = ftr_train.merge(df_sub[["smiles"]], on="smiles", how="inner")
                if len(ftr_sub) != len(df_sub):
                    missing = set(df_sub["smiles"]) - set(ftr_train["smiles"])
                    raise ValueError(
                        f"{len(missing)} SMILES in subsample missing from features_train.csv "
                        f"(e.g., {list(missing)[:3]})"
                    )
            else:
                # No 'smiles' col: assume row-wise alignment with smiles_target_train.csv and index-select
                # use the original indices captured in `idx` and preserve the same row order as df_sub
                ftr_sub = ftr_train.iloc[idx].reset_index(drop=True)
                if len(ftr_sub) != len(df_sub):
                    raise ValueError("Row-aligned features selection produced a length mismatch.")

            ftr_sub.to_csv(out_dir / "features_train.csv", index=False)

        # Save indices + meta
        np.savetxt(out_dir / "indices_train.txt", idx, fmt="%d")
        write_subset_manifest(df_sub, idx, out_dir / "train_manifest.csv")

        materialize_valtest = args.copy_valtest or args.symlink_valtest
        if materialize_valtest:
            def place(src: Path, dstname: str):
                dst = out_dir / dstname
                if args.copy_valtest:
                    if dst.exists() or dst.is_symlink():
                        dst.unlink()
                    shutil.copy2(src, dst)
                else:
                    symlink_or_copy(src, dst)

            # val/test smiles
            place(st_val, "smiles_target_val.csv")
            place(st_test, "smiles_target_test.csv")

            # val/test features (if present)
            if have_features:
                place(ft_val, "features_val.csv")
                place(ft_test, "features_test.csv")

        meta = dict(
            created=time.strftime("%Y-%m-%d %H:%M:%S"),
            base=str(base),
            out=str(out_dir),
            N=n,
            seed=args.seed,
            alignment="row-index"
            if have_features and not has_smiles_in_features
            else "merge-on-smiles",
            files={
                "train_smiles": "smiles_target_train.csv",
                "train_manifest": "train_manifest.csv",
                "val_smiles": "smiles_target_val.csv" if materialize_valtest else None,
                "test_smiles": "smiles_target_test.csv" if materialize_valtest else None,
                "train_features": "features_train.csv" if have_features else None,
                "val_features": "features_val.csv" if have_features and materialize_valtest else None,
                "test_features": "features_test.csv" if have_features and materialize_valtest else None,
            },
            inherits_from_canonical_split=not materialize_valtest,
        )
        write_meta(out_dir, meta)
        print(f"[ok] wrote {out_dir}")


if __name__ == "__main__":
    main()
