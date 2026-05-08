#!/usr/bin/env python3
"""
Train and ensemble downstream regression heads on one frozen embedding source.

This script drives the frozen-head comparison used for the supporting
analysis. It takes a single pretrained Chemprop checkpoint, dumps frozen
embeddings for train/val/test, trains one or more downstream regression
heads on top of those embeddings, and writes the averaged predictions.

The encoder stays fixed throughout. Only the downstream head changes. In the
paper workflow this was used to compare XGBoost, random forest, MLP, and ridge
on the same embedding source before settling on the downstream model used later
in the study.

Embeddings are cached once per comparison root and shared across head families.
If `--outdir` points at a model-specific directory like `.../mlp`, the default
embedding cache is written to the sibling directory `.../embeddings`.

For models with tunable hyperparameters (`mlp`, `rf`, `xgb`), the script first
runs one validation-based tuning pass if the expected best-params file is not
already present in `--best-params-dir`. The seeded ensemble members then reuse
those saved params.

Prediction aggregation uses strict alignment across potential duplicate SMILES
by constructing a stable `(smiles, occ)` key from the split CSVs.

Example:
  python scripts/frozen_head_comparison/run_frozen_embedding_head_ensemble.py \
    --split-dir data/uvvisml/abs/deep4chem/splits/scaffold \
    --ckpt checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42/fold_0/model_4/model.pt \
    --outdir results/frozen_head_comparison/deep4chem/scaffold/morgan_fp/ensembles/mlp_k5 \
    --model mlp \
    --best-params-dir results/abs/phase1A/deep4chem/scaffold/morgan_fp/mlp \
    --members 5 \
    --base-seed 100 \
    --device cuda
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def run(cmd: List[str], cwd: str | None = None) -> None:
    """Run a subprocess and fail loudly if it exits nonzero."""
    print("[CMD]", " ".join(map(str, cmd)))
    res = subprocess.run(cmd, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(map(str, cmd))}")


def find_repo_root(start: Path) -> Path:
    """Walk upward until the project root is found."""
    cur = start.resolve()
    for _ in range(10):
        if (cur / "uvvis_hybrid").is_dir():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the small metric bundle used throughout this comparison."""
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train and ensemble K downstream regression heads on frozen "
            "Chemprop embeddings using reused tuned hyperparameters."
        )
    )
    p.add_argument("--split-dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument(
        "--outdir",
        required=True,
        help="Model-specific output directory for member predictions and ensemble files.",
    )
    p.add_argument(
        "--embeddings-dir",
        default=None,
        help=(
            "Optional shared embedding cache directory. Defaults to a sibling "
            "'embeddings/' directory next to --outdir."
        ),
    )
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument("--target-col", default="peakwavs_max")
    p.add_argument("--model", choices=["xgb", "rf", "mlp", "ridge"], default="xgb")
    p.add_argument("--members", type=int, default=5, help="Number of ensemble members (K).")
    p.add_argument("--base-seed", type=int, default=0, help="Base seed; member k uses base_seed + k.")
    p.add_argument(
        "--best-params-dir",
        required=True,
        help=(
            "Directory containing the tuned downstream-head hyperparameters to reuse. "
            "If the expected params file is missing, the script will tune once and "
            "save it here before running the seeded ensemble."
        ),
    )
    p.add_argument(
        "--tune-trials",
        type=int,
        default=30,
        help="Number of Optuna trials to use when the script needs to tune a head family.",
    )
    p.add_argument(
        "--force-retune",
        action="store_true",
        help="Ignore any saved best-params file and rerun the tuning step before the ensemble.",
    )
    p.add_argument("--device", default=None, help="'cpu' or 'cuda' or index")
    p.add_argument("--force-redump", action="store_true")
    return p.parse_args()


def best_params_path(model: str, params_dir: Path) -> Path | None:
    """Return the expected best-params file for this head family."""
    if model == "xgb":
        return params_dir / "xgb_best_params_head0.json"
    if model == "mlp":
        return params_dir / "mlp_best_params_head0.json"
    if model == "rf":
        return params_dir / "rf_best_params_head0.json"
    return None


def main():
    args = parse_args()
    repo_root = find_repo_root(Path(__file__).parent)
    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)
    params_dir = Path(args.best_params_dir)
    params_dir.mkdir(parents=True, exist_ok=True)

    split_dir = Path(args.split_dir)
    train_csv = split_dir / "smiles_target_train.csv"
    val_csv   = split_dir / "smiles_target_val.csv"
    test_csv  = split_dir / "smiles_target_test.csv"
    for pth in (train_csv, val_csv, test_csv):
        if not pth.exists():
            raise FileNotFoundError(f"Missing split CSV: {pth}")

    def build_key(df: pd.DataFrame, smiles_col: str, target_col: str) -> pd.DataFrame:
        """Assign a stable occurrence index so duplicate SMILES can be aligned safely."""
        df = df[[smiles_col, target_col]].copy()
        df["occ"] = df.groupby(smiles_col).cumcount()
        return df

    key_val = build_key(pd.read_csv(val_csv), args.smiles_col, args.target_col)
    key_test = build_key(pd.read_csv(test_csv), args.smiles_col, args.target_col)

    emb_dir = Path(args.embeddings_dir) if args.embeddings_dir else out_root.parent / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    emb_train = emb_dir / "train.parquet"
    emb_val = emb_dir / "val.parquet"
    emb_test = emb_dir / "test.parquet"

    need_dump = args.force_redump or not (emb_train.exists() and emb_val.exists() and emb_test.exists())
    if need_dump:
        print("[step 1] Dumping embeddings...")
        for split_name, in_csv, out_parquet in [
            ("train", train_csv, emb_train),
            ("val",   val_csv,   emb_val),
            ("test",  test_csv,  emb_test),
        ]:
            cmd = [
                sys.executable, "-m", "uvvis_hybrid.scripts.dump_embeddings",
                "--ckpt", str(args.ckpt),
                "--csv", str(in_csv),
                "--smiles-col", args.smiles_col,
                "--stage", "pre_ffn",
                "--out", str(out_parquet)
            ]
            if args.device is not None:
                cmd += ["--device", args.device]
            run(cmd, cwd=str(repo_root))
        print("[step 1] Embeddings written to:", emb_dir)
    else:
        print("[step 1] Skipping embedding dump (exists).")

    params_path = best_params_path(args.model, params_dir)
    if params_path is not None:
        need_tune = args.force_retune or not params_path.exists()
        if need_tune:
            print(f"[step 2] Tuning {args.model} once to select reusable hyperparameters...")
            tune_cmd = [
                sys.executable, "-m", "uvvis_hybrid.scripts.fit_tabular",
                "--train-csv", str(train_csv),
                "--val-csv", str(val_csv),
                "--test-csv", str(test_csv),
                "--smiles-col", args.smiles_col,
                "--target-col", args.target_col,
                "--emb-train", str(emb_train),
                "--emb-val", str(emb_val),
                "--emb-test", str(emb_test),
                "--include-solvent-fp",
                "--model", args.model,
                "--seed", str(args.base_seed),
                "--outdir", str(params_dir),
                "--tune",
                "--tune-trials", str(args.tune_trials),
            ]
            if args.device is not None:
                tune_cmd += ["--device", args.device]
            run(tune_cmd, cwd=str(repo_root))
            if not params_path.exists():
                raise FileNotFoundError(
                    f"Tuning step finished but did not create the expected params file: {params_path}"
                )
            print(f"[step 2] Saved tuned params to: {params_path}")
        else:
            print(f"[step 2] Reusing tuned params from: {params_path}")
    else:
        print(f"[step 2] {args.model} does not use a tuning step; proceeding with defaults.")

    member_dirs: List[Path] = []
    for k in range(args.members):
        seed = args.base_seed + k
        member_dir = out_root / f"member_{k:02d}"
        member_dir.mkdir(parents=True, exist_ok=True)
        member_dirs.append(member_dir)

        cmd = [
            sys.executable, "-m", "uvvis_hybrid.scripts.fit_tabular",
            "--train-csv", str(train_csv),
            "--val-csv", str(val_csv),
            "--test-csv", str(test_csv),
            "--smiles-col", args.smiles_col,
            "--target-col", args.target_col,
            "--emb-train", str(emb_train),
            "--emb-val", str(emb_val),
            "--emb-test", str(emb_test),
            "--include-solvent-fp",
            "--model", args.model,
            "--seed", str(seed),
            "--outdir", str(member_dir),
            "--best-params-dir", str(args.best_params_dir),
        ]
        if args.device is not None:
            cmd += ["--device", args.device]
        run(cmd, cwd=str(repo_root))
        print(f"[member {k}] finished -> {member_dir}")

    def read_member_preds(path: Path, smiles_col: str) -> pd.DataFrame:
        """Load one member prediction file and attach occurrence indices in file order."""
        df = pd.read_csv(path)
        pred_col = None
        if "pred_mean" in df.columns:
            pred_col = "pred_mean"
        elif "y_pred" in df.columns:
            pred_col = "y_pred"
        else:
            head_cols = [c for c in df.columns if c.startswith("pred_head")]
            if head_cols:
                pred_col = head_cols[0]
            else:
                raise ValueError(f"Could not find prediction column in {path} (expected pred_mean or y_pred).")

        for col in (smiles_col, "y_true", pred_col):
            if col not in df.columns:
                raise ValueError(f"Missing column '{col}' in {path}")

        df["occ"] = df.groupby(smiles_col).cumcount()
        df = df[[smiles_col, "occ", "y_true", pred_col]].copy()
        df.rename(columns={pred_col: "y_pred"}, inplace=True)
        return df

    def aggregate_split(split_name: str, key_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
        """Align member predictions to the canonical key and average them."""
        member_pred_list = []
        for midx, mdir in enumerate(member_dirs):
            p = mdir / f"preds_{split_name}.csv"
            if not p.exists():
                raise FileNotFoundError(f"Missing preds file: {p}")
            df = read_member_preds(p, args.smiles_col)
            if len(df) != len(key_df):
                raise ValueError(f"Row mismatch in {p}: got {len(df)} rows, expected {len(key_df)}")
            df = df.merge(
                key_df[[args.smiles_col, "occ", args.target_col]].rename(columns={args.target_col: "y_true_key"}),
                on=[args.smiles_col, "occ"],
                how="inner",
                validate="one_to_one"
            )
            if not np.allclose(df["y_true"].values, df["y_true_key"].values, rtol=0, atol=1e-6):
                df["y_true"] = df["y_true_key"].values
            df.drop(columns=["y_true_key"], inplace=True)
            df = df.sort_values([args.smiles_col, "occ"]).reset_index(drop=True)
            df = df.rename(columns={"y_pred": f"pred_m{midx}"})
            member_pred_list.append(df)

        merged = member_pred_list[0]
        for j in range(1, len(member_pred_list)):
            merged = merged.merge(
                member_pred_list[j][[args.smiles_col, "occ", f"pred_m{j}"]],
                on=[args.smiles_col, "occ"],
                how="inner",
                validate="one_to_one"
            )
        assert len(merged) == len(key_df), "Merged rows != canonical key rows; alignment issue."

        pred_cols = [c for c in merged.columns if c.startswith("pred_m")]
        P = merged[pred_cols].to_numpy(dtype=float)
        ens_mean = P.mean(axis=1)
        ens_std = P.std(axis=1, ddof=0)
        y_true = merged["y_true"].to_numpy(dtype=float)

        out_df = merged[[args.smiles_col, "occ", "y_true"]].copy()
        for c in pred_cols:
            out_df[c] = merged[c].values
        out_df["pred_mean"] = ens_mean
        out_df["pred_std"] = ens_std

        return out_df, metrics(y_true, ens_mean)

    ens_report = {
        "model": args.model,
        "members": args.members,
        "base_seed": args.base_seed,
        "members_dirs": [str(p) for p in member_dirs],
    }

    for split_name, key_df in [("val", key_val), ("test", key_test)]:
        out_df, stats = aggregate_split(split_name, key_df)
        out_df.to_csv(out_root / f"ensemble_preds_{split_name}.csv", index=False)
        ens_report[f"{split_name}_metrics"] = stats

    with open(out_root / "ensemble_metrics.json", "w") as f:
        json.dump(ens_report, f, indent=2)

    print("[done] Ensemble results written to:", out_root)
    print(json.dumps(ens_report, indent=2))


if __name__ == "__main__":
    main()
