#!/usr/bin/env python3
"""Build the global PCA basis used for counterfactual latent-space plots."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a PCA basis on reference training embeddings and save it for reuse."
    )
    parser.add_argument(
        "--train-emb",
        default="local_runs/counterfactuals_cache/train.parquet",
        help="Reference training embedding parquet used to fit the PCA basis.",
    )
    parser.add_argument(
        "--train-csv",
        default="data/uvvisml/abs/deep4chem/splits/scaffold/smiles_target_train.csv",
        help="Canonical train split CSV used if --train-emb must be created from a checkpoint.",
    )
    parser.add_argument(
        "--ckpt",
        default="checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42/fold_0/model_0/model.pt",
        help="Checkpoint used to dump train embeddings when --train-emb does not already exist.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="'cpu', 'cuda', or CUDA device index passed through to dump_embeddings.",
    )
    parser.add_argument("--n-components", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", default="figures/latent_interpretation")
    parser.add_argument("--basis-name", default="global_pca_basis.joblib")
    parser.add_argument("--curve-name", default="global_pca_basis_cumvar.pdf")
    return parser.parse_args()


def embedding_cols(df: pd.DataFrame) -> list[str]:
    """Return all embedding columns in a parquet table."""
    return [col for col in df.columns if col.startswith("emb_")]


def ensure_train_embeddings(args: argparse.Namespace) -> Path:
    train_emb_path = Path(args.train_emb)
    if train_emb_path.exists():
        return train_emb_path

    ckpt_path = Path(args.ckpt)
    train_csv_path = Path(args.train_csv)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint for embedding dump: {ckpt_path}")
    if not train_csv_path.exists():
        raise FileNotFoundError(f"Missing canonical train CSV for embedding dump: {train_csv_path}")

    train_emb_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "uvvis_hybrid.scripts.dump_embeddings",
        "--ckpt",
        str(ckpt_path),
        "--csv",
        str(train_csv_path),
        "--smiles-col",
        "smiles",
        "--stage",
        "pre_ffn",
        "--out",
        str(train_emb_path),
        "--device",
        str(args.device),
    ]
    print("[build_global_pca_basis]", " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError("Failed to dump train embeddings for global PCA basis.")
    return train_emb_path


def main() -> None:
    args = parse_args()
    train_emb_path = ensure_train_embeddings(args)

    df = pd.read_parquet(train_emb_path)
    cols = embedding_cols(df)
    if not cols:
        raise ValueError(f"No emb_* columns found in {train_emb_path}")

    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    n_components = min(args.n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    pca.fit(X)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    basis_path = outdir / args.basis_name
    curve_path = outdir / args.curve_name

    joblib.dump(
        {
            "pca": pca,
            "emb_cols": cols,
            "train_emb_path": str(train_emb_path),
            "n_components": n_components,
        },
        basis_path,
    )

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", markersize=3)
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance ratio")
    ax.set_title("Global PCA Basis on Deep4Chem Train Embeddings")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    fig.tight_layout()
    fig.savefig(curve_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[write] {basis_path}")
    print(f"[write] {curve_path}")


if __name__ == "__main__":
    main()
