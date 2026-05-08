#!/usr/bin/env python3
"""Build compact latent-space interpretation summaries across embedding heads.

This script reuses the learned-embedding utilities to:
- load train/val/test tables for multiple heads
- fit per-head PCA bases on the train split
- summarize cumulative explained variance
- measure cross-head stability of PC0 against HL-gap and the prediction target
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.emb_tools import RunSpec, add_pcs, fit_pca_on_train, load_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact PCA summaries for learned-embedding interpretation."
    )
    parser.add_argument("--target", default="abs")
    parser.add_argument("--target-col", default="peakwavs_max")
    parser.add_argument("--dataset", default="deep4chem")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--N", default="N11816")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heads", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--n-components", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--embedding-cache-root", default="local_runs/latent_interpretation_cache")
    parser.add_argument("--outdir", default="figures/latent_interpretation")
    parser.add_argument("--variance-csv", default="pca_explained_variance.csv")
    parser.add_argument("--stability-csv", default="pc0_stability_summary.csv")
    parser.add_argument("--stability-summary-json", default="pc0_stability_aggregate.json")
    parser.add_argument("--cumvar-fig", default="pca_cumulative_variance.pdf")
    parser.add_argument("--reference-pca", default="reference_pca_basis.joblib")
    return parser.parse_args()


def spearman_corr(x: pd.Series, y: pd.Series) -> float:
    """Compute a Spearman correlation between two aligned series."""
    return float(pd.Series(x).corr(pd.Series(y), method="spearman"))


def build_head_spec(args: argparse.Namespace, head: int) -> RunSpec:
    """Build one RunSpec for a given head/member."""
    return RunSpec(
        dataset=args.dataset,
        split=args.split,
        N=args.N,
        seed=args.seed,
        target=args.target,
        target_col=args.target_col,
        head=head,
        member=0,
        embedding_cache_root=args.embedding_cache_root,
        device=args.device,
    )


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cumvar_rows = []
    stability_rows = []
    pcas: dict[int, object] = {}
    splits_by_head: dict[int, dict[str, pd.DataFrame]] = {}

    for head in args.heads:
        spec = build_head_spec(args, head)
        splits = load_splits(spec, include_hl_gap=True)
        splits_by_head[head] = splits

        pca = fit_pca_on_train(splits["train"], n_components=args.n_components)
        pcas[head] = pca

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)
        for idx, (evr, cum) in enumerate(zip(explained, cumulative), start=1):
            cumvar_rows.append(
                {
                    "head": head,
                    "component": idx,
                    "explained_variance_ratio": float(evr),
                    "cumulative_explained_variance_ratio": float(cum),
                }
            )

        projected = add_pcs(splits, pca)
        train_df = projected["train"]
        stability_rows.append(
            {
                "head": head,
                "abs_rho_pc0_target": abs(spearman_corr(train_df["pc_0"], train_df[args.target_col])),
                "abs_rho_pc0_HLgap": abs(spearman_corr(train_df["pc_0"], train_df["HL_gap"])),
            }
        )

    reference_head = args.heads[0]
    ref_payload = {
        "pca": pcas[reference_head],
        "ref_head": reference_head,
        "dataset": args.dataset,
        "split": args.split,
        "N": args.N,
        "target": args.target,
        "target_col": args.target_col,
    }
    joblib.dump(ref_payload, outdir / args.reference_pca)

    cumvar_df = pd.DataFrame(cumvar_rows)
    stability_df = pd.DataFrame(stability_rows).sort_values("head").reset_index(drop=True)
    cumvar_df.to_csv(outdir / args.variance_csv, index=False)
    stability_df.to_csv(outdir / args.stability_csv, index=False)

    stability_stats = (
        stability_df[["abs_rho_pc0_target", "abs_rho_pc0_HLgap"]]
        .agg(["mean", "std"])
    )
    stability_payload = {
        "per_head": stability_df.to_dict(orient="records"),
        "aggregate": {
            "abs_rho_pc0_target_mean": float(stability_stats.loc["mean", "abs_rho_pc0_target"]),
            "abs_rho_pc0_target_std": float(stability_stats.loc["std", "abs_rho_pc0_target"]),
            "abs_rho_pc0_HLgap_mean": float(stability_stats.loc["mean", "abs_rho_pc0_HLgap"]),
            "abs_rho_pc0_HLgap_std": float(stability_stats.loc["std", "abs_rho_pc0_HLgap"]),
        },
    }
    (outdir / args.stability_summary_json).write_text(json.dumps(stability_payload, indent=2))

    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    for head in args.heads:
        sub = cumvar_df[cumvar_df["head"] == head]
        ax.plot(
            sub["component"],
            sub["cumulative_explained_variance_ratio"],
            marker="o",
            markersize=2.5,
            linewidth=1.3,
            alpha=0.85,
            label=f"Head {head:02d}",
        )
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA Cumulative Explained Variance Across Heads", pad=12, fontweight="bold")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / args.cumvar_fig, bbox_inches="tight")
    plt.close(fig)

    print(f"[write] {outdir / args.variance_csv}")
    print(f"[write] {outdir / args.stability_csv}")
    print(f"[write] {outdir / args.stability_summary_json}")
    print(f"[write] {outdir / args.cumvar_fig}")
    print(f"[write] {outdir / args.reference_pca}")


if __name__ == "__main__":
    main()
