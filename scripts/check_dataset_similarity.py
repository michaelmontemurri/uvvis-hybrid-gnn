#!/usr/bin/env python3
"""Compare a molecular dataset to the Deep4Chem reference set.

This CLI computes a small set of chemical-space similarity summaries,
including unique valid molecule counts, sampled within-set and between-set
Tanimoto similarity, nearest-neighbor Tanimoto in both directions, and
Murcko scaffold overlap.

It is intended as a quick practical check of whether a new dataset looks
relatively similar to the Deep4Chem pretraining set or substantially
shifted away from it.
"""


from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.similarity_utils import canonicalize_smiles, ecdf, nn_tanimoto_scores, smiles_to_morgan_fps


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compare a query dataset against the canonical Deep4Chem reference "
            "set using Morgan-fingerprint similarity summaries."
        )
    )
    p.add_argument("--query-csv", required=True, help="CSV for the external dataset.")
    p.add_argument("--query-smiles-col", default="smiles", help="SMILES column in the query CSV.")
    p.add_argument(
        "--reference-csv",
        default=None,
        help=(
            "Optional reference CSV. If omitted, the script uses the canonical "
            "processed UVVisML master filtered to source=deep4chem."
        ),
    )
    p.add_argument("--reference-smiles-col", default="smiles", help="SMILES column in the reference CSV.")
    p.add_argument(
        "--reference-target",
        choices=["abs", "em"],
        default="abs",
        help="Which canonical Deep4Chem master to use when --reference-csv is omitted.",
    )
    p.add_argument(
        "--reference-source",
        default="deep4chem",
        help="Value in the processed master 'source' column to use as the reference subset.",
    )
    p.add_argument("--radius", type=int, default=2, help="Morgan fingerprint radius.")
    p.add_argument("--n-bits", type=int, default=2048, help="Morgan fingerprint size.")
    p.add_argument(
        "--max-pairs",
        type=int,
        default=20000,
        help="Maximum random pairwise similarities for within/between-set summaries.",
    )
    p.add_argument(
        "--nn-sample-size",
        type=int,
        default=5000,
        help="Maximum number of molecules to score in each nearest-neighbor direction.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    p.add_argument("--outdir", required=True, help="Directory for summary outputs.")
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the optional nearest-neighbor ECDF figure.",
    )
    return p.parse_args()


def default_reference_csv(reference_target: str) -> Path:
    if reference_target == "abs":
        return REPO_ROOT / "data" / "processed" / "uvvis_abs_master_full.csv"
    return REPO_ROOT / "data" / "processed" / "uvvis_em_master_full.csv"


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "p05": float("nan"), "p95": float("nan")}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p05": float(np.quantile(arr, 0.05)),
        "p95": float(np.quantile(arr, 0.95)),
    }


def sample_pairwise_sims(
    fps_a: list,
    fps_b: list | None = None,
    max_pairs: int = 20000,
    seed: int = 42,
) -> np.ndarray:
    rng = random.Random(seed)
    if fps_b is None:
        if len(fps_a) < 2:
            return np.array([], dtype=float)
        possible = len(fps_a) * (len(fps_a) - 1) // 2
        n_pairs = min(max_pairs, possible)
        sims = []
        seen: set[tuple[int, int]] = set()
        while len(sims) < n_pairs:
            i = rng.randrange(len(fps_a))
            j = rng.randrange(len(fps_a))
            if i >= j or (i, j) in seen:
                continue
            seen.add((i, j))
            sims.append(DataStructs.TanimotoSimilarity(fps_a[i], fps_a[j]))
        return np.asarray(sims, dtype=float)

    if not fps_a or not fps_b:
        return np.array([], dtype=float)

    sims = np.empty(max_pairs, dtype=float)
    for idx in range(max_pairs):
        i = rng.randrange(len(fps_a))
        j = rng.randrange(len(fps_b))
        sims[idx] = DataStructs.TanimotoSimilarity(fps_a[i], fps_b[j])
    return sims


def murcko_scaffolds(smiles_list: list[str]) -> set[str]:
    scaffolds: set[str] = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        if scaffold:
            scaffolds.add(scaffold)
    return scaffolds


def load_unique_smiles(csv_path: Path, smiles_col: str, reference_source: str | None = None) -> tuple[list[str], dict[str, int]]:
    df = pd.read_csv(csv_path)
    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in {csv_path}")

    if reference_source is not None:
        if "source" not in df.columns:
            raise ValueError(f"{csv_path} does not contain a 'source' column for filtering.")
        df = df.loc[df["source"].astype(str) == reference_source].copy()

    raw_n = int(len(df))
    smiles = df[smiles_col].dropna().astype(str).tolist()
    canon = [canonicalize_smiles(s) for s in smiles]
    canon = [s for s in canon if s is not None]
    unique = sorted(set(canon))
    return unique, {"rows": raw_n, "valid_smiles": len(canon), "unique_molecules": len(unique)}


def heuristic_guidance(nn_query_to_ref: dict[str, float | int], scaffold_overlap_fraction: float) -> dict[str, str]:
    median_nn = float(nn_query_to_ref["median"])
    if median_nn >= 0.65 and scaffold_overlap_fraction >= 0.50:
        label = "relatively_close"
        text = (
            "The query dataset looks relatively close to Deep4Chem by nearest-neighbor "
            "similarity and scaffold overlap. Starting with the frozen pretrained "
            "encoder is a reasonable first experiment."
        )
    elif median_nn >= 0.45 and scaffold_overlap_fraction >= 0.20:
        label = "moderately_shifted"
        text = (
            "The query dataset shows a moderate chemical shift relative to Deep4Chem. "
            "A frozen encoder is still a reasonable baseline, but finetuning is also "
            "worth testing."
        )
    else:
        label = "substantially_shifted"
        text = (
            "The query dataset appears substantially shifted from Deep4Chem. This does "
            "not rule out frozen reuse, but it makes finetuning a more important path "
            "to test."
        )
    return {
        "label": label,
        "text": text,
        "caution": (
            "This is a practical heuristic based on fingerprint similarity and scaffold "
            "overlap. It is not a guarantee of downstream model performance."
        ),
    }


def maybe_write_plot(
    outdir: Path,
    query_to_ref: np.ndarray,
    ref_to_query: np.ndarray,
) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    x_q, y_q = ecdf(query_to_ref)
    x_r, y_r = ecdf(ref_to_query)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(x_q, y_q, label="Query -> Deep4Chem", linewidth=2.0)
    ax.plot(x_r, y_r, label="Deep4Chem -> Query", linewidth=2.0)
    ax.set_xlabel("Nearest-neighbor Tanimoto")
    ax.set_ylabel("ECDF")
    ax.set_title("Dataset similarity relative to Deep4Chem")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(frameon=False)
    ax.set_xlim(0.0, 1.0)
    fig.tight_layout()
    outpath = outdir / "nearest_neighbor_similarity_cdf.png"
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return outpath.name


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    query_csv = Path(args.query_csv)
    ref_csv = Path(args.reference_csv) if args.reference_csv else default_reference_csv(args.reference_target)

    query_smiles, query_counts = load_unique_smiles(query_csv, args.query_smiles_col)
    ref_filter = None if args.reference_csv else args.reference_source
    ref_smiles, ref_counts = load_unique_smiles(ref_csv, args.reference_smiles_col, reference_source=ref_filter)

    if not query_smiles:
        raise ValueError("No valid query molecules were found after SMILES canonicalization.")
    if not ref_smiles:
        raise ValueError("No valid reference molecules were found after SMILES canonicalization.")

    query_fps, _, _ = smiles_to_morgan_fps(query_smiles, radius=args.radius, n_bits=args.n_bits)
    ref_fps, _, _ = smiles_to_morgan_fps(ref_smiles, radius=args.radius, n_bits=args.n_bits)

    query_within = sample_pairwise_sims(query_fps, max_pairs=args.max_pairs, seed=args.seed)
    ref_within = sample_pairwise_sims(ref_fps, max_pairs=args.max_pairs, seed=args.seed + 1)
    between = sample_pairwise_sims(query_fps, ref_fps, max_pairs=args.max_pairs, seed=args.seed + 2)
    nn_query_to_ref = nn_tanimoto_scores(query_fps, ref_fps, sample_size=args.nn_sample_size, seed=args.seed)
    nn_ref_to_query = nn_tanimoto_scores(ref_fps, query_fps, sample_size=args.nn_sample_size, seed=args.seed + 1)

    query_scaffolds = murcko_scaffolds(query_smiles)
    ref_scaffolds = murcko_scaffolds(ref_smiles)
    overlap = query_scaffolds & ref_scaffolds
    scaffold_summary = {
        "query_unique_scaffolds": len(query_scaffolds),
        "reference_unique_scaffolds": len(ref_scaffolds),
        "shared_scaffolds": len(overlap),
        "query_scaffold_overlap_fraction": float(len(overlap) / len(query_scaffolds)) if query_scaffolds else 0.0,
        "reference_scaffold_overlap_fraction": float(len(overlap) / len(ref_scaffolds)) if ref_scaffolds else 0.0,
    }

    nn_query_summary = summarize(nn_query_to_ref)
    nn_ref_summary = summarize(nn_ref_to_query)
    guidance = heuristic_guidance(nn_query_summary, scaffold_summary["query_scaffold_overlap_fraction"])

    summary = {
        "query_csv": str(query_csv),
        "reference_csv": str(ref_csv),
        "reference_source_filter": ref_filter,
        "fingerprint": {"radius": args.radius, "n_bits": args.n_bits},
        "query_counts": query_counts,
        "reference_counts": ref_counts,
        "within_query_tanimoto": summarize(query_within),
        "within_reference_tanimoto": summarize(ref_within),
        "between_query_reference_tanimoto": summarize(between),
        "nearest_neighbor_query_to_reference": nn_query_summary,
        "nearest_neighbor_reference_to_query": nn_ref_summary,
        "scaffold_overlap": scaffold_summary,
        "heuristic_guidance": guidance,
    }

    plot_name = None if args.no_plot else maybe_write_plot(outdir, nn_query_to_ref, nn_ref_to_query)
    if plot_name is not None:
        summary["nearest_neighbor_plot"] = plot_name

    pd.DataFrame(
        {
            "direction": ["query_to_reference"] * len(nn_query_to_ref) + ["reference_to_query"] * len(nn_ref_to_query),
            "nn_tanimoto": np.concatenate([nn_query_to_ref, nn_ref_to_query]),
        }
    ).to_csv(outdir / "nearest_neighbor_scores.csv", index=False)

    with open(outdir / "similarity_summary.json", "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"[done] wrote {outdir / 'similarity_summary.json'}")


if __name__ == "__main__":
    main()
