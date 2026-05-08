#!/usr/bin/env python
"""Compare two molecular sets using Morgan-fingerprint similarity statistics.

The script reports within-set similarity, random between-set similarity, and
nearest-neighbor similarity using sampled Tanimoto comparisons.
"""

import argparse
import random
from collections import namedtuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

SimStats = namedtuple("SimStats", ["mean", "median", "p05", "p95"])

def canon_smiles(smi: str):
    """Return a canonical RDKit SMILES string, or `None` if parsing fails."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)

def smiles_to_fps(smiles, radius=2, n_bits=2048):
    """Convert a sequence of SMILES strings into Morgan fingerprints."""
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    fps = []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        fps.append((fp, i))
    return fps

def sample_pairs(n, max_pairs):
    """Yield up to max_pairs random distinct index pairs (i, j) with i<j from range(n)."""
    if n < 2:
        return []
    possible = n * (n - 1) // 2
    num = min(max_pairs, possible)
    pairs = set()
    while len(pairs) < num:
        i = random.randrange(n)
        j = random.randrange(n)
        if i >= j:
            continue
        pairs.add((i, j))
    return list(pairs)

def summarize_sims(sims):
    """Summarize a list of similarity scores with a few robust statistics."""
    sims = np.asarray(sims, dtype=float)
    if sims.size == 0:
        return SimStats(np.nan, np.nan, np.nan, np.nan)
    return SimStats(
        float(sims.mean()),
        float(np.median(sims)),
        float(np.quantile(sims, 0.05)),
        float(np.quantile(sims, 0.95)),
    )

def within_set_stats(fps, max_pairs=20000):
    """Estimate the within-set similarity distribution from sampled pairs."""
    n = len(fps)
    pairs = sample_pairs(n, max_pairs)
    sims = []
    for i, j in pairs:
        sims.append(DataStructs.TanimotoSimilarity(fps[i][0], fps[j][0]))
    return summarize_sims(sims)

def between_set_stats(fps_a, fps_b, max_pairs=40000):
    """Estimate the between-set similarity distribution from sampled pairs."""
    if len(fps_a) == 0 or len(fps_b) == 0:
        return summarize_sims([])
    sims = []
    for _ in range(max_pairs):
        i = random.randrange(len(fps_a))
        j = random.randrange(len(fps_b))
        sims.append(DataStructs.TanimotoSimilarity(fps_a[i][0], fps_b[j][0]))
    return summarize_sims(sims)

def nn_stats(fps_ref, fps_query, max_ref=3000, max_query=3000):
    """For each query fp, compute its max Tanimoto to any ref fp."""
    if len(fps_ref) == 0 or len(fps_query) == 0:
        return summarize_sims([])

    # Subsample to keep computation reasonable
    ref_subset = [fp for fp, _ in random.sample(
        fps_ref, min(max_ref, len(fps_ref))
    )]
    query_subset = [fp for fp, _ in random.sample(
        fps_query, min(max_query, len(fps_query))
    )]

    sims = []
    for fp_q in query_subset:
        arr = DataStructs.BulkTanimotoSimilarity(fp_q, ref_subset)
        sims.append(max(arr))
    return summarize_sims(sims)

def print_block(title, stats: SimStats):
    print(f"\n{title}")
    print(f"  mean  : {stats.mean:.4f}")
    print(f"  median: {stats.median:.4f}")
    print(f"  p05   : {stats.p05:.4f}")
    print(f"  p95   : {stats.p95:.4f}")


# -----------------------------
# Main logic
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare chemical similarity between two datasets/splits "
                    "using RDKit Morgan fingerprints and Tanimoto metrics."
    )
    parser.add_argument("--csv_a", required=True, help="First CSV file")
    parser.add_argument("--csv_b", required=True, help="Second CSV file")
    parser.add_argument("--name_a", default="A", help="Label for dataset A")
    parser.add_argument("--name_b", default="B", help="Label for dataset B")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    parser.add_argument("--max_pairs_within", type=int, default=20000,
                        help="Max random within-set pairs per dataset")
    parser.add_argument("--max_pairs_between", type=int, default=40000,
                        help="Max random between-set pairs")
    parser.add_argument("--max_ref_nn", type=int, default=3000,
                        help="Max reference FPS for NN stats")
    parser.add_argument("--max_query_nn", type=int, default=3000,
                        help="Max query FPS for NN stats")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    df_a = pd.read_csv(args.csv_a)
    df_b = pd.read_csv(args.csv_b)

    if args.smiles_col not in df_a.columns or args.smiles_col not in df_b.columns:
        raise ValueError(f"SMILES column '{args.smiles_col}' must exist in both CSVs.")

    # Canonicalize first so equivalent strings are compared consistently.
    smi_a = df_a[args.smiles_col].astype(str).apply(canon_smiles)
    smi_b = df_b[args.smiles_col].astype(str).apply(canon_smiles)
    smi_a = smi_a.dropna().tolist()
    smi_b = smi_b.dropna().tolist()

    print(f"{args.name_a}: {len(smi_a)} valid SMILES")
    print(f"{args.name_b}: {len(smi_b)} valid SMILES")

    fps_a = smiles_to_fps(smi_a)
    fps_b = smiles_to_fps(smi_b)

    print(f"{args.name_a}: {len(fps_a)} fingerprints")
    print(f"{args.name_b}: {len(fps_b)} fingerprints")

    stats_within_a = within_set_stats(fps_a, max_pairs=args.max_pairs_within)
    stats_within_b = within_set_stats(fps_b, max_pairs=args.max_pairs_within)

    print_block(f"Within-set Tanimoto ({args.name_a})", stats_within_a)
    print_block(f"Within-set Tanimoto ({args.name_b})", stats_within_b)

    stats_between = between_set_stats(fps_a, fps_b, max_pairs=args.max_pairs_between)
    print_block(f"Between-set Tanimoto ({args.name_a} vs {args.name_b})", stats_between)

    nn_b_to_a = nn_stats(
        fps_ref=fps_a,
        fps_query=fps_b,
        max_ref=args.max_ref_nn,
        max_query=args.max_query_nn,
    )
    nn_a_to_b = nn_stats(
        fps_ref=fps_b,
        fps_query=fps_a,
        max_ref=args.max_ref_nn,
        max_query=args.max_query_nn,
    )

    print_block(
        f"Nearest-neighbor Tanimoto: {args.name_b} queries vs {args.name_a} ref",
        nn_b_to_a,
    )
    print_block(
        f"Nearest-neighbor Tanimoto: {args.name_a} queries vs {args.name_b} ref",
        nn_a_to_b,
    )

if __name__ == "__main__":
    main()
