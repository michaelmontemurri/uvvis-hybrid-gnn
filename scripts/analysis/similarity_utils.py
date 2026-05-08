"""Shared helpers for dataset-level chemical-space and similarity analysis."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from uvvis_hybrid.scripts.fit_tabular import normalize_smiles_column


def canonicalize_smiles(smiles: str) -> str | None:
    """Return a canonical RDKit SMILES string, or `None` if parsing fails."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_split_smiles(split_dir: Path, combine_train_val: bool = True) -> tuple[list[str], list[str]]:
    """Load train/reference and test SMILES lists from one split directory."""
    split_dir = Path(split_dir)
    train_df = normalize_smiles_column(pd.read_csv(split_dir / "smiles_target_train.csv"), "smiles")
    val_df = normalize_smiles_column(pd.read_csv(split_dir / "smiles_target_val.csv"), "smiles")
    test_df = normalize_smiles_column(pd.read_csv(split_dir / "smiles_target_test.csv"), "smiles")

    ref_df = pd.concat([train_df, val_df], ignore_index=True) if combine_train_val else train_df
    return ref_df["smiles"].astype(str).tolist(), test_df["smiles"].astype(str).tolist()


def load_full_split_smiles(split_dir: Path) -> list[str]:
    """Load all SMILES from train/val/test and return them as a single list."""
    ref_smiles, test_smiles = load_split_smiles(split_dir, combine_train_val=True)
    return ref_smiles + test_smiles


def smiles_to_morgan_fps(
    smiles_list: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    use_chirality: bool = True,
    use_bond_types: bool = True,
) -> tuple[list, np.ndarray, np.ndarray]:
    """Convert SMILES strings into RDKit Morgan fingerprints and a binary matrix."""
    gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=n_bits,
        includeChirality=use_chirality,
        useBondTypes=use_bond_types,
        includeRingMembership=True,
    )

    fps = []
    X = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    mask = np.zeros(len(smiles_list), dtype=bool)

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        fp = gen.GetFingerprint(mol)
        fps.append(fp)
        mask[i] = True

        arr = np.zeros((n_bits,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        X[i] = arr

    return fps, X[mask], mask


def nn_tanimoto_scores(
    query_fps: Sequence,
    ref_fps: Sequence,
    sample_size: int | None = None,
    seed: int = 0,
    batch_size: int = 512,
) -> np.ndarray:
    """Compute test-to-train nearest-neighbor Tanimoto scores."""
    if not query_fps:
        raise ValueError("query_fps is empty")
    if not ref_fps:
        raise ValueError("ref_fps is empty")

    rng = np.random.default_rng(seed)
    query_idx = np.arange(len(query_fps))
    if sample_size is not None and sample_size < len(query_idx):
        query_idx = rng.choice(query_idx, size=int(sample_size), replace=False)

    scores = np.empty(len(query_idx), dtype=np.float32)
    for start in range(0, len(query_idx), batch_size):
        batch = query_idx[start : start + batch_size]
        for offset, qi in enumerate(batch):
            sims = DataStructs.BulkTanimotoSimilarity(query_fps[int(qi)], ref_fps)
            scores[start + offset] = float(max(sims)) if sims else 0.0
    return scores


def ecdf(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y arrays for an empirical CDF."""
    x = np.sort(np.asarray(list(values), dtype=float))
    if x.size == 0:
        return x, x
    y = np.arange(1, x.size + 1) / x.size
    return x, y
