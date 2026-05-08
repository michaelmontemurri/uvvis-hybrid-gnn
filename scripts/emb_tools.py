from __future__ import annotations
"""Utilities for inspecting learned-embedding runs and derived feature tables.

This module centralizes path helpers, data loading, and a few analysis routines
used when working with frozen-embedding experiments. It is intended as a shared
utility module rather than a user-facing command-line entry point.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Literal

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from uvvis_hybrid.scripts.fit_tabular import (
    normalize_smiles_column,
    ensure_float32_embeddings,
    build_solvent_fp_from_column,
    merge_features_on_smiles,
)

CorrMethod = Literal["pearson", "spearman"]


@dataclass(frozen=True)
class RunSpec:
    """Minimal metadata needed to resolve one embedding-based experiment run."""
    dataset: str = "deep4chem"
    split: str = "scaffold"
    N: str = "N11816"
    seed: int = 42
    datasource: str = "uvvisml"
    target: str = "abs"            # "abs" or "em"
    target_col: str = "peakwavs_max"

    # Embedding and downstream-head layout in the results tree.
    phase: str = "phase1C"
    ensemble_tag: str = "xgb_{N}_s{seed}_single_perhead"
    head: int = 0
    member: int = 0

    repo_root: Optional[str] = None  # if None, use cwd
    embedding_cache_root: str = "local_runs/latent_interpretation_cache"
    solvent_featurizer: str = "morgan_fingerprint"
    device: str = "cpu"


def set_repo_root(path: str) -> None:
    """Change the working directory so relative repo paths resolve correctly."""
    os.chdir(os.path.expanduser(path))


def _fmt_head_member(head: int, member: int) -> Tuple[str, str]:
    return f"head_{head:02d}", f"member_{member:02d}"


def get_base_csv_dir(spec: RunSpec) -> str:
    subset_base = (
        f"data/{spec.datasource}/{spec.target}/{spec.dataset}/subsets/"
        f"{spec.split}/{spec.N}_s{spec.seed}"
    )
    if os.path.isdir(subset_base):
        return subset_base
    return f"data/{spec.datasource}/{spec.target}/{spec.dataset}/splits/{spec.split}"


def get_split_csv_paths(spec: RunSpec) -> Dict[str, str]:
    """Return the canonical train/val/test split CSV paths for a run spec."""
    base = get_base_csv_dir(spec)
    return {
        "train": f"{base}/smiles_target_train.csv",
        "val":   f"{base}/smiles_target_val.csv",
        "test":  f"{base}/smiles_target_test.csv",
    }


def get_emb_dir(spec: RunSpec) -> str:
    """Return the directory containing saved embedding parquet files."""
    head_cache = (
        f"results/{spec.target}/{spec.phase}/{spec.dataset}/{spec.split}/heads/"
        f"head_{spec.head:02d}_{spec.N}_s{spec.seed}/embeddings"
    )
    if os.path.isdir(head_cache):
        return head_cache

    head_dir, member_dir = _fmt_head_member(spec.head, spec.member)
    ensemble = spec.ensemble_tag.format(N=spec.N, seed=spec.seed)
    return (
        f"results/{spec.target}/{spec.phase}/{spec.dataset}/{spec.split}/ensembles/"
        f"{ensemble}/{head_dir}/{member_dir}/embeddings"
    )


def get_cache_emb_dir(spec: RunSpec) -> str:
    return (
        f"{spec.embedding_cache_root}/{spec.target}/{spec.dataset}/{spec.split}/"
        f"{spec.N}_s{spec.seed}/head_{spec.head:02d}/embeddings"
    )


def get_checkpoint_path(spec: RunSpec) -> Optional[str]:
    ckpt = (
        f"checkpoints/{spec.target}/{spec.dataset}/{spec.split}/{spec.solvent_featurizer}/"
        f"fromscratch_{spec.N}_s{spec.seed}/fold_0/model_{spec.head}/model.pt"
    )
    return ckpt if os.path.exists(ckpt) else None



def get_split_emb_paths(spec: RunSpec) -> Dict[str, str]:
    """Return train/val/test embedding parquet paths for a run spec."""
    emb_dir = get_emb_dir(spec)
    if not os.path.isdir(emb_dir):
        emb_dir = get_cache_emb_dir(spec)
    return {
        "train": f"{emb_dir}/train.parquet",
        "val":   f"{emb_dir}/val.parquet",
        "test":  f"{emb_dir}/test.parquet",
    }


def get_xgb_model_path(spec: RunSpec) -> str:
    """Return the saved downstream XGBoost model path for one run member."""
    head_dir, member_dir = _fmt_head_member(spec.head, spec.member)
    ensemble = spec.ensemble_tag.format(N=spec.N, seed=spec.seed)
    return (
        f"results/{spec.target}/{spec.phase}/{spec.dataset}/{spec.split}/ensembles/"
        f"{ensemble}/{head_dir}/{member_dir}/xgb_model_head0.joblib"
    )


def load_master_hl_gap(
    master_csv: str = "data/processed/uvvis_abs_master_full.csv",
    *,
    smiles_col: str = "smiles",
    homo_col: str = "homo",
    lumo_col: str = "lumo",
    out_col: str = "HL_gap",
    filter_failed: bool = True,
) -> pd.DataFrame:
    """
    Loads and returns a 2-col DataFrame: [smiles, HL_gap], de-duplicated on smiles.

    The filter intentionally mirrors the cleanup used elsewhere in the repo.
    """
    m = pd.read_csv(master_csv)[[smiles_col, homo_col, lumo_col]].copy()

    if filter_failed:
        m = m[
            (m[homo_col] < 0) &
            (m[lumo_col] < 0) &
            (m[homo_col] > -20) &
            (m[lumo_col] > -10)
        ].copy()

    m[out_col] = m[lumo_col] - m[homo_col]
    m = m[[smiles_col, out_col]]
    m = normalize_smiles_column(m, smiles_col)
    m = m.drop_duplicates(smiles_col, keep="first").reset_index(drop=True)
    return m


# -----------------------------
# Build split table (csv + embeddings + solvent fp + hlgap)
# -----------------------------
def build_split_table(
    csv_path: str,
    emb_path: str,
    *,
    master_hl_gap: Optional[pd.DataFrame] = None,
    smiles_col: str = "smiles",
    solvent_col: str = "solvent",
) -> pd.DataFrame:
    """
    Reconstructs a split dataframe with:
      - original CSV columns
      - embedding columns (emb_*)
      - solvent Morgan fingerprints (solvfp_*)
      - optional HL_gap merged on smiles
    """
    df = pd.read_csv(csv_path)
    df = normalize_smiles_column(df, smiles_col)

    # unique smiles+solvent table
    U = df[[smiles_col, solvent_col]].drop_duplicates(smiles_col).reset_index(drop=True)

    # solvent FPs on unique table
    solv_fp = build_solvent_fp_from_column(U, smiles_col, solvent_col)
    X_ex = merge_features_on_smiles(U, [solv_fp], smiles_col)

    # embeddings
    E = pd.read_parquet(emb_path)
    E = ensure_float32_embeddings(normalize_smiles_column(E, smiles_col))
    E_u = E.drop_duplicates(smiles_col, keep="first")

    # restore original order
    M = df.copy()
    M["__row_id"] = np.arange(len(M), dtype=np.int64)

    M = M.merge(E_u, on=smiles_col, how="left", validate="m:1")

    if not X_ex.empty:
        M = M.merge(X_ex, on=smiles_col, how="left", validate="m:1")

    if master_hl_gap is not None:
        M = M.merge(master_hl_gap, on=smiles_col, how="left", validate="m:1")

    M.sort_values("__row_id", kind="stable", inplace=True)
    M.drop(columns="__row_id", inplace=True)
    return M


def ensure_split_embeddings(spec: RunSpec) -> Dict[str, str]:
    """
    Ensure train/val/test embedding parquet files exist for a run spec.

    Resolution order:
    1. Existing results/ cache under the historical experiment layout
    2. Existing local cache under spec.embedding_cache_root
    3. Dump embeddings from a tracked compatible checkpoint into the local cache
    """
    emb_paths = get_split_emb_paths(spec)
    if all(os.path.exists(path) for path in emb_paths.values()):
        return emb_paths

    ckpt_path = get_checkpoint_path(spec)
    if ckpt_path is None:
        missing = [p for p in emb_paths.values() if not os.path.exists(p)]
        raise FileNotFoundError(
            "Missing embedding parquet files and no compatible tracked checkpoint was found.\n"
            f"Missing:\n- " + "\n- ".join(missing)
        )

    csv_paths = get_split_csv_paths(spec)
    cache_dir = get_cache_emb_dir(spec)
    os.makedirs(cache_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        out_path = os.path.join(cache_dir, f"{split}.parquet")
        if os.path.exists(out_path):
            continue
        cmd = [
            sys.executable,
            "-m",
            "uvvis_hybrid.scripts.dump_embeddings",
            "--ckpt",
            ckpt_path,
            "--csv",
            csv_paths[split],
            "--smiles-col",
            "smiles",
            "--stage",
            "pre_ffn",
            "--out",
            out_path,
            "--device",
            spec.device,
        ]
        print("[emb_tools] dumping embeddings:", " ".join(cmd))
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(f"Embedding dump failed for head {spec.head}, split {split}")

    return {
        "train": os.path.join(cache_dir, "train.parquet"),
        "val": os.path.join(cache_dir, "val.parquet"),
        "test": os.path.join(cache_dir, "test.parquet"),
    }


def load_splits(
    spec: RunSpec,
    *,
    include_hl_gap: bool = True,
    master_hl_gap: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Returns dict with keys train/val/test -> DataFrames.
    """
    if include_hl_gap and master_hl_gap is None:
        master_hl_gap = load_master_hl_gap()

    csv_paths = get_split_csv_paths(spec)
    emb_paths = ensure_split_embeddings(spec)

    out = {}
    for split in ["train", "val", "test"]:
        out[split] = build_split_table(
            csv_paths[split],
            emb_paths[split],
            master_hl_gap=master_hl_gap if include_hl_gap else None,
        )
    return out


# -----------------------------
# XGB model + importances
# -----------------------------
def load_xgb_model(spec: RunSpec):
    path = get_xgb_model_path(spec)
    return joblib.load(path)


def xgb_feature_importance(
    model,
    feature_names: Sequence[str],
) -> pd.Series:
    """
    Uses model.feature_importances_ (gain-based for XGBRegressor default).
    Returns sorted Series.
    """
    imp = pd.Series(model.feature_importances_, index=list(feature_names))
    return imp.sort_values(ascending=False)


# -----------------------------
# Embeddings + PCA
# -----------------------------
def embedding_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("emb_")]


def fit_pca_on_train(
    Mtr: pd.DataFrame,
    *,
    n_components: int = 32,
) -> PCA:
    X = Mtr[embedding_cols(Mtr)].to_numpy(dtype=np.float32, copy=False)
    pca = PCA(n_components=n_components)
    pca.fit(X)
    return pca


def add_pcs(
    splits: Dict[str, pd.DataFrame],
    pca: PCA,
    *,
    prefix: str = "pc_",
) -> Dict[str, pd.DataFrame]:
    """
    Returns NEW dict of DataFrames with pc_0..pc_{k-1} added to each split.
    """
    out = {}
    emb_cols = embedding_cols(splits["train"])
    k = pca.n_components_

    for name, df in splits.items():
        X = df[emb_cols].to_numpy(dtype=np.float32, copy=False)
        PCs = pca.transform(X)
        d2 = df.copy()
        for j in range(k):
            d2[f"{prefix}{j}"] = PCs[:, j]
        out[name] = d2
    return out


# -----------------------------
# Correlations + plotting
# -----------------------------
def corr_table(
    df: pd.DataFrame,
    rows: Sequence[str],
    cols: Sequence[str],
    *,
    method: CorrMethod = "pearson",
) -> pd.DataFrame:
    """
    Returns correlation matrix slice: rows x cols.
    """
    use = list(dict.fromkeys(list(rows) + list(cols)))  # stable unique
    C = df[use].corr(method=method)
    return C.loc[list(rows), list(cols)]


def top_k_features(series: pd.Series, k: int = 20) -> List[str]:
    return series.head(k).index.tolist()


def plot_corr_heatmap(
    C: pd.DataFrame,
    *,
    title: str = "",
    figsize: Optional[Tuple[float, float]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    if figsize is None:
        figsize = (6, max(4, 0.25 * len(C.index)))

    plt.figure(figsize=figsize)
    im = plt.imshow(C.values, aspect="auto", vmin=vmin, vmax=vmax, cmap="coolwarm")
    plt.colorbar(im, label="Correlation")
    plt.yticks(np.arange(len(C.index)), C.index)
    plt.xticks(np.arange(len(C.columns)), C.columns, rotation=45, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_cumvar(pca: PCA, *, title: str = "PCA cumulative explained variance") -> None:
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(cum) + 1), cum, marker="o", markersize=3)
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative explained variance ratio")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def analyze_run(
    spec: RunSpec,
    *,
    k_top: int = 20,
    use_pcs: bool = False,
    n_pcs: int = 24,
    corr_against: Sequence[str] = ("HL_gap",),
    corr_method: CorrMethod = "pearson",
    include_hl_gap: bool = True,
) -> Dict[str, object]:
    """
    Loads splits + xgb model, picks top-k features, and returns correlation tables.

    If use_pcs=True: fits PCA on train embeddings and correlates top PCs (by XGB importance if present)
    else: correlates top embedding dims (emb_*) by XGB importance.
    """
    splits = load_splits(spec, include_hl_gap=include_hl_gap)
    Mtr, Mva, Mte = splits["train"], splits["val"], splits["test"]

    model = load_xgb_model(spec)

    # Restrict importance ranking to the numeric inputs passed into the downstream model.
    model_feat_cols = [c for c in Mtr.columns if c.startswith(("emb_", "solvfp_"))]
    imp = xgb_feature_importance(model, model_feat_cols)

    target_cols = list(dict.fromkeys([spec.target_col] + list(corr_against)))

    if use_pcs:
        pca = fit_pca_on_train(Mtr, n_components=n_pcs)
        splits_pc = add_pcs(splits, pca)
        Mtr_pc = splits_pc["train"]

        plot_pca_cumvar(pca, title=f"PCA cumvar ({spec.dataset}, {spec.target}, {spec.N})")

        pc_cols = [f"pc_{i}" for i in range(pca.n_components_)]
        # Without a model trained directly on PCs, use the leading PCs by order.
        top_rows = pc_cols[:k_top]
        C = corr_table(Mtr_pc, top_rows, target_cols, method=corr_method)

        return {
            "spec": spec,
            "splits": splits,
            "model": model,
            "importance": imp,
            "pca": pca,
            "splits_pc": splits_pc,
            "corr": C,
        }

    else:
        top_emb = [c for c in top_k_features(imp, k_top) if c.startswith("emb_")]
        if len(top_emb) == 0:
            raise ValueError("No emb_* features found in top-k; check your feature list/model path.")

        C = corr_table(Mtr, top_emb, target_cols, method=corr_method)
        return {
            "spec": spec,
            "splits": splits,
            "model": model,
            "importance": imp,
            "top_emb": top_emb,
            "corr": C,
        }


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    set_repo_root("~/projects/uvvis-benchmark")

    spec = RunSpec(
        dataset="deep4chem",
        split="scaffold",
        N="N11816",
        seed=42,
        target="abs",
        target_col="peakwavs_max",
        head=0,
        member=0,
    )

    out = analyze_run(
        spec,
        k_top=20,
        use_pcs=False,
        corr_against=("HL_gap",),
        corr_method="pearson",
        include_hl_gap=True,
    )

    print(out["importance"].head(20))
    print(out["corr"].head(10))
    plot_corr_heatmap(out["corr"], title="Top embedding dims vs target + HL_gap")
