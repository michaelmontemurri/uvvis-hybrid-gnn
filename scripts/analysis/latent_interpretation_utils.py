#!/usr/bin/env python3
"""Shared helpers for latent-space interpretation analyses."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.emb_tools import RunSpec, add_pcs, fit_pca_on_train, load_splits
from scripts.physchem import add_physchem_descriptors


def build_spec(
    *,
    dataset: str,
    split: str,
    N: str,
    seed: int,
    target: str,
    target_col: str,
    head: int,
) -> RunSpec:
    return RunSpec(
        dataset=dataset,
        split=split,
        N=N,
        seed=seed,
        target=target,
        target_col=target_col,
        head=head,
        member=0,
    )


def load_head_projected_splits(
    *,
    dataset: str,
    split: str,
    N: str,
    seed: int,
    target: str,
    target_col: str,
    head: int,
    n_pcs: int = 24,
    include_physchem: bool = False,
) -> tuple[dict[str, pd.DataFrame], object, list[str]]:
    """Load train/val/test splits for one head and add local PCA columns."""
    spec = build_spec(
        dataset=dataset,
        split=split,
        N=N,
        seed=seed,
        target=target,
        target_col=target_col,
        head=head,
    )
    splits = load_splits(spec, include_hl_gap=True)
    pca = fit_pca_on_train(splits["train"], n_components=n_pcs)
    splits_pc = add_pcs(splits, pca)
    physchem_cols: list[str] = []
    if include_physchem:
        updated = {}
        for name, df in splits_pc.items():
            updated[name], physchem_cols = add_physchem_descriptors(df.copy(), verbose=False)
        splits_pc = updated
    return splits_pc, pca, physchem_cols


def latent_feature_columns(df: pd.DataFrame, *, n_pcs: int) -> list[str]:
    pc_cols = [f"pc_{i}" for i in range(n_pcs) if f"pc_{i}" in df.columns]
    solvfp_cols = [c for c in df.columns if c.startswith("solvfp_")]
    return pc_cols + solvfp_cols


def fit_head_xgb_member0(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    *,
    target_col: str,
    n_pcs: int,
    model_seed: int = 0,
) -> tuple[XGBRegressor, list[str]]:
    """Fit the notebook-style member_00 PCA-XGB model for one head."""
    feat_cols = latent_feature_columns(train_df, n_pcs=n_pcs)
    Xtr = train_df[feat_cols].astype(np.float32)
    Xva = val_df[feat_cols].astype(np.float32)
    ytr = train_df[target_col].astype(np.float32)
    yva = val_df[target_col].astype(np.float32)

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=model_seed,
        n_jobs=8,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return model, feat_cols
