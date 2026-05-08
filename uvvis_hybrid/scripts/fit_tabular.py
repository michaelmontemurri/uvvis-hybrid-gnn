#!/usr/bin/env python3
"""
Train tabular models on frozen Chemprop embeddings for UV-Vis prediction.

This script loads train/validation/test split CSVs together with one or more
Parquet files of precomputed Chemprop pre-FFN embeddings, optionally appends
additional molecular or solvent features, and fits a downstream tabular model.

Optional extra features include:
- solvent descriptor lookups (`solv_*`)
- RDKit physicochemical descriptors (`pc_*`)
- molecule Morgan fingerprints (`fp_*`)
- solvent Morgan fingerprints (`solvfp_*`)

Dense descriptor features are fit-scaled on the training split only and then
applied to validation and test splits. Binary fingerprint features are not
scaled.

When multiple embedding heads are provided, the script trains one model per
head and aggregates predictions across heads by mean or median.

Supported models include XGBoost, random forest, MLP, and ridge regression.
The script saves metrics, validation/test predictions, and tuning artifacts
when applicable.

Assumes embedding files are keyed by SMILES and are consistent with the split
CSVs, with duplicate SMILES mapping to identical embedding vectors.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from packaging import version
from sklearn.ensemble import RandomForestRegressor as SkRandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Local modules
from uvvis_hybrid.features import rdkit_feats as rdmod
from uvvis_hybrid.features import solvent as solv_mod
from uvvis_hybrid.features.scaler import ZScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MIN_XGBOOST_VERSION = version.parse("2.0.0")


def _tree_gpu_requested(device: str | None) -> bool:
    if device is None:
        return False
    hint = str(device).lower()
    if hint == "cpu":
        return False
    if hint == "cuda" or hint.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return True
    raise ValueError(f"Unsupported --device value: {device}")


def _mlp_device(device: str | None) -> str:
    if device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"

    hint = str(device).lower()
    if hint == "cpu":
        return "cpu"
    if hint == "cuda" or hint.isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available.")
        return "cuda"
    raise ValueError(f"Unsupported --device value: {device}")


def _require_cuml_random_forest():
    try:
        import cupy as cp
        from cuml.ensemble import RandomForestRegressor as CuMLRandomForestRegressor
    except ImportError as exc:
        raise RuntimeError(
            "GPU random-forest mode requires cupy and cuml. "
            "Use --device cpu or install the Linux CUDA environment."
        ) from exc
    return cp, CuMLRandomForestRegressor

# Utilities
def load_split(csv_path: str, smiles_col: str, target_col: str) -> pd.DataFrame:
    """Load a split CSV with at least [smiles_col, target_col]."""
    df = pd.read_csv(csv_path)
    needed = [smiles_col, target_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    df = df[needed + ([c for c in ["solvent"] if c in df.columns])].copy()
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    return df


def normalize_smiles_column(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """Coerce smiles column to plain string (handles list/tuple/ndarray)."""
    def _to_str(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            x = x[0] if len(x) > 0 else ""
        return str(x)
    out = df.copy()
    out[smiles_col] = out[smiles_col].apply(_to_str)
    return out


def ensure_float32_embeddings(E: pd.DataFrame) -> pd.DataFrame:
    """Downcast emb_* columns to float32, keep others as-is."""
    E = E.copy()
    for c in E.columns:
        if c.startswith("emb_"):
            E[c] = E[c].astype(np.float32, copy=False)
    return E


def check_embedding_duplicates(E: pd.DataFrame, smiles_col: str, tol: float = 5e-5, tag: str = "") -> None:
    """Ensure duplicate SMILES (if any) have identical embeddings."""
    emb_cols = [c for c in E.columns if c.startswith("emb_")]
    if not emb_cols:
        raise ValueError(f"No emb_* columns found in {tag or 'embeddings'}")
    dup = E.duplicated(smiles_col, keep=False)
    if not dup.any():
        print(f"[{tag or 'emb'}] no duplicate SMILES.")
        return
    def spread(g: pd.DataFrame) -> float:
        return float((g[emb_cols].max() - g[emb_cols].min()).abs().max())
    spreads = E.loc[dup].groupby(smiles_col, sort=False).apply(spread)
    offenders = spreads[spreads > tol]
    if len(offenders):
        print(offenders.sort_values(ascending=False).head(10))
        raise ValueError(f"Duplicate SMILES with differing embeddings in {tag or 'emb'} (> {tol}).")
    print(f"[{tag or 'emb'}] duplicates present but identical within tol.")


# Feature builders
def build_solvent_descriptors(df_unique: pd.DataFrame,
                              smiles_col: str,
                              lookup: pd.DataFrame | None) -> pd.DataFrame:
    """Attach solv_* descriptor columns using a preloaded lookup table; requires 'solvent' col."""
    if lookup is None or "solvent" not in df_unique.columns:
        return pd.DataFrame({smiles_col: df_unique[smiles_col].values})
    solv_df = solv_mod.build_solvent_features(
        df_unique, left_key="solvent", lookup=lookup, right_key="solvent"
    )
    keep = [smiles_col] + [c for c in solv_df.columns if c.startswith("solv_")]
    return solv_df[keep].copy()


def build_physchem(df_unique: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """pc_* descriptor columns from RDKit."""
    pc = rdmod.physchem_dataframe(
        df_unique.rename(columns={smiles_col: "smiles"}), smiles_col="smiles", key_col="smiles"
    ).rename(columns={"smiles": smiles_col})
    return pc


def build_mol_fp(df_unique: pd.DataFrame, smiles_col: str,
                 radius: int = 2, nbits: int = 1024) -> pd.DataFrame:
    """fp_* molecule Morgan fingerprint bit columns."""
    fp = rdmod.morgan_fp_dataframe(
        df_unique.rename(columns={smiles_col: "smiles"}),
        smiles_col="smiles",
        key_col="smiles",
        radius=radius,
        nbits=nbits
    ).rename(columns={"smiles": smiles_col})
    return fp


def build_solvent_fp_from_column(df: pd.DataFrame, smiles_key: str, solvent_col: str) -> pd.DataFrame:
    """
    Given a split dataframe with columns [smiles_key, solvent_col], where solvent_col already
    contains solvent SMILES strings, return a dataframe keyed by smiles_key with solvfp_* cols.
    """
    if solvent_col not in df.columns:
        # Nothing to do; return just the key to keep merge code simple
        return pd.DataFrame({smiles_key: df[smiles_key].values})

    # Unique (molecule, solvent_smiles) per molecule
    tmp = (
        df[[smiles_key, solvent_col]]
        .dropna()
        .drop_duplicates(subset=[smiles_key])
        .reset_index(drop=True)
    )

    # Avoid name collision if the molecule key is also called "smiles"
    if smiles_key == "smiles":
        tmp2 = tmp.rename(columns={solvent_col: "solvent_smiles"})
        smiles_col_for_fp = "solvent_smiles"
    else:
        tmp2 = tmp.rename(columns={solvent_col: "smiles"})
        smiles_col_for_fp = "smiles"

    # Compute Morgan FP over solvent SMILES, keep the molecule key for merge-back
    fp = rdmod.morgan_fp_dataframe(
        tmp2,
        smiles_col=smiles_col_for_fp,
        key_col=smiles_key,
        radius=2,
        nbits=1024,
    )

    # Keep only key and fp cols, and prefix fp_* -> solvfp_*
    fp_cols = [c for c in fp.columns if c.startswith("fp_")]
    fp = fp[[smiles_key] + fp_cols].copy()
    fp.rename(columns={c: c.replace("fp_", "solvfp_") for c in fp_cols}, inplace=True)
    return fp


def merge_features_on_smiles(base: pd.DataFrame, parts: List[pd.DataFrame], smiles_col: str) -> pd.DataFrame:
    """Inner-merge all parts onto base by [smiles_col]."""
    out = base[[smiles_col]].drop_duplicates(smiles_col).copy()
    for p in parts:
        if p is None or p.empty:
            continue
        out = out.merge(p, on=smiles_col, how="inner", validate="1:1")
    return out


# Modeling helpers
def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def _resolve_xgb_backend(device: Optional[str]) -> Dict:
    """
    Decide CPU vs GPU kwargs for the artifact's XGBoost 2.x environment.
    device: None | "cpu" | "cuda" | "0" | "1" ...
    """
    xgb_ver = version.parse(getattr(xgb, "__version__", "2.0.0"))
    if xgb_ver < MIN_XGBOOST_VERSION:
        raise RuntimeError(
            f"XGBoost {xgb.__version__} is too old for this artifact. "
            "Use XGBoost >= 2.0.0 from the tracked environments."
        )

    if _tree_gpu_requested(device):
        print("[xgb] using CUDA histogram backend.")
        return {"device": "cuda", "tree_method": "hist"}

    print("[xgb] using CPU histogram backend.")
    return {"device": "cpu", "tree_method": "hist"}


def fit_xgb(
    Xtr, ytr, Xva, yva, seed: int, device: str | None = "cuda"
) -> xgb.XGBRegressor:
    """
    Reasonable default XGB config with early stopping on val.
    Uses GPU if device="cuda" (XGBoost >= 2.0).
    """
    backend = _resolve_xgb_backend(device)

    params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_estimators=5000,
        early_stopping_rounds=200,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.7,
        reg_alpha=0.0,
        reg_lambda=10.0,
        n_jobs=0,  # use all available CPU threads
        **backend,
    )

    print(f"[xgb] default backend={params.get('device','cpu')} "
          f"| tree_method={params.get('tree_method')} | n_estimators={params['n_estimators']}")

    model = xgb.XGBRegressor(**params)
    model.fit(
    Xtr, ytr,
    eval_set=[(Xva, yva)],
    verbose=False,
)

    return model


def fit_xgb_optuna(
    Xtr, ytr, Xva, yva, seed: int, n_trials: int, device: str | None = "cuda"
):
    """
    Lightweight Optuna tuner for XGBoost. Same backend as default().
    Returns (best_model, best_params_dict).
    """
    print(f"[xgb] Detected XGBoost version: {xgb.__version__}")
    backend = _resolve_xgb_backend(device)

    def objective(trial):
        params = dict(
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=seed,
            n_estimators=trial.suggest_int("n_estimators", 100, 10000),
            early_stopping_rounds=200,
            # search space
            max_depth=trial.suggest_categorical("max_depth", [4, 6, 8, 10, 12]),
            learning_rate=trial.suggest_float("learning_rate", 1e-3, 3e-1, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 1e2, log=True),
            gamma=trial.suggest_float("gamma", 1e-8, 10.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            n_jobs=0,
            **backend,
        )

        # Only add hist-only knobs when relevant
        tree_method = str(params.get("tree_method", "")).lower()
        if tree_method in {"hist", "gpu_hist"}:
            params["max_bin"] = trial.suggest_int("max_bin", 128, 1024)
            params["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
            if params["grow_policy"] == "lossguide":
                params["max_leaves"] = trial.suggest_int("max_leaves", 16, 512)

        model = xgb.XGBRegressor(**params)
        # enforce early stopping via sklearn-compatible arg during trials
        model.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            verbose=False,
        )
        pred = model.predict(Xva)
        return float(np.sqrt(mean_squared_error(yva, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"[optuna] best val RMSE: {study.best_value:.4f}")
    best_params = study.best_trial.params

    final_params = dict(
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        early_stopping_rounds=200,
        n_jobs=-1,
        **backend,
        **best_params,
    )

    print(f"[xgb] optuna backend={final_params.get('device','cpu')} "
          f"| tree_method={final_params.get('tree_method')} | n_estimators={final_params['n_estimators']}")

    best_model = xgb.XGBRegressor(**final_params)
    # final fit
    best_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return best_model, best_params


def fit_mlp(Xtr, ytr, Xva, yva, params: dict, seed: int, device: str = None):
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _mlp_device(device)

    # unpack params with defaults
    layers = params.get("layers", 2)
    width  = params.get("width", 512)
    dropout= params.get("dropout", 0.0)
    lr     = params.get("lr", 1e-3)
    wd     = params.get("weight_decay", 1e-4)
    bs     = params.get("batch_size", 1024)
    max_epochs = params.get("epochs", 300)
    patience   = params.get("patience", 20)
    amp        = bool(params.get("amp", True))

    # tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr.reshape(-1,1), dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva.reshape(-1,1), dtype=torch.float32, device=device)

    train_loader = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=int(bs), shuffle=True)
    val_loader   = DataLoader(TensorDataset(Xva_t, yva_t), batch_size=int(bs), shuffle=False)

    # model
    dims = [Xtr.shape[1]] + [width]*int(layers)
    blocks = []
    for i in range(len(dims)-1):
        blocks.append(nn.Linear(dims[i], dims[i+1]))
        blocks.append(nn.ReLU())
        if dropout and dropout > 0:
            blocks.append(nn.Dropout(dropout))
    blocks.append(nn.Linear(dims[-1], 1))
    model = nn.Sequential(*blocks).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()
    scaler = torch.amp.GradScaler(enabled=(amp and device=="cuda"))

    best_rmse = float("inf")
    best_state = None
    waited = 0

    for epoch in range(int(max_epochs)):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(enabled=(amp and device=="cuda")):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # val RMSE
        model.eval()
        se_sum = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                pred = model(xb)
                se_sum += torch.sum((pred - yb)**2).item()
                n += yb.numel()
        v_rmse = math.sqrt(se_sum/max(n,1))
        if v_rmse + 1e-6 < best_rmse:
            best_rmse = v_rmse
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            waited = 0
        else:
            waited += 1
            if waited >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_rmse


def fit_mlp_optuna(Xtr, ytr, Xva, yva, seed: int, n_trials: int, device: str = None):
    """Tune MLP hyperparams with Optuna on (Xtr,ytr)->(Xva,yva). Returns (model, best_params)."""

    def suggest(trial):
        return {
            "layers": trial.suggest_int("layers", 1, 4),
            "width": trial.suggest_categorical("width", [256, 384, 512, 768, 1024]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.4),
            "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024, 2048]),
            "epochs": trial.suggest_int("epochs", 80, 300),
            "patience": trial.suggest_int("patience", 10, 40),
            "amp": True,
        }

    def objective(trial):
        params = suggest(trial)
        # vary intra-run seed
        _seed = seed + trial.number
        model, v_rmse = fit_mlp(Xtr, ytr, Xva, yva, params, _seed, device=device)
        # free GPU RAM between trials
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return float(v_rmse)

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_trial.params

    # Train final model with best params
    best_model, _ = fit_mlp(Xtr, ytr, Xva, yva, best_params, seed, device=device)
    return best_model, best_params


def predict_mlp(model, X):
    device = next(model.parameters()).device
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        y = model(Xt).squeeze(-1).detach().cpu().numpy()
    return y


def fit_rf_with_params(Xtr, ytr, params: dict, seed: int, device: str | None = None):
    """
    Fit a RandomForest using provided params dict.
    Uses cuML only when CUDA is requested explicitly; otherwise uses sklearn.
    Expected params keys: n_estimators, max_depth, min_samples_leaf, max_features, bootstrap
    """
    use_gpu = _tree_gpu_requested(device)

    n_estimators = int(params.get("n_estimators", 500))
    max_depth = None if params.get("max_depth", None) is None else int(params.get("max_depth"))
    min_samples_leaf = int(params.get("min_samples_leaf", 1))
    max_features = params.get("max_features", "sqrt")
    bootstrap = bool(params.get("bootstrap", True))

    if use_gpu:
        cp, CuMLRandomForestRegressor = _require_cuml_random_forest()
        Xtr_c = cp.asarray(Xtr, dtype=cp.float32)
        ytr_c = cp.asarray(ytr, dtype=cp.float32)
        rf = CuMLRandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth is not None else 0,
            n_bins=128,
            bootstrap=bootstrap,
            min_samples_leaf=min_samples_leaf,
            random_state=seed,
        )
        rf.fit(Xtr_c, ytr_c)
        return rf

    rf = SkRandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(Xtr, ytr)
    return rf


def fit_rf_optuna(Xtr, ytr, Xva, yva, seed: int, n_trials: int, device: str | None = None):
    """Optuna tuner for RandomForest (sklearn/cuml). Returns (model, best_params)."""

    def objective(trial):
        params = dict(
            n_estimators=int(trial.suggest_categorical("n_estimators", [100, 200, 500, 800])),
            max_depth=trial.suggest_categorical("max_depth", [8, 16, 32, 1000]),
            min_samples_leaf=int(trial.suggest_int("min_samples_leaf", 1, 8)),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            bootstrap=trial.suggest_categorical("bootstrap", [True, False]),
        )
        model = fit_rf_with_params(Xtr, ytr, params, seed + trial.number, device=device)
        yp = predict_rf(model, Xva)
        rmse = float(np.sqrt(mean_squared_error(yva, yp)))
        return rmse

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=max(3, min(10, n_trials//4))))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_trial.params

    # convert categorical choices to proper types for final fit
    final_params = dict(
        n_estimators=int(best_params.get("n_estimators", 500)),
        max_depth=best_params.get("max_depth", None),
        min_samples_leaf=int(best_params.get("min_samples_leaf", 1)),
        max_features=best_params.get("max_features", "sqrt"),
        bootstrap=bool(best_params.get("bootstrap", True)),
    )
    # rebuild final model with best params
    model = fit_rf_with_params(Xtr, ytr, final_params, seed=seed, device=device)
    return model, final_params


def fit_ridge_with_params(Xtr, ytr, params: dict, seed: int):
    alpha = float(params.get("alpha", 1.0))
    model = Ridge(alpha=alpha, random_state=seed)
    model.fit(Xtr, ytr)
    return model


def predict_rf(model, X):
    if "cuml" in type(model).__module__:
        cp, _ = _require_cuml_random_forest()
        return model.predict(cp.asarray(X, dtype=cp.float32)).get()
    return model.predict(X)


def parse_args():
    default_master_csv = str(PROJECT_ROOT / "data" / "processed" / "uvvis_abs_master_full.csv")
    default_solvent_tables = str(PROJECT_ROOT / "data" / "solvents")

    ap = argparse.ArgumentParser(description="Fit tabular model on frozen pre-FFN embeddings.")

    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--emb-train", required=True, nargs="+", help="One or more train parquet paths (heads).")
    ap.add_argument("--emb-val", required=True, nargs="+", help="One or more val parquet paths (heads).")
    ap.add_argument("--emb-test", required=True, nargs="+", help="One or more test parquet paths (heads).")
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--target-col", required=True)

    # solvent handling
    ap.add_argument(
        "--master-csv",
        default=default_master_csv,
        help="Repo-owned processed master CSV used only to attach 'solvent' if missing from splits.",
    )
    ap.add_argument(
        "--include-solvent-descriptor",
        choices=["chemfluor", "minnesota_descriptors"],
        default=None,
        help="If set, add solv_* descriptor columns via lookup.",
    )
    ap.add_argument("--include-physchem", action="store_true", help="Add pc_* molecule descriptors.")
    ap.add_argument("--include-mol-fp", action="store_true", help="Add fp_* molecule Morgan FP.")
    ap.add_argument(
        "--include-solvent-fp",
        action="store_true",
        help="Append solvent Morgan fingerprints (concatenated with molecule embedding).",
    )
    ap.add_argument(
        "--solvent-tables-path",
        default=default_solvent_tables,
        help="Path to repo-owned solvent descriptor tables.",
    )

    # Modeling
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tune-trials", type=int, default=30)
    ap.add_argument(
        "--ensemble-agg",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate predictions across multiple heads.",
    )
    ap.add_argument("--device", default=None, help="'cpu', 'cuda', or index like '0'")
    ap.add_argument(
        "--model",
        choices=["xgb", "rf", "mlp", "ridge"],
        default="xgb",
        help="Which tabular model to fit for each head.",
    )
    ap.add_argument(
        "--best-params-dir",
        default=None,
        help="Directory containing per-head xgb_best_params_head{k}.json produced by a tuning run; "
             "when provided and model==xgb, these params are used (no tuning).",
    )

    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("fit_tabular.py")
    print("Outdir:", args.outdir)
    print("Heads:", len(args.emb_train), "train paths,", len(args.emb_val), "val paths,", len(args.emb_test), "test paths")

    # load splits
    df_tr = load_split(args.train_csv, args.smiles_col, args.target_col)
    df_va = load_split(args.val_csv, args.smiles_col, args.target_col)
    df_te = load_split(args.test_csv, args.smiles_col, args.target_col)
    print(f"[splits] n_train={len(df_tr)}  n_val={len(df_va)}  n_test={len(df_te)}")

    # If we need solvent info but it's missing, attach from master
    need_solvent = (args.include_solvent_descriptor is not None) or args.include_solvent_fp
    if need_solvent and (
        ("solvent" not in df_tr.columns) or
        ("solvent" not in df_va.columns) or
        ("solvent" not in df_te.columns)
    ):
        print("[solvent] 'solvent' not present in all splits; attaching from master CSV...")
        master = (
            pd.read_csv(args.master_csv, usecols=[args.smiles_col, "solvent"])
            .dropna()
            .drop_duplicates(args.smiles_col)
        )

        def attach(df):
            return df.merge(master, on=args.smiles_col, how="left", validate="m:1")

        df_tr, df_va, df_te = attach(df_tr), attach(df_va), attach(df_te)

    # preload solvent descriptor table if requested
    lookup = None
    if args.include_solvent_descriptor is not None:
        lookup = solv_mod.load_solvent_table(
            solvent_descriptor=args.include_solvent_descriptor,
            solvent_tables_path=args.solvent_tables_path,
            key_col="solvent",
        )
        print(f"[solv desc] loaded lookup ({args.include_solvent_descriptor}), rows={len(lookup)}")

    # Build unique rows per split to construct extras cheaply
    def uniques(df):
        cols = [c for c in [args.smiles_col, "solvent"] if c in df.columns]
        return df[cols].drop_duplicates(args.smiles_col).reset_index(drop=True)

    U_tr, U_va, U_te = uniques(df_tr), uniques(df_va), uniques(df_te)

    # Construct extras on unique tables
    parts_tr, parts_va, parts_te = [], [], []

    # solvent descriptors via lookup (solv_*)
    if lookup is not None:
        parts_tr.append(build_solvent_descriptors(U_tr, args.smiles_col, lookup))
        parts_va.append(build_solvent_descriptors(U_va, args.smiles_col, lookup))
        parts_te.append(build_solvent_descriptors(U_te, args.smiles_col, lookup))

    # molecule physchem (pc_*)
    if args.include_physchem:
        parts_tr.append(build_physchem(U_tr, args.smiles_col))
        parts_va.append(build_physchem(U_va, args.smiles_col))
        parts_te.append(build_physchem(U_te, args.smiles_col))

    # molecule Morgan FP (fp_*)
    if args.include_mol_fp:
        parts_tr.append(build_mol_fp(U_tr, args.smiles_col))
        parts_va.append(build_mol_fp(U_va, args.smiles_col))
        parts_te.append(build_mol_fp(U_te, args.smiles_col))

    # solvent Morgan FP from solvent SMILES column in the splits (solvfp_*)
    if args.include_solvent_fp:
        if "solvent" not in U_tr.columns or "solvent" not in U_va.columns or "solvent" not in U_te.columns:
            raise ValueError("Requested --include-solvent-fp but no 'solvent' column found in split CSVs.")
        parts_tr.append(build_solvent_fp_from_column(U_tr, args.smiles_col, "solvent"))
        parts_va.append(build_solvent_fp_from_column(U_va, args.smiles_col, "solvent"))
        parts_te.append(build_solvent_fp_from_column(U_te, args.smiles_col, "solvent"))

    # Merge extras for each split (keys by smiles)
    Xtr_ex = merge_features_on_smiles(U_tr, parts_tr, args.smiles_col)
    Xva_ex = merge_features_on_smiles(U_va, parts_va, args.smiles_col)
    Xte_ex = merge_features_on_smiles(U_te, parts_te, args.smiles_col)

    # Identify dense columns to scale (solv_* descriptors, pc_* physchem), but dont scale binary FPs
    dense_cols = [c for c in Xtr_ex.columns if c.startswith(("solv_", "pc_")) and not c.startswith("solvfp_")]
    if dense_cols:
        scaler = ZScaler.fit(Xtr_ex, dense_cols)
        scaler.to_json(os.path.join(args.outdir, "dense_scaler.json"))
        Xtr_ex = scaler.apply(scaler.impute_means(Xtr_ex))
        Xva_ex = scaler.apply(scaler.impute_means(Xva_ex))
        Xte_ex = scaler.apply(scaler.impute_means(Xte_ex))
        for X in (Xtr_ex, Xva_ex, Xte_ex):
            for c in dense_cols:
                X[c] = X[c].astype(np.float32, copy=False)
        print(f"[scaler] applied to {len(dense_cols)} dense columns.")
    else:
        print("[scaler] no dense columns to scale (solv_*, pc_*).")

    # loop over heads
    n_heads = len(args.emb_train)
    assert len(args.emb_val) == n_heads and len(args.emb_test) == n_heads, "Mismatch in number of head files per split."

    preds_tr_heads, preds_va_heads, preds_te_heads = [], [], []
    head_metrics = []

    for k in range(n_heads):
        print(f"\n=== Head {k} ===")

        # load embeddings
        Etr = pd.read_parquet(args.emb_train[k])
        Eva = pd.read_parquet(args.emb_val[k])
        Ete = pd.read_parquet(args.emb_test[k])

        # basic cleaning and checks
        Etr = ensure_float32_embeddings(normalize_smiles_column(Etr, args.smiles_col))
        Eva = ensure_float32_embeddings(normalize_smiles_column(Eva, args.smiles_col))
        Ete = ensure_float32_embeddings(normalize_smiles_column(Ete, args.smiles_col))
        check_embedding_duplicates(Etr, args.smiles_col, tag="train emb")
        check_embedding_duplicates(Eva, args.smiles_col, tag="val emb")
        check_embedding_duplicates(Ete, args.smiles_col, tag="test emb")

        # unique per smiles
        Etr_u = Etr.drop_duplicates(args.smiles_col, keep="first")
        Eva_u = Eva.drop_duplicates(args.smiles_col, keep="first")
        Ete_u = Ete.drop_duplicates(args.smiles_col, keep="first")

        # reconstruct full splits in original order
        def reconstruct(base_df, Euniq, Xex):
            base = base_df.copy()
            base["__row_id"] = np.arange(len(base), dtype=np.int64)
            M = base.merge(Euniq, on=args.smiles_col, how="left", validate="m:1")
            if not Xex.empty:
                M = M.merge(Xex, on=args.smiles_col, how="left", validate="m:1")
            M.sort_values("__row_id", kind="stable", inplace=True)
            M.drop(columns="__row_id", inplace=True)
            return M

        Mtr = reconstruct(df_tr, Etr_u, Xtr_ex)
        Mva = reconstruct(df_va, Eva_u, Xva_ex)
        Mte = reconstruct(df_te, Ete_u, Xte_ex)

        # feature columns
        emb_cols = [c for c in Mtr.columns if c.startswith("emb_")]
        dense_cols_2 = [c for c in Mtr.columns if c.startswith(("solv_", "pc_")) and not c.startswith("solvfp_")]
        mol_fp_cols = [c for c in Mtr.columns if c.startswith("fp_")]
        solv_fp_cols = [c for c in Mtr.columns if c.startswith("solvfp_")]
        feat_cols = emb_cols + dense_cols_2 + mol_fp_cols + solv_fp_cols
        if not feat_cols:
            raise ValueError("No features found (embeddings and/or extras).")

        print(
            f"[cols] emb={len(emb_cols)} dense={len(dense_cols_2)} "
            f"mol_fp={len(mol_fp_cols)} solv_fp={len(solv_fp_cols)} total={len(feat_cols)}"
        )

        Xtr = Mtr[feat_cols].to_numpy(dtype=np.float32, copy=False)
        ytr = Mtr[args.target_col].to_numpy(dtype=np.float32, copy=False)
        Xva = Mva[feat_cols].to_numpy(dtype=np.float32, copy=False)
        yva = Mva[args.target_col].to_numpy(dtype=np.float32, copy=False)
        Xte = Mte[feat_cols].to_numpy(dtype=np.float32, copy=False)
        yte = Mte[args.target_col].to_numpy(dtype=np.float32, copy=False)
        print(f"[shapes] Xtr={Xtr.shape} Xva={Xva.shape} Xte={Xte.shape}")

        # fit model
        if args.model == "xgb":
            if args.best_params_dir is not None:
                bp_path = os.path.join(args.best_params_dir, f"xgb_best_params_head{k}.json")
                if not os.path.exists(bp_path):
                    raise FileNotFoundError(f"Requested best-params for head {k} but file not found: {bp_path}")
                print(f"[xgb] Using saved best params for head {k} from: {bp_path}")
                with open(bp_path, "r") as f:
                    loaded_params = json.load(f)
                final_params = dict(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    random_state=args.seed,
                    early_stopping_rounds=200,
                    **_resolve_xgb_backend(args.device),
                    **loaded_params,
                )
                model = xgb.XGBRegressor(**final_params)
                model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
                joblib.dump(model, os.path.join(args.outdir, f"xgb_model_head{k}.joblib"))
            elif args.tune:
                print(f"[xgb] Running Optuna tuning for head {k} (seed={args.seed + k})...")
                model, best_params = fit_xgb_optuna(
                    Xtr, ytr, Xva, yva,
                    seed=args.seed + k,
                    n_trials=args.tune_trials,
                    device=args.device,
                )
                bp_out = os.path.join(args.outdir, f"xgb_best_params_head{k}.json")
                with open(bp_out, "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"[xgb] Saved best params for head {k} to: {bp_out}")
            else:
                print(f"[xgb] Using default XGB config for head {k} (no tuning).")
                model = fit_xgb(
                    Xtr,
                    ytr,
                    Xva,
                    yva,
                    seed=args.seed + k,
                    device=args.device,
                )

        elif args.model == "mlp":
            params_dir = args.best_params_dir if getattr(args, "best_params_dir", None) else args.outdir
            os.makedirs(params_dir, exist_ok=True)
            mlp_params_path = os.path.join(params_dir, f"mlp_best_params_head{k}.json")

            if args.tune and (not os.path.exists(mlp_params_path)):
                model, best_params = fit_mlp_optuna(
                    Xtr, ytr, Xva, yva,
                    seed=args.seed + k,
                    n_trials=args.tune_trials,
                    device=args.device,
                )
                with open(mlp_params_path, "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"[mlp] Tuned best params saved to {mlp_params_path}")
            else:
                if os.path.exists(mlp_params_path):
                    with open(mlp_params_path, "r") as f:
                        best_params = json.load(f)
                    print(f"[mlp] Loading best params from {mlp_params_path}")
                    print(json.dumps(best_params, indent=2))
                    model, _ = fit_mlp(Xtr, ytr, Xva, yva, best_params, seed=args.seed + k, device=args.device)
                else:
                    default_params = dict(
                        layers=2,
                        width=512,
                        dropout=0.10,
                        lr=1e-3,
                        weight_decay=1e-4,
                        batch_size=1024,
                        epochs=200,
                        patience=20,
                        amp=True,
                    )
                    print("[mlp] No tuned params found; using default params:")
                    print(json.dumps(default_params, indent=2))
                    model, _ = fit_mlp(Xtr, ytr, Xva, yva, default_params, seed=args.seed + k, device=args.device)

            used_params_src = mlp_params_path if os.path.exists(mlp_params_path) else None
            if used_params_src is not None:
                used_params_dst = os.path.join(args.outdir, "best_params_used.json")
                if os.path.abspath(used_params_src) != os.path.abspath(used_params_dst):
                    shutil.copy(used_params_src, used_params_dst)

        elif args.model == "rf":
            rf_params_path = None
            if getattr(args, "best_params_dir", None) is not None:
                rf_params_path = os.path.join(args.best_params_dir, f"rf_best_params_head{k}.json")
            else:
                rf_params_path = os.path.join(args.outdir, f"rf_best_params_head{k}.json")

            if rf_params_path is not None and os.path.exists(rf_params_path):
                with open(rf_params_path, "r") as f:
                    loaded_params = json.load(f)
                print(f"[rf] Using saved RF params from: {rf_params_path}")
                model = fit_rf_with_params(Xtr, ytr, loaded_params, seed=args.seed + k, device=args.device)
            elif args.tune:
                print(f"[rf] Running Optuna tuning for RF (seed={args.seed + k})...")
                model, best_params = fit_rf_optuna(
                    Xtr, ytr, Xva, yva,
                    seed=args.seed + k,
                    n_trials=args.tune_trials,
                    device=args.device,
                )
                bp_out = os.path.join(args.outdir, f"rf_best_params_head{k}.json")
                with open(bp_out, "w") as f:
                    json.dump(best_params, f, indent=2)
                print(f"[rf] Saved best params to: {bp_out}")
            else:
                default_rf = dict(
                    n_estimators=500,
                    max_depth=32,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    bootstrap=True,
                )
                print("[rf] Using default RF params:")
                print(json.dumps(default_rf, indent=2))
                model = fit_rf_with_params(Xtr, ytr, default_rf, seed=args.seed + k, device=args.device)

        elif args.model == "ridge":
            print(f"[ridge] Using default Ridge regression for head {k} (no tuning).")
            model = fit_ridge_with_params(Xtr, ytr, params={"alpha": 1.0}, seed=args.seed + k)
            pred_tr = model.predict(Xtr)
            pred_va = model.predict(Xva)
            pred_te = model.predict(Xte)

        # predictions
        if args.model == "xgb":
            pred_tr = model.predict(Xtr)
            pred_va = model.predict(Xva)
            pred_te = model.predict(Xte)
        elif args.model == "mlp":
            pred_tr = predict_mlp(model, Xtr)
            pred_va = predict_mlp(model, Xva)
            pred_te = predict_mlp(model, Xte)
        elif args.model == "rf":
            pred_tr = predict_rf(model, Xtr)
            pred_va = predict_rf(model, Xva)
            pred_te = predict_rf(model, Xte)
        elif args.model == "ridge":
            pred_tr = model.predict(Xtr)
            pred_va = model.predict(Xva)
            pred_te = model.predict(Xte)
        head_metrics.append({
            "train": metrics(ytr, pred_tr),
            "val": metrics(yva, pred_va),
            "test": metrics(yte, pred_te),
        })
        preds_tr_heads.append(pred_tr.astype(np.float32, copy=False))
        preds_va_heads.append(pred_va.astype(np.float32, copy=False))
        preds_te_heads.append(pred_te.astype(np.float32, copy=False))

    # Aggregate heads
    P_tr = np.vstack(preds_tr_heads)
    P_va = np.vstack(preds_va_heads)
    P_te = np.vstack(preds_te_heads)

    if args.ensemble_agg == "median":
        ens_tr = np.median(P_tr, axis=0)
        ens_va = np.median(P_va, axis=0)
        ens_te = np.median(P_te, axis=0)
    else:
        ens_tr = P_tr.mean(axis=0)
        ens_va = P_va.mean(axis=0)
        ens_te = P_te.mean(axis=0)

    ytr = df_tr[args.target_col].to_numpy(dtype=np.float32, copy=False)
    yva = df_va[args.target_col].to_numpy(dtype=np.float32, copy=False)
    yte = df_te[args.target_col].to_numpy(dtype=np.float32, copy=False)

    report = {
        "ensemble": {
            "train": metrics(ytr, ens_tr),
            "val": metrics(yva, ens_va),
            "test": metrics(yte, ens_te),
        },
        "per_head": head_metrics,
        "n_heads": len(preds_tr_heads),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "n_test": int(len(yte)),
        "seed": args.seed,
        "config": {
            "include_solvent_descriptor": args.include_solvent_descriptor,
            "include_physchem": bool(args.include_physchem),
            "include_mol_fp": bool(args.include_mol_fp),
            "include_solvent_fp": bool(args.include_solvent_fp),
            "agg": args.ensemble_agg,
        },
    }

    # save artifacts
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)

    out_te = pd.DataFrame({
        args.smiles_col: df_te[args.smiles_col].values,
        "y_true": yte,
    })
    for k in range(P_te.shape[0]):
        out_te[f"pred_head{k}"] = P_te[k]
    out_te["pred_mean"] = P_te.mean(axis=0)
    out_te["pred_std"] = P_te.std(axis=0, ddof=0)
    out_te.to_csv(os.path.join(args.outdir, "preds_test.csv"), index=False)

    out_va = pd.DataFrame({
        args.smiles_col: df_va[args.smiles_col].values,
        "y_true": yva,
    })
    for k in range(P_va.shape[0]):
        out_va[f"pred_head{k}"] = P_va[k]
    out_va["pred_mean"] = P_va.mean(axis=0)
    out_va["pred_std"] = P_va.std(axis=0, ddof=0)
    out_va.to_csv(os.path.join(args.outdir, "preds_val.csv"), index=False)

    print("=== Done ===")
    print(json.dumps(report["ensemble"], indent=2))


if __name__ == "__main__":
    main()
