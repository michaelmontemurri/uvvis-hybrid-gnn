#!/usr/bin/env python3
"""Run simple linear and shallow tabular baselines on saved embedding splits.

The script reconstructs train/val/test feature matrices from split CSVs,
learned embeddings, solvent fingerprints, and optional physicochemical
descriptors, then fits a small collection of baseline models and saves their
outputs in the format expected by the N-vs-RMSE plotting pipeline.
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer
import torch

from uvvis_hybrid.scripts.fit_tabular import (
        normalize_smiles_column,
        ensure_float32_embeddings,
        build_solvent_fp_from_column,
        merge_features_on_smiles,
    )

from scripts.physchem import add_physchem_descriptors

def save_for_plotter(y_true, y_pred, output_dir, split="test"):
    """Save metrics and predictions in the plotting pipeline's expected format."""
    os.makedirs(output_dir, exist_ok=True)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    metrics_data = {
        "ensemble": {
            split: {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2)
            }
        }
    }
    with open(Path(output_dir) / "metrics.json", "w") as f:
        json.dump(metrics_data, f, indent=2)
    
    df_preds = pd.DataFrame({
        "target": y_true,
        "pred_mean": y_pred,
        "pred_std": 0.0 
    })
    df_preds.to_csv(Path(output_dir) / f"preds_{split}_allseeds.csv", index=False)

def load_and_prep_master(master_path):
    """Load the master table and derive a cleaned HOMO-LUMO gap lookup."""
    Master = pd.read_csv(master_path)
    Master = Master[(Master["homo"] < 0) & (Master["lumo"] < 0) & 
                    (Master["homo"] > -20) & (Master["lumo"] > -10)].copy()
    Master["HL_gap"] = Master["lumo"] - Master["homo"]
    Master = normalize_smiles_column(Master[["smiles", "HL_gap"]], "smiles").drop_duplicates("smiles")
    return Master

def build_split(csv_path, emb_path, Master_DF):
    """Reconstruct one split's feature matrix from CSVs, embeddings, and solvent features."""
    df = normalize_smiles_column(pd.read_csv(csv_path), "smiles")
    U = df[["smiles", "solvent"]].drop_duplicates("smiles").reset_index(drop=True)
    solv_fp = build_solvent_fp_from_column(U, "smiles", "solvent")
    X_ex = merge_features_on_smiles(U, [solv_fp], "smiles")
    E = ensure_float32_embeddings(normalize_smiles_column(pd.read_parquet(emb_path), "smiles")).drop_duplicates("smiles")
    M = df.merge(E, on="smiles", how="left").merge(X_ex, on="smiles", how="left").merge(Master_DF, on="smiles", how="left")
    return M

def run_baseline_models(Mtr, Mva, Mte, N_size, target_col, output_root):
    """Fit the baseline suite and save each model's outputs under `output_root`."""
    n_tag = f"fp_baseline_N{str(N_size).zfill(5)}_s42"

    # Build the handcrafted descriptor blocks once, then reuse them across models.
    Mtr_pc, pc_cols = add_physchem_descriptors(pd.concat([Mtr, Mva], axis=0))
    Mte_pc, _ = add_physchem_descriptors(Mte.copy())
    solv_cols = [c for c in Mtr_pc.columns if c.startswith("solvfp_")]
    emb_cols = [c for c in Mtr_pc.columns if c.startswith("emb_")]

    # Keep HL-gap-based baselines robust even if a few values are missing.
    imputer = SimpleImputer(strategy="median")
    Mtr_pc[["HL_gap"]] = imputer.fit_transform(Mtr_pc[["HL_gap"]])
    Mte_pc[["HL_gap"]] = imputer.transform(Mte_pc[["HL_gap"]])

    y_tr = Mtr_pc[target_col]
    y_te = Mte_pc[target_col]

    print("Fitting tabular_fp_physchem_hl...")
    X_tr = Mtr_pc[["HL_gap"] + pc_cols + solv_cols]
    X_te = Mte_pc[["HL_gap"] + pc_cols + solv_cols]
    xgb_full = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05).fit(X_tr, y_tr)
    save_for_plotter(y_te, xgb_full.predict(X_te), Path(output_root) / "baselines" / f"{n_tag}_with_physchem_hl" / "xgb_with_physchem_hl")

    print("Fitting Ridge...")
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    ridge = Ridge(alpha=1.0).fit(X_tr_sc, y_tr)
    save_for_plotter(y_te, ridge.predict(X_te_sc), Path(output_root) / "baselines" / f"{n_tag}_ridge" / "ridge")

    # physics-informed (InvGap)
    print("Fitting InvGap OLS...")
    inv_tr = (1.0 / np.maximum(Mtr_pc[["HL_gap"]].values, 1e-6))
    inv_te = (1.0 / np.maximum(Mte_pc[["HL_gap"]].values, 1e-6))
    phys = LinearRegression().fit(inv_tr, y_tr)
    save_for_plotter(y_te, phys.predict(inv_te), Path(output_root) / "baselines" / f"{n_tag}_phys" / "invgap")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split-type", required=True)
    parser.add_argument("--target", default="abs")
    parser.add_argument("--target-col", default="peakwavs_max")
    parser.add_argument("--n-size", required=True)
    parser.add_argument("--datasource", required=True)
    parser.add_argument("--master-path", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    Master = load_and_prep_master(args.master_path)
    
    n_str = str(args.n_size).zfill(5)
    data_path = f"data/{args.datasource}/{args.target}/{args.dataset}/subsets/{args.split_type}/N{n_str}_s42"
    emb_path = f"results/{args.target}/phase1C/{args.dataset}/{args.split_type}/ensembles/xgb_N{args.n_size}_s42_single_perhead/head_00/embeddings"

    Mtr = build_split(f"{data_path}/smiles_target_train.csv", f"{emb_path}/train.parquet", Master)
    Mva = build_split(f"{data_path}/smiles_target_val.csv", f"{emb_path}/val.parquet", Master)
    Mte = build_split(f"{data_path}/smiles_target_test.csv", f"{emb_path}/test.parquet", Master)

    # The output root is the split-level result directory for the selected experiment.
    run_baseline_models(Mtr, Mva, Mte, args.n_size, args.target_col, args.outdir)
    print("\n[OK] All baseline results saved.")
