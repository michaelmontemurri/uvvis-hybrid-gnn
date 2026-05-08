#!/usr/bin/env python3
"""Recompute metrics for one transfer-learning run as a sanity check.

This is a one-off diagnostic script: it reads a hardcoded run directory,
reconstructs the matching test split, recomputes RMSE/MAE/correlation from the
saved predictions, and compares those values against the logged metrics file.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

RUN_DIR = Path(
    "results/abs/transfer/chemfluor/scaffold/e2e_finetune/policy_full/runs/run_20260201_113831_job3283_N01000_s42_lrscale0.1"
)

assert RUN_DIR.exists(), f"Run dir not found: {RUN_DIR}"

cfg = {}
with open(RUN_DIR / "config.yaml") as f:
    for line in f:
        if ":" in line:
            k, v = line.split(":", 1)
            cfg[k.strip()] = v.strip()

dataset     = cfg["dataset"]
datasource  = cfg["datasource"]
split       = cfg["split"]
subset      = cfg["subset"]
target_col  = cfg["target_col"]
target_task = cfg["target"]

DATA_DIR = Path(
    f"data/{datasource}/{target_task}/{dataset}/subsets/{split}/{subset}"
)
TEST_CSV = DATA_DIR / "smiles_target_test.csv"
PRED_CSV = RUN_DIR / "preds_test.csv"
METRICS  = RUN_DIR / "metrics.json"

assert TEST_CSV.exists(), f"Missing test CSV: {TEST_CSV}"
assert PRED_CSV.exists(), f"Missing preds CSV: {PRED_CSV}"

test = pd.read_csv(TEST_CSV)
pred = pd.read_csv(PRED_CSV)

print("=== FILE CHECK ===")
print("Test rows:", len(test))
print("Pred rows:", len(pred))
print("Pred columns:", list(pred.columns))
print()

pred_col = None
for c in [f"pred_{target_col}", target_col, "prediction", "pred", "preds"]:
    if c in pred.columns:
        pred_col = c
        break

if pred_col is None:
    num_cols = pred.select_dtypes(include="number").columns.tolist()
    if len(num_cols) != 1:
        raise RuntimeError(f"Ambiguous numeric columns: {num_cols}")
    pred_col = num_cols[0]

print(f"Using prediction column: {pred_col}")
print()

# Chemprop usually preserves row order, but realign by SMILES if needed.
if "smiles" in pred.columns and "smiles" in test.columns:
    if not (pred["smiles"].values == test["smiles"].values).all():
        print("WARNING: SMILES order mismatch — realigning by SMILES")
        merged = test.merge(pred, on="smiles", suffixes=("_true", "_pred"))
        y = merged[target_col].to_numpy(float)
        yhat = merged[pred_col].to_numpy(float)
    else:
        y = test[target_col].to_numpy(float)
        yhat = pred[pred_col].to_numpy(float)
else:
    y = test[target_col].to_numpy(float)
    yhat = pred[pred_col].to_numpy(float)

rmse = np.sqrt(np.mean((yhat - y) ** 2))
mae  = np.mean(np.abs(yhat - y))
corr = np.corrcoef(y, yhat)[0, 1]

print("=== METRICS (RECOMPUTED) ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"Corr: {corr:.4f}")
print()

print("=== TARGET DISTRIBUTION ===")
print(f"y    min / mean / max: {y.min():.2f} / {y.mean():.2f} / {y.max():.2f}")
print(f"yhat min / mean / max: {yhat.min():.2f} / {yhat.mean():.2f} / {yhat.max():.2f}")
print()

print("=== SANITY FLAGS ===")
print("Any NaNs in preds:", np.isnan(yhat).any())
print("Pred std:", np.std(yhat))
print("True std:", np.std(y))
print()

if METRICS.exists():
    with open(METRICS) as f:
        logged = json.load(f)
    print("=== LOGGED metrics.json ===")
    print(json.dumps(logged, indent=2))
    print(f"ΔRMSE (recomputed - logged): {rmse - logged['rmse']:.6f}")
else:
    print("No metrics.json found")
