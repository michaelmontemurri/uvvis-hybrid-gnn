#!/bin/bash
# Zero-shot transfer evaluation for a pretrained Deep4Chem model.
# Runs the end-to-end pretrained checkpoint directly on the Jeffries test set
# without finetuning and records the resulting transfer RMSE.
# Inputs: Jeffries split CSVs and pretrained Deep4Chem checkpoints
# Outputs: results/.../transfer/... zero-shot predictions and metrics
#SBATCH -J pure-deep4chem-on-jeffries
#SBATCH -p gpu_debug_tesla
#SBATCH --qos=gpu_debug_tesla
#SBATCH --gpus=1
#SBATCH -c 2
#SBATCH --mem=12G
#SBATCH -t 1-00:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

# ---------- Micromamba ----------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

echo "JOBID=${SLURM_JOB_ID:-LOCAL} HOST=$(hostname) DATE=$(date)"

# ---------- Threads/CUDA ----------
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

nvidia-smi || true

"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ================================================================
# Config
# ================================================================
SPLIT="${SPLIT:-scaffold}"
PRETRAIN_SPLIT="${PRETRAIN_SPLIT:-scaffold}"
DATASET="${DATASET:-jeffries}"
DATASOURCE="${DATASOURCE:-custom}"
TARGET="${TARGET:-em}"
TARGET_COL="${TARGET_COL:-empeakwavs_max}"
SUBSET="${SUBSET:-N00034_s42}"

if [[ "${TARGET}" == "em" ]]; then
  PRETRAIN_SIZE="${PRETRAIN_SIZE:-11502}"
else
  PRETRAIN_SIZE="${PRETRAIN_SIZE:-11816}"
fi

# Path to Deep4Chem end-to-end checkpoint directory
DEEP4CHEM_E2E_CKPT_DIR=checkpoints/${TARGET}/deep4chem/${PRETRAIN_SPLIT}/morgan_fingerprint/fromscratch_N${PRETRAIN_SIZE}_s42

# Where Jeffries split CSVs live
SUBSET_DIR=data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${SUBSET}

[[ -d "${SUBSET_DIR}" ]] || { echo "[ERROR] Missing SUBSET_DIR=${SUBSET_DIR}" >&2; exit 1; }
[[ -d "${DEEP4CHEM_E2E_CKPT_DIR}" ]] || { echo "[ERROR] Missing DEEP4CHEM_E2E_CKPT_DIR=${DEEP4CHEM_E2E_CKPT_DIR}" >&2; exit 1; }

# Output dirs
OUT_ROOT=results/${TARGET}/transfer/${DATASET}/${SPLIT}/${SUBSET}
E2E_OUTDIR=${OUT_ROOT}/deep4chem_end2end

mkdir -p "$E2E_OUTDIR"

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT} pretrain_split=${PRETRAIN_SPLIT}"
echo "[CONFIG] checkpoint_dir=${DEEP4CHEM_E2E_CKPT_DIR}"

set -x

# ================================================================
# 1) End-to-end Deep4Chem model on Jeffries test set
# ================================================================
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
  chemprop_predict \
    --test_path "${SUBSET_DIR}/smiles_target_test.csv" \
    --features_path "${SUBSET_DIR}/features_test.csv" \
    --preds_path "${E2E_OUTDIR}/preds_test.csv" \
    --checkpoint_dir "${DEEP4CHEM_E2E_CKPT_DIR}" \
    --smiles_columns smiles

# Compute RMSE on the fly
export SUBSET_DIR E2E_OUTDIR TARGET_COL
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import json
import math
import os
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_squared_error

subset_dir = Path(os.environ["SUBSET_DIR"])
e2e_outdir = Path(os.environ["E2E_OUTDIR"])
target_col = os.environ["TARGET_COL"]

test_df = pd.read_csv(subset_dir / "smiles_target_test.csv")
preds_df = pd.read_csv(e2e_outdir / "preds_test.csv")

if target_col not in preds_df.columns:
    raise RuntimeError(f"Expected prediction column {target_col!r} in preds_test.csv; found {list(preds_df.columns)}")
if len(preds_df) != len(test_df):
    raise RuntimeError(f"Row count mismatch: preds={len(preds_df)} test={len(test_df)}")
if "smiles" in preds_df.columns and "smiles" in test_df.columns:
    pred_smiles = preds_df["smiles"].astype(str).to_numpy()
    test_smiles = test_df["smiles"].astype(str).to_numpy()
    if not (pred_smiles == test_smiles).all():
        idx = next(i for i, (lhs, rhs) in enumerate(zip(pred_smiles, test_smiles)) if lhs != rhs)
        raise RuntimeError(
            f"SMILES order mismatch at row {idx}: pred={pred_smiles[idx]!r} test={test_smiles[idx]!r}"
        )

y_true = test_df[target_col].astype(float).to_numpy()
y_pred = preds_df[target_col].astype(float).to_numpy()
rmse = math.sqrt(mean_squared_error(y_true, y_pred))

metrics = {"metric": "rmse", "pred_col": target_col, "test_rmse": rmse}
with open(e2e_outdir / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=2, sort_keys=True)

print(f"[End-to-end] Deep4Chem-only on Jeffries test RMSE: {rmse:.3f} nm")
PY

echo
echo "Artifacts:"
echo "  ${E2E_OUTDIR}/preds_test.csv"
echo "  ${E2E_OUTDIR}/metrics.json"
