#!/bin/bash
# Frozen-encoder transfer workflow for Jeffries subsets.
# Reuses a pretrained Deep4Chem encoder, fits one XGBoost head per encoder
# member on the target subset, and then ensembles predictions across heads.
# Inputs: pretrained checkpoints and Jeffries subset split CSVs
# Outputs: results/.../transfer/... hybrid predictions and metrics
#SBATCH -J frozenenc-hybrid-jeffries
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

# ---------- Micromamba Setup ----------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/uvvisml"

set -e -o pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source "batch_jobs/_shared/subsets.sh"

# ---------- Config ----------
TARGET="${TARGET:-em}"
TARGET_COL="${TARGET_COL:-empeakwavs_max}"
DATASET="${DATASET:-jeffries}"
DATASOURCE="${DATASOURCE:-custom}"
SPLIT="${SPLIT:-scaffold}"
PRETRAIN_SPLIT="${PRETRAIN_SPLIT:-scaffold}"
REGIME="${REGIME:-frozen_xgb_transfer}"
MODEL="${MODEL:-xgb}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

CKPT_DIR=checkpoints/${TARGET}/deep4chem/${PRETRAIN_SPLIT}/morgan_fingerprint/fromscratch_N11502_s42

# Consistent Base Path
BASE_OUT="results/${TARGET}/transfer/${DATASET}/${SPLIT}/${REGIME}"

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] pretrain_split=${PRETRAIN_SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"

set -x

for subset in "${SUBSET_LIST[@]}"; do
  SUBSET_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}"
  
  # ================================================================
  # 1) Setup Model Parameters (Dynamic for Ridge or XGB)
  # ================================================================
  PARAMS_DIR="${BASE_OUT}/params/${MODEL}_${subset}"
  PARAMS_JSON="${PARAMS_DIR}/${MODEL}_best_params_head0.json"
  mkdir -p "$PARAMS_DIR"

  if [ ! -f "$PARAMS_JSON" ]; then
    export PARAMS_JSON MODEL_TYPE="$MODEL"
    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import json, os
out_path = os.environ["PARAMS_JSON"]
m_type = os.environ["MODEL_TYPE"]

if m_type == "xgb":
    params = {
        "n_estimators": 50, "max_depth": 2, "min_child_weight": 5.0,
        "subsample": 0.5, "colsample_bytree": 0.5, "learning_rate": 0.03,
        "reg_lambda": 100.0, "reg_alpha": 10.0
    }
else: # ridge
    params = {"alpha": 10.0}

with open(out_path, "w") as f:
    json.dump(params, f)
PY
  fi

  # ================================================================
  # 2) Fit frozen-encoder head for all 5 pretrained heads
  # ================================================================
  ENSEMBLE_BASE="${BASE_OUT}/ensembles/${MODEL}_${subset}_single_perhead"
  
  for i in {0..4}; do
    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python scripts/frozen_head_comparison/run_frozen_embedding_head_ensemble.py \
        --split-dir "${SUBSET_DIR}" \
        --ckpt      "${CKPT_DIR}/fold_0/model_${i}/model.pt" \
        --outdir    "${ENSEMBLE_BASE}/head_0${i}" \
        --model     "${MODEL}" \
        --members   1 \
        --target-col "${TARGET_COL}" \
        --base-seed 200 \
        --best-params-dir "$PARAMS_DIR" \
        --device    cuda
  done

  # ================================================================
  # 3) Cross-head ensemble (mean)
  # ================================================================
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    python scripts/ensemble_cross_heads.py \
      --inputs  "${ENSEMBLE_BASE}"/head_0*/ensemble_preds_test.csv \
      --split   test \
      --outdir  "${BASE_OUT}/ensembles/${MODEL}_${subset}_crosshead_mean" \
      --blend   mean

  echo "[$(date)] Completed ${MODEL} for subset ${subset}"
done

set +x
