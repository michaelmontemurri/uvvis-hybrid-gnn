#!/bin/bash
# Paper-specific transfer workflow:
# finetune each member of the Deep4Chem pretrained ensemble with a gradual
# unfreeze schedule, then save per-member Stage A and Stage B predictions.
# Outputs: results/.../transfer/.../gradual_unfreeze_pretrain_ensemble/job*/

#SBATCH -J transfer-gradunf-ens
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
  export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/uvvisml"

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source "batch_jobs/_shared/subsets.sh"

TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
DATASET="${DATASET:-chemfluor}"
DATASOURCE="${DATASOURCE:-uvvisml}"
SPLIT="${SPLIT:-scaffold}"
PRETRAIN_SPLIT="${PRETRAIN_SPLIT:-scaffold}"
SMILES_COL="${SMILES_COL:-smiles}"
TRAIN_SEED="${TRAIN_SEED:-42}"

if [[ "${TARGET}" == "em" ]]; then
  PRETRAIN_SIZE="${PRETRAIN_SIZE:-11502}"
else
  PRETRAIN_SIZE="${PRETRAIN_SIZE:-11816}"
fi

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

if [[ -z "${DOWNSTREAM_FULL_SIZE:-}" ]]; then
  LAST_SUBSET="${SUBSET_LIST[-1]}"
  DOWNSTREAM_FULL_SIZE="${LAST_SUBSET#N}"
  DOWNSTREAM_FULL_SIZE="${DOWNSTREAM_FULL_SIZE%_s*}"
fi

PRETRAIN_DIR="checkpoints/${TARGET}/deep4chem/${PRETRAIN_SPLIT}/morgan_fingerprint/fromscratch_N${PRETRAIN_SIZE}_s42/fold_0"
BASE_HPARAMS_JSON="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/hyperopt_N${DOWNSTREAM_FULL_SIZE}_s42/best_config.json"
CONFIG_SCALER="scripts/make_finetune_config.py"

PRETRAIN_MODELS=(model_0 model_1 model_2 model_3 model_4)
STAGEA_EPOCHS="${STAGEA_EPOCHS:-30}"
STAGEB_EPOCHS="${STAGEB_EPOCHS:-170}"
STAGEA_LR_SCALE_DEFAULT="${STAGEA_LR_SCALE_DEFAULT:-0.5}"
STAGEB_LR_SCALE_DEFAULT="${STAGEB_LR_SCALE_DEFAULT:-0.05}"
USE_PERN_LR="${USE_PERN_LR:-1}"
FFN_HIDDEN="${FFN_HIDDEN:-1700}"
FFN_LAYERS="${FFN_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.10}"
HIDDEN_SIZE="${HIDDEN_SIZE:-482}"

RUN_ROOT="results/${TARGET}/transfer/${DATASET}/${SPLIT}/gradual_unfreeze_pretrain_ensemble/job${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

for m in "${PRETRAIN_MODELS[@]}"; do
  [[ -f "${PRETRAIN_DIR}/${m}/model.pt" ]] || { echo "[ERROR] Missing pretrained ckpt: ${PRETRAIN_DIR}/${m}/model.pt"; exit 11; }
done
[[ -f "${BASE_HPARAMS_JSON}" ]] || { echo "[ERROR] Missing BASE_HPARAMS_JSON=${BASE_HPARAMS_JSON}"; exit 12; }

rmse_py='import pandas as pd, numpy as np, sys
gt_path, pred_path, target = sys.argv[1], sys.argv[2], sys.argv[3]
gt = pd.read_csv(gt_path)
preds = pd.read_csv(pred_path)
y = gt[target].to_numpy(dtype=float)
if target not in preds.columns:
    raise RuntimeError(f"Expected prediction column {target!r} in {pred_path}; found {list(preds.columns)}")
if len(preds) != len(gt):
    raise RuntimeError(f"Row count mismatch: preds={len(preds)} test={len(gt)}")
if "smiles" in preds.columns and "smiles" in gt.columns:
    pred_smiles = preds["smiles"].astype(str).to_numpy()
    test_smiles = gt["smiles"].astype(str).to_numpy()
    if not (pred_smiles == test_smiles).all():
        idx = int(np.flatnonzero(pred_smiles != test_smiles)[0])
        raise RuntimeError(
            f"SMILES order mismatch at row {idx}: pred={pred_smiles[idx]!r} test={test_smiles[idx]!r}"
        )
yhat = preds[target].to_numpy(dtype=float)
print(float(np.sqrt(np.mean((y - yhat) ** 2))))
'

choose_lrs() {
  local subset="$1"
  local nstr="${subset#N}"
  nstr="${nstr%_s*}"
  local a="${STAGEA_LR_SCALE_DEFAULT}"
  local b="${STAGEB_LR_SCALE_DEFAULT}"
  if [[ "${USE_PERN_LR}" == "1" ]]; then
    case "${nstr}" in
      00050) a="0.5";  b="0.05" ;;
      00100) a="0.5";  b="0.05" ;;
      00250) a="0.2";  b="0.05" ;;
      01000) a="0.05"; b="0.05" ;;
      01376|03072) a="0.1"; b="0.05" ;;
      *)     a="0.1";  b="0.05" ;;
    esac
  fi
  echo "${a} ${b}"
}

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] pretrain_split=${PRETRAIN_SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"
echo "[CONFIG] out=${RUN_ROOT}"

for subset in "${SUBSET_LIST[@]}"; do
  DATA_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}"
  [[ -d "${DATA_DIR}" ]] || { echo "[ERROR] Missing DATA_DIR=${DATA_DIR}"; exit 2; }

  read -r STAGEA_LR_SCALE STAGEB_LR_SCALE < <(choose_lrs "${subset}")
  OUT_SUB="${RUN_ROOT}/${subset}"
  mkdir -p "${OUT_SUB}"

  for i in 0 1 2 3 4; do
    MEMBER_DIR="${OUT_SUB}/member_model_${i}"
    PRETRAIN_CKPT="${PRETRAIN_DIR}/${PRETRAIN_MODELS[$i]}/model.pt"
    mkdir -p "${MEMBER_DIR}"

    STAGEA_DIR="${MEMBER_DIR}/stageA_frozen"
    STAGEA_CFG="${STAGEA_DIR}/config_resolved.json"
    STAGEA_MODELS="${STAGEA_DIR}/models"
    mkdir -p "${STAGEA_DIR}" "${STAGEA_MODELS}"

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python "${CONFIG_SCALER}" \
        --base "${BASE_HPARAMS_JSON}" \
        --out "${STAGEA_CFG}" \
        --lr-scale "${STAGEA_LR_SCALE}" \
        --epochs "${STAGEA_EPOCHS}" \
        --ffn-hidden-size "${FFN_HIDDEN}" \
        --ffn-num-layers "${FFN_LAYERS}" \
        --dropout "${DROPOUT}" \
      2>&1 | tee -a "${STAGEA_DIR}/train.log"

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_train \
        --data_path "${DATA_DIR}/smiles_target_train.csv" \
        --separate_val_path "${DATA_DIR}/smiles_target_val.csv" \
        --separate_test_path "${DATA_DIR}/smiles_target_test.csv" \
        --features_path "${DATA_DIR}/features_train.csv" \
        --separate_val_features_path "${DATA_DIR}/features_val.csv" \
        --separate_test_features_path "${DATA_DIR}/features_test.csv" \
        --dataset_type regression \
        --target_columns "${TARGET_COL}" \
        --smiles_columns "${SMILES_COL}" \
        --metric rmse \
        --config_path "${STAGEA_CFG}" \
        --epochs "${STAGEA_EPOCHS}" \
        --save_dir "${STAGEA_MODELS}" \
        --num_workers 2 \
        --seed "${TRAIN_SEED}" --pytorch_seed "${TRAIN_SEED}" \
        --gpu 0 \
        --hidden_size "${HIDDEN_SIZE}" \
        --hidden_size_solvent 0 \
        --depth_solvent 0 \
        --checkpoint_frzn "${PRETRAIN_CKPT}" \
      2>&1 | tee -a "${STAGEA_DIR}/train.log"

    STAGEA_BEST="${STAGEA_MODELS}/fold_0/model_0/model.pt"
    [[ -f "${STAGEA_BEST}" ]] || { echo "[ERROR] Missing StageA ckpt: ${STAGEA_BEST}"; exit 3; }

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_predict \
        --checkpoint_path "${STAGEA_BEST}" \
        --test_path "${DATA_DIR}/smiles_target_test.csv" \
        --features_path "${DATA_DIR}/features_test.csv" \
        --smiles_columns "${SMILES_COL}" \
        --preds_path "${STAGEA_DIR}/preds_test.csv" \
        --num_workers 2 --gpu 0 \
      2>&1 | tee -a "${STAGEA_DIR}/train.log"

    STAGEB_DIR="${MEMBER_DIR}/stageB_unfreeze"
    STAGEB_CFG="${STAGEB_DIR}/config_resolved.json"
    STAGEB_MODELS="${STAGEB_DIR}/models"
    mkdir -p "${STAGEB_DIR}" "${STAGEB_MODELS}"

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python "${CONFIG_SCALER}" \
        --base "${BASE_HPARAMS_JSON}" \
        --out "${STAGEB_CFG}" \
        --lr-scale "${STAGEB_LR_SCALE}" \
        --epochs "${STAGEB_EPOCHS}" \
        --ffn-hidden-size "${FFN_HIDDEN}" \
        --ffn-num-layers "${FFN_LAYERS}" \
        --dropout "${DROPOUT}" \
      2>&1 | tee -a "${STAGEB_DIR}/train.log"

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_train \
        --data_path "${DATA_DIR}/smiles_target_train.csv" \
        --separate_val_path "${DATA_DIR}/smiles_target_val.csv" \
        --separate_test_path "${DATA_DIR}/smiles_target_test.csv" \
        --features_path "${DATA_DIR}/features_train.csv" \
        --separate_val_features_path "${DATA_DIR}/features_val.csv" \
        --separate_test_features_path "${DATA_DIR}/features_test.csv" \
        --dataset_type regression \
        --target_columns "${TARGET_COL}" \
        --smiles_columns "${SMILES_COL}" \
        --metric rmse \
        --config_path "${STAGEB_CFG}" \
        --epochs "${STAGEB_EPOCHS}" \
        --save_dir "${STAGEB_MODELS}" \
        --num_workers 2 \
        --seed "${TRAIN_SEED}" --pytorch_seed "${TRAIN_SEED}" \
        --gpu 0 \
        --hidden_size "${HIDDEN_SIZE}" \
        --hidden_size_solvent 0 \
        --depth_solvent 0 \
        --checkpoint_path "${STAGEA_BEST}" \
      2>&1 | tee -a "${STAGEB_DIR}/train.log"

    STAGEB_BEST="${STAGEB_MODELS}/fold_0/model_0/model.pt"
    [[ -f "${STAGEB_BEST}" ]] || { echo "[ERROR] Missing StageB ckpt: ${STAGEB_BEST}"; exit 4; }
    echo "${STAGEB_BEST}" > "${MEMBER_DIR}/stageB_best_path.txt"

    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_predict \
        --checkpoint_path "${STAGEB_BEST}" \
        --test_path "${DATA_DIR}/smiles_target_test.csv" \
        --features_path "${DATA_DIR}/features_test.csv" \
        --smiles_columns "${SMILES_COL}" \
        --preds_path "${STAGEB_DIR}/preds_test.csv" \
        --num_workers 2 --gpu 0 \
      2>&1 | tee -a "${STAGEB_DIR}/train.log"

    STAGEA_RMSE=$("$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python -c "$rmse_py" "${DATA_DIR}/smiles_target_test.csv" "${STAGEA_DIR}/preds_test.csv" "${TARGET_COL}")
    STAGEB_RMSE=$("$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python -c "$rmse_py" "${DATA_DIR}/smiles_target_test.csv" "${STAGEB_DIR}/preds_test.csv" "${TARGET_COL}")

    {
      echo "subset = ${subset}"
      echo "member = ${i}"
      echo "policy = gradual_unfreeze"
      echo "stageA_rmse = ${STAGEA_RMSE}"
      echo "stageB_rmse = ${STAGEB_RMSE}"
    } | tee "${MEMBER_DIR}/metrics.txt"
  done
done

echo "[DONE] Wrote gradual-unfreeze ensemble results under: ${RUN_ROOT}"
