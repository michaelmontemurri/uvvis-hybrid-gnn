#!/bin/bash
# Paper-specific transfer workflow:
# consume the Stage B checkpoints from a gradual-unfreeze pretrained-ensemble
# run and fit the downstream hybrid tabular model on top of each finetuned
# encoder, then blend across the finetuned members.
# Outputs: results/.../transfer/.../hybrid_on_finetuned_pretrain_ensemble/job*/

#SBATCH -J transfer-hybrid-ft-ens
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-12:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
  export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
fi

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/uvvisml"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source "batch_jobs/_shared/subsets.sh"

TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
DATASET="${DATASET:-chemfluor}"
DATASOURCE="${DATASOURCE:-uvvisml}"
SPLIT="${SPLIT:-scaffold}"
MODEL="${MODEL:-xgb}"
DEVICE="${DEVICE:-cuda}"
TUNE_TRIALS="${TUNE_TRIALS:-100}"
SKIP_TUNING="${SKIP_TUNING:-0}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

FT_RUN_ROOT="${FT_RUN_ROOT:-}"
[[ -n "${FT_RUN_ROOT}" ]] || {
  echo "[ERROR] FT_RUN_ROOT is required. Point it at a completed gradual-unfreeze run, e.g. results/${TARGET}/transfer/${DATASET}/${SPLIT}/gradual_unfreeze_pretrain_ensemble/job123456" >&2
  exit 1
}
[[ -d "${FT_RUN_ROOT}" ]] || { echo "[ERROR] FT_RUN_ROOT not found: ${FT_RUN_ROOT}" >&2; exit 1; }

RUN_ROOT="results/${TARGET}/transfer/${DATASET}/${SPLIT}/hybrid_on_finetuned_pretrain_ensemble/job${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"
echo "[CONFIG] finetuned source=${FT_RUN_ROOT}"
echo "[CONFIG] out=${RUN_ROOT}"

for subset in "${SUBSET_LIST[@]}"; do
  DATA_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}"
  [[ -d "${DATA_DIR}" ]] || { echo "[ERROR] Missing DATA_DIR=${DATA_DIR}"; exit 2; }

  declare -a CKPTS=()
  for i in 0 1 2 3 4; do
    CKPT="${FT_RUN_ROOT}/${subset}/member_model_${i}/stageB_unfreeze/models/fold_0/model_0/model.pt"
    [[ -f "${CKPT}" ]] || { echo "[ERROR] Missing finetuned member checkpoint: ${CKPT}" >&2; exit 3; }
    CKPTS+=("${CKPT}")
  done

  CMD=(
    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH"
    python scripts/run_pretrained_hybrid.py
    --checkpoints "${CKPTS[@]}"
    --train-csv "${DATA_DIR}/smiles_target_train.csv"
    --val-csv "${DATA_DIR}/smiles_target_val.csv"
    --test-csv "${DATA_DIR}/smiles_target_test.csv"
    --target-col "${TARGET_COL}"
    --outdir "${RUN_ROOT}/${subset}"
    --model "${MODEL}"
    --device "${DEVICE}"
    --tune-trials "${TUNE_TRIALS}"
  )

  if [[ "${SKIP_TUNING}" == "1" ]]; then
    CMD+=(--skip-tuning)
  fi

  echo "[CMD] ${CMD[*]}"
  "${CMD[@]}"
done

echo "[DONE] Wrote hybrid-on-finetuned ensemble results under: ${RUN_ROOT}"
