#!/bin/bash
# End-to-end Chemprop subset baseline training.
# Runs hyperparameter search on each requested subset, then trains a
# 5-member ensemble from scratch with the selected config.
# Inputs: data/uvvisml/... subset split CSVs
# Outputs: checkpoints/.../hyperopt_* and checkpoints/.../fromscratch_*

#SBATCH -J subN-hopt+ens-abs-chemfluor-scaffold
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-00:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

# --- Slurm-or-local shim ---
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
  export SLURM_JOB_NAME="${SLURM_JOB_NAME:-$(basename "$0")}"
  export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
  export SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
fi


# ---------- Env ----------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

echo "JOBID=${SLURM_JOB_ID}  HOST=$(hostname)  DATE=$(date)"
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs

source "batch_jobs/_shared/subsets.sh"

# Graceful walltime / TERM notice
trap 'echo "[WARN] Received termination signal at $(date). Exiting gracefully." >&2' TERM

# ---------- Threads & CUDA ----------
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "[INFO] CUDA devices:"
nvidia-smi || true

echo "[INFO] Micromamba envs:"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" env list || true

echo "[INFO] Python & CUDA sanity (inside $ENV_NAME):"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", None), "is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# ---------- Config ----------
SPLIT="${SPLIT:-scaffold}"
TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
METRIC="${METRIC:-rmse}"
DATASET="${DATASET:-chemfluor}"
DATASOURCE="${DATASOURCE:-uvvisml}"
FEATURE_VARIANT="${FEATURE_VARIANT:-}"
SAVE_SUFFIX="${SAVE_SUFFIX:-}"

# Pick the fixed encoder width
FIXED_H="${FIXED_H:-482}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a NSTRS <<< "${SUBSETS}"
else
  read -r -a NSTRS <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${NSTRS[*]}"

run_cmd() {
  echo "[CMD] $*"
  "$@"
}

summaries=()   # collect one-line summaries per N

for NSTR in "${NSTRS[@]}"; do

  echo
  echo "====================  START ${NSTR}  ($(date))  ===================="

  DATA_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${NSTR}"
  FEATURE_DIR="${DATA_DIR}"
  CHECKPOINT_ROOT="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint${SAVE_SUFFIX}"
  if [[ -n "${FEATURE_VARIANT}" ]]; then
    FEATURE_DIR="${DATA_DIR}/feature_variants/${FEATURE_VARIANT}"
    if [[ -z "${SAVE_SUFFIX}" ]]; then
      CHECKPOINT_ROOT="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint_${FEATURE_VARIANT}"
    fi
  fi

  HOPT_DIR="${CHECKPOINT_ROOT}/hyperopt_${NSTR}"
  BEST_CFG="${HOPT_DIR}/best_config.json"
  SAVE_DIR="${CHECKPOINT_ROOT}/fromscratch_${NSTR}"

  echo "[DATA] ${DATA_DIR}"
  echo "[FEAT] ${FEATURE_DIR}"
  echo "[HOPT] ${HOPT_DIR}"
  echo "[BEST] ${BEST_CFG}"
  echo "[SAVE] ${SAVE_DIR}"

  for feat_file in features_train.csv features_val.csv features_test.csv; do
    if [[ ! -f "${FEATURE_DIR}/${feat_file}" ]]; then
      echo "[ERROR] Missing feature file: ${FEATURE_DIR}/${feat_file}" >&2
      exit 1
    fi
  done

  # Clean per-N dirs
  rm -rf "${HOPT_DIR}" "${SAVE_DIR}"
  mkdir -p "${HOPT_DIR}"

  start_ts=$(date +%s)

  # 1) Hyperparameter search with hidden_size FIXED and solvent MLP disabled
  echo "[STEP] Hyperopt (100 iters) for ${NSTR} with fixed hidden_size=${FIXED_H} and raw solvent concat (no solvent MLP)"
  if ! run_cmd "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_hyperopt \
      --data_path                     "${DATA_DIR}/smiles_target_train.csv" \
      --separate_val_path             "${DATA_DIR}/smiles_target_val.csv" \
      --separate_test_path            "${DATA_DIR}/smiles_target_test.csv" \
      --features_path                 "${FEATURE_DIR}/features_train.csv" \
      --separate_val_features_path    "${FEATURE_DIR}/features_val.csv" \
      --separate_test_features_path   "${FEATURE_DIR}/features_test.csv" \
      --dataset_type regression \
      --target_columns "${TARGET_COL}" \
      --smiles_columns smiles \
      --metric "${METRIC}" \
      --num_iters 100 \
      --config_save_path "${BEST_CFG}" \
      --gpu 0 \
      --search_parameter_keywords \
          depth dropout ffn_hidden_size ffn_num_layers \
          init_lr max_lr final_lr batch_size warmup_epochs \
          activation aggregation aggregation_norm \
      --hidden_size "${FIXED_H}" \
      --hidden_size_solvent 0 \
      --depth_solvent 0
  then
    echo "[ERROR] Hyperopt failed for ${NSTR}. Skipping to next N." >&2
    summaries+=("${NSTR}: HYPEROPT FAILED")
    continue
  fi

  # 2) Train 5-member ensemble using best config, again forcing hidden_size and no solvent MLP
  if ! run_cmd "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_train \
      --data_path                     "${DATA_DIR}/smiles_target_train.csv" \
      --separate_val_path             "${DATA_DIR}/smiles_target_val.csv" \
      --separate_test_path            "${DATA_DIR}/smiles_target_test.csv" \
      --features_path                 "${FEATURE_DIR}/features_train.csv" \
      --separate_val_features_path    "${FEATURE_DIR}/features_val.csv" \
      --separate_test_features_path   "${FEATURE_DIR}/features_test.csv" \
      --dataset_type regression \
      --target_columns "${TARGET_COL}" \
      --smiles_columns smiles \
      --metric "${METRIC}" \
      --config_path "${BEST_CFG}" \
      --epochs 200 \
      --ensemble_size 5 \
      --save_dir "${SAVE_DIR}" \
      --show_individual_scores \
      --num_workers 2 \
      --seed 0 --pytorch_seed 0 \
      --gpu 0 \
      --hidden_size "${FIXED_H}" \
      --hidden_size_solvent 0 \
      --depth_solvent 0
  then
    echo "[ERROR] Train failed for ${NSTR}." >&2
    summaries+=("${NSTR}: TRAIN FAILED")
    continue
  fi

  # Try to extract the final ensemble test RMSE from test_scores.csv if present
  one_line="OK"
  if [[ -f "${SAVE_DIR}/test_scores.csv" ]]; then
    # Expecting a column rmse 
    ens_rmse=$(awk -F, 'NR==1{for(i=1;i<=NF;i++){h[$i]=i}} NR>1 && $1=="ensemble"{print $h["rmse"]; exit}' "${SAVE_DIR}/test_scores.csv" || true)
    if [[ -n "${ens_rmse:-}" ]]; then
      one_line="ensemble_rmse=${ens_rmse}"
    fi
  fi

  end_ts=$(date +%s)
  dur_min=$(( (end_ts - start_ts) / 60 ))
  summaries+=("${NSTR}: ${one_line} (duration ~${dur_min} min)")

  echo "====================  DONE  ${NSTR}  ($(date))  ===================="
done

echo
echo "====================  NIGHT RUN SUMMARY  ===================="
for s in "${summaries[@]}"; do
  echo " - ${s}"
done
echo "=============================================================="
