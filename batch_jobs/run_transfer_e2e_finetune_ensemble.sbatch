#!/bin/bash
# End-to-end transfer-learning ensemble for subset experiments.
# Starts from a pretrained Chemprop checkpoint, scales the learning-rate
# schedule, finetunes multiple members on the target subset, and evaluates
# the resulting ensemble on the held-out test split.
# Inputs: subset split CSVs, pretrained checkpoint, and base hyperparameter JSON
# Outputs: results/.../transfer/... run directories, predictions, and metrics
#SBATCH -J abs-tl-e2e-dsscdb-scaffold
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

# -----------------------
# Local fallback (debug)
# -----------------------
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
  export SLURM_JOB_NAME="${SLURM_JOB_NAME:-$(basename "$0")}"
  export SLURM_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
  export SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST:-$(hostname)}"
fi

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source "batch_jobs/_shared/subsets.sh"

trap 'echo "[WARN] Received TERM at $(date). Exiting." >&2' TERM

# -----------------------
# Env (micromamba)
# -----------------------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

# -----------------------
# Threads & CUDA behavior
# -----------------------
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "JOBID=$SLURM_JOB_ID  HOST=$(hostname)  DATE=$(date)"
echo "[INFO] CUDA:"
nvidia-smi || true

echo "[INFO] Python sanity:"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", None), "avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu0:", torch.cuda.get_device_name(0))
PY

# -----------------------
# Experiment factors
# -----------------------
TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
METRIC="${METRIC:-rmse}"

DATASET="${DATASET:-dsscdb}"
DATASOURCE="${DATASOURCE:-uvvisml}"
SPLIT="${SPLIT:-scaffold}"
PRETRAIN_SPLIT="${PRETRAIN_SPLIT:-scaffold}"
REGIME="${REGIME:-e2e_finetune}"

# Subset(s)
if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
elif [[ -n "${SUBSET:-}" ]]; then
  SUBSET_LIST=("${SUBSET}")
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

# Ensemble members
N_MEMBERS="${N_MEMBERS:-5}"
EPOCHS="${EPOCHS:-200}"
FIXED_H="${FIXED_H:-482}"

# LR scaling for finetuning (applies to resolved chemprop LR schedule)
LR_SCALE="${LR_SCALE:-0.001}"

# Overwrite control (0 = refuse overwrite; 1 = overwrite run dirs if they exist)
OVERWRITE="${OVERWRITE:-0}"

# Policy: full | ffn_only | last_layer
FT_POLICY="${FT_POLICY:-full}"

# FFN overrides (used for frozen-encoder policies)
FFN_HIDDEN_SIZE="${FFN_HIDDEN_SIZE:-512}"
FFN_NUM_LAYERS="${FFN_NUM_LAYERS:-2}"
DROPOUT="${DROPOUT:-0.10}"

# If FT_POLICY=last_layer, freeze all but last FFN layer.
# By default: FRZN_FFN_LAYERS = FFN_NUM_LAYERS - 1
FRZN_FFN_LAYERS="${FRZN_FFN_LAYERS:-AUTO}"

if [[ "${TARGET}" == "em" ]]; then
  DEFAULT_FULL_SIZE="11502"
else
  DEFAULT_FULL_SIZE="11816"
fi
FULL_SIZE="${FULL_SIZE:-${DEFAULT_FULL_SIZE}}"

if [[ -z "${DOWNSTREAM_FULL_SIZE:-}" ]]; then
  LAST_SUBSET="${SUBSET_LIST[$((${#SUBSET_LIST[@]} - 1))]}"
  DOWNSTREAM_FULL_SIZE="${LAST_SUBSET#N}"
  DOWNSTREAM_FULL_SIZE="${DOWNSTREAM_FULL_SIZE%_s*}"
fi

PRETRAIN_CKPT="checkpoints/${TARGET}/deep4chem/${PRETRAIN_SPLIT}/morgan_fingerprint/fromscratch_N${FULL_SIZE}_s42/fold_0/model_0/model.pt"
if [[ ! -f "${PRETRAIN_CKPT}" ]]; then
  echo "[ERROR] PRETRAIN_CKPT not found: ${PRETRAIN_CKPT}" >&2
  exit 1
fi

# -----------------------
# Base hyperparams from best scratch run (this dataset)
# -----------------------
BASE_HPARAMS_JSON="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/hyperopt_N${DOWNSTREAM_FULL_SIZE}_s42/best_config.json"
if [[ ! -f "${BASE_HPARAMS_JSON}" ]]; then
  echo "[ERROR] BASE_HPARAMS_JSON not found: ${BASE_HPARAMS_JSON}" >&2
  exit 1
fi

# -----------------------
# Config scaling script
# -----------------------
CONFIG_SCALER="scripts/make_finetune_config.py"
if [[ ! -f "${CONFIG_SCALER}" ]]; then
  echo "[ERROR] Config scaling script not found: ${CONFIG_SCALER}" >&2
  exit 1
fi

# -----------------------
# Resolve policy -> chemprop args
# -----------------------
FREEZE_ARGS=()
CKPT_LOAD_ARGS=()

if [[ "${FT_POLICY}" == "full" ]]; then
  # Full finetune: load checkpoint normally; everything trainable.
  CKPT_LOAD_ARGS+=( --checkpoint_path "${PRETRAIN_CKPT}" )
elif [[ "${FT_POLICY}" == "ffn_only" ]]; then
  # Frozen encoder, train entire FFN.
  FREEZE_ARGS+=( --checkpoint_frzn "${PRETRAIN_CKPT}" --frzn_ffn_layers 0 )
elif [[ "${FT_POLICY}" == "last_layer" ]]; then
  if [[ "${FFN_NUM_LAYERS}" -lt 2 ]]; then
    echo "[ERROR] last_layer policy requires FFN_NUM_LAYERS>=2 (got ${FFN_NUM_LAYERS})." >&2
    exit 1
  fi
  if [[ "${FRZN_FFN_LAYERS}" == "AUTO" ]]; then
    FRZN_FFN_LAYERS=$((FFN_NUM_LAYERS - 1))
  fi
  FREEZE_ARGS+=( --checkpoint_frzn "${PRETRAIN_CKPT}" --frzn_ffn_layers "${FRZN_FFN_LAYERS}" )
else
  echo "[ERROR] Unknown FT_POLICY=${FT_POLICY}. Use full|ffn_only|last_layer." >&2
  exit 1
fi

echo "[INFO] Policy=${FT_POLICY}"
echo "[INFO] FFN_NUM_LAYERS=${FFN_NUM_LAYERS}  FRZN_FFN_LAYERS=${FRZN_FFN_LAYERS}"
echo "[INFO] LR_SCALE=${LR_SCALE}"

# FFN overrides:
# - For frozen-encoder policies, changing the FFN can be okay, but it means FFN isn't pretrained.
# - For "full", if you change FFN shape you will get mismatch warnings and effectively retrain FFN.
CFG_ARGS=( --ffn-hidden-size "$FFN_HIDDEN_SIZE" --ffn-num-layers "$FFN_NUM_LAYERS" --dropout "$DROPOUT" )

# -----------------------
# Results root
# -----------------------
RUNS_ROOT="results/${TARGET}/transfer/${DATASET}/${SPLIT}/${REGIME}/policy_${FT_POLICY}/runs"
mkdir -p "${RUNS_ROOT}"

run_cmd() { echo "[CMD] $*"; "$@"; }

summaries=()

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] pretrain_split=${PRETRAIN_SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"

for NSTR in "${SUBSET_LIST[@]}"; do
  echo
  echo "==================== START ${DATASET} ${SPLIT} ${FT_POLICY} ${NSTR} ($(date)) ===================="

  DATA_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${NSTR}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] DATA_DIR not found: ${DATA_DIR}" >&2
    exit 1
  fi

  RUN_ID="$(date +%Y%m%d_%H%M%S)_job${SLURM_JOB_ID}_${NSTR}_policy${FT_POLICY}_lrscale${LR_SCALE}"
  RUN_DIR="${RUNS_ROOT}/run_${RUN_ID}"

  if [[ -e "${RUN_DIR}" && "${OVERWRITE}" != "1" ]]; then
    echo "[ERROR] RUN_DIR exists and OVERWRITE!=1: ${RUN_DIR}" >&2
    exit 1
  fi
  rm -rf "${RUN_DIR}"
  mkdir -p "${RUN_DIR}"

  git rev-parse HEAD > "${RUN_DIR}/git_commit.txt" 2>/dev/null || true
  env | sort > "${RUN_DIR}/env.txt"
  printf "chemprop_train + chemprop_predict\n" > "${RUN_DIR}/cmd.txt"

  cat > "${RUN_DIR}/config.yaml" <<EOF
dataset: ${DATASET}
datasource: ${DATASOURCE}
split: ${SPLIT}
pretrain_split: ${PRETRAIN_SPLIT}
regime: ${REGIME}
target: ${TARGET}
target_col: ${TARGET_COL}
subset: ${NSTR}

n_members: ${N_MEMBERS}
epochs: ${EPOCHS}
hidden_size: ${FIXED_H}

base_hparams_json: ${BASE_HPARAMS_JSON}
lr_scale: ${LR_SCALE}

ft_policy: ${FT_POLICY}
ffn_hidden_size: ${FFN_HIDDEN_SIZE}
ffn_num_layers: ${FFN_NUM_LAYERS}
dropout: ${DROPOUT}
frzn_ffn_layers: ${FRZN_FFN_LAYERS}

pretrain_ckpt: ${PRETRAIN_CKPT}
EOF

  RESOLVED_CONFIG="${RUN_DIR}/config_resolved.json"
  run_cmd "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    python "${CONFIG_SCALER}" \
      --base "$BASE_HPARAMS_JSON" \
      --out  "$RESOLVED_CONFIG" \
      --lr-scale "$LR_SCALE" \
      --epochs "$EPOCHS" \
      "${CFG_ARGS[@]}" \
    2>&1 | tee -a "${RUN_DIR}/train.log"

  if [[ ! -f "${RESOLVED_CONFIG}" ]]; then
    echo "[ERROR] Resolved config not created: ${RESOLVED_CONFIG}" >&2
    exit 1
  fi

  echo "[DATA] ${DATA_DIR}"
  echo "[SAVE] ${RUN_DIR}"
  echo "[INFO] Using resolved config: ${RESOLVED_CONFIG}"

  start_ts=$(date +%s)

  MODELS_DIR="${RUN_DIR}/models/fold_0"
  mkdir -p "${MODELS_DIR}"

  ok=1
  for i in $(seq 0 $((N_MEMBERS-1))); do
    MEMBER_DIR="${RUN_DIR}/tmp_member_${i}"
    rm -rf "${MEMBER_DIR}"

    echo "[INFO] Training member ${i}/${N_MEMBERS}..."
    if ! run_cmd "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      chemprop_train \
        --data_path                     "${DATA_DIR}/smiles_target_train.csv" \
        --separate_val_path             "${DATA_DIR}/smiles_target_val.csv" \
        --separate_test_path            "${DATA_DIR}/smiles_target_test.csv" \
        --features_path                 "${DATA_DIR}/features_train.csv" \
        --separate_val_features_path    "${DATA_DIR}/features_val.csv" \
        --separate_test_features_path   "${DATA_DIR}/features_test.csv" \
        --dataset_type regression \
        --target_columns "${TARGET_COL}" \
        --smiles_columns smiles \
        --metric "${METRIC}" \
        --config_path "${RESOLVED_CONFIG}" \
        --epochs "${EPOCHS}" \
        --save_dir "${MEMBER_DIR}" \
        --show_individual_scores \
        --num_workers 2 \
        --seed "$i" --pytorch_seed "$i" \
        --gpu 0 \
        --hidden_size "${FIXED_H}" \
        --hidden_size_solvent 0 \
        --depth_solvent 0 \
        "${CKPT_LOAD_ARGS[@]}" \
        "${FREEZE_ARGS[@]}" \
        2>&1 | tee -a "${RUN_DIR}/train.log"
    then
      echo "[ERROR] Member ${i} failed for ${NSTR}." >&2
      ok=0; break
    fi

    if [[ ! -d "${MEMBER_DIR}/fold_0/model_0" ]]; then
      echo "[ERROR] Expected ${MEMBER_DIR}/fold_0/model_0 not found." >&2
      ok=0; break
    fi

    mv "${MEMBER_DIR}/fold_0/model_0" "${MODELS_DIR}/model_${i}"
    rm -rf "${MEMBER_DIR}"
  done

  if [[ $ok -ne 1 ]]; then
    summaries+=("${NSTR}: TRAIN FAILED")
    echo "==================== DONE ${NSTR} ($(date)) ===================="
    continue
  fi

  PRED_PATH_RAW="${RUN_DIR}/preds_test_raw.csv"
  if ! run_cmd "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    chemprop_predict \
      --checkpoint_dir "${MODELS_DIR}" \
      --test_path      "${DATA_DIR}/smiles_target_test.csv" \
      --features_path  "${DATA_DIR}/features_test.csv" \
      --smiles_columns smiles \
      --preds_path     "${PRED_PATH_RAW}" \
      --num_workers 2 --gpu 0 \
      2>&1 | tee -a "${RUN_DIR}/train.log"
  then
    summaries+=("${NSTR}: PREDICT FAILED")
    echo "==================== DONE ${NSTR} ($(date)) ===================="
    continue
  fi

  cp "${PRED_PATH_RAW}" "${RUN_DIR}/preds_test.csv"

  export RUN_DIR DATA_DIR TARGET_COL
  rmse="$("$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import os, pandas as pd, numpy as np, json, datetime
run_dir = os.environ["RUN_DIR"]
data_dir = os.environ["DATA_DIR"]
target = os.environ["TARGET_COL"]

preds = pd.read_csv(os.path.join(run_dir, "preds_test.csv"))
test  = pd.read_csv(os.path.join(data_dir, "smiles_target_test.csv"))

if target not in preds.columns:
    raise RuntimeError(f"Expected prediction column {target!r} in preds_test.csv; found {list(preds.columns)}")
if len(preds) != len(test):
    raise RuntimeError(f"Row count mismatch: preds={len(preds)} test={len(test)}")
if "smiles" in preds.columns and "smiles" in test.columns:
    pred_smiles = preds["smiles"].astype(str).to_numpy()
    test_smiles = test["smiles"].astype(str).to_numpy()
    if not (pred_smiles == test_smiles).all():
        idx = int(np.flatnonzero(pred_smiles != test_smiles)[0])
        raise RuntimeError(
            f"SMILES order mismatch at row {idx}: pred={pred_smiles[idx]!r} test={test_smiles[idx]!r}"
        )

yhat = preds[target].to_numpy(dtype=float)
y = test[target].to_numpy(dtype=float)
rmse = float(np.sqrt(np.mean((yhat - y)**2)))

out = {
  "metric": "rmse",
  "rmse": rmse,
  "n_test": int(len(y)),
  "pred_col": target,
  "timestamp": datetime.datetime.now().isoformat()
}
with open(os.path.join(run_dir, "metrics.json"), "w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(f"{rmse:.6f}")
PY
)"

  end_ts=$(date +%s)
  dur_min=$(( (end_ts - start_ts) / 60 ))
  summaries+=("${NSTR}: rmse=${rmse} (~${dur_min} min)")

  echo "==================== DONE ${NSTR} ($(date)) ===================="
done

echo
echo "==================== RUN SUMMARY ===================="
for s in "${summaries[@]}"; do
  echo " - ${s}"
done
echo "====================================================="
