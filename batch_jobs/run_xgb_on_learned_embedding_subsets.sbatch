#!/bin/bash
# Frozen-encoder downstream-head workflow across subset sizes.
# Dumps embeddings from Chemprop checkpoints, fits one downstream head per
# encoder member, and then ensembles predictions across encoder heads.
# Inputs: subset split CSVs and pretrained Chemprop checkpoints
# Outputs: results/.../xgb_on_learned_embedding_subsets/... head-level and cross-head ensemble metrics
#SBATCH -J xgb-on-learned-embeddings
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

# ---------- Micromamba setup ----------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

set -e -o pipefail

echo "JOBID=$SLURM_JOB_ID  HOST=$(hostname)  DATE=$(date)"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source "batch_jobs/_shared/subsets.sh"

# ---------- Threading & CUDA ----------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

echo "SLURM env: GPU=${SLURM_JOB_GPUS:-} GRES=${SLURM_JOB_GRES:-}"
nvidia-smi || true
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" env list || true

echo "Python & CUDA sanity (inside $ENV_NAME):"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", getattr(torch.version, "cuda", None), "is_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

# ---------- Config ----------
DATASOURCE="${DATASOURCE:-uvvisml}"
SPLIT="${SPLIT:-scaffold}"
DATASET="${DATASET:-chemfluor}"
REGIME="${REGIME:-xgb_on_learned_embedding_subsets}"
MODEL="${MODEL:-xgb}"
TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
INCLUDE_PHYSCHEM="${INCLUDE_PHYSCHEM:-false}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"
echo "[CONFIG] include_physchem=${INCLUDE_PHYSCHEM}"

# 1 = use fixed XGB hyperparams (no Optuna)
# 0 = run Optuna tuning as before
USE_FIXED_PARAMS=1

FEATURE_ARGS=(--include-solvent-fp)
if [[ "${INCLUDE_PHYSCHEM}" == "true" ]]; then
  FEATURE_ARGS+=(--include-physchem)
fi

for subset in "${SUBSET_LIST[@]}"; do
  SUBSET_DIR=data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}
  CKPT_DIR=checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/fromscratch_${subset}

  # ---------- 1) Dump embeddings for each head ----------
  set -x
  for i in 0 1 2 3 4; do
    OUTDIR=results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/heads/head_0${i}_${subset}
    mkdir -p "$OUTDIR/embeddings"

    for split in train val test; do
      "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
        python -m uvvis_hybrid.scripts.dump_embeddings \
          --ckpt ${CKPT_DIR}/fold_0/model_${i}/model.pt \
          --csv  ${SUBSET_DIR}/smiles_target_${split}.csv \
          --smiles-col smiles --stage pre_ffn \
          --out $OUTDIR/embeddings/${split}.parquet
    done
  done

  # ---------- 2) Get XGB hyperparams for HEAD 0 ----------
  # PARAMS_DIR is where run_frozen_embedding_head_ensemble will look for best params
  PARAMS_DIR=results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/heads/head_00_${subset}/${MODEL}
  mkdir -p "$PARAMS_DIR"
  BEST_PARAMS_JSON="${PARAMS_DIR}/xgb_best_params_head0.json"

  if [[ "$USE_FIXED_PARAMS" -eq 1 ]]; then
    echo "Using FIXED XGB hyperparameters, writing to ${BEST_PARAMS_JSON}"
    cat > "$BEST_PARAMS_JSON" <<'EOF'
  {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.022773099002946575,
    "min_child_weight": 32,
    "reg_alpha": 0.1833181064969345,
    "reg_lambda": 0.013,
    "gamma": 0.054,
    "subsample": 0.5,
    "colsample_bytree": 0.90
  }
EOF
  else
    echo "Running Optuna tuning for XGB (head 0) into ${PARAMS_DIR}"
    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python -m uvvis_hybrid.scripts.fit_tabular \
        --train-csv ${SUBSET_DIR}/smiles_target_train.csv \
        --val-csv   ${SUBSET_DIR}/smiles_target_val.csv \
        --test-csv  ${SUBSET_DIR}/smiles_target_test.csv \
        --smiles-col smiles --target-col ${TARGET_COL} \
        --emb-train results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/heads/head_00_${subset}/embeddings/train.parquet \
        --emb-val   results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/heads/head_00_${subset}/embeddings/val.parquet \
        --emb-test  results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/heads/head_00_${subset}/embeddings/test.parquet \
        "${FEATURE_ARGS[@]}" \
        --model ${MODEL} \
        --seed 100 \
        --tune --tune-trials 100 \
        --device cuda \
        --outdir "$PARAMS_DIR"
  fi

  # ---------- 3) Fit 1-per-head using tuned or fixed params ----------
  for i in 0 1 2 3 4; do
    "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
      python scripts/frozen_head_comparison/run_frozen_embedding_head_ensemble_experiment.py \
        --split-dir ${SUBSET_DIR} \
        --ckpt      ${CKPT_DIR}/fold_0/model_${i}/model.pt \
        --outdir    results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_0${i} \
        --model ${MODEL} \
        --target-col ${TARGET_COL} \
        --members 1 \
        --base-seed 100 \
        --best-params-dir "$PARAMS_DIR" \
        "${FEATURE_ARGS[@]}" \
        --device cuda
  done

  # ---------- 4) Cross-head ensemble (mean) ----------
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    python scripts/ensemble_cross_heads.py \
      --inputs \
        results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_00/ensemble_preds_test.csv \
        results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_01/ensemble_preds_test.csv \
        results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_02/ensemble_preds_test.csv \
        results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_03/ensemble_preds_test.csv \
        results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_single_perhead/head_04/ensemble_preds_test.csv \
      --split test \
      --outdir results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_crosshead_singleseed_mean \
      --blend mean
  set +x

  echo
  echo "[$(date)] Done learned-embedding XGB run for ${subset}"
  echo "Results in: results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}/ensembles/${MODEL}_${subset}_crosshead_singleseed_mean"
done
