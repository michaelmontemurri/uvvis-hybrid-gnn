#!/bin/bash
# Variance-decomposition analysis for frozen-head XGBoost runs.
# Tunes and reruns XGBoost across encoder heads, downstream heads, and seeds
# to separate variability from the embedding source versus the regression head.
# Inputs: pre-dumped embeddings under results/.../xgb_on_learned_embedding_subsets/...
# Outputs: nested results under results/.../variance_decomp/...
#SBATCH -J variance-decomp-xgb
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=0
#SBATCH -c 16
#SBATCH --mem=10G
#SBATCH -t 1-00:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

# ---------- Micromamba ----------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs
source "batch_jobs/_shared/subsets.sh"

echo "JOBID=${SLURM_JOB_ID:-LOCAL} HOST=$(hostname) DATE=$(date)"

# ---------- Threads / CPU-only ----------
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Force CPU-only even on GPU nodes
export CUDA_VISIBLE_DEVICES=""

nvidia-smi || true

"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import torch, sys
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
PY

# ---------- Config ----------
SPLIT="${SPLIT:-scaffold}"
DATASET="${DATASET:-deep4chem}"
TARGET="${TARGET:-abs}"
TARGET_COL="${TARGET_COL:-peakwavs_max}"
DATASOURCE="${DATASOURCE:-uvvisml}"
MODEL="${MODEL:-xgb}"
EMB_REGIME="${EMB_REGIME:-xgb_on_learned_embedding_subsets}"
OUT_REGIME="${OUT_REGIME:-variance_decomp}"
TUNE_TRIALS="${TUNE_TRIALS:-10}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"

for subset in "${SUBSET_LIST[@]}"; do
  SUBSET_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}"
  BASE_OUT="results/${TARGET}/${OUT_REGIME}/${DATASET}/${SPLIT}/subsets/${subset}/${MODEL}"
  mkdir -p "$BASE_OUT"

  for i in 0 1 2 3 4; do
    EMB_DIR="results/${TARGET}/${EMB_REGIME}/${DATASET}/${SPLIT}/heads/head_0${i}_${subset}"
    OUT_HD_ROOT="${BASE_OUT}/encoder_0${i}"

    for h in 0 1 2 3 4; do
      HEADDIR="${OUT_HD_ROOT}/head_0${h}"
      TUNE_DIR="${HEADDIR}/tune"
      mkdir -p "$TUNE_DIR"

      echo "Tuning Encoder $i, Head $h..."
      "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
        python -m uvvis_hybrid.scripts.fit_tabular \
          --train-csv "${SUBSET_DIR}/smiles_target_train.csv" \
          --val-csv   "${SUBSET_DIR}/smiles_target_val.csv" \
          --test-csv  "${SUBSET_DIR}/smiles_target_test.csv" \
          --smiles-col smiles --target-col ${TARGET_COL} \
          --emb-train "${EMB_DIR}/embeddings/train.parquet" \
          --emb-val   "${EMB_DIR}/embeddings/val.parquet" \
          --emb-test  "${EMB_DIR}/embeddings/test.parquet" \
          --model "${MODEL}" \
          --tune --tune-trials ${TUNE_TRIALS} \
          --seed "$((42 + h))" \
          --device cpu \
          --outdir "$TUNE_DIR"

      for s in 100 101 102 103 104; do
        OUTDIR_RUN="${HEADDIR}/seed_${s}"
        mkdir -p "$OUTDIR_RUN"

        "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
          python -m uvvis_hybrid.scripts.fit_tabular \
            --train-csv "${SUBSET_DIR}/smiles_target_train.csv" \
            --val-csv   "${SUBSET_DIR}/smiles_target_val.csv" \
            --test-csv  "${SUBSET_DIR}/smiles_target_test.csv" \
            --smiles-col smiles --target-col ${TARGET_COL} \
            --emb-train "${EMB_DIR}/embeddings/train.parquet" \
            --emb-val   "${EMB_DIR}/embeddings/val.parquet" \
            --emb-test  "${EMB_DIR}/embeddings/test.parquet" \
            --model "${MODEL}" \
            --seed "$s" \
            --device cpu \
            --best-params-dir "$TUNE_DIR" \
            --outdir "$OUTDIR_RUN"
      done
    done
  done
done

echo "[$(date)] Completed variance decomposition run."
