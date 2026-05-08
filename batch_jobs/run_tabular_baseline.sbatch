#!/usr/bin/env bash
# Tabular baseline workflow on Morgan fingerprints and optional descriptors.
# Builds fingerprint parquet files for each subset, tunes the requested
# tabular model once, and reruns it across many random seeds with fixed params.
# Inputs: subset split CSVs and optional solvent/physchem features
# Outputs: results/... baseline metrics, predictions, and tuned params
#SBATCH -J em-fp-tabular-baseline-dccsdb-scaffold
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

# ---------------- Env ----------------
export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"
source "batch_jobs/_shared/subsets.sh"

mkdir -p logs
echo "JOBID=${SLURM_JOB_ID:-LOCAL} HOST=$(hostname) DATE=$(date)"

# Graceful walltime notice
trap 'echo "[WARN] Received TERM at $(date). Exiting gracefully." >&2' TERM

# Threads & CUDA sanity
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

echo "[INFO] CUDA devices:"
nvidia-smi || true

echo "[INFO] Micromamba envs:"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" env list || true

echo "[INFO] Python & XGBoost in ${ENV_NAME}:"
"$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python - <<'PY'
import sys, xgboost
print("python:", sys.version.split()[0])
print("xgboost:", xgboost.__version__)
PY

# Helper to run Python inside env
run_py() {
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python "$@"
}

# ---------------- Config ----------------
MODEL="${MODEL:-xgb}"  # Set to 'xgb' or 'ridge'
INCLUDE_PHYSCHEM="${INCLUDE_PHYSCHEM:-false}"
SPLIT="${SPLIT:-scaffold}"
DATASET="${DATASET:-dsscdb}"
DATASOURCE="${DATASOURCE:-uvvisml}"
TARGET="${TARGET:-em}"
TARGET_COL="${TARGET_COL:-empeakwavs_max}"
REGIME="${REGIME:-tabular_baselines}"
N_BITS="${N_BITS:-2048}"
RADIUS="${RADIUS:-2}"
MKFP=./scripts/make_morgan_emb_parquet.py

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

DATA_ROOT="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}"
RESULTS_ROOT="results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}"

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"

# ---------------- Logic ----------------

# Determine directory suffixes based on model and physchem
PHYS_SUFFIX=$([[ "$INCLUDE_PHYSCHEM" == "true" ]] && echo "_with_physchem" || echo "")
PHYS_FLAG=$([[ "$INCLUDE_PHYSCHEM" == "true" ]] && echo "--include-physchem" || echo "")

for SUB in "${SUBSET_LIST[@]}"; do
  echo "=============== Subset: ${SUB} | Model: ${MODEL} ================"

  SUBSET_DIR="${DATA_ROOT}/${SUB}"
  
  # 1) Setup Paths
  # We separate the embeddings (shared) from the model results (specific)
  BASE_TAG="fp_baseline_${SUB}${PHYS_SUFFIX}"
  EMB_DIR="${RESULTS_ROOT}/baselines/${BASE_TAG}/embeddings"
  OUT_BASE="${RESULTS_ROOT}/baselines/${BASE_TAG}/${MODEL}${PHYS_SUFFIX}"
  
  mkdir -p "$EMB_DIR" "$OUT_BASE"

  # 2) Build Morgan-FP (only if missing)
  for SPL in train val test; do
    OUT_PQ="${EMB_DIR}/${SPL}.parquet"
    if [[ ! -s "$OUT_PQ" ]]; then
      run_py "$MKFP" --csv "${SUBSET_DIR}/smiles_target_${SPL}.csv" \
        --smiles-col smiles --out "$OUT_PQ" --n-bits ${N_BITS} --radius ${RADIUS}
    fi
  done

  # 3) Check for solvent features
  INCLUDE_SOLV=""
  if [[ -s "${SUBSET_DIR}/features_train.csv" ]]; then
    INCLUDE_SOLV="--include-solvent-fp"
  fi

  # 4) (A) TUNE on seed 0
  echo "[tune] ${MODEL} on seed 0..."
  run_py -m uvvis_hybrid.scripts.fit_tabular \
    --train-csv "${SUBSET_DIR}/smiles_target_train.csv" \
    --val-csv   "${SUBSET_DIR}/smiles_target_val.csv" \
    --test-csv  "${SUBSET_DIR}/smiles_target_test.csv" \
    --smiles-col smiles --target-col ${TARGET_COL} \
    --emb-train "${EMB_DIR}/train.parquet" \
    --emb-val   "${EMB_DIR}/val.parquet" \
    --emb-test  "${EMB_DIR}/test.parquet" \
    $INCLUDE_SOLV $PHYS_FLAG \
    --model "${MODEL}" \
    --seed 0 \
    --tune --tune-trials 50 \
    --device cuda \
    --outdir "$OUT_BASE"

  # 5) (B) REUSE best params for seeds 1..29
  echo "[run] Seeds 1-29 using best ${MODEL} params..."
  for s in $(seq 1 29); do
    SEED_DIR="${OUT_BASE}/seed_$(printf "%04d" $s)"
    run_py -m uvvis_hybrid.scripts.fit_tabular \
      --train-csv "${SUBSET_DIR}/smiles_target_train.csv" \
      --val-csv   "${SUBSET_DIR}/smiles_target_val.csv" \
      --test-csv  "${SUBSET_DIR}/smiles_target_test.csv" \
      --smiles-col smiles --target-col ${TARGET_COL} \
      --emb-train "${EMB_DIR}/train.parquet" \
      --emb-val   "${EMB_DIR}/val.parquet" \
      --emb-test  "${EMB_DIR}/test.parquet" \
      $INCLUDE_SOLV $PHYS_FLAG \
      --model "${MODEL}" \
      --seed $s \
      --best-params-dir "$OUT_BASE" \
      --device cuda \
      --outdir "$SEED_DIR"
  done
done
