#!/usr/bin/env bash
# Linear-style baseline comparison on saved representations.
# Rebuilds split-level feature tables and evaluates simple downstream models
# such as ridge and inverse-gap baselines on the requested dataset/subsets.
# Inputs: processed masters, split CSVs, and saved embeddings
# Outputs: results/... baseline metrics and predictions
#SBATCH -J em-fp-linear-baseline-dsscdb-scaffold
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

# CRITICAL FIX: Add the current directory to PYTHONPATH 
# so 'uvvis_hybrid' can be found as a module.
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
source "batch_jobs/_shared/subsets.sh"

mkdir -p logs
echo "JOBID=${SLURM_JOB_ID:-LOCAL} HOST=$(hostname) DATE=$(date)"

# Helper to run Python inside env
run_py() {
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" python "$@"
}

# ---------------- Config ----------------
DATASET="${DATASET:-dsscdb}"
SPLIT="${SPLIT:-scaffold}"
TARGET="${TARGET:-em}"
TARGET_COL="${TARGET_COL:-empeakwavs_max}"
DATASOURCE="${DATASOURCE:-uvvisml}"
REGIME="${REGIME:-linear_baselines}"
MASTER_PATH="${MASTER_PATH:-data/processed/uvvis_${TARGET}_master_full.csv}"
OUTDIR="results/${TARGET}/${REGIME}/${DATASET}/${SPLIT}"

if [[ -n "${SUBSETS:-}" ]]; then
  read -r -a SUBSET_LIST <<< "${SUBSETS}"
else
  read -r -a SUBSET_LIST <<< "$(get_subsets "${TARGET}" "${DATASET}" "${SPLIT}")"
fi

echo "Starting baseline runs for ${DATASET}..."
echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"

for subset in "${SUBSET_LIST[@]}"
do
    echo "------------------------------------------------"
    echo "Running subset = ${subset}"
    echo "------------------------------------------------"
    n="${subset#N}"
    n="${n%_s42}"
    
    # Use the module path relative to the project root
    run_py -m scripts.run_linear_baselines \
      --dataset "${DATASET}" \
      --split-type "${SPLIT}" \
      --target "${TARGET}" \
      --target-col "${TARGET_COL}" \
      --n-size "${n}" \
      --datasource "${DATASOURCE}" \
      --master-path "${MASTER_PATH}" \
      --outdir "${OUTDIR}"
done

echo "Done!"
