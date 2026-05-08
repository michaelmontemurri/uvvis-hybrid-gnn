#!/bin/bash
# Zero-shot transfer baseline from the tracked pretrained Deep4Chem checkpoint.
# Evaluates the pretrained model directly on target-dataset subset test splits
# without any finetuning.
# Outputs: results/.../transfer/.../zeroshot_deep4chem/job*/

#SBATCH -J transfer-zeroshot
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 0-02:00:00
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

PRETRAIN_CKPT="checkpoints/${TARGET}/deep4chem/scaffold/morgan_fingerprint/fromscratch_N${PRETRAIN_SIZE}_s42/fold_0/model_0/model.pt"
[[ -f "${PRETRAIN_CKPT}" ]] || { echo "[ERROR] missing PRETRAIN_CKPT=${PRETRAIN_CKPT}" >&2; exit 1; }

RUN_ROOT="results/${TARGET}/transfer/${DATASET}/${SPLIT}/zeroshot_deep4chem/job${SLURM_JOB_ID}"
mkdir -p "${RUN_ROOT}"

rmse_py='import pandas as pd, numpy as np, sys, json, datetime
out_dir, data_dir, target = sys.argv[1], sys.argv[2], sys.argv[3]
preds = pd.read_csv(f"{out_dir}/preds_test.csv")
test  = pd.read_csv(f"{data_dir}/smiles_target_test.csv")

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

yhat = preds[target].to_numpy(float)
y    = test[target].to_numpy(float)
rmse = float(np.sqrt(np.mean((yhat - y)**2)))

out = {
  "metric": "rmse",
  "rmse": rmse,
  "n_test": int(len(y)),
  "pred_col": target,
  "timestamp": datetime.datetime.now().isoformat()
}
with open(f"{out_dir}/metrics.json","w") as f:
    json.dump(out, f, indent=2, sort_keys=True)
print(rmse)
'

echo "[CONFIG] target=${TARGET} dataset=${DATASET} split=${SPLIT}"
echo "[CONFIG] subsets=${SUBSET_LIST[*]}"
echo "[CONFIG] ckpt=${PRETRAIN_CKPT}"
echo "[CONFIG] out=${RUN_ROOT}"

for subset in "${SUBSET_LIST[@]}"; do
  DATA_DIR="data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${subset}"
  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[WARN] missing DATA_DIR=${DATA_DIR} (skipping)"
    continue
  fi

  OUT_DIR="${RUN_ROOT}/${subset}"
  mkdir -p "${OUT_DIR}"

  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    chemprop_predict \
      --checkpoint_path "${PRETRAIN_CKPT}" \
      --test_path "${DATA_DIR}/smiles_target_test.csv" \
      --features_path "${DATA_DIR}/features_test.csv" \
      --smiles_columns smiles \
      --preds_path "${OUT_DIR}/preds_test.csv" \
      --num_workers 2 --gpu 0 \
    2>&1 | tee "${OUT_DIR}/predict.log"

  RMSE=$("$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" \
    python -c "$rmse_py" "${OUT_DIR}" "${DATA_DIR}" "${TARGET_COL}" | tail -n 1)

  {
    echo "SUBSET = ${subset}"
    echo "rmse = ${RMSE}"
  } | tee "${OUT_DIR}/metrics.txt"
done

echo "[DONE] Wrote zero-shot results under: ${RUN_ROOT}"
