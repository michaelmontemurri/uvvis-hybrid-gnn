#!/usr/bin/env bash
# Historical 5-member zero-shot Jeffries workflow.
# Evaluates each pretrained Deep4Chem scaffold member on the corrected Jeffries
# holdout split and writes per-member predictions plus an aggregated ensemble.

#SBATCH -J em-deep4chem-zeroshot-ens
#SBATCH -p gpu_debug_tesla
#SBATCH --qos=gpu_debug_tesla
#SBATCH --gpus=1
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 0-02:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail
mkdir -p logs

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/uvvisml"

RUN_ENV() {
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" "$@"
}

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TARGET="${TARGET:-em}"
TARGET_COL="${TARGET_COL:-empeakwavs_max}"
DATASOURCE="${DATASOURCE:-custom}"
DATASET="${DATASET:-jeffries}"
SPLIT="${SPLIT:-holdout}"
SUBSET_TAG="${SUBSET_TAG:-N00038_s42}"
PRETRAIN_ROOT="${PRETRAIN_ROOT:-checkpoints/${TARGET}/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11502_s42/fold_0}"
MODELS_STR="${MODELS:-model_0 model_1 model_2 model_3 model_4}"
read -r -a MODELS <<< "$MODELS_STR"

SUBSET_DIR="${SUBSET_DIR:-data/${DATASOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${SUBSET_TAG}}"
[[ -d "$SUBSET_DIR" ]] || { echo "[ERROR] missing SUBSET_DIR=$SUBSET_DIR" >&2; exit 1; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
fi

OUT_ROOT="${OUT_ROOT:-results/${TARGET}/transfer/${DATASET}/${SPLIT}/deep4chem_zeroshot_ensemble/job${SLURM_JOB_ID}}"
mkdir -p "$OUT_ROOT"

TEST_CSV="${SUBSET_DIR}/smiles_target_test.csv"
TEST_FEAT="${SUBSET_DIR}/features_test.csv"

echo "OUT_ROOT=$OUT_ROOT"
echo "TEST_CSV=$TEST_CSV"
echo "PRETRAIN_ROOT=$PRETRAIN_ROOT"
echo "MODELS=${MODELS[*]}"

for model_name in "${MODELS[@]}"; do
  ckpt="${PRETRAIN_ROOT}/${model_name}/model.pt"
  [[ -f "$ckpt" ]] || { echo "[ERROR] missing checkpoint $ckpt" >&2; exit 2; }

  out_dir="${OUT_ROOT}/${model_name}"
  mkdir -p "$out_dir"

  echo "=============================="
  echo "Zero-shot: ${model_name}"
  echo "CKPT=${ckpt}"
  echo "=============================="

  RUN_ENV chemprop_predict \
    --checkpoint_path "$ckpt" \
    --test_path "$TEST_CSV" \
    --features_path "$TEST_FEAT" \
    --smiles_columns smiles \
    --preds_path "${out_dir}/preds_test.csv" \
    --num_workers 2 \
    --gpu 0 \
    2>&1 | tee "${out_dir}/predict.log"
done

RUN_ENV python - <<'PY' "$OUT_ROOT" "$TEST_CSV" "$TARGET_COL"
import glob
import os
import sys

import numpy as np
import pandas as pd

out_root, test_csv, target = sys.argv[1], sys.argv[2], sys.argv[3]

gt = pd.read_csv(test_csv)
assert "smiles" in gt.columns, "GT missing smiles column"
y = gt[target].astype(float).to_numpy()

pred_paths = sorted(glob.glob(os.path.join(out_root, "model_*", "preds_test.csv")))
if not pred_paths:
    raise SystemExit(f"No preds_test.csv found under {out_root}")

member_names = []
member_rmses = []
all_preds = []

for p in pred_paths:
    member = os.path.basename(os.path.dirname(p))
    df = pd.read_csv(p)

    if "smiles" not in df.columns:
        raise SystemExit(f"{p} missing smiles column; cannot verify ordering")
    if not (df["smiles"].astype(str).values == gt["smiles"].astype(str).values).all():
        a = gt["smiles"].astype(str).values
        b = df["smiles"].astype(str).values
        idx = next(i for i, (x, y2) in enumerate(zip(a, b)) if x != y2)
        raise SystemExit(f"SMILES order mismatch in {p} at index {idx}\\nGT={a[idx]}\\nPR={b[idx]}")

    if target not in df.columns:
        raise SystemExit(f"{p} missing target column {target}")

    pred = df[target].astype(float).to_numpy()
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))

    member_names.append(member)
    member_rmses.append(rmse)
    all_preds.append(pred)

member_df = pd.DataFrame({"member": member_names, "rmse": member_rmses}).sort_values("member")
member_path = os.path.join(out_root, "member_metrics.csv")
member_df.to_csv(member_path, index=False)

P = np.stack(all_preds, axis=0)
p_mean = P.mean(axis=0)
p_median = np.median(P, axis=0)
p_std = P.std(axis=0)

rmse_mean = float(np.sqrt(np.mean((y - p_mean) ** 2)))
rmse_median = float(np.sqrt(np.mean((y - p_median) ** 2)))

ens = gt[["smiles"]].copy()
if "solvent" in gt.columns:
    ens["solvent"] = gt["solvent"]
ens[target] = y
ens[f"{target}_mean"] = p_mean
ens[f"{target}_median"] = p_median
ens[f"{target}_std"] = p_std

ens_path = os.path.join(out_root, "ensemble_test.csv")
ens.to_csv(ens_path, index=False)

txt_path = os.path.join(out_root, "posthoc_metrics.txt")
with open(txt_path, "w") as f:
    f.write("Per-member RMSEs:\\n")
    for _, row in member_df.iterrows():
        f.write(f"  {row['member']}: {row['rmse']:.6f}\\n")
    f.write("\\nAcross-member ensemble RMSE:\\n")
    f.write(f"  mean  : {rmse_mean:.6f}\\n")
    f.write(f"  median: {rmse_median:.6f}\\n")
    f.write("\\nSanity (GT vs ensemble-mean preds):\\n")
    for i in range(min(10, len(y))):
        f.write(f"{i} {float(y[i])} {float(p_mean[i])}\\n")

print("\\nPer-member RMSEs:")
for _, row in member_df.iterrows():
    print(f"  {row['member']}: {row['rmse']:.6f}")

print("\\nAcross-member ensemble RMSE:")
print(f"  mean  : {rmse_mean:.6f}")
print(f"  median: {rmse_median:.6f}")

print("\\nSanity (GT vs ensemble-mean preds):")
for i in range(min(10, len(y))):
    print(i, float(y[i]), float(p_mean[i]))

print("\\nWrote:")
print(" ", member_path)
print(" ", ens_path)
print(" ", txt_path)
PY

echo "Done."
echo "Artifacts:"
echo "  $OUT_ROOT/member_metrics.csv"
echo "  $OUT_ROOT/ensemble_test.csv"
echo "  $OUT_ROOT/posthoc_metrics.txt"
