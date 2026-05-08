#!/usr/bin/env bash
# Historical two-stage Jeffries transfer workflow.
# Uses a single pretrained Deep4Chem member, performs a head-warmup stage
# followed by full finetuning, and writes outputs under twostage_ft_ensemble.

#SBATCH -J em-jeffries-2stage-ft-ens
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-04:00:00
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

TARGET_COL="${TARGET_COL:-empeakwavs_max}"
DATA_DIR="${DATA_DIR:-data/custom/em/jeffries/subsets/holdout/N00038_s42}"

TRAIN_CSV="${TRAIN_CSV:-$DATA_DIR/smiles_target_train.csv}"
VAL_CSV="${VAL_CSV:-$DATA_DIR/smiles_target_val.csv}"
TEST_CSV="${TEST_CSV:-$DATA_DIR/smiles_target_test.csv}"

FEAT_TRAIN="${FEAT_TRAIN:-$DATA_DIR/features_train.csv}"
FEAT_VAL="${FEAT_VAL:-$DATA_DIR/features_val.csv}"
FEAT_TEST="${FEAT_TEST:-$DATA_DIR/features_test.csv}"

PRETRAIN_CKPT="${PRETRAIN_CKPT:-checkpoints/em/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11502_s42/fold_0/model_0/model.pt}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
fi

OUT_ROOT="${OUT_ROOT:-results/em/transfer/jeffries/holdout/twostage_ft_ensemble/job${SLURM_JOB_ID}}"
mkdir -p "$OUT_ROOT"

ARCH_ARGS=(
  --hidden_size "${HIDDEN_SIZE:-482}"
  --ffn_hidden_size "${FFN_HIDDEN_SIZE:-900}"
  --ffn_num_layers "${FFN_NUM_LAYERS:-2}"
  --depth "${DEPTH:-3}"
  --dropout "${DROPOUT:-0.05}"
  --activation "${ACTIVATION:-ReLU}"
  --aggregation "${AGGREGATION:-norm}"
  --aggregation_norm "${AGGREGATION_NORM:-90}"
  --hidden_size_solvent "${HIDDEN_SIZE_SOLVENT:-0}"
  --depth_solvent "${DEPTH_SOLVENT:-0}"
)

COMMON_DATA_ARGS=(
  --dataset_type regression
  --target_columns "$TARGET_COL"
  --smiles_columns smiles
  --metric rmse
  --num_workers 2
  --gpu 0
  --data_path "$TRAIN_CSV"
  --separate_val_path "$VAL_CSV"
  --separate_test_path "$TEST_CSV"
  --features_path "$FEAT_TRAIN"
  --separate_val_features_path "$FEAT_VAL"
  --separate_test_features_path "$FEAT_TEST"
)

HELP_TXT="$(RUN_ENV chemprop_train -h 2>&1 || true)"

HAS_FRZN_FFN_LAYERS=0
echo "$HELP_TXT" | grep -q -- "--frzn_ffn_layers" && HAS_FRZN_FFN_LAYERS=1

HAS_FREEZE_FIRST_ONLY=0
echo "$HELP_TXT" | grep -q -- "--freeze_first_only" && HAS_FREEZE_FIRST_ONLY=1

STAGE1_EPOCHS="${STAGE1_EPOCHS:-30}"
STAGE1_BS="${STAGE1_BS:-8}"
STAGE1_INIT_LR="${STAGE1_INIT_LR:-2e-5}"
STAGE1_MAX_LR="${STAGE1_MAX_LR:-2e-4}"
STAGE1_FINAL_LR="${STAGE1_FINAL_LR:-2e-5}"
STAGE1_WARMUP="${STAGE1_WARMUP:-3}"

STAGE2_EPOCHS="${STAGE2_EPOCHS:-50}"
STAGE2_BS="${STAGE2_BS:-8}"
STAGE2_INIT_LR="${STAGE2_INIT_LR:-3e-5}"
STAGE2_MAX_LR="${STAGE2_MAX_LR:-1e-4}"
STAGE2_FINAL_LR="${STAGE2_FINAL_LR:-3e-5}"
STAGE2_WARMUP="${STAGE2_WARMUP:-5}"

SEEDS_STR="${SEEDS:-0 1 2 3 4}"
read -r -a SEEDS <<< "$SEEDS_STR"

echo "OUT_ROOT=$OUT_ROOT"
echo "Using PRETRAIN_CKPT=$PRETRAIN_CKPT"
echo "HAS_FRZN_FFN_LAYERS=$HAS_FRZN_FFN_LAYERS HAS_FREEZE_FIRST_ONLY=$HAS_FREEZE_FIRST_ONLY"

for SEED in "${SEEDS[@]}"; do
  RUN_DIR="$OUT_ROOT/seed_${SEED}"
  STAGE1_DIR="$RUN_DIR/stage1_headwarm"
  STAGE2_DIR="$RUN_DIR/stage2_fullft"
  mkdir -p "$STAGE1_DIR" "$STAGE2_DIR"

  echo "=============================="
  echo "Seed $SEED - Stage 1 (head warmup)"
  echo "=============================="

  STAGE1_ARGS=(
    "${COMMON_DATA_ARGS[@]}"
    "${ARCH_ARGS[@]}"
    --epochs "$STAGE1_EPOCHS"
    --batch_size "$STAGE1_BS"
    --init_lr "$STAGE1_INIT_LR"
    --max_lr "$STAGE1_MAX_LR"
    --final_lr "$STAGE1_FINAL_LR"
    --warmup_epochs "$STAGE1_WARMUP"
    --seed "$SEED"
    --pytorch_seed "$SEED"
    --save_dir "$STAGE1_DIR/models"
    --checkpoint_path "$PRETRAIN_CKPT"
    --checkpoint_frzn "$PRETRAIN_CKPT"
  )

  if [[ "$HAS_FREEZE_FIRST_ONLY" -eq 1 ]]; then
    STAGE1_ARGS+=( --freeze_first_only )
  fi
  if [[ "$HAS_FRZN_FFN_LAYERS" -eq 1 ]]; then
    STAGE1_ARGS+=( --frzn_ffn_layers 0 )
  fi

  RUN_ENV chemprop_train "${STAGE1_ARGS[@]}"

  STAGE1_CKPT="$STAGE1_DIR/models/fold_0/model_0/model.pt"
  if [[ ! -f "$STAGE1_CKPT" ]]; then
    echo "ERROR: Stage 1 checkpoint not found at $STAGE1_CKPT"
    exit 1
  fi

  echo "=============================="
  echo "Seed $SEED - Stage 2 (full finetune)"
  echo "=============================="

  STAGE2_ARGS=(
    "${COMMON_DATA_ARGS[@]}"
    "${ARCH_ARGS[@]}"
    --epochs "$STAGE2_EPOCHS"
    --batch_size "$STAGE2_BS"
    --init_lr "$STAGE2_INIT_LR"
    --max_lr "$STAGE2_MAX_LR"
    --final_lr "$STAGE2_FINAL_LR"
    --warmup_epochs "$STAGE2_WARMUP"
    --seed "$SEED"
    --pytorch_seed "$SEED"
    --save_dir "$STAGE2_DIR/models"
    --checkpoint_path "$STAGE1_CKPT"
  )

  RUN_ENV chemprop_train "${STAGE2_ARGS[@]}"

  STAGE2_CKPT="$STAGE2_DIR/models/fold_0/model_0/model.pt"
  if [[ ! -f "$STAGE2_CKPT" ]]; then
    echo "ERROR: Stage 2 checkpoint not found at $STAGE2_CKPT"
    exit 1
  fi

  RUN_ENV chemprop_predict \
    --checkpoint_path "$STAGE2_CKPT" \
    --test_path "$TEST_CSV" \
    --features_path "$FEAT_TEST" \
    --smiles_columns smiles \
    --preds_path "$STAGE2_DIR/preds_test.csv" \
    --gpu 0 --num_workers 2
done

echo "=============================="
echo "Post-hoc evaluation (per seed + ensemble)"
echo "=============================="

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

pred_paths = sorted(glob.glob(os.path.join(out_root, "seed_*", "stage2_fullft", "preds_test.csv")))
if not pred_paths:
    raise SystemExit(f"No preds_test.csv found under {out_root}")

seed_names = []
seed_rmses = []
all_preds = []

for p in pred_paths:
    seed = os.path.basename(os.path.dirname(os.path.dirname(p)))
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

    seed_names.append(seed)
    seed_rmses.append(rmse)
    all_preds.append(pred)

per_seed = pd.DataFrame({"seed": seed_names, "rmse": seed_rmses}).sort_values("seed")
per_seed_path = os.path.join(out_root, "per_seed_rmse.csv")
per_seed.to_csv(per_seed_path, index=False)

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
    f.write("Per-seed RMSEs:\\n")
    for _, row in per_seed.iterrows():
        f.write(f"  {row['seed']}: {row['rmse']:.6f}\\n")
    f.write("\\nAcross-seed ensemble RMSE:\\n")
    f.write(f"  mean   : {rmse_mean:.6f}\\n")
    f.write(f"  median : {rmse_median:.6f}\\n")
    f.write("\\nSanity (GT vs ensemble-mean preds):\\n")
    for i in range(min(10, len(y))):
        f.write(f"{i} {float(y[i])} {float(p_mean[i])}\\n")

print("\\nPer-seed RMSEs:")
for _, row in per_seed.iterrows():
    print(f"  {row['seed']}: {row['rmse']:.6f}")

print("\\nAcross-seed ensemble RMSE:")
print(f"  mean   : {rmse_mean:.6f}")
print(f"  median : {rmse_median:.6f}")

print("\\nSanity (GT vs ensemble-mean preds):")
for i in range(min(10, len(y))):
    print(i, float(y[i]), float(p_mean[i]))

print("\\nWrote:")
print(" ", per_seed_path)
print(" ", ens_path)
print(" ", txt_path)
PY

echo "Done."
echo "Artifacts:"
echo "  $OUT_ROOT/per_seed_rmse.csv"
echo "  $OUT_ROOT/ensemble_test.csv"
echo "  $OUT_ROOT/posthoc_metrics.txt"
