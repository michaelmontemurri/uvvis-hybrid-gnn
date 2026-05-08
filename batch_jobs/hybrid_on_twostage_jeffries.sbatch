#!/bin/bash
# Historical hybrid-on-finetuned-Jeffries workflow.
# Starts from a completed twostage_ft_ensemble job, dumps finetuned embeddings,
# tunes the hybrid head on member_0, then fits one hybrid model per member.

#SBATCH -J em-hybrid-on-ft-jeffries-ens
#SBATCH -p gpu_h100_pro
#SBATCH --qos=gpu_h100_pro
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 0-08:00:00
#SBATCH -A kolaczyk-2025
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --signal=B:TERM@300

set -euo pipefail

echo "JOBID=${SLURM_JOB_ID:-LOCAL}  HOST=$(hostname)  DATE=$(date)"
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export PATH="$HOME/bin:$PATH"
export MAMBA_ROOT_PREFIX="$HOME/.conda"
export MAMBA_EXE="${MAMBA_EXE:-$HOME/bin/micromamba}"
ENV_NAME="uvvisml"
ENV_PATH="$MAMBA_ROOT_PREFIX/envs/$ENV_NAME"

RUN_ENV() {
  "$MAMBA_EXE" -r "$MAMBA_ROOT_PREFIX" run -p "$ENV_PATH" "$@"
}

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0
export TOKENIZERS_PARALLELISM=false

nvidia-smi || true

datasource="${DATASOURCE:-custom}"
target_type="${TARGET_TYPE:-em}"
dataset="${DATASET:-jeffries}"
split_type="${SPLIT_TYPE:-holdout}"
target="${TARGET_COL:-empeakwavs_max}"
smiles_col="${SMILES_COL:-smiles}"

SUBSET_TAG="${SUBSET_TAG:-N00038_s42}"
SUBSET_DIR="${SUBSET_DIR:-data/${datasource}/${target_type}/${dataset}/subsets/${split_type}/${SUBSET_TAG}}"
[[ -d "$SUBSET_DIR" ]] || { echo "[ERROR] Missing SUBSET_DIR=$SUBSET_DIR"; exit 2; }

FT_JOB_DIR="${FT_JOB_DIR:-results/${target_type}/transfer/${dataset}/${split_type}/twostage_ft_ensemble/job3408}"
[[ -d "$FT_JOB_DIR" ]] || { echo "[ERROR] Missing FT_JOB_DIR=$FT_JOB_DIR"; exit 3; }

SEEDS_STR="${SEEDS:-0 1 2 3 4}"
read -r -a SEEDS <<< "$SEEDS_STR"

HIDDEN_SIZE="${HIDDEN_SIZE:-482}"
model="${MODEL:-xgb}"
HYBRID_SEED="${HYBRID_SEED:-100}"
TUNE_TRIALS="${TUNE_TRIALS:-100}"
DEVICE="${DEVICE:-cuda}"
BEST_PARAMS_DIR_OVERRIDE="${BEST_PARAMS_DIR:-}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  export SLURM_JOB_ID="LOCAL"
fi

RUN_ROOT="${RUN_ROOT:-results/${target_type}/transfer/${dataset}/${split_type}/hybrid_on_finetuned_jeffries_embeddings/job${SLURM_JOB_ID}}"
mkdir -p "$RUN_ROOT"

echo "[INFO] RUN_ROOT=$RUN_ROOT"
echo "[INFO] SUBSET_DIR=$SUBSET_DIR"
echo "[INFO] FT_JOB_DIR=$FT_JOB_DIR"
echo "[INFO] SEEDS=${SEEDS[*]}"
echo "[INFO] model=$model HYBRID_SEED=$HYBRID_SEED TUNE_TRIALS=$TUNE_TRIALS DEVICE=$DEVICE"

CKPT_LIST=()
for s in "${SEEDS[@]}"; do
  ckpt="${FT_JOB_DIR}/seed_${s}/stage2_fullft/models/fold_0/model_0/model.pt"
  [[ -f "$ckpt" ]] || { echo "[ERROR] Missing finetuned ckpt for seed_${s}: $ckpt"; exit 4; }
  CKPT_LIST+=("$ckpt")
done

MANIFEST="$RUN_ROOT/finetuned_ckpts.txt"
: > "$MANIFEST"
for i in "${!SEEDS[@]}"; do
  echo "member_${i} seed_${SEEDS[$i]} ${CKPT_LIST[$i]}" >> "$MANIFEST"
done
echo "[INFO] Wrote ckpt manifest: $MANIFEST"

echo
echo "============================================================"
echo "1) Dump finetuned Jeffries embeddings (train/val/test)"
echo "============================================================"

for i in "${!CKPT_LIST[@]}"; do
  CKPT="${CKPT_LIST[$i]}"

  EMB_DIR="$RUN_ROOT/hybrid_members/member_${i}/embeddings"
  mkdir -p "$EMB_DIR"

  echo "[INFO] member_${i} ckpt=$CKPT -> $EMB_DIR"

  for sp in train val test; do
    RUN_ENV python -m uvvis_hybrid.scripts.dump_embeddings \
      --ckpt "$CKPT" \
      --csv "${SUBSET_DIR}/smiles_target_${sp}.csv" \
      --smiles-col "$smiles_col" \
      --stage pre_ffn \
      --out "$EMB_DIR/${sp}.parquet"
  done
done

echo
echo "============================================================"
echo "2) HPO for hybrid head on member_0 embeddings"
echo "============================================================"

PARAMS_DIR="$RUN_ROOT/hpo/member_0/${model}"
if [[ -n "${BEST_PARAMS_DIR_OVERRIDE}" ]]; then
  PARAMS_DIR="${BEST_PARAMS_DIR_OVERRIDE}"
fi
mkdir -p "$PARAMS_DIR"
BEST_PARAMS_JSON="$PARAMS_DIR/xgb_best_params_head0.json"

if [[ -f "$BEST_PARAMS_JSON" ]]; then
  echo "[INFO] Reusing tuned params: $BEST_PARAMS_JSON"
else
  RUN_ENV python -m uvvis_hybrid.scripts.fit_tabular \
    --train-csv "${SUBSET_DIR}/smiles_target_train.csv" \
    --val-csv "${SUBSET_DIR}/smiles_target_val.csv" \
    --test-csv "${SUBSET_DIR}/smiles_target_test.csv" \
    --smiles-col "$smiles_col" \
    --target-col "$target" \
    --emb-train "$RUN_ROOT/hybrid_members/member_0/embeddings/train.parquet" \
    --emb-val "$RUN_ROOT/hybrid_members/member_0/embeddings/val.parquet" \
    --emb-test "$RUN_ROOT/hybrid_members/member_0/embeddings/test.parquet" \
    --include-solvent-fp \
    --model "$model" \
    --seed "$HYBRID_SEED" \
    --tune --tune-trials "$TUNE_TRIALS" \
    --device "$DEVICE" \
    --outdir "$PARAMS_DIR"

  [[ -f "$BEST_PARAMS_JSON" ]] || { echo "[ERROR] HPO did not produce $BEST_PARAMS_JSON"; exit 20; }
  echo "[INFO] Using tuned params: $BEST_PARAMS_JSON"
fi

echo
echo "============================================================"
echo "3) Fit hybrid on each finetuned member (shared tuned params)"
echo "============================================================"

for i in "${!CKPT_LIST[@]}"; do
  OUTDIR="$RUN_ROOT/ensembles/${model}_${SUBSET_TAG}_single_permemb/member_${i}"
  EMB_DIR="$RUN_ROOT/hybrid_members/member_${i}/embeddings"
  mkdir -p "$OUTDIR"

  RUN_ENV python scripts/frozen_head_comparison/run_frozen_embedding_head_ensemble.py \
    --split-dir "${SUBSET_DIR}" \
    --ckpt "${CKPT_LIST[$i]}" \
    --outdir "$OUTDIR" \
    --embeddings-dir "$EMB_DIR" \
    --model "$model" \
    --target-col "$target" \
    --members 1 \
    --base-seed "$HYBRID_SEED" \
    --best-params-dir "$PARAMS_DIR" \
    --device "$DEVICE"
done

echo
echo "============================================================"
echo "4) Cross-member ensemble (mean)"
echo "============================================================"

INPUTS=()
for i in "${!CKPT_LIST[@]}"; do
  INPUTS+=("$RUN_ROOT/ensembles/${model}_${SUBSET_TAG}_single_permemb/member_${i}/ensemble_preds_test.csv")
done

RUN_ENV python scripts/ensemble_cross_heads.py \
  --inputs "${INPUTS[@]}" \
  --split test \
  --outdir "$RUN_ROOT/ensembles/${model}_${SUBSET_TAG}_crossmember_singleseed_mean" \
  --blend mean

echo
echo "[$(date)] Done (${dataset} ${split_type} ${SUBSET_TAG})"
echo "Results in: $RUN_ROOT/ensembles/${model}_${SUBSET_TAG}_crossmember_singleseed_mean"
