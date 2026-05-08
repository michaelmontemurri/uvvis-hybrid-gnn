#!/usr/bin/env bash

# Collect prediction artifacts used during figure assembly.
# This script can:
# 1. run Chemprop prediction for each ensemble member,
# 2. collate those member-level predictions into one CSV, and
# 3. optionally collect saved seed-level predictions for tabular baselines.

DATASET="deep4chem"   # deep4chem | chemfluor | dsscdb | jeffries
SPLIT="scaffold"     # random | scaffold
DATA_SOURCE="uvvisml"
TARGET="em"
TARGET_COL="empeakwavs_max"
SEED_TAG="s42"
GPU=0

N_MEMBERS=5          # Chemprop ensemble size
N_SEEDS_FP=30        # Number of saved tabular baseline seed runs


get_n_list() {
  local dataset="$1"
  case "$dataset" in
    deep4chem) echo "N00050 N00100 N00250 N01000 N04000 N11817" ;;
    chemfluor) echo "N00050 N00100 N00250 N01000 N03162" ;;
    dsscdb)    echo "N00050 N00100 N00250 N00552" ;; 
    *) echo "Unknown DATASET: $dataset" >&2; exit 1 ;;
  esac
}

N_LIST="$(get_n_list "${DATASET}")"

echo "[config] DATASET=${DATASET} SPLIT=${SPLIT} N_LIST=${N_LIST}"

# 1) Generate per-member Chemprop predictions for each subset.
run_chemprop_predicts() {
  for NSTR in ${N_LIST}; do
    DATA_DIR="data/${DATA_SOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${NSTR}_${SEED_TAG}"
    CKPT_DIR="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/fromscratch_${NSTR}_${SEED_TAG}/fold_0"
    OUT_ROOT="checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/fromscratch_${NSTR}_${SEED_TAG}"
    OUT_PRED_DIR="${OUT_ROOT}/preds"

    mkdir -p "${OUT_PRED_DIR}"

    for i in $(seq 0 $((N_MEMBERS-1))); do
      echo "[chemprop_predict] ${DATASET} ${SPLIT} ${NSTR} model_${i}"
      chemprop_predict \
        --test_path "${DATA_DIR}/smiles_target_test.csv" \
        --features_path "${DATA_DIR}/features_test.csv" \
        --checkpoint_path "${CKPT_DIR}/model_${i}/model.pt" \
        --preds_path "${OUT_PRED_DIR}/preds_test_model_${i}.csv" \
        --gpu "${GPU}"
    done
  done
}

# 2) Assemble those per-member predictions into one summary CSV.
collect_chemprop_ensembles() {
  for NSTR in ${N_LIST}; do
    echo "[collect_ensemble] ${DATASET} ${SPLIT} ${NSTR}"
    python scripts/collect_chemprop_ensemble_preds.py \
      --data-dir "data/${DATA_SOURCE}/${TARGET}/${DATASET}/subsets/${SPLIT}/${NSTR}_${SEED_TAG}" \
      --target-col "${TARGET_COL}" \
      --pred-dir "checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/fromscratch_${NSTR}_${SEED_TAG}/preds" \
      --n-members "${N_MEMBERS}" \
      --out "checkpoints/${TARGET}/${DATASET}/${SPLIT}/morgan_fingerprint/fromscratch_${NSTR}_${SEED_TAG}/preds_test_allmodels.csv"
  done
}

# 3) Optionally collect saved seed-level predictions for tabular baselines.
collect_fp_baselines() {
  for NSTR in ${N_LIST}; do
    base="results/${TARGET}/phase1C/${DATASET}/${SPLIT}/baselines/fp_baseline_${NSTR}_${SEED_TAG}"

    if [[ -d "${base}/xgb" ]]; then
      echo "[collect_fp_seeds] ${base}/xgb"
      python scripts/collect_fp_seed_preds.py \
        --xgb-dir "${base}/xgb" \
        --n-seeds "${N_SEEDS_FP}" \
        --out-name "preds_test_allseeds.csv"
    fi

    if [[ -d "${base}_with_physchem/xgb_with_physchem" ]]; then
      echo "[collect_fp_seeds] ${base}_with_physchem/xgb_with_physchem"
      python scripts/collect_fp_seed_preds.py \
        --xgb-dir "${base}_with_physchem/xgb_with_physchem" \
        --n-seeds "${N_SEEDS_FP}" \
        --out-name "preds_test_allseeds.csv"
    fi
  done
}


run_chemprop_predicts
collect_chemprop_ensembles
collect_fp_baselines          
