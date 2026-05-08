# Hybrid GNN-Tree Models for Optical Property Prediction

Artifact repository for the paper:

**"Hybrid GNN-Tree Models for Low-Data Optical Property Prediction"**

This repo contains:

- processed public data artifacts
- curated pretrained Deep4Chem Chemprop checkpoints
- scripts for frozen-encoder, finetuning, and transfer workflows
- cluster batch jobs for paper reproduction
- tracked figures and notebooks

The main empirical result is the frozen D-MPNN encoder + XGBoost workflow,
which is consistently strong in low-to-mid data scaffold-split regimes.

## Key Result

- On the tracked absorption benchmarks, frozen D-MPNN encoder + XGBoost improves performance in small-N settings.
- Transfer performance is dataset-dependent. Deep4Chem pretraining helps most
  when the target dataset is chemically similar to Deep4Chem.
- The repo includes similarity tooling so users can check that before choosing
  between frozen reuse, finetuning, or scratch training.

![Abs N vs RMSE](figures/abs/abs_multipanel_bootstrap_ci.png)

## Quickstart

For the main public reuse path:

1. Create the environment.
2. Materialize train/val/test split CSVs.
3. Run the dataset similarity check.
4. Start with the pretrained frozen-encoder hybrid workflow.

### Environment

Tracked environments:

- Linux / CUDA: [envs/uvvisml.yml](envs/uvvisml.yml)
- macOS: [envs/uvvisml-macos.yml](envs/uvvisml-macos.yml)

```bash
conda env create -f envs/uvvisml.yml
conda activate uvvisml
```

### Tracked Pretrained Bundles

The curated public bundles are:

- [`checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42`](checkpoints/abs/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11816_s42)
- [`checkpoints/em/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11502_s42`](checkpoints/em/deep4chem/scaffold/morgan_fingerprint/fromscratch_N11502_s42)

Each tracked bundle includes:

- `args.json`
- `fold_0/model_i/model.pt` for `i = 0..4`

See:

- [`pretrained/README.md`](pretrained/README.md)
- [`pretrained/deep4chem_ensemble_manifest.json`](pretrained/deep4chem_ensemble_manifest.json)

## Repository Layout

```text
uvvis-benchmark/
├── batch_jobs/      # Cluster workflows
├── data/            # Public processed data artifacts and split metadata
├── envs/            # Conda environment files
├── figures/         # Tracked figures
├── notebooks/       # Notebooks for reproducing results
├── pretrained/      # Metadata for curated checkpoint bundles
├── scripts/         # Scripts and helpers
├── uvvis_hybrid/    # Python package and CLIs
└── checkpoints/     # Local checkpoint cache
```

## Data

- [`data/processed`](data/processed)
- [`data/custom`](data/custom)
- [`data/uvvisml`](data/uvvisml)
- [`data/README.md`](data/README.md)

The tracked split trees are metadata-first. Need to materialize splits before using them as Chemprop inputs:

```bash
python scripts/materialize_uvvisml_splits.py \
  --target abs \
  --dataset deep4chem \
  --split-type scaffold
```

## External Dataset Workflow

### 1. Prepare split CSVs

Prepare:

- `smiles_target_train.csv`
- `smiles_target_val.csv`
- `smiles_target_test.csv`

At minimum these should contain:

- `smiles`
- one target column such as `peakwavs_max`

If you want the default solvent fingerprint path, also include:

- `solvent` as solvent SMILES

For Chemprop finetuning, also provide:

- `features_train.csv`
- `features_val.csv`
- `features_test.csv`

### 2. Check dataset similarity

```bash
python scripts/check_dataset_similarity.py \
  --query-csv /path/to/your/smiles_target_train.csv \
  --query-smiles-col smiles \
  --reference-target abs \
  --outdir local_runs/dataset_similarity
```

If the dataset looks close to Deep4Chem, start with the frozen path. If it
looks shifted, test finetuning and scratch training as well.

### 3. Run the frozen hybrid workflow

Primary wrapper:

- [`scripts/run_pretrained_hybrid.py`](scripts/run_pretrained_hybrid.py)

Example:

```bash
python scripts/run_pretrained_hybrid.py \
  --manifest-id deep4chem_abs_scaffold_full_ensemble \
  --train-csv /path/to/your/smiles_target_train.csv \
  --val-csv /path/to/your/smiles_target_val.csv \
  --test-csv /path/to/your/smiles_target_test.csv \
  --target-col peakwavs_max \
  --outdir local_runs/deep4chem_frozen_xgb \
  --model xgb \
  --device cuda
```

This resolves the checkpoint ensemble, dumps pre-FFN embeddings, tunes one
tabular head family if needed, fits one head per encoder member, and blends the
ensemble.

### 4. Optional finetuning

The cluster-oriented finetuning path is:

- [`batch_jobs/run_transfer_e2e_finetune_ensemble.sbatch`](batch_jobs/run_transfer_e2e_finetune_ensemble.sbatch)

The pretrained transfer baselines are:

- [`batch_jobs/run_transfer_zero_shot_pretrained.sbatch`](batch_jobs/run_transfer_zero_shot_pretrained.sbatch)
- [`batch_jobs/run_transfer_gradual_unfreeze_pretrain_ensemble.sbatch`](batch_jobs/run_transfer_gradual_unfreeze_pretrain_ensemble.sbatch)
- [`batch_jobs/run_transfer_hybrid_on_finetuned_pretrain_ensemble.sbatch`](batch_jobs/run_transfer_hybrid_on_finetuned_pretrain_ensemble.sbatch)

Lower-level utilities:

- [`uvvis_hybrid/scripts/dump_embeddings.py`](uvvis_hybrid/scripts/dump_embeddings.py)
- [`uvvis_hybrid/scripts/fit_tabular.py`](uvvis_hybrid/scripts/fit_tabular.py)

## Paper Reproduction

Main cluster jobs:

- [`batch_jobs/run_e2e_chemprop_subsample_ensemble.sbatch`](batch_jobs/run_e2e_chemprop_subsample_ensemble.sbatch)
- [`batch_jobs/run_xgb_on_learned_embedding_subsets.sbatch`](batch_jobs/run_xgb_on_learned_embedding_subsets.sbatch)
- [`batch_jobs/run_transfer_e2e_finetune_ensemble.sbatch`](batch_jobs/run_transfer_e2e_finetune_ensemble.sbatch)
- [`batch_jobs/run_finetune_xgb_subsampled_ensemble.sbatch`](batch_jobs/run_finetune_xgb_subsampled_ensemble.sbatch)
- [`batch_jobs/run_transfer_zero_shot_pretrained.sbatch`](batch_jobs/run_transfer_zero_shot_pretrained.sbatch)
- [`batch_jobs/run_transfer_gradual_unfreeze_pretrain_ensemble.sbatch`](batch_jobs/run_transfer_gradual_unfreeze_pretrain_ensemble.sbatch)
- [`batch_jobs/run_transfer_hybrid_on_finetuned_pretrain_ensemble.sbatch`](batch_jobs/run_transfer_hybrid_on_finetuned_pretrain_ensemble.sbatch)


## Figures and Notebooks

Tracked notebooks:

- [`notebooks/chemical_space_and_split_similarity.ipynb`](notebooks/chemical_space_and_split_similarity.ipynb)
- [`notebooks/variance_decomposition.ipynb`](notebooks/variance_decomposition.ipynb)
- [`notebooks/counterfactuals.ipynb`](notebooks/counterfactuals.ipynb)
- [`notebooks/latent_space_interpretation.ipynb`](notebooks/latent_space_interpretation.ipynb)

For the frozen-head comparison workflow, see:

- [`scripts/frozen_head_comparison/README.md`](scripts/frozen_head_comparison/README.md)

## Related Work

- Chemprop / D-MPNN: Yang et al. (2019)
- UVVis benchmark data lineage: UVVisML / Greenman et al. (2022)
- XGraphBoost: Deng et al. (2021)
- Jeffries evaluation data: public Jeffries-EL dataset

## License

Released under the MIT License.
