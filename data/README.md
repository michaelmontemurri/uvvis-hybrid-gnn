# Data

This folder contains the public data artifacts for the paper release.

We keep the frozen base tables that the experiments start from, and we keep the compact metadata needed to reconstruct the exact train/validation/test splits and fixed-size training subsets used in the paper. We do not track all of the derived split CSVs.

## What's included

### `processed/`

This directory contains the processed UVVisML master tables used as the starting point for the benchmark experiments.

These files were copied from the Greenman et al. 2022 UVVisML workflow after applying their processing pipeline. They are Greenman et al.'s compiled and processed versions of the Deep4Chem, ChemFluor, and DSSCDB datasets, but trimmed down for the public artifact to the fields used here: identifiers, target values, source labels, the `homo`/`lumo` TD-DFT descriptors, and solvent fingerprint (`sfp*`) columns. The rest of the larger TD-DFT block is intentionally omitted. These are not raw source exports; they are the cleaned master CSVs that this repository actually works from.

The two canonical target columns are:

- absorption: `peakwavs_max`
- emission: `empeakwavs_max`

### `custom/`

This directory contains the project-specific CSVs derived from the publicly available Jeffries-EL dataset.

These are the local base tables used for the transfer and external-evaluation experiments in this repository. They are included so the experiments can be reproduced from the same starting point without requiring users to redo the Jeffries-EL conversion step themselves.

### `uvvisml/`

This directory now stores compact split metadata for the UVVisML experiments rather than fully materialized split datasets.

For each target, dataset, and split strategy, we track:

- `train_idx.txt`, `val_idx.txt`, `test_idx.txt`: the most compact split definition, stored as row indices into the fully processed dataframe immediately before splitting
- `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`: the same split membership written in a more readable form, with `row_idx`, `smiles`, `solvent`, and the target value

The index files and manifest CSVs are intentionally redundant. The `.txt` files are the compact and rely on the canonical row ordering of the processed dataframe. The manifest CSVs make the split contents easy to inspect and sanity-check against the underlying data without completely materializing the full derived split trees.

For each fixed-size training subset, we track:

- `indices_train.txt`: the selected training rows
- `train_manifest.csv`: a compact manifest for that subset
- `meta.json`: subset metadata

### split metadata under `custom/`

The Jeffries-EL-derived datasets follow the same pattern.

Alongside each base CSV, the repository tracks canonical split metadata and fixed-subset metadata so that the transfer experiments can be rebuilt deterministically from the frozen base tables.

For the Jeffries-EL data, the absorption measurements are TD-DFT computed, whereas the emission wavelengths are experimental. We only really consider the emission measurments for this reason. In order to allow fair comparison to the method reported by Tannir et al. 2024, in the data/custom/em/jeffries/splits/holdout we force the test set to be the 4 molecules for which they reported their holdout set results on. 

### `metadata/feature_names.json`

This file defines the feature columns expected by the data preparation scripts. 

## Reproducing Results

If you want to reproduce the paper data pipeline, the intended flow is:

1. Start from the frozen base CSVs in `processed/` or `custom/`.
2. Use the tracked split and subset metadata to recover the exact train/validation/test membership.
3. Run the provided scripts to materialize any derived split files needed by a training or analysis step.