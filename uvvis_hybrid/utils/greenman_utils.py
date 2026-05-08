"""
Vendored Greenman-compatible preprocessing utilities used by this artifact repo.

This module contains a minimal subset of helper functions adapted from the
Greenman codebase while preserving the original preprocessing behavior
used for dataset preparation and splitting.

Included here only are the functions needed for:
- duplicate handling / aggregation,
- Morgan fingerprint generation, and
- train/validation/test dataset splitting.

These utilities were copied and minimally adapted for local use in the
artifact repository. They should remain behaviorally aligned with the
original pipeline unless explicitly documented otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from rdkit.Chem import MolFromSmiles
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.DataStructs import ConvertToNumpyArray

from .greenman_scaffold_splits import scaffold_split


def _write_split_metadata(
    split_name: str,
    split_df: pd.DataFrame,
    target_names: list[str],
    manifest_columns: list[str] | None = None,
) -> None:
    """Write compact metadata for a split relative to the processed pre-split dataframe.

    The saved `row_idx` values refer to the row positions of the fully processed
    dataframe immediately before train/val/test splitting.
    """
    row_idx = split_df.index.to_numpy(dtype=int, copy=True)
    np.savetxt(f"{split_name}_idx.txt", row_idx, fmt="%d")

    manifest_cols = ["smiles", "solvent"] + list(target_names)
    if manifest_columns:
        manifest_cols = list(manifest_columns)

    missing = [col for col in manifest_cols if col not in split_df.columns]
    if missing:
        raise KeyError(f"Missing manifest columns for split '{split_name}': {missing}")

    manifest = split_df.loc[:, manifest_cols].copy()
    manifest.insert(0, "row_idx", row_idx)
    manifest.to_csv(f"{split_name}_manifest.csv", index=False)


def get_morgan_fingerprints(df, mol_or_solv="molecules", nbits=None):
    """Gets Morgan fingerprints for a dataframe with SMILES columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A DataFrame that contains columns 'smiles' and/or 'solvent'.
    mol_or_solv : {"molecules", "solvents"}
        Which column to featurize.
    nbits : int or None
        Number of fingerprint bits. Defaults to 1024 for molecules and
        256 for solvents.

    Returns
    -------
    df : pandas.DataFrame
        Updated DataFrame including one fingerprint bit per column.
    col_names_list : list[str]
        Names of the generated fingerprint columns.
    """
    if mol_or_solv == "molecules":
        radius = 4
        nbits = 1024 if nbits is None else nbits
        col_name = "mfp"
        smiles_col = "smiles"
    elif mol_or_solv == "solvents":
        radius = 4
        nbits = 256 if nbits is None else nbits
        col_name = "sfp"
        smiles_col = "solvent"
    else:
        raise ValueError("mol_or_solv must be either 'molecules' or 'solvents'")

    unique_df = df.drop_duplicates(subset=[smiles_col])[[smiles_col]].copy()
    smiles_list = list(unique_df[smiles_col])
    generator = GetMorganGenerator(radius=radius, fpSize=nbits)

    fps = []
    for smi in smiles_list:
        mol = MolFromSmiles(smi)
        arr = np.zeros((nbits,), dtype=np.int8)
        fp = generator.GetFingerprint(mol)
        ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    col_names_list = [f"{col_name}{i}" for i in range(1, nbits + 1)]
    fp_df = pd.DataFrame(fps, columns=col_names_list, index=unique_df.index)
    unique_df = pd.concat([unique_df, fp_df], axis=1)

    df = df.merge(unique_df, on=smiles_col)

    return df, col_names_list


def handle_duplicates(df, target_col="empeakwavs_max", cutoff=5, agg_source_col="multiple"):
    """Aggregates duplicate measurements in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with required columns including:
        'smiles', 'solvent', target_col, and optionally 'source'.
    target_col : str
        Name of the wavelength target column, e.g. 'peakwavs_max' or
        'empeakwavs_max'.
    cutoff : int or float
        Wavelength cutoff in nm. Duplicate measurements of the same
        smiles-solvent pair with standard deviation less than cutoff
        are averaged. Those with standard deviation greater than cutoff
        are dropped.
    agg_source_col : {"multiple", "random"}
        How to aggregate the source column when duplicates are present.

    Returns
    -------
    pandas.DataFrame
        Updated DataFrame with duplicates aggregated or removed.
    """
    required_cols = {"smiles", "solvent", target_col}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise KeyError(f"Missing required columns for duplicate handling: {sorted(missing_required)}")

    cols_to_average = [
        col for col in df.columns
        if col not in {"smiles", "solvent", target_col, "source"}
    ]

    agg_dict = {target_col: ["mean", "std"]}

    if "source" in df.columns:
        if agg_source_col == "multiple":
            agg_dict["source"] = lambda x: "multiple" if len(x) > 1 else x.iloc[0]
        elif agg_source_col == "random":
            rng = np.random.default_rng(seed=0)
            agg_dict["source"] = lambda x: rng.choice(x.to_numpy())
        else:
            raise ValueError("agg_source_col must be either 'multiple' or 'random'")

    for col in cols_to_average:
        agg_dict[col] = "mean"

    grouped = df.groupby(["smiles", "solvent"]).agg(agg_dict).reset_index()

    high_std_idx = grouped[grouped[target_col]["std"] > cutoff].index
    grouped = grouped.drop(index=high_std_idx)

    grouped = grouped.drop(columns="std", level=1)
    grouped.columns = grouped.columns.get_level_values(0)

    return grouped


def data_split_and_write(
    X,
    feature_names=None,
    target_names=None,
    solvation=False,
    sizes=(0.8, 0.1, 0.1),
    split_type="scaffold",
    scale_targets=False,
    write_files=False,
    random_seed=0,
    write_split_metadata=False,
):
    """Writes train/val/test CSV files for Chemprop with a given dataset.

    Parameters
    ----------
    X : pandas.DataFrame
        DataFrame to be split. Must contain 'smiles', target_names, optional
        feature_names, and 'solvent' if solvation=True.
    feature_names : list[str] or None
        Feature columns to write to feature CSVs.
    target_names : list[str] or None
        Target columns to include in target CSVs.
    solvation : bool
        Whether to include the solvent column in target CSVs.
    sizes : tuple[float, float, float]
        Train/val/test fractions.
    split_type : {"scaffold", "group_by_smiles", "random"}
        Splitting strategy.
    scale_targets : bool
        Whether to z-score targets using the training split statistics.
    write_files : bool
        Whether to write the resulting splits to CSVs.
    random_seed : int
        Random seed for reproducible splitting.
    write_split_metadata : bool
        Whether to write compact split metadata (`*_idx.txt` and `*_manifest.csv`)
        alongside the split CSVs. The saved indices refer to the processed
        dataframe `X` in its final row order immediately before splitting.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
        Train, validation, and test splits.
    """
    if target_names is None:
        target_names = ["empeakwavs_max"]

    if split_type == "scaffold":
        X_train, X_val, X_test = scaffold_split(X, sizes=sizes, balanced=True, seed=random_seed)
    elif split_type == "group_by_smiles":
        gss1 = GroupShuffleSplit(n_splits=2, train_size=sizes[0], random_state=random_seed)
        train_idx, temp_idx = list(gss1.split(X, groups=X["smiles"]))[0]
        X_train, X_temp = X.iloc[train_idx, :].copy(), X.iloc[temp_idx, :].copy()

        relative_val_size = sizes[1] / (sizes[1] + sizes[2])
        gss2 = GroupShuffleSplit(n_splits=2, train_size=relative_val_size, random_state=random_seed)
        val_idx, test_idx = list(gss2.split(X_temp, groups=X_temp["smiles"]))[0]
        X_val, X_test = X_temp.iloc[val_idx, :].copy(), X_temp.iloc[test_idx, :].copy()
    elif split_type == "random":
        X_train = X.sample(frac=sizes[0], random_state=random_seed).copy()
        X_temp = X.drop(X_train.index).copy()
        val_frac = sizes[1] / (sizes[1] + sizes[2])
        X_val = X_temp.sample(frac=val_frac, random_state=random_seed).copy()
        X_test = X_temp.drop(X_val.index).copy()
    else:
        raise ValueError("split_type must be one of: 'scaffold', 'group_by_smiles', 'random'")

    if scale_targets:
        scaler = StandardScaler()
        scaler.fit(X_train[target_names])
        X_train.loc[:, target_names] = scaler.transform(X_train[target_names])
        X_val.loc[:, target_names] = scaler.transform(X_val[target_names])
        X_test.loc[:, target_names] = scaler.transform(X_test[target_names])

    if write_files:
        train_target_file = "smiles_target_train.csv"
        val_target_file = "smiles_target_val.csv"
        test_target_file = "smiles_target_test.csv"
        train_features_file = "features_train.csv"
        val_features_file = "features_val.csv"
        test_features_file = "features_test.csv"

        if solvation:
            X_train[["smiles", "solvent"] + target_names].to_csv(train_target_file, index=False)
            X_val[["smiles", "solvent"] + target_names].to_csv(val_target_file, index=False)
            X_test[["smiles", "solvent"] + target_names].to_csv(test_target_file, index=False)
        else:
            X_train[["smiles"] + target_names].to_csv(train_target_file, index=False)
            X_val[["smiles"] + target_names].to_csv(val_target_file, index=False)
            X_test[["smiles"] + target_names].to_csv(test_target_file, index=False)

        if feature_names:
            X_train[feature_names].to_csv(train_features_file, index=False)
            X_val[feature_names].to_csv(val_features_file, index=False)
            X_test[feature_names].to_csv(test_features_file, index=False)

    if write_split_metadata:
        _write_split_metadata("train", X_train, target_names)
        _write_split_metadata("val", X_val, target_names)
        _write_split_metadata("test", X_test, target_names)

    return X_train, X_val, X_test
