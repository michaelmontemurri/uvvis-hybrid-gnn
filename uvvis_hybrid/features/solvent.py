"""
Solvent feature utilities for tabular UV-Vis models.

This module loads Greenman et al. solvent descriptor tables and joins
solvent properties onto dataset rows. It also includes small helpers for
scaling and assembling solvent feature matrices.

This is mainly useful for CGSD/ChemFluor or Minnesota descriptor-based
experiments. In our workflow, we use solvent
fingerprints instead, so this module is kept primarily for when we reproduce
Greenman et al. results. The CGSD and Minnesota descriptors need scaling.

Usage:
lookup = load_solvent_table(
    solvent_descriptor='chemfluor_cgsd',  # or 'mn' for minnesota
    solvent_tables_path='thirdparty/greenman/data_solvents',
    key_col='solvent'
)
solv_feats = build_solvent_features(df, key_col='solvent', lookup=lookup)
from uvvis_hybrid.features.scaler import ZScaler
scaler = ZScaler.fit(solv_feats, [c for c in solv_feats.columns if c != 'solvent'])
solv_scaled = scaler.apply(solv_feats)
"""

from __future__ import annotations
from os import path
from typing import Iterable, List
import pandas as pd
from uvvis_hybrid.features.scaler import ZScaler


def load_solvent_table(
    solvent_descriptor: str,  # either 'chemfluor_cgsd' or 'mn'
    solvent_tables_path: str,
    key_col: str,
) -> pd.DataFrame:
    """
    Load the solvent descriptor table from the Greenman data directory.

    Parameters
    ----------
    solvent_descriptor : str
        Which table to load: 'chemfluor_cgsd' or 'mn'.
    solvent_tables_path : str
        Path to 'thirdparty/greenman/data_solvents'.
    key_col : str
        Column used as solvent key (usually 'solvent').

    Returns
    -------
    DataFrame with columns [key_col, <descriptor columns>].
    """
    if solvent_descriptor in ("chemfluor_cgsd", "chemfluor"):
        solvent_descriptor = "chemfluor_cgsd"
        df = pd.read_csv(path.join(solvent_tables_path, f"{solvent_descriptor}_solvent_db.csv"))
    elif solvent_descriptor in ("mn", "minnesota", "minnesota_descriptors"):
        solvent_descriptor = "mn"
        df = pd.read_csv(path.join(solvent_tables_path, f"{solvent_descriptor}_solvent_db.csv"))

    if solvent_descriptor == "chemfluor_cgsd":
        feature_cols = ["ET(30) (kcal mol-1)", "SP", "SdP", "SA", "SB"]
    elif solvent_descriptor == "mn":
        feature_cols = ["eps", "n", "alpha", "beta", "gamma", "phi**2", "psi**2"]
    else:
        raise ValueError("solvent_descriptor must be 'chemfluor_cgsd' or 'mn'")

    cols = [key_col] + feature_cols
    df = df.loc[:, cols].drop_duplicates(subset=[key_col]).reset_index(drop=True)
    return df


def build_solvent_features(
    df: pd.DataFrame,
    left_key: str,
    lookup: pd.DataFrame,
    right_key: str | None = None,
    feature_cols: Iterable[str] | None = None,
    prefix: str = "solv_",
) -> pd.DataFrame:
    """
    Left-join solvent descriptors from `lookup` onto `df`.

    Parameters
    ----------
    df : DataFrame
        Source data containing the join key on the left (e.g., 'smiles' or 'solvent').
    left_key : str
        Column in `df` to join on.
    lookup : DataFrame
        Solvent descriptor lookup table.
    right_key : str, optional
        Column in `lookup` to join on. Defaults to `left_key`.
    feature_cols : Iterable[str], optional
        Descriptor columns to bring from `lookup`; if None, uses all columns except `right_key`.
    prefix : str
        Prefix applied to descriptor columns in the merged output.

    Returns
    -------
    DataFrame with columns:
        [left_key, f"{prefix}{feat1}", f"{prefix}{feat2}", ...]
    """

    # Defaults
    if right_key is None:
        right_key = left_key
    if right_key not in lookup.columns:
        raise KeyError(f"right_key '{right_key}' not found in lookup.columns: {list(lookup.columns)}")

    if feature_cols is None:
        feature_cols = [c for c in lookup.columns if c != right_key]
        if not feature_cols:
            raise ValueError(
                f"No descriptor columns to join: lookup contains only the right_key '{right_key}'."
            )

    # Build a lookup with prefixed columns
    lk = lookup[[right_key] + list(feature_cols)].copy()
    lk.columns = [right_key] + [f"{prefix}{c}" for c in feature_cols]

    merged = df.merge(lk, left_on=left_key, right_on=right_key, how="left")

    # If keys differ, drop the duplicate right_key column after merge
    if right_key != left_key and right_key in merged.columns:
        merged = merged.drop(columns=[right_key])

    return merged


def fit_solvent_scaler(train_df: pd.DataFrame, feature_cols: Iterable[str]) -> ZScaler:
    """
    Fit a ZScaler on solvent descriptor columns (TRAIN ONLY).
    Returns a reusable ZScaler instance.
    """
    return ZScaler.fit(train_df, feature_cols)


def impute_missing_with_train_means(df: pd.DataFrame, scaler: ZScaler) -> pd.DataFrame:
    """
    Impute NaNs in solvent columns using TRAIN means stored in the ZScaler.
    """
    return scaler.impute_means(df)

#Helpers
def assemble_tabular_matrix(
    embeddings: pd.DataFrame,
    extras: pd.DataFrame,
    key_col: str,
) -> pd.DataFrame:
    """
    Join embedding vectors with solvent features by a shared key.

    Returns DataFrame columns:
        [key_col, emb_*, solv_*]
    """
    merged = embeddings.merge(extras, on=key_col, how="inner")
    return merged


def select_dense_and_binary_columns(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """
    Split columns into dense (float) vs. binary (0/1) features.
    Use to decide which to scale before tabular training.
    """
    dense = []
    binary = []
    for c in df.columns:
        if c.startswith(("emb_", "solv_", "pc_")):
            dense.append(c)
        elif c.startswith("fp_"):
            binary.append(c)
    return dense, binary
