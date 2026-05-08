"""
Add RDKit-based molecular features for UVVis hybrid experiments.

Usage:
pc = physchem_dataframe(df, smiles_col="smiles", key_col="smiles")
fp = morgan_fp_dataframe(df, smiles_col="smiles", key_col="smiles",
                         radius=3, nbits=2048)
"""

from __future__ import annotations
from typing import Iterable, List
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import rdMolDescriptors as rdMD
from rdkit import DataStructs
from rdkit.Chem import Lipinski
from rdkit.Chem import Crippen


PHYSCHM_COLS = [
    "pc_MolWt",
    "pc_MolLogP",
    "pc_TPSA",
    "pc_HBA",
    "pc_HBD",
    "pc_AromRings",
    "pc_AliphRings",
    "pc_RotBonds",
    "pc_FractionCSP3",
]

def _get_morgan_gen(radius: int, nbits: int):
    return GetMorganGenerator(radius=radius, fpSize=nbits)

def _mol_from_smiles(smiles: str) -> Chem.Mol | None:
    """Construct an RDKit Mol; returns None if parsing fails."""
    return Chem.MolFromSmiles(smiles)


def physchem_from_mol(mol: Chem.Mol) -> dict:
    """Compute 9 standard physchem descriptors from an RDKit Mol."""
    return {
        "pc_MolWt": rdMD.CalcExactMolWt(mol),
        "pc_MolLogP": Crippen.MolLogP(mol),
        "pc_TPSA": rdMD.CalcTPSA(mol),
        "pc_HBA": Lipinski.NumHAcceptors(mol),
        "pc_HBD": Lipinski.NumHDonors(mol),
        "pc_AromRings": rdMD.CalcNumAromaticRings(mol),
        "pc_AliphRings": rdMD.CalcNumAliphaticRings(mol),
        "pc_RotBonds": Lipinski.NumRotatableBonds(mol),
        "pc_FractionCSP3": rdMD.CalcFractionCSP3(mol),
    }


def physchem_dataframe(df: pd.DataFrame,
                       smiles_col: str = "smiles",
                       key_col: str = "smiles") -> pd.DataFrame:
    """
    Compute physchem features for each row and return [key_col + physchem].
    """
    smiles = df[smiles_col].astype(str).tolist()
    mols = [_mol_from_smiles(s) for s in smiles]
    rows = [physchem_from_mol(m) for m in mols]

    out = pd.DataFrame(rows)
    out.insert(0, key_col, df[key_col].values)
    return out


def morgan_fp_bits_from_mol(mol, radius=2, nbits=2048) -> np.ndarray:
    gen = _get_morgan_gen(radius, nbits)
    bv = gen.GetFingerprint(mol)           # ExplicitBitVect
    arr = np.zeros((nbits,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr


def morgan_fp_dataframe(df: pd.DataFrame,
                        smiles_col: str = "smiles",
                        key_col: str = "smiles",
                        radius: int = 2,
                        nbits: int = 2048,
                        prefix: str = "fp_") -> pd.DataFrame:
    """
    Compute Morgan fingerprints and return as a wide DataFrame.

    Returns columns:
        [key_col, fp_0, fp_1, ..., fp_{nbits-1}]
    """
    smiles = df[smiles_col].astype(str).tolist()
    mols = [_mol_from_smiles(s) for s in smiles]

    fp_mat = np.stack([morgan_fp_bits_from_mol(m, radius, nbits) for m in mols], axis=0)
    colnames = [f"{prefix}{i}" for i in range(nbits)]

    out = pd.DataFrame(fp_mat, columns=colnames)
    out.insert(0, key_col, df[key_col].values)
    return out


def assemble_extras(df: pd.DataFrame,
                    include_physchem: bool = True,
                    include_fp: bool = False,
                    smiles_col: str = "smiles",
                    key_col: str = "smiles",
                    fp_radius: int = 2,
                    fp_nbits: int = 2048) -> pd.DataFrame:
    """
    Build a combined extras frame keyed by `key_col`.
    """
    parts: List[pd.DataFrame] = []

    if include_physchem:
        parts.append(physchem_dataframe(df, smiles_col=smiles_col, key_col=key_col))

    if include_fp:
        parts.append(
            morgan_fp_dataframe(
                df,
                smiles_col=smiles_col,
                key_col=key_col,
                radius=fp_radius,
                nbits=fp_nbits,
            )
        )

    if not parts:
        return df[[key_col]].copy()

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=key_col, how="inner")
    return out
