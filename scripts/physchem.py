"""RDKit-based physicochemical descriptor helpers for tabular baselines.

The main entry point is `add_physchem_descriptors`, which augments a DataFrame
containing SMILES strings with a fixed set of handcrafted descriptors used in
the repo's non-neural baseline models.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors as rdmd
from collections import deque
from typing import Optional, Tuple, List

# Pre-compile SMARTS patterns once so descriptor generation stays cheap.
_METHOXY = Chem.MolFromSmarts("[CH3][O][!$(C=O)]")
_ETHER_O = Chem.MolFromSmarts("[O;X2;!$(O=C)]([C])[C]")
_METHYL_ESTER = Chem.MolFromSmarts("[CX3](=O)[O][CH3]")

def num_conjugated_bonds(mol: Optional[Chem.Mol]) -> Optional[int]:
    """Counts total number of conjugated bonds in the molecule."""
    if mol is None: return None
    return sum(1 for b in mol.GetBonds() if b.GetIsConjugated())

def longest_conjugated_chain(mol: Optional[Chem.Mol]) -> Optional[int]:
    """Approximate longest conjugated chain length using BFS diameter."""
    if mol is None: return None
    adj = {}
    for bond in mol.GetBonds():
        if bond.GetIsConjugated():
            a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adj.setdefault(a1, []).append(a2)
            adj.setdefault(a2, []).append(a1)
    if not adj: return 0

    visited = set()
    max_diameter = 0

    def bfs(start):
        dist = {start: 0}
        q = deque([start])
        while q:
            v = q.popleft()
            for w in adj.get(v, []):
                if w not in dist:
                    dist[w] = dist[v] + 1
                    q.append(w)
        farthest = max(dist.items(), key=lambda t: t[1])[0]
        return farthest, dist

    for node in list(adj.keys()):
        if node in visited: continue
        comp_nodes = set([node])
        q_comp = deque([node])
        visited.add(node)
        while q_comp:
            v = q_comp.popleft()
            for w in adj.get(v, []):
                if w not in visited:
                    visited.add(w); comp_nodes.add(w); q_comp.append(w)
        start = next(iter(comp_nodes))
        far, _ = bfs(start)
        _, dist2 = bfs(far)
        if dist2:
            max_diameter = max(max_diameter, max(dist2.values()))
    return int(max_diameter)

def count_methoxy(mol: Optional[Chem.Mol]) -> Optional[int]:
    return len(mol.GetSubstructMatches(_METHOXY)) if mol else None

def count_ether_oxygens(mol: Optional[Chem.Mol]) -> Optional[int]:
    return len(mol.GetSubstructMatches(_ETHER_O)) if mol else None

def count_methyl_esters(mol: Optional[Chem.Mol]) -> Optional[int]:
    return len(mol.GetSubstructMatches(_METHYL_ESTER)) if mol else None

def add_physchem_descriptors(df: pd.DataFrame, smiles_col: str = "smiles", verbose: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Adds RDKit physicochemical and conjugation descriptors to a DataFrame.
    Returns (modified_df, list_of_new_column_names).
    """
    if smiles_col not in df.columns:
        raise ValueError(f"DataFrame must contain a '{smiles_col}' column.")

    df = df.copy()
    if verbose: print(f"Processing {len(df)} molecules...")

    # Build RDKit molecule objects once and reuse them across descriptor calls.
    df["_mol_temp"] = df[smiles_col].apply(
        lambda smi: Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    )

    descriptor_funcs = {
        "NumAromaticRings":        rdmd.CalcNumAromaticRings,
        "FractionCSP3":            rdmd.CalcFractionCSP3,
        "NumConjugatedBonds":      num_conjugated_bonds,
        "LongestConjDiameterBonds":        longest_conjugated_chain,
        "TPSA":                    rdmd.CalcTPSA,
        "NumHBA":                  rdmd.CalcNumHBA,
        "NumHBD":                  rdmd.CalcNumHBD,
        "ExactMolWt":              Descriptors.ExactMolWt,
        "HeavyAtomCount":          Descriptors.HeavyAtomCount,
        "NumRotatableBonds":       Descriptors.NumRotatableBonds,
        "NumEtherO":               count_ether_oxygens,
        "NumMethoxy":              count_methoxy,
        "NumMethylEster":          count_methyl_esters,
    }

    for name, fn in descriptor_funcs.items():
        if verbose: print(f"  - Computing {name}")
        df[name] = df["_mol_temp"].apply(fn)
        
    df = df.drop(columns=["_mol_temp"])
    return df, list(descriptor_funcs.keys())
