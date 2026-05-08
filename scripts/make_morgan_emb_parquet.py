#!/usr/bin/env python3
"""Encode a SMILES CSV as a Morgan-fingerprint parquet file.

This script is mainly used as a lightweight baseline generator so tabular
models can consume fixed-size molecular fingerprints through the same embedding
interface used by learned representations.
"""

import argparse, pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from pathlib import Path

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def morgan_fp_bits(smiles: str, radius=2, n_bits=2048):
    """Convert one SMILES string into a float32 Morgan bit vector."""
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int32)
    DataStructs.ConvertToNumpyArray(bv, arr)
    return arr.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-bits", type=int, default=2048)
    ap.add_argument("--radius", type=int, default=2)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    smiles = df[args.smiles_col].astype(str).tolist()
    fps = np.vstack([morgan_fp_bits(s, args.radius, args.n_bits) for s in smiles])
    emb_cols = [f"emb_{i}" for i in range(args.n_bits)]
    out_df = pd.DataFrame(fps, columns=emb_cols)
    # Keep the original SMILES column so downstream merges preserve row alignment.
    out_df.insert(0, args.smiles_col, smiles)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out, index=False)
    print(f"[done] wrote {args.out} with shape {out_df.shape}")

if __name__ == "__main__":
    main()
