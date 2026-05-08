"""
Smoke test for precomputed molecule embeddings.

This script checks that duplicated SMILES in a dataset split map to
identical pre-FFN embedding vectors after merging with a saved embeddings
table. It is mainly used to confirm that repeated molecules receive
consistent embedding features.
"""

import pandas as pd
import numpy as np

# --- config ---
# point to train csv, embedding .parquet to check
SPLIT_CSV = "data/uvvisml/splits/scaffold/chemfluor/train.csv"
EMB_PARQUET = "uvvis_hybrid/embedding_dumps/deep4chem/chemfluor_cgsd/scaffold/chemprop_pre_ffn_emb_train.parquet"

SMILES_COL = "smiles"
TOL = 1e-6  

def _normalize_smiles_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    def _to_str(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            x = x[0] if len(x) else ""
        return str(x)
    out = df.copy()
    out[col] = out[col].apply(_to_str)
    return out

# load base split (keep duplicates)
base = pd.read_csv(SPLIT_CSV, usecols=[SMILES_COL]).astype({SMILES_COL: "object"})

# load embeddings
E = pd.read_parquet(EMB_PARQUET)
E = _normalize_smiles_column(E, SMILES_COL)

# sanity: must have only one row per SMILES in embeddings for pre-ffn
dups_in_E = E.duplicated(SMILES_COL).sum()
print(f"[emb] rows={len(E)}, unique_smiles={E[SMILES_COL].nunique()}, duplicates_in_emb={dups_in_E}")

# merge base -> embeddings (left; keeps all base rows, including duplicates)
M = base.merge(E, on=SMILES_COL, how="left")
emb_cols = [c for c in M.columns if c.startswith("emb_")]
assert emb_cols, "No embedding columns found."

# find duplicated smiles in base
dup_mask = M.duplicated(SMILES_COL, keep=False)
Md = M.loc[dup_mask, [SMILES_COL] + emb_cols].copy()
print(f"[base] duplicated rows: {len(Md)} across {Md[SMILES_COL].nunique()} duplicated SMILES")

# group by smiles, compute the max per-dimension spread; then the max over dims
def max_spread(g: pd.DataFrame) -> float:
    arr = g[emb_cols].to_numpy(dtype=np.float32, copy=False)
    if arr.shape[0] <= 1:
        return 0.0
    spread = np.nanmax(arr, axis=0) - np.nanmin(arr, axis=0)
    return float(np.nanmax(np.abs(spread)))

spreads = Md.groupby(SMILES_COL, sort=False).apply(max_spread)

# report
ok = (spreads <= TOL).sum()
bad = (spreads > TOL).sum()
print(f"[check] duplicated SMILES with identical embeddings (≤ {TOL}): {ok}")
print(f"[check] duplicated SMILES with differing embeddings   (> {TOL}): {bad}")

if bad:
    print("\nTop offenders (smiles -> max_abs_dim_diff):")
    print(spreads[spreads > TOL].sort_values(ascending=False).head(10))

    bad_smiles = spreads[spreads > TOL].index[0]
    print(f"\nExample differing rows for: {bad_smiles}")
    print(Md[Md[SMILES_COL] == bad_smiles].head())
else:
    print("All duplicated SMILES map to identical pre-FFN embeddings (within tolerance).")
