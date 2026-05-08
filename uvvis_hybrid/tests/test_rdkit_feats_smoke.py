"""
Simple smoke test for `uvvis_hybrid.features.rdkit_feats`.

Builds physchem descriptors and Morgan FPs from a small sample of SMILES,
checks that the expected feature columns are produced, and verifies the
train-only impute-and-scale workflow used for dense physchem features.
"""

import numpy as np
import pandas as pd

from uvvis_hybrid.features import rdkit_feats
from uvvis_hybrid.features.scaler import ZScaler

DEEP4CHEM_PATH = "data/processed/uvvis_abs_master_full.csv"

def run_smoke_test():
    print("Running rdkit_feats.py smoke test")

    # load a small sample of SMILES
    df_all = pd.read_csv(DEEP4CHEM_PATH, usecols=["smiles"], nrows=200)
    df_all = df_all.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
    assert len(df_all) > 0, "No SMILES loaded"
    print(f"Loaded {len(df_all)} unique SMILES")

    # Create tiny train/val slices to simulate pipeline flow
    n_train = max(20, int(0.6 * len(df_all)))
    df_train = df_all.iloc[:n_train].copy()
    df_val   = df_all.iloc[n_train:n_train + 20].copy()  # small val slice

    # Physchem features (dense, will be scaled)
    pc_train = rdkit_feats.physchem_dataframe(df_train, smiles_col="smiles", key_col="smiles")
    pc_val   = rdkit_feats.physchem_dataframe(df_val,   smiles_col="smiles", key_col="smiles")

    # Basic physchem checks
    assert "smiles" in pc_train.columns
    for c in rdkit_feats.PHYSCHM_COLS:
        assert c in pc_train.columns, f"Missing physchem column: {c}"
    assert pc_train[rdkit_feats.PHYSCHM_COLS].notna().all().all()

    # Fingerprints (binary, do NOT scale)
    fp_train = rdkit_feats.morgan_fp_dataframe(df_train, smiles_col="smiles", key_col="smiles",
                                               radius=2, nbits=1024)
    fp_val   = rdkit_feats.morgan_fp_dataframe(df_val,   smiles_col="smiles", key_col="smiles",
                                               radius=2, nbits=1024)
    assert "smiles" in fp_train.columns
    fp_cols = [c for c in fp_train.columns if c.startswith("fp_")]
    assert len(fp_cols) == 1024, "Unexpected fingerprint width"
    uniqs = np.unique(fp_train[fp_cols].values)
    assert set(uniqs).issubset({0, 1}), "Fingerprints are not binary {0,1}"

    # Scale ONLY physchem with ZScaler (fit on TRAIN), and test impute scale order
    pc_cols = rdkit_feats.PHYSCHM_COLS
    scaler = ZScaler.fit(pc_train, pc_cols)

    # inject a nan in VAL physchem to test imputation pathway
    if len(pc_cols) > 0 and len(pc_val) > 0:
        pc_val.loc[pc_val.index[0], pc_cols[0]] = None

    pc_val_imputed = scaler.impute_means(pc_val)  #impute in raw space
    pc_train_s = scaler.apply(pc_train)
    pc_val_s = scaler.apply(pc_val_imputed)

    # check that no nans post-impute or post-scale; imputed cell ~0 after scaling
    assert pc_val_imputed[pc_cols].notna().all().all(), "Imputation failed"
    assert pc_train_s[pc_cols].notna().all().all(), "NaNs after scaling (train)"
    assert pc_val_s[pc_cols].notna().all().all(), "NaNs after scaling (val)"
    if len(pc_cols) > 0 and len(pc_val_s) > 0:
        assert abs(pc_val_s.iloc[0][pc_cols[0]]) < 1e-6, "Imputed value not centered after scaling"

    # Merge scaled physchem + (unscaled) fingerprints
    extras_train = pc_train_s.merge(fp_train, on="smiles", how="inner")
    extras_val   = pc_val_s.merge(fp_val,     on="smiles", how="inner")

    assert len(extras_train) == len(df_train), "Row loss after merging extras (train)"
    assert len(extras_val)   == len(df_val),   "Row loss after merging extras (val)"
    print("Merged extras shapes:", extras_train.shape, extras_val.shape)

    # distribution check (std ~ 1 for physchem columns with >1 unique value)
    stds_train = extras_train[pc_cols].std(ddof=0)
    ok = stds_train[(extras_train[pc_cols].nunique() > 1)]
    assert (ok > 0.5).all() and (ok < 1.5).all(), "Scaled physchem std out of expected range"
    print("Scaled physchem stats OK (std ~ 1).")


    print("TRAIN extras features:")
    print(extras_train.head())
    print("VAL extras features:")
    print(extras_val.head())

    print("rdkit_feats.py test passed")

if __name__ == "__main__":
    run_smoke_test()
