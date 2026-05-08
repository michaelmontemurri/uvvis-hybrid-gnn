"""Simple smoke test for `uvvis_hybrid.features.solvent`.
Runs a small sample from the Deep4Chem dataset.
run this from the head of the repo: python -m uvvis_hybrid.tests.test_solvent_smoke
"""

import pandas as pd
import numpy as np
from uvvis_hybrid.features import solvent
from uvvis_hybrid.features.scaler import ZScaler
def run_smoke_test():
    print("Running solvent.py smoke test on Deep4Chem sample")

    # Load a small sample
    deep4chem_path = "data/processed/uvvis_abs_master_full.csv"
    df_all = pd.read_csv(deep4chem_path, nrows=20000)
    assert "solvent" in df_all.columns
    print(f"Loaded {len(df_all)} rows")

    # Test two descriptor sets
        
    for desc in ["chemfluor_cgsd", "mn"]:
        print(f"Testing {desc}")

        df_lookup = solvent.load_solvent_table(
            solvent_descriptor=desc,
            solvent_tables_path="third_party/greenman/data_solvents",
            key_col="solvent",
        )

        # sample a few solvents (ensures left_key='solvent' path works)
        df_subset_train = df_all[["smiles", "solvent"]].drop_duplicates("solvent").head(5).copy()
        df_subset_val   = df_subset_train.copy()

        # --- Case A: join using solvent→solvent (left_key == right_key)
        solv_feats_train = solvent.build_solvent_features(
            df_subset_train, left_key="solvent", lookup=df_lookup, right_key="solvent"
        )
        solv_feats_val = solvent.build_solvent_features(
            df_subset_val, left_key="solvent", lookup=df_lookup, right_key="solvent"
        )

        feature_cols = [c for c in solv_feats_train.columns if c.startswith("solv_")]
        assert feature_cols, "No solvent descriptor columns found."

        # Inject NaN in VAL, fit scaler on TRAIN, then impute+scale VAL
        solv_feats_val.loc[0, feature_cols[0]] = np.nan
        scaler = ZScaler.fit(solv_feats_train, feature_cols)
        val_imputed_raw = solvent.impute_missing_with_train_means(solv_feats_val, scaler)
        val_scaled = scaler.apply(val_imputed_raw)

        # Check no NaNs remain and imputed cell ~0 after scaling
        assert not val_imputed_raw[feature_cols].isna().any().any()
        assert abs(val_scaled.loc[0, feature_cols[0]]) < 1e-6

        # --- Optional Case B: demonstrate smiles(left) → solvent(right) join
        # Build a smiles→solvent frame and join with right_key='solvent'
        solv_feats_by_smiles = solvent.build_solvent_features(
            df_subset_train[["smiles", "solvent"]],
            left_key="smiles",
            lookup=df_lookup,
            right_key="solvent",
        )
        assert "smiles" in solv_feats_by_smiles.columns
        assert any(c.startswith("solv_") for c in solv_feats_by_smiles.columns)


        print("Unscaled TRAIN solvent features:")
        print(solv_feats_train.head())
        print("VAL solvent features (before impute):")
        print(solv_feats_val.head())
        print("VAL solvent features (after raw impute, then scaled):")
        print(val_scaled.head())



if __name__ == "__main__":
    run_smoke_test()
