#!/usr/bin/env python3
"""
Copy the minimal processed UVVisML master CSVs into `data/processed/`.

The artifact repo only needs the canonical identifiers, targets, source label,
HOMO/LUMO values, and solvent fingerprint columns from the Greenman-processed
masters. 
"""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uvvis_hybrid.utils.greenman_utils import get_morgan_fingerprints

GREENMAN_DIR = ROOT / "third_party" / "greenman" / "data_processed"
OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EM_INPUT = GREENMAN_DIR / "em_master_df_all_features.csv"
ABS_INPUT = GREENMAN_DIR / "master_df_all_features.csv"


TARGET_COLUMNS = {
    "uvvis_em_master_full.csv": "empeakwavs_max",
    "uvvis_abs_master_full.csv": "peakwavs_max",
}


def select_output_columns(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    keep_columns = ["smiles", "solvent", target_col, "source", "homo", "lumo"]
    missing = [col for col in keep_columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    sfp_columns = [col for col in df.columns if col.startswith("sfp")]
    if not sfp_columns:
        df, sfp_columns = get_morgan_fingerprints(df.copy(), mol_or_solv="solvents")

    ordered_sfp_columns = sorted(
        sfp_columns,
        key=lambda col: int(col[3:]) if col[3:].isdigit() else col,
    )
    return df.loc[:, keep_columns + ordered_sfp_columns].copy()


def process_file(input_path: Path, output_path: Path) -> None:
    df = pd.read_csv(input_path)
    target_col = TARGET_COLUMNS[output_path.name]
    trimmed = select_output_columns(df, target_col)
    trimmed.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(trimmed)} rows, {len(trimmed.columns)} cols)")


def main() -> None:
    process_file(
        EM_INPUT,
        OUT_DIR / "uvvis_em_master_full.csv",
    )

    process_file(
        ABS_INPUT,
        OUT_DIR / "uvvis_abs_master_full.csv",
    )


if __name__ == "__main__":
    main()
