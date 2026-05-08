#!/usr/bin/env python3
"""
Build the processed Jeffries dataset from the public Excel source.

This script reads the raw Jeffries workbook, trims it to the rows used in
this project, reshapes solvent-specific measurements into long format,
applies the manual CHCl3 holdout corrections used in the study, converts
solvent labels to solvent SMILES, and generates solvent Morgan
fingerprints.

The processed outputs are written under `data/custom/{target}/jeffries/`.
The same long-format table is saved under both the `abs` and `em` target
roots because it contains both `peakwavs_max` and `empeakwavs_max`.
"""


from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from uvvis_hybrid.utils.greenman_utils import get_morgan_fingerprints


RAW_DEFAULT = ROOT / "data" / "raw" / "jeffries.xlsx"
OUT_DEFAULT = ROOT / "data" / "custom"

ROWS_TO_KEEP = 58
HOLDOUT_INDICES = [54, 55, 56, 57] # hardcode to match tannir et al.
HOLDOUT_CHCL3_FIXES = [414, 410, 470, 454]

EMISSION_COLUMNS = [
    "EXPT Lmax EMIS (nm) DMF ",
    "EXPT Lmax EMIS (nm) CHCl3 ",
    "EXPT Lmax EMIS (nm) THF ",
]

SOLVENT_TO_SMILES = {
    "CHCl3": "ClC(Cl)Cl",
    "THF": "C1CCOC1",
    "DMF": "CN(C)C=O",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build processed Jeffries CSVs from the public Excel source."
    )
    parser.add_argument(
        "--input-xlsx",
        type=Path,
        default=RAW_DEFAULT,
        help="Path to the raw Jeffries Excel file.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=OUT_DEFAULT,
        help="Root directory where processed Jeffries CSVs will be written.",
    )
    parser.add_argument(
        "--nbits",
        type=int,
        default=256,
        help="Number of solvent Morgan fingerprint bits to generate.",
    )
    return parser.parse_args()


def load_source_table(input_xlsx: Path) -> pd.DataFrame:
    """Load the raw Jeffries workbook and keep only the rows used in the study."""
    df = pd.read_excel(input_xlsx)
    return df.iloc[:ROWS_TO_KEEP].copy()


def build_long_table(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Reshape the Jeffries worksheet into a solvent-resolved long table."""
    holdout_smiles = df.iloc[HOLDOUT_INDICES]["SMILES"].tolist()

    cols_to_keep = [
        "SMILES",
        "HOMO (ev)",
        "LUMO (ev)",
        "DFT Lmax ABS (nm)",
        *EMISSION_COLUMNS,
    ]
    long = df.loc[:, cols_to_keep].melt(
        id_vars=["SMILES", "HOMO (ev)", "LUMO (ev)", "DFT Lmax ABS (nm)"],
        value_vars=EMISSION_COLUMNS,
        var_name="solvent_raw",
        value_name="empeakwavs_max",
    )

    long["solvent"] = long["solvent_raw"].str.extract(r"EMIS \(nm\)\s*(.*)$")
    long["solvent"] = long["solvent"].str.strip()
    long = long.drop(columns=["solvent_raw"]).rename(
        columns={
            "SMILES": "smiles",
            "HOMO (ev)": "homo",
            "LUMO (ev)": "lumo",
            "DFT Lmax ABS (nm)": "peakwavs_max",
        }
    )

    # These CHCl3 measurements were manually corrected for the designated holdout molecules.
    for smiles, value in zip(holdout_smiles, HOLDOUT_CHCL3_FIXES):
        mask = (long["smiles"] == smiles) & (long["solvent"] == "CHCl3")
        long.loc[mask, "empeakwavs_max"] = value

    long["solvent"] = long["solvent"].map(SOLVENT_TO_SMILES)
    if long["solvent"].isna().any():
        raise ValueError("Encountered a solvent label without a SMILES mapping.")

    return long, holdout_smiles


def finalize_table(long: pd.DataFrame, nbits: int) -> pd.DataFrame:
    """Generate solvent fingerprints and apply the final emission filters."""
    long, _ = get_morgan_fingerprints(long, mol_or_solv="solvents", nbits=nbits)
    long = long.dropna(subset=["empeakwavs_max"]).reset_index(drop=True)
    long = long[long["empeakwavs_max"] > 200].reset_index(drop=True)
    return long


def write_outputs(df: pd.DataFrame, holdout_smiles: list[str], out_root: Path) -> None:
    """Write the processed Jeffries tables into the repo's target-specific layout."""
    holdout_df = df[df["smiles"].isin(holdout_smiles)].reset_index(drop=True)
    trainval_df = df[~df["smiles"].isin(holdout_smiles)].reset_index(drop=True)

    for target in ["abs", "em"]:
        target_dir = out_root / target / "jeffries"
        target_dir.mkdir(parents=True, exist_ok=True)

        trainval_df.to_csv(target_dir / "jeffries.csv", index=False)
        holdout_df.to_csv(target_dir / "jeffries_holdout.csv", index=False)
        print(f"Wrote {target_dir / 'jeffries.csv'} ({len(trainval_df)} rows)")
        print(f"Wrote {target_dir / 'jeffries_holdout.csv'} ({len(holdout_df)} rows)")


def main() -> None:
    args = parse_args()

    if not args.input_xlsx.exists():
        raise FileNotFoundError(f"Missing input workbook: {args.input_xlsx}")

    source = load_source_table(args.input_xlsx)
    long, holdout_smiles = build_long_table(source)
    processed = finalize_table(long, nbits=args.nbits)
    write_outputs(processed, holdout_smiles, args.out_root)


if __name__ == "__main__":
    main()
