# uvvis_hybrid/scripts/dump_embeddings.py
"""
Extract frozen Chemprop embeddings for UVVis hybrid experiments.

This script loads a trained Chemprop checkpoint and a CSV containing SMILES,
runs the molecules through the encoder, and writes the resulting molecular
embeddings to a Parquet file. These embeddings correspond to the pooled GNN
representation before the Chemprop feed-forward network (FFN), which is the
representation used for downstream tabular models in the hybrid experiments.

Inputs
------
--ckpt
    Path to a trained Chemprop checkpoint (.pt).

--csv
    CSV file containing at least a SMILES column.

--smiles-col
    Name of the SMILES column in the CSV (default: "smiles").

--stage
    Only `pre_ffn` is supported in the public artifact. The argument remains
    for compatibility with existing job wrappers.

--out
    Output Parquet file containing:
        smiles
        emb_0 ... emb_d

Example
-------
Example loop for extracting embeddings from a set of trained checkpoints
across data splits and ensemble heads.

ckpt="checkpoints/.../model_${f}.pt"

# for a train split
python uvvis_hybrid/scripts/dump_embeddings.py \
    --ckpt "${ckpt}" \
    --csv  data/.../train.csv \
    --smiles-col smiles \
    --out uvvis_hybrid/embedding_dumps/.../pre_ffn_embeddings/fold_${f}_train.parquet
"""

from __future__ import annotations
import argparse
import os
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm

from chemprop.data.utils import get_data
from chemprop.data import MoleculeDataLoader
from chemprop.utils import load_checkpoint 

OPTIONAL_FEATURE_FLAGS = (
    "use_input_features",
    "use_atom_descriptors",
    "use_atom_features",
    "use_bond_features",
    "features_scaling",
)


def _device_from_arg(arg: Optional[str]) -> torch.device:
    if arg is not None:
        a = arg.lower()
        if a in ("cpu", "cuda"):
            return torch.device(a)
        if a.isdigit():
            return torch.device(f"cuda:{a}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _disable_optional_feature_paths(model) -> None:
    """
    Disable Chemprop's optional feature paths before calling the frozen encoder.

    The released checkpoints were trained with tabular input features enabled,
    but the public artifact only extracts the pooled GNN representation before
    the FFN. We therefore switch off the feature-gated branches explicitly and
    then call the encoder in that fixed pre-FFN mode.
    """
    targets = []
    encoder = getattr(model, "encoder", None)
    if encoder is not None:
        targets.append(("model.encoder", encoder))
    mpn = getattr(model, "mpn", None)
    if mpn is not None and mpn is not encoder:
        targets.append(("model.mpn", mpn))
    args_obj = getattr(model, "args", None)
    if args_obj is not None:
        targets.append(("model.args", args_obj))

    updated = []
    for target_name, target in targets:
        for flag_name in OPTIONAL_FEATURE_FLAGS:
            if not hasattr(target, flag_name):
                continue
            try:
                setattr(target, flag_name, False)
            except AttributeError as exc:
                raise RuntimeError(
                    f"Could not disable {target_name}.{flag_name} for pre-FFN extraction."
                ) from exc
            updated.append(f"{target_name}.{flag_name}")

    if not updated:
        raise RuntimeError(
            "Checkpoint model does not expose the expected Chemprop feature-gating flags."
        )

    print("[dump_embeddings] disabled optional feature paths for pre-FFN extraction")


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to Chemprop checkpoint .pt")
    p.add_argument("--csv", required=True, help="CSV with the SMILES column")
    p.add_argument("--smiles-col", default="smiles")
    p.add_argument(
        "--stage",
        choices=["pre_ffn"],
        default="pre_ffn",
        help="Only pre_ffn extraction is supported in the public artifact.",
    )
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", default=None, help="'cpu', 'cuda', or cuda index (e.g., '0')")
    p.add_argument("--out", required=True, help="Output Parquet path")
    args = p.parse_args()

    device = _device_from_arg(args.device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    base_df = pd.read_csv(args.csv) #for row count checks

    # build dataset/loader 
    dataset = get_data(
        path=args.csv,
        smiles_columns=[args.smiles_col],
        target_columns=[]
    )
    loader = MoleculeDataLoader(dataset=dataset, batch_size=args.batch_size, num_workers=0)

    # load model
    model = load_checkpoint(args.ckpt, device=device)
    model.eval()

    # extract only the pooled encoder representation before the FFN.
    _disable_optional_feature_paths(model)

    # collect embeddings
    smiles_out: List[str] = []
    embs: List[torch.Tensor] = []

    for batch in tqdm(loader, total=len(loader), desc="Embedding"):
        mol_batch = batch.batch_graph()  # BatchMolGraph type

        # Call the encoder directly in its pre-FFN path with all optional
        # feature inputs disabled.
        features_batch = None
        atom_desc_batch = None
        atom_feats_batch = None
        bond_feats_batch = None

        gnn_rep = model.encoder(
            mol_batch,
            features_batch,
            atom_desc_batch,
            atom_feats_batch,
            bond_feats_batch,
        )  # -> [B, hidden]

        embs.append(gnn_rep.detach().cpu())
        smiles_out.extend(batch.smiles())

    # Stack and write
    emb_mat = torch.cat(embs, dim=0).numpy()
    dim = emb_mat.shape[1]
    out_df = pd.DataFrame(emb_mat, columns=[f"emb_{i}" for i in range(dim)])
    out_df.insert(0, args.smiles_col, smiles_out)

    # checks
    print(f"[dump_embeddings] emb_dim={dim}, out_rows={len(out_df)}")
    if len(out_df) != len(base_df):
        print("[dump_embeddings/WARN] row count mismatch vs input CSV "
              f"({len(out_df)} vs {len(base_df)}). Check invalid SMILES filtering.")

    print("[dump_embeddings] head:\n", out_df.head(3))
    out_df.to_parquet(args.out, index=False)
    print(f"[dump_embeddings] wrote {args.out}")


if __name__ == "__main__":
    main()
