#!/usr/bin/env python3
"""Blend predictions from multiple frozen-model heads into one ensemble result.

The inputs are per-head `ensemble_preds_{split}.csv` files. The script aligns
rows across heads, blends predictions by mean, median, or validation-weighted
average, and writes both merged predictions and summary metrics.
"""

import argparse, json, pandas as pd, numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_preds(path):
    """Load one per-head prediction file and choose stable alignment keys."""
    df = pd.read_csv(path)
    # If duplicate molecules are disambiguated with an occurrence index, preserve it.
    if "occ" in df.columns:
        df = df.drop_duplicates(subset=["smiles","occ"], keep="first")
        key = ["smiles","occ"]
    else:
        df = df.drop_duplicates(subset=["smiles"], keep="first")
        key = ["smiles"]
    return df[key + ["y_true","pred_mean"]], key

def metrics(y, p):
    """Compute the standard regression metrics used across the repo."""
    return dict(
        rmse=float(np.sqrt(((y - p)**2).mean())),
        mae=float(mean_absolute_error(y, p)),
        r2=float(r2_score(y, p)),
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="paths to per-head ensemble_preds_{split}.csv")
    ap.add_argument("--split", choices=["val","test"], required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--blend", choices=["mean","median","wval"], default="mean",
                    help="mean/median; wval=weighted by 1/RMSE(val)")
    ap.add_argument("--val-metrics-jsons", nargs="*", help="for --blend wval: per-head val metrics jsons")
    args = ap.parse_args()

    dfs, key_cols = [], None
    for p in args.inputs:
        df, k = load_preds(p)
        if key_cols is None: key_cols = k
        dfs.append(df.set_index(key_cols))
    base = dfs[0].copy()
    Ys = base["y_true"].to_numpy()
    preds = [d["pred_mean"].reindex(base.index).to_numpy() for d in dfs]
    P = np.vstack(preds)

    if args.blend == "mean":
        P_ens = P.mean(0)
    elif args.blend == "median":
        P_ens = np.median(P, axis=0)
    else:
        # Weight each head by the inverse of its validation RMSE.
        assert args.val_metrics_jsons and len(args.val_metrics_jsons) == len(args.inputs)
        inv = []
        for jm in args.val_metrics_jsons:
            with open(jm) as f:
                m = json.load(f)
            inv.append(1.0 / float(m["rmse"]))
        w = np.array(inv, dtype=float)
        w = w / w.sum()
        P_ens = (w[:,None] * P).sum(0)

    out = base.reset_index().rename(columns={"pred_mean":"pred_member"})
    for i in range(P.shape[0]):
        out[f"pred_head{i}"] = P[i]
    out["pred_mean"] = P_ens
    out["pred_std"]  = P.std(0)

    m = metrics(Ys, P_ens)
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.outdir)/f"crosshead_preds_{args.split}.csv", index=False)
    with open(Path(args.outdir)/f"crosshead_metrics_{args.split}.json","w") as f:
        json.dump(m, f, indent=2)
    print(args.split, m)

if __name__ == "__main__":
    main()
