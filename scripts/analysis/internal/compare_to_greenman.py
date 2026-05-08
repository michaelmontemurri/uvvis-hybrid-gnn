"""Compare frozen tabular-model results against Greenman Chemprop baselines.

This is a historical comparison script that reads one set of tabular-model
metrics, parses Greenman `quiet.log` files, and writes a side-by-side summary
table plus a small comparison plot.
"""

import json, re, sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUR_BASE = Path("uvvis_hybrid/models/tabular_model_outputs")
GM_BASE  = Path("third_party/greenman/results/split_type")
SPLITS   = ["scaffold", "group_by_smiles", "random"]
FEATS    = ["chemfluor", "minnesota_descriptors", "morgan_fingerprint"]
OUR_SUBDIR = "ensemble_xgb_fixedparams"  # the directory you used
OUT_CSV  = Path("uvvis_hybrid/analysis/tabular_vs_greenman.csv")
OUT_PNG  = Path("uvvis_hybrid/analysis/tabular_vs_greenman.png")

gm_model_line = re.compile(
    r"Model\s+(\d+)\s+best validation rmse\s*=\s*([0-9.]+).*?Model \1 test rmse\s*=\s*([0-9.]+)",
    re.IGNORECASE,
)
gm_ens_line = re.compile(r"Ensemble test rmse\s*=\s*([0-9.]+)", re.IGNORECASE)

def read_our_metrics(split: str, feat: str):
    """Read one of our saved metrics bundles for a split/featurizer pair."""
    mpath = OUR_BASE / split / feat / OUR_SUBDIR / "metrics.json"
    if not mpath.exists():
        return None
    with mpath.open() as f:
        m = json.load(f)
    out = {
        "our_train_rmse": m["ensemble"]["train"]["rmse"],
        "our_val_rmse":   m["ensemble"]["val"]["rmse"],
        "our_test_rmse":  m["ensemble"]["test"]["rmse"],
        "our_n_heads":    m.get("n_heads"),
        "our_n_train":    m.get("n_train"),
        "our_n_val":      m.get("n_val"),
        "our_n_test":     m.get("n_test"),
        "our_include_physchem": m.get("include_physchem"),
        "our_include_fp":       m.get("include_fp"),
    }
    ph = m.get("per_head", [])
    if ph:
        out["our_heads_test_rmse_mean"] = float(pd.Series([h["test"]["rmse"] for h in ph]).mean())
        out["our_heads_test_rmse_std"]  = float(pd.Series([h["test"]["rmse"] for h in ph]).std(ddof=1))
        out["our_heads_val_rmse_mean"]  = float(pd.Series([h["val"]["rmse"] for h in ph]).mean())
        out["our_heads_val_rmse_std"]   = float(pd.Series([h["val"]["rmse"] for h in ph]).std(ddof=1))
    return out

def read_greenman_quiet(split: str, feat: str):
    """Parse the relevant Greenman `quiet.log` file for a split/feature pair."""
    logp = GM_BASE / split / "chemprop" / feat / "fp" / "deep4chem" / "quiet.log"
    if not logp.exists():
        return None
    text = logp.read_text()
    models = gm_model_line.findall(text)
    ens = gm_ens_line.findall(text)
    out = {}
    if ens:
        out["gm_ens_test_rmse"] = float(ens[-1])
    if models:
        v = [float(x[1]) for x in models]
        t = [float(x[2]) for x in models]
        out["gm_heads_val_rmse_mean"] = float(pd.Series(v).mean())
        out["gm_heads_val_rmse_std"]  = float(pd.Series(v).std(ddof=1))
        out["gm_heads_test_rmse_mean"]= float(pd.Series(t).mean())
        out["gm_heads_test_rmse_std"] = float(pd.Series(t).std(ddof=1))
        out["gm_n_heads"] = len(t)
    return out if out else None

rows = []
for split in SPLITS:
    for feat in FEATS:
        our = read_our_metrics(split, feat)
        gm  = read_greenman_quiet(split, feat)
        row = {"split": split, "featurizer": feat}
        if our:
            row.update(our)
        if gm:
            row.update(gm)
        if our and gm and "gm_ens_test_rmse" in gm:
            row["delta_test_rmse_vs_gm_ens"] = row["our_test_rmse"] - row["gm_ens_test_rmse"]
        rows.append(row)

df = pd.DataFrame(rows).sort_values(["split", "featurizer"]).reset_index(drop=True)

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT_CSV, index=False)
print(f"[write] {OUT_CSV}")

plt.figure(figsize=(12, 6))
labels = [f"{s}\n{f}" for s, f in zip(df["split"], df["featurizer"])]
x = range(len(df))
y_ours = df["our_test_rmse"]
y_gm = df.get("gm_ens_test_rmse", pd.Series([float("nan")]*len(df)))

width = 0.35
plt.bar([i - width/2 for i in x], y_ours, width, label="Ours (XGB ensemble)")
plt.bar([i + width/2 for i in x], y_gm,   width, label="Greenman (Chemprop ensemble)")
plt.xticks(list(x), labels, rotation=45, ha="right")
plt.ylabel("Test RMSE (nm)")
plt.title("Frozen-embed Tabular (XGB) vs Greenman Chemprop — Test RMSE")
plt.legend()
plt.tight_layout()

OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT_PNG, dpi=150)
print(f"[write] {OUT_PNG}")

requested_cols = [
    "split", "featurizer",
    "our_test_rmse", "gm_ens_test_rmse", "delta_test_rmse_vs_gm_ens",
    "our_val_rmse", "gm_heads_val_rmse_mean",
]

cols = [c for c in requested_cols if c in df.columns]

print("\n=== Summary ===")
print(df[cols].to_string(index=False))
