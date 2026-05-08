"""Resolve a finetuning config from a base Chemprop hyperparameter JSON.

This helper copies a base config, optionally scales the learning-rate schedule,
applies a small set of overrides, and writes the resolved JSON used by a
finetuning job. Used in the transfer learning experiments.
"""

import json
import argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base config JSON")
    ap.add_argument("--out", required=True, help="Output resolved config JSON")
    ap.add_argument("--lr-scale", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--ffn-hidden-size", type=int, default=None)
    ap.add_argument("--ffn-num-layers", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    args = ap.parse_args()

    with open(args.base) as f:
        cfg = json.load(f)

    # Scale the finetuning learning-rate schedule while leaving other keys intact.
    for k in ["max_lr", "final_lr"]:
        if k in cfg:
            cfg[k] = float(cfg[k]) * args.lr_scale

    # Apply any explicit command-line overrides.
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    if args.ffn_hidden_size is not None:
        cfg["ffn_hidden_size"] = int(args.ffn_hidden_size)

    if args.ffn_num_layers is not None:
        cfg["ffn_num_layers"] = int(args.ffn_num_layers)

    if args.dropout is not None:
        cfg["dropout"] = float(args.dropout)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    print(f"[INFO] Wrote resolved config to {out}")
    print(f"[INFO] LR scale = {args.lr_scale}")
    for k in ["init_lr", "max_lr", "final_lr"]:
        if k in cfg:
            print(f"  {k} = {cfg[k]}")

if __name__ == "__main__":
    main()
