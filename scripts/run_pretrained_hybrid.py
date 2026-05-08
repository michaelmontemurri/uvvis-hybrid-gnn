#!/usr/bin/env python3
"""Run the local frozen-encoder hybrid workflow from a checkpoint ensemble.

This wrapper is meant for external reuse on a local machine. It starts from a
compatible Chemprop encoder ensemble, dumps pre-FFN embeddings for one
train/val/test split, trains one downstream tabular model per encoder member,
and then blends those head-level predictions across the ensemble.

The checkpoints may come from:

- a curated bundle entry in `pretrained/deep4chem_ensemble_manifest.json`
- an explicit checkpoint directory
- an explicit list of checkpoint files
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Dump embeddings from a compatible Chemprop encoder ensemble and "
            "fit one downstream tabular head per encoder member."
        )
    )
    p.add_argument("--train-csv", required=True, help="Train split CSV.")
    p.add_argument("--val-csv", required=True, help="Validation split CSV.")
    p.add_argument("--test-csv", required=True, help="Test split CSV.")
    p.add_argument("--smiles-col", default="smiles", help="SMILES column name.")
    p.add_argument(
        "--target-col",
        default=None,
        help="Target column. Defaults to the manifest target column when --manifest-id is used.",
    )
    p.add_argument(
        "--manifest-path",
        default="pretrained/deep4chem_ensemble_manifest.json",
        help="Manifest used to resolve curated checkpoint bundles.",
    )
    p.add_argument(
        "--manifest-id",
        default="deep4chem_abs_scaffold_full_ensemble",
        help="Manifest bundle id to use when --checkpoint-dir/--checkpoints are omitted.",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory containing a Chemprop ensemble with fold_0/model_i/model.pt files.",
    )
    p.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="Explicit list of model.pt files. Overrides --checkpoint-dir and --manifest-id.",
    )
    p.add_argument("--outdir", required=True, help="Output directory for embeddings, head models, and ensemble metrics.")
    p.add_argument("--model", choices=["xgb", "rf", "mlp", "ridge"], default="xgb")
    p.add_argument(
        "--blend",
        choices=["mean", "median"],
        default="mean",
        help="Cross-head blending rule for the final val/test predictions.",
    )
    p.add_argument("--device", default=None, help="'cpu', 'cuda', or CUDA index.")
    p.add_argument("--base-seed", type=int, default=100, help="Base seed for downstream tabular fits.")
    p.add_argument("--tune-trials", type=int, default=50, help="Optuna trials used when tuning xgb/mlp/rf on head 0.")
    p.add_argument(
        "--best-params-dir",
        default=None,
        help=(
            "Directory containing precomputed downstream best-params JSON files "
            "(e.g. xgb_best_params_head0.json). When provided, tuning is skipped "
            "and these params are forwarded to fit_tabular."
        ),
    )
    p.add_argument(
        "--skip-tuning",
        action="store_true",
        help="Skip the one-time head-0 tuning pass and use fit_tabular defaults.",
    )
    p.add_argument(
        "--no-solvent-fp",
        action="store_true",
        help="Do not add solvent fingerprint columns in fit_tabular.",
    )
    p.add_argument("--force-redump", action="store_true", help="Redump embeddings even if cached parquet files exist.")
    return p.parse_args()


def run(cmd: list[str]) -> None:
    print("[CMD]", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(REPO_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def load_manifest(path: Path) -> dict[str, dict]:
    with open(path) as f:
        payload = json.load(f)
    bundles = payload.get("bundles", [])
    return {bundle["id"]: bundle for bundle in bundles}


def resolve_checkpoints(args: argparse.Namespace) -> tuple[list[Path], dict | None]:
    if args.checkpoints:
        ckpts = [Path(p) for p in args.checkpoints]
        return ckpts, None

    if args.checkpoint_dir:
        ckpt_dir = Path(args.checkpoint_dir)
        return discover_checkpoints(ckpt_dir), None

    manifest_path = REPO_ROOT / args.manifest_path
    bundles = load_manifest(manifest_path)
    if args.manifest_id not in bundles:
        raise KeyError(f"Manifest id '{args.manifest_id}' not found in {manifest_path}")
    bundle = bundles[args.manifest_id]
    ckpt_dir = REPO_ROOT / bundle["expected_local_dir"]
    return discover_checkpoints(ckpt_dir), bundle


def discover_checkpoints(ckpt_dir: Path) -> list[Path]:
    if not ckpt_dir.exists():
        raise FileNotFoundError(
            f"Checkpoint directory not found: {ckpt_dir}\n"
            "If you are using a curated pretrained bundle, extract it under checkpoints/ "
            "or pass --checkpoint-dir explicitly."
        )
    ckpts = sorted(ckpt_dir.glob("fold_0/model_*/model.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No member checkpoints were found under {ckpt_dir}")
    return ckpts


def fit_tabular_cmd(
    split_args: list[str],
    emb_paths: dict[str, Path],
    outdir: Path,
    model: str,
    target_col: str,
    smiles_col: str,
    seed: int,
    use_solvent_fp: bool,
    device: str | None,
    best_params_dir: Path | None = None,
    tune: bool = False,
    tune_trials: int = 50,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "uvvis_hybrid.scripts.fit_tabular",
        *split_args,
        "--smiles-col",
        smiles_col,
        "--target-col",
        target_col,
        "--emb-train",
        str(emb_paths["train"]),
        "--emb-val",
        str(emb_paths["val"]),
        "--emb-test",
        str(emb_paths["test"]),
        "--model",
        model,
        "--seed",
        str(seed),
        "--outdir",
        str(outdir),
    ]
    if use_solvent_fp:
        cmd.append("--include-solvent-fp")
    if device is not None:
        cmd.extend(["--device", device])
    if best_params_dir is not None:
        cmd.extend(["--best-params-dir", str(best_params_dir)])
    if tune:
        cmd.extend(["--tune", "--tune-trials", str(tune_trials)])
    return cmd


def main() -> None:
    args = parse_args()
    ckpts, bundle = resolve_checkpoints(args)
    target_col = args.target_col or (bundle["target_column"] if bundle is not None else None)
    if target_col is None:
        raise ValueError("Provide --target-col when using --checkpoint-dir or --checkpoints.")

    outdir = Path(args.outdir)
    emb_root = outdir / "embeddings"
    heads_root = outdir / "heads"
    ensemble_root = outdir / "ensemble"
    params_dir = Path(args.best_params_dir) if args.best_params_dir else (outdir / "best_params")
    emb_root.mkdir(parents=True, exist_ok=True)
    heads_root.mkdir(parents=True, exist_ok=True)
    ensemble_root.mkdir(parents=True, exist_ok=True)

    split_args = [
        "--train-csv",
        args.train_csv,
        "--val-csv",
        args.val_csv,
        "--test-csv",
        args.test_csv,
    ]

    use_solvent_fp = not args.no_solvent_fp
    head_prediction_paths: dict[str, list[str]] = {"val": [], "test": []}

    for idx, ckpt in enumerate(ckpts):
        head_name = f"head_{idx:02d}"
        head_emb_dir = emb_root / head_name
        head_outdir = heads_root / head_name
        head_emb_dir.mkdir(parents=True, exist_ok=True)
        head_outdir.mkdir(parents=True, exist_ok=True)

        emb_paths = {
            "train": head_emb_dir / "train.parquet",
            "val": head_emb_dir / "val.parquet",
            "test": head_emb_dir / "test.parquet",
        }

        for split_name, split_csv in [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]:
            out_parquet = emb_paths[split_name]
            if out_parquet.exists() and not args.force_redump:
                continue
            run(
                [
                    sys.executable,
                    "-m",
                    "uvvis_hybrid.scripts.dump_embeddings",
                    "--ckpt",
                    str(ckpt),
                    "--csv",
                    split_csv,
                    "--smiles-col",
                    args.smiles_col,
                    "--stage",
                    "pre_ffn",
                    "--out",
                    str(out_parquet),
                    *(["--device", args.device] if args.device is not None else []),
                ]
            )

        if idx == 0 and args.best_params_dir is None and not args.skip_tuning and args.model in {"xgb", "mlp", "rf"}:
            params_dir.mkdir(parents=True, exist_ok=True)
            run(
                fit_tabular_cmd(
                    split_args=split_args,
                    emb_paths=emb_paths,
                    outdir=params_dir,
                    model=args.model,
                    target_col=target_col,
                    smiles_col=args.smiles_col,
                    seed=args.base_seed,
                    use_solvent_fp=use_solvent_fp,
                    device=args.device,
                    best_params_dir=None,
                    tune=True,
                    tune_trials=args.tune_trials,
                )
            )

        best_params_dir = params_dir if params_dir.exists() and any(params_dir.glob("*best_params*.json")) else None
        run(
            fit_tabular_cmd(
                split_args=split_args,
                emb_paths=emb_paths,
                outdir=head_outdir,
                model=args.model,
                target_col=target_col,
                smiles_col=args.smiles_col,
                seed=args.base_seed + idx,
                use_solvent_fp=use_solvent_fp,
                device=args.device,
                best_params_dir=best_params_dir,
                tune=False,
                tune_trials=args.tune_trials,
            )
        )

        head_prediction_paths["val"].append(str(head_outdir / "preds_val.csv"))
        head_prediction_paths["test"].append(str(head_outdir / "preds_test.csv"))

    for split_name in ("val", "test"):
        run(
            [
                sys.executable,
                "scripts/ensemble_cross_heads.py",
                "--inputs",
                *head_prediction_paths[split_name],
                "--split",
                split_name,
                "--outdir",
                str(ensemble_root),
                "--blend",
                args.blend,
            ]
        )

    run_summary = {
        "manifest_id": args.manifest_id if bundle is not None else None,
        "resolved_checkpoints": [str(p) for p in ckpts],
        "target_col": target_col,
        "smiles_col": args.smiles_col,
        "model": args.model,
        "blend": args.blend,
        "used_solvent_fp": use_solvent_fp,
        "outdir": str(outdir),
    }
    with open(outdir / "run_summary.json", "w") as f:
        json.dump(run_summary, f, indent=2, sort_keys=True)

    print(json.dumps(run_summary, indent=2, sort_keys=True))
    print(f"[done] wrote {outdir}")


if __name__ == "__main__":
    main()
