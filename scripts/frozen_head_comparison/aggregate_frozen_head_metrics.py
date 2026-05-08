#!/usr/bin/env python3
"""
Aggregate frozen-head comparison metrics across ensemble members.

This script reads the per-member outputs written by the frozen-head comparison
runs and summarizes them into compact JSON and CSV files. The summaries are
what the plotting script reads later when building the comparison figures.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from glob import glob

import numpy as np


def load_member_metrics(metrics_path: str) -> dict:
    """Load one member metrics file."""
    with open(metrics_path, "r") as f:
        return json.load(f)


def _iter_member_dirs(model_dir: str) -> list[str]:
    """Return the per-member run directories produced by the public workflow."""
    run_dirs = sorted(glob(os.path.join(model_dir, "member_*")))
    if not run_dirs:
        raise FileNotFoundError(f"No member_* directories found under {model_dir}")
    return run_dirs


def _per_head_entries(per_head: object) -> list[tuple[int, dict]]:
    """Require the list-shaped per-head metrics written by fit_tabular.py."""
    if not isinstance(per_head, list):
        raise TypeError("Expected per_head to be a list of per-head metric dicts.")
    entries: list[tuple[int, dict]] = []
    for idx, entry in enumerate(per_head):
        if not isinstance(entry, dict):
            raise TypeError(f"Expected per_head[{idx}] to be a dict, got {type(entry).__name__}.")
        entries.append((idx, entry))
    return entries


def collect_model_metrics(model_dir: str) -> dict:
    """Collect ensemble- and head-level metrics for one head family."""
    collected = {
        "ensemble": defaultdict(lambda: defaultdict(list)),
        "head": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    run_dirs = _iter_member_dirs(model_dir)
    for run_dir in run_dirs:
        mfile = os.path.join(run_dir, "metrics.json")
        if not os.path.exists(mfile):
            raise FileNotFoundError(f"Missing metrics.json in {run_dir}")
        data = load_member_metrics(mfile)

        ens = data.get("ensemble")
        if not isinstance(ens, dict):
            raise TypeError(f"Expected 'ensemble' metrics dict in {mfile}")
        for split in ("train", "val", "test"):
            ent = ens.get(split, {})
            for metric in ("rmse", "mae", "r2"):
                if metric in ent:
                    collected["ensemble"][split][metric].append(float(ent[metric]))

        ph = data.get("per_head")
        for h_idx, h in _per_head_entries(ph):
            for split in ("train", "val", "test"):
                hv = h.get(split, {})
                for metric in ("rmse", "mae", "r2"):
                    if metric in hv:
                        collected["head"][h_idx][split][metric].append(float(hv[metric]))
    return collected


def summarize_list(vals: list[float]) -> dict[str, float | int | None]:
    """Summarize one list of metric values."""
    arr = np.array(vals, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)) if arr.size else None,
        "median": float(np.median(arr)) if arr.size else None,
        "var": float(np.var(arr, ddof=0)) if arr.size else None,
        "std": float(np.std(arr, ddof=0)) if arr.size else None,
        "min": float(np.min(arr)) if arr.size else None,
        "max": float(np.max(arr)) if arr.size else None,
    }


def build_summary(collected: dict) -> dict:
    """Convert raw collected metric lists into a nested summary dict."""
    summary = {"ensemble": {}, "per_head": {}}
    for split, metrics in collected["ensemble"].items():
        summary["ensemble"][split] = {}
        for metric, vals in metrics.items():
            summary["ensemble"][split][metric] = summarize_list(vals)

    for h_idx, splits in collected["head"].items():
        summary["per_head"][int(h_idx)] = {}
        for split, metrics in splits.items():
            summary["per_head"][int(h_idx)][split] = {}
            for metric, vals in metrics.items():
                summary["per_head"][int(h_idx)][split][metric] = summarize_list(vals)
    return summary


def write_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(summary: dict, out_csv: str) -> None:
    """Flatten the nested summary into a compact tabular export."""
    rows = []
    for split, metrics in summary.get("ensemble", {}).items():
        for metric, stats in metrics.items():
            rows.append(
                (
                    "ensemble",
                    "ensemble",
                    split,
                    metric,
                    stats["count"],
                    stats["mean"],
                    stats["median"],
                    stats["var"],
                    stats["std"],
                    stats["min"],
                    stats["max"],
                )
            )

    for h_idx, splits in sorted(summary.get("per_head", {}).items()):
        for split, metrics in splits.items():
            for metric, stats in metrics.items():
                rows.append(
                    (
                        "per_head",
                        int(h_idx),
                        split,
                        metric,
                        stats["count"],
                        stats["mean"],
                        stats["median"],
                        stats["var"],
                        stats["std"],
                        stats["min"],
                        stats["max"],
                    )
                )

    header = ["level", "head_idx", "split", "metric", "count", "mean", "median", "var", "std", "min", "max"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Aggregate frozen-head comparison metrics across ensemble members."
    )
    ap.add_argument(
        "--results-root",
        required=False,
        default="results/frozen_head_comparison/deep4chem/scaffold/morgan_fp",
        help="Root directory containing per-model result folders such as mlp, xgb, and rf.",
    )
    ap.add_argument("--models", nargs="+", default=["mlp","xgb","rf"], help="Models to aggregate.")
    ap.add_argument(
        "--outdir",
        default=None,
        help="Directory for summary outputs. Defaults to <results-root>/aggregates.",
    )
    args = ap.parse_args()

    root = args.results_root
    outdir = args.outdir or os.path.join(root, "aggregates")
    os.makedirs(outdir, exist_ok=True)

    overall = {}
    for model in args.models:
        model_dir = os.path.join(root, model)
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"Model dir not found: {model_dir}")
        collected = collect_model_metrics(model_dir)
        summary = build_summary(collected)
        overall[model] = summary
        json_out = os.path.join(outdir, f"{model}_metrics_summary.json")
        csv_out = os.path.join(outdir, f"{model}_metrics_summary.csv")
        write_json(summary, json_out)
        write_csv(summary, csv_out)
        print(f"[ok] wrote summaries for {model}: {json_out}, {csv_out}")

    combined_json = os.path.join(outdir, "combined_metrics_summary.json")
    write_json(overall, combined_json)
    print("[done] combined summary:", combined_json)


if __name__ == "__main__":
    main()
