#!/usr/bin/env python3
"""
Summarize nested and pipeline-level variance analyses for learned embeddings.

This script collects three related variance analyses: manual nested-ANOVA
variance components across encoder, head, and seed factors; a corresponding
statsmodels MixedLM variance-component estimate; and pipeline-level
single-model and ensemble RMSE comparisons between the frozen-encoder
hybrid pipeline and end-to-end Chemprop models.
"""


from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter
from scipy.interpolate import PchipInterpolator
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "legend.fontsize": 12,
        "lines.linewidth": 2.5,
        "lines.markersize": 7,
        "figure.titlesize": 18,
        "savefig.dpi": 600,
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


COMPONENT_LABELS = [
    "Encoders ($A$)",
    "Heads ($B(A)$)",
    "Residual ($E$)",
]
COMPONENT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize nested variance decomposition and pipeline-level variance."
    )
    parser.add_argument("--target", default="abs")
    parser.add_argument("--dataset", default="deep4chem")
    parser.add_argument("--split", default="scaffold")
    parser.add_argument("--model", default="xgb")
    parser.add_argument(
        "--results-root",
        default=None,
        help="Optional nested-variance root containing subset directories.",
    )
    parser.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional explicit subset tags such as N00050_s42 N00100_s42.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--include-absolute",
        action="store_true",
        help="Also write the absolute variance-component stackplots.",
    )
    parser.add_argument("--outdir", default="figures/variance_decomposition")
    return parser.parse_args()


def parse_subset_size(tag: str) -> int:
    """Extract the integer N from a tag like N00100_s42."""
    return int(tag.split("_")[0].replace("N", ""))


def extract_test_metrics(metrics_path: Path) -> dict[str, float]:
    """Load one metrics file and return test metrics."""
    with metrics_path.open("r") as handle:
        data = json.load(handle)

    if "ensemble" in data and "test" in data["ensemble"]:
        test = data["ensemble"]["test"]
    elif "test" in data:
        test = data["test"]
    else:
        test = data
    return {
        "rmse": float(test["rmse"]),
        "mae": float(test["mae"]) if "mae" in test else np.nan,
        "r2": float(test["r2"]) if "r2" in test else np.nan,
    }


def resolve_nested_root(args: argparse.Namespace) -> Path:
    """Find the subset-results root for nested variance analysis."""
    if args.results_root:
        return Path(args.results_root)

    candidates = [
        Path(f"results/{args.target}/variance_decomp/{args.dataset}/{args.split}/subsets"),
        Path(f"results/{args.target}/phase1B/{args.dataset}/{args.split}/subsets"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate a nested variance root. Set --results-root explicitly if needed."
    )


def resolve_subset_model_dir(root: Path, subset: str, model: str) -> Path:
    """Resolve one subset/model directory across old and new layouts."""
    candidates = [
        root / subset / model,
        root / subset / "variance" / model,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find nested variance results for {subset} under {root}")


def discover_subsets(root: Path, explicit: list[str] | None) -> list[str]:
    """Return sorted subset tags."""
    if explicit:
        return explicit
    subsets = sorted(
        [path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("N")],
        key=parse_subset_size,
    )
    if not subsets:
        raise FileNotFoundError(f"No subset directories found under {root}")
    return subsets


def collect_nested_rows(subset_dir: Path, subset: str) -> pd.DataFrame:
    """Collect encoder/head/seed RMSE rows from one subset directory."""
    rows = []
    for enc_dir in sorted(subset_dir.glob("encoder_*")):
        encoder = int(enc_dir.name.split("_")[1])
        for head_dir in sorted(enc_dir.glob("head_*")):
            head = int(head_dir.name.split("_")[1])
            for seed_dir in sorted(head_dir.glob("seed_*")):
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                seed = int(seed_dir.name.split("_")[1])
                metrics = extract_test_metrics(metrics_path)
                rows.append(
                    {
                        "subset": subset,
                        "encoder": encoder,
                        "head": head,
                        "seed": seed,
                        "rmse_test": metrics["rmse"],
                        "path": str(metrics_path),
                    }
                )
    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No nested seed metrics found under {subset_dir}")
    return df.sort_values(["encoder", "head", "seed"]).reset_index(drop=True)


def get_design_dims(df: pd.DataFrame) -> tuple[int, int, int, bool]:
    """Return a, b, n and whether the nested design is balanced."""
    a = df["encoder"].nunique()
    heads_per_encoder = df.groupby("encoder")["head"].nunique()
    reps_per_cell = df.groupby(["encoder", "head"])["seed"].nunique()
    balanced = heads_per_encoder.nunique() == 1 and reps_per_cell.nunique() == 1
    b = int(heads_per_encoder.max())
    n = int(reps_per_cell.max())
    return a, b, n, balanced


def compute_manual_components(df: pd.DataFrame, a: int, b: int, n: int) -> dict[str, float]:
    """Compute balanced nested-ANOVA variance components by hand."""
    grand_mean = float(df["rmse_test"].mean())
    enc_means = df.groupby("encoder")["rmse_test"].mean()
    cell_means = df.groupby(["encoder", "head"])["rmse_test"].mean()

    ss_a = (b * n) * float(((enc_means - grand_mean) ** 2).sum())
    cell_frame = cell_means.rename("cell_mean").reset_index()
    cell_frame["enc_mean"] = cell_frame["encoder"].map(enc_means)
    ss_ba = float((n * (cell_frame["cell_mean"] - cell_frame["enc_mean"]) ** 2).sum())

    df_with_cell = df.merge(
        cell_means.rename("cell_mean").reset_index(),
        on=["encoder", "head"],
        how="left",
    )
    ss_e = float(((df_with_cell["rmse_test"] - df_with_cell["cell_mean"]) ** 2).sum())

    df_a = max(1, a - 1)
    df_ba = max(1, a * (b - 1))
    df_e = max(1, a * b * (n - 1))

    ms_a = ss_a / df_a
    ms_ba = ss_ba / df_ba
    ms_e = ss_e / df_e

    sig_e = ms_e
    sig_ba = max(0.0, (ms_ba - ms_e) / n)
    sig_a = max(0.0, (ms_a - ms_ba) / (b * n))
    total = sig_a + sig_ba + sig_e

    return {
        "manual_encoder_variance": sig_a,
        "manual_head_variance": sig_ba,
        "manual_residual_variance": sig_e,
        "manual_total_variance": total,
        "manual_encoder_prop": sig_a / total if total else 0.0,
        "manual_head_prop": sig_ba / total if total else 0.0,
        "manual_residual_prop": sig_e / total if total else 0.0,
    }


def bootstrap_manual_proportions(
    df: pd.DataFrame,
    a: int,
    b: int,
    n: int,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """Estimate bootstrap confidence intervals for the manual proportions."""
    rng = np.random.default_rng(seed)
    boot_props = []
    unique_encoders = df["encoder"].unique()

    for _ in range(n_boot):
        resampled_encoders = rng.choice(unique_encoders, size=len(unique_encoders), replace=True)
        boot_frames = []

        for new_encoder, original_encoder in enumerate(resampled_encoders):
            encoder_df = df[df["encoder"] == original_encoder]
            unique_heads = encoder_df["head"].unique()
            resampled_heads = rng.choice(unique_heads, size=len(unique_heads), replace=True)

            for new_head, original_head in enumerate(resampled_heads):
                head_df = encoder_df[encoder_df["head"] == original_head]
                sampled = head_df.sample(n=len(head_df), replace=True, random_state=int(rng.integers(1 << 31))).copy()
                sampled["encoder"] = new_encoder
                sampled["head"] = new_head
                boot_frames.append(sampled)

        boot_df = pd.concat(boot_frames, ignore_index=True)
        comp = compute_manual_components(boot_df, a, b, n)
        boot_props.append(
            [
                comp["manual_encoder_prop"],
                comp["manual_head_prop"],
                comp["manual_residual_prop"],
            ]
        )

    boot_arr = np.asarray(boot_props)
    lower = np.percentile(boot_arr, 2.5, axis=0)
    upper = np.percentile(boot_arr, 97.5, axis=0)
    return {
        "manual_encoder_ci_low": float(lower[0]),
        "manual_encoder_ci_high": float(upper[0]),
        "manual_head_ci_low": float(lower[1]),
        "manual_head_ci_high": float(upper[1]),
        "manual_residual_ci_low": float(lower[2]),
        "manual_residual_ci_high": float(upper[2]),
    }


def compute_statsmodels_components(df: pd.DataFrame, a: int, b: int, n: int) -> dict[str, float]:
    """Estimate variance components from a statsmodels ANOVA table."""
    model_df = df.copy()
    model_df["encoder"] = model_df["encoder"].astype("category")
    model_df["head"] = model_df["head"].astype("category")
    fit = smf.ols("rmse_test ~ C(encoder) + C(encoder):C(head)", data=model_df).fit()
    anova = anova_lm(fit, typ=1)

    encoder_row = anova.loc["C(encoder)"]
    head_row = anova.loc["C(encoder):C(head)"]
    residual_row = anova.loc["Residual"]

    ms_a = float(encoder_row["sum_sq"] / encoder_row["df"])
    ms_ba = float(head_row["sum_sq"] / head_row["df"])
    ms_e = float(residual_row["sum_sq"] / residual_row["df"])

    encoder_var = max((ms_a - ms_ba) / (b * n), 0.0)
    head_var = max((ms_ba - ms_e) / n, 0.0)
    residual_var = max(ms_e, 0.0)
    total = encoder_var + head_var + residual_var

    return {
        "statsmodels_encoder_variance": encoder_var,
        "statsmodels_head_variance": head_var,
        "statsmodels_residual_variance": residual_var,
        "statsmodels_total_variance": total,
        "statsmodels_encoder_prop": encoder_var / total if total else 0.0,
        "statsmodels_head_prop": head_var / total if total else 0.0,
        "statsmodels_residual_prop": residual_var / total if total else 0.0,
        "statsmodels_converged": True,
    }


def build_nested_summary(args: argparse.Namespace) -> pd.DataFrame:
    """Collect all nested-variance subset summaries."""
    root = resolve_nested_root(args)
    subsets = discover_subsets(root, args.subsets)
    rows = []

    for idx, subset in enumerate(subsets):
        subset_dir = resolve_subset_model_dir(root, subset, args.model)
        df = collect_nested_rows(subset_dir, subset)
        a, b, n, balanced = get_design_dims(df)

        row = {
            "subset": subset,
            "N": parse_subset_size(subset),
            "n_rows": len(df),
            "n_encoders": a,
            "n_heads_per_encoder": b,
            "n_replicates_per_cell": n,
            "balanced_design": balanced,
            "results_dir": str(subset_dir),
        }

        manual = compute_manual_components(df, a, b, n)
        row.update(manual)
        row.update(
            bootstrap_manual_proportions(
                df,
                a,
                b,
                n,
                n_boot=args.bootstrap_samples,
                seed=args.seed + idx,
            )
        )

        try:
            row.update(compute_statsmodels_components(df, a, b, n))
        except Exception as exc:
            row.update(
                {
                    "statsmodels_encoder_variance": np.nan,
                    "statsmodels_head_variance": np.nan,
                    "statsmodels_residual_variance": np.nan,
                    "statsmodels_total_variance": np.nan,
                    "statsmodels_encoder_prop": np.nan,
                    "statsmodels_head_prop": np.nan,
                    "statsmodels_residual_prop": np.nan,
                    "statsmodels_converged": False,
                    "statsmodels_error": str(exc),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def resolve_hybrid_root(target: str, dataset: str, split: str, subset: str, model: str) -> Path:
    """Resolve the single-per-head hybrid result directory for one subset."""
    candidates = [
        Path(f"results/{target}/xgb_on_learned_embedding_subsets/{dataset}/{split}/ensembles/{model}_{subset}_single_perhead"),
        Path(f"results/{target}/phase1C/{dataset}/{split}/ensembles/{model}_{subset}_single_perhead"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find hybrid single-per-head results for {subset}")


def collect_pipeline_summary(target: str, dataset: str, split: str, model: str, subsets: list[str]) -> pd.DataFrame:
    """Compare pipeline-level single-model and ensemble performance across subset size."""
    rows = []
    for subset in subsets:
        hybrid_root = resolve_hybrid_root(target, dataset, split, subset, model)
        hybrid_metrics = []
        hybrid_preds = []

        for encoder in range(5):
            member_dir = hybrid_root / f"head_0{encoder}" / "member_00"
            metrics = extract_test_metrics(member_dir / "metrics.json")
            hybrid_metrics.append(metrics)

            pred_path = member_dir / "preds_test.csv"
            if pred_path.exists():
                pred_df = pd.read_csv(pred_path)
                pred_col = "pred_mean" if "pred_mean" in pred_df.columns else "y_pred"
                hybrid_preds.append(pred_df[pred_col].to_numpy(dtype=float))

        test_csv = Path(f"data/uvvisml/{target}/{dataset}/subsets/{split}/{subset}/smiles_target_test.csv")
        target_col = "peakwavs_max" if target == "abs" else "empeakwavs_max"
        y_true = pd.read_csv(test_csv)[target_col].to_numpy(dtype=float)

        e2e_ckpt = Path(f"checkpoints/{target}/{dataset}/{split}/morgan_fingerprint/fromscratch_{subset}")
        e2e_metrics = []
        e2e_preds = []
        for member in range(5):
            pred_path = e2e_ckpt / "preds" / f"preds_test_model_{member}.csv"
            pred_df = pd.read_csv(pred_path)
            y_pred = pred_df[target_col].to_numpy(dtype=float)
            e2e_preds.append(y_pred)
            e2e_metrics.append(
                {
                    "rmse": float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
                    "mae": float(np.mean(np.abs(y_pred - y_true))),
                    "r2": float(
                        1.0 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
                    ),
                }
            )

        hybrid_rmse = np.asarray([row["rmse"] for row in hybrid_metrics], dtype=float)
        e2e_rmse = np.asarray([row["rmse"] for row in e2e_metrics], dtype=float)
        hybrid_ens = float(np.sqrt(np.mean((np.mean(np.stack(hybrid_preds, axis=1), axis=1) - y_true) ** 2)))
        e2e_ens = float(np.sqrt(np.mean((np.mean(np.stack(e2e_preds, axis=1), axis=1) - y_true) ** 2)))

        rows.append(
            {
                "subset": subset,
                "N": parse_subset_size(subset),
                "hybrid_mean_rmse": float(hybrid_rmse.mean()),
                "hybrid_std_rmse": float(hybrid_rmse.std(ddof=1)),
                "hybrid_var_rmse": float(hybrid_rmse.var(ddof=1)),
                "hybrid_ensemble_rmse": hybrid_ens,
                "e2e_mean_rmse": float(e2e_rmse.mean()),
                "e2e_std_rmse": float(e2e_rmse.std(ddof=1)),
                "e2e_var_rmse": float(e2e_rmse.var(ddof=1)),
                "e2e_ensemble_rmse": e2e_ens,
            }
        )

    return pd.DataFrame(rows).sort_values("N").reset_index(drop=True)


def plot_stack(
    df: pd.DataFrame,
    value_cols: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    """Plot smoothed stacked variance components across subset size."""
    df = df.sort_values("N").reset_index(drop=True)
    x = df["N"].to_numpy(dtype=float)
    ys = [df[col].to_numpy(dtype=float) for col in value_cols]

    x_log = np.log10(x)
    x_smooth_log = np.linspace(x_log.min(), x_log.max(), 300)
    x_smooth = 10 ** x_smooth_log

    y_smooth = []
    for y in ys:
        interpolator = PchipInterpolator(x_log, y)
        y_interp = np.clip(interpolator(x_smooth_log), 0.0, None)
        y_smooth.append(y_interp)

    if "Proportion" in ylabel:
        y_sum = np.sum(y_smooth, axis=0)
        y_sum[y_sum == 0] = 1.0
        y_smooth = [y / y_sum for y in y_smooth]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(
        x_smooth,
        *y_smooth,
        labels=labels,
        colors=COMPONENT_COLORS,
        alpha=0.85,
        edgecolor="none",
        linewidth=0.0,
    )

    cumulative = np.cumsum(ys, axis=0)
    for series in cumulative[:-1]:
        ax.scatter(x, series, color="white", edgecolor="black", s=30, zorder=5, linewidth=0.8)

    ax.set_title(title, pad=15, fontweight="bold", fontsize=14)
    ax.set_xlabel("Training Samples ($N$)", labelpad=10)
    ax.set_ylabel(ylabel, labelpad=10)
    ax.set_xscale("log")
    ax.set_xticks(x)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlim(x.min(), x.max())
    if "Proportion" in ylabel:
        ax.set_ylim(0.0, 1.0)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline_comparison(df: pd.DataFrame, out_path: Path) -> None:
    """Plot pipeline-level single-model vs ensemble RMSE comparison."""
    x = df["N"].to_numpy(dtype=float)
    gain_e2e = df["e2e_mean_rmse"] - df["e2e_ensemble_rmse"]
    gain_hybrid = df["hybrid_mean_rmse"] - df["hybrid_ensemble_rmse"]

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(7, 8),
        dpi=300,
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1]},
    )

    c_hybrid = "#d62728"
    c_e2e = "#000000"

    ax1.plot(x, df["hybrid_mean_rmse"], color=c_hybrid, marker="s", markersize=5, linestyle="-", lw=2, label="Hybrid (Mean of 5 Seeds)")
    ax1.fill_between(
        x,
        df["hybrid_mean_rmse"] - df["hybrid_std_rmse"],
        df["hybrid_mean_rmse"] + df["hybrid_std_rmse"],
        color=c_hybrid,
        alpha=0.15,
        lw=0,
    )
    ax1.plot(x, df["hybrid_ensemble_rmse"], color=c_hybrid, marker="s", markersize=5, linestyle="--", lw=2, mfc="white", label="Hybrid (Ensemble)")

    ax1.plot(x, df["e2e_mean_rmse"], color=c_e2e, marker="o", markersize=5, linestyle="-", lw=2, label="End-to-End (Mean of 5 Seeds)")
    ax1.fill_between(
        x,
        df["e2e_mean_rmse"] - df["e2e_std_rmse"],
        df["e2e_mean_rmse"] + df["e2e_std_rmse"],
        color="gray",
        alpha=0.2,
        lw=0,
    )
    ax1.plot(x, df["e2e_ensemble_rmse"], color=c_e2e, marker="o", markersize=5, linestyle="--", lw=2, mfc="white", label="End-to-End (Ensemble)")
    ax1.set_ylabel("Test RMSE (nm)", fontsize=12, fontweight="bold")
    ax1.set_title("A. Convergence of Architectures", loc="left", fontsize=12, fontweight="bold")
    ax1.grid(False)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor="none")

    ax2.plot(x, gain_e2e, color=c_e2e, marker="o", markersize=5, lw=2, label="End-to-End Gain")
    ax2.plot(x, gain_hybrid, color=c_hybrid, marker="s", markersize=5, lw=2, label="Hybrid Gain")
    ax2.fill_between(x, 0, gain_e2e, color=c_e2e, alpha=0.1)
    ax2.fill_between(x, 0, gain_hybrid, color=c_hybrid, alpha=0.1)
    ax2.set_ylabel("RMSE Reduction\nfrom Ensembling (nm)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Training Set Size $N$", fontsize=12, fontweight="bold")
    ax2.set_title("B. Magnitude of Ensemble Gain", loc="left", fontsize=12, fontweight="bold")
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9, loc="upper right", framealpha=0.95, edgecolor="none")

    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(ScalarFormatter())
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(int(v)) for v in x])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=600)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    nested_df = build_nested_summary(args)
    subsets = nested_df["subset"].tolist()
    pipeline_df = collect_pipeline_summary(args.target, args.dataset, args.split, args.model, subsets)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manual_csv = outdir / "variance_components_manual_vs_statsmodels.csv"
    manual_abs_fig = outdir / "variance_components_manual_absolute.png"
    manual_prop_fig = outdir / "variance_components_manual_proportion.png"
    stats_abs_fig = outdir / "variance_components_statsmodels_absolute.png"
    stats_prop_fig = outdir / "variance_components_statsmodels_proportion.png"
    pipeline_csv = outdir / "pipeline_variance_summary.csv"
    pipeline_fig = outdir / "pipeline_variance_comparison.png"

    nested_df.to_csv(manual_csv, index=False)
    pipeline_df.to_csv(pipeline_csv, index=False)

    plot_stack(
        nested_df,
        ["manual_encoder_prop", "manual_head_prop", "manual_residual_prop"],
        labels=COMPONENT_LABELS,
        ylabel="Proportion of Total Variance",
        title="Variance Decomposition vs. Training Size $N$",
        out_path=manual_prop_fig,
    )
    plot_stack(
        nested_df,
        ["statsmodels_encoder_prop", "statsmodels_head_prop", "statsmodels_residual_prop"],
        labels=COMPONENT_LABELS,
        ylabel="Proportion of Total Variance",
        title="Variance Decomposition vs. Training Size $N$",
        out_path=stats_prop_fig,
    )
    if args.include_absolute:
        plot_stack(
            nested_df,
            ["manual_encoder_variance", "manual_head_variance", "manual_residual_variance"],
            labels=["Encoders ($\\sigma_A^2$)", "Heads ($\\sigma_{B(A)}^2$)", "Residual ($\\sigma^2$)"],
            ylabel="Estimated Variance Component",
            title="Manual Nested-ANOVA Variance Components vs. Training Size $N$",
            out_path=manual_abs_fig,
        )
        plot_stack(
            nested_df,
            ["statsmodels_encoder_variance", "statsmodels_head_variance", "statsmodels_residual_variance"],
            labels=["Encoders ($\\sigma_A^2$)", "Heads ($\\sigma_{B(A)}^2$)", "Residual ($\\sigma^2$)"],
            ylabel="Estimated Variance Component",
            title="Statsmodels Variance Components vs. Training Size $N$",
            out_path=stats_abs_fig,
        )
    plot_pipeline_comparison(pipeline_df, pipeline_fig)

    print(f"[write] {manual_csv}")
    print(f"[write] {manual_prop_fig}")
    print(f"[write] {stats_prop_fig}")
    print(f"[write] {pipeline_csv}")
    print(f"[write] {pipeline_fig}")
    if args.include_absolute:
        print(f"[write] {manual_abs_fig}")
        print(f"[write] {stats_abs_fig}")


if __name__ == "__main__":
    main()
