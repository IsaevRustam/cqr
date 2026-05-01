"""
Plot results from results_real_data.csv — Global CQR vs Localized CQR.
ICML publication-ready figures.

Usage:
    python plot_results.py
    python plot_results.py --csv results_real_data.csv --save_dir figures_cqr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from math import ceil
import re
import argparse

# ============================================================================
# STYLE CONFIGURATION — ICML publication
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.2,
})

COLORS = {
    'bar': '#4575b4',
    'error': '#2c3e50',
    'mean': '#e74c3c',
    'baseline': '#95a5a6',
    'target': '#27ae60',
}

# Method-specific colours
METHOD_COLORS = {
    "Global CQR": "#3498db",       # Blue
    "Localized CQR": "#e74c3c",    # Red
}

METHOD_ORDER = ["Global CQR", "Localized CQR"]

# Metrics to plot and their configuration
# Each entry: column for mean, column for std (or None), display name, kwargs
METRICS_CONFIG = {
    "Coverage": {
        "mean_col": "Coverage (mean)",
        "std_col": "Coverage (std)",
        "display": "Coverage (target = 90%)",
        "target_line": 0.9,
        "baseline": "auto",
    },
    "Avg Width": {
        "mean_col": "Avg Width (mean)",
        "std_col": "Avg Width (std)",
        "display": "Average Interval Width (standardized)",
        "target_line": None,
        "baseline": "auto",
    },
    "Worst-Bin Coverage": {
        "mean_col": "Worst-Bin Cov (mean)",
        "std_col": "Worst-Bin Cov (std)",
        "display": "Worst-Bin Coverage",
        "target_line": 0.9,
        "baseline": "auto",
    },
    "Winkler Score": {
        "mean_col": "Winkler Score (mean)",
        "std_col": "Winkler Score (std)",
        "display": "Winkler Score (lower is better)",
        "target_line": None,
        "baseline": "auto",
    },
    "Width-Error Correlation": {
        "mean_col": "Width-Error Corr (mean)",
        "std_col": "Width-Error Corr (std)",
        "display": "Width–Error Correlation (higher is better)",
        "target_line": None,
        "baseline": "auto",
    },
    "CCV": {
        "mean_col": "CCV (mean)",
        "std_col": "CCV (std)",
        "display": "Conditional Coverage Violation (lower is better)",
        "target_line": 0.0,
        "baseline": "auto",
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def load_results(csv_path: str = "results_real_data.csv") -> pd.DataFrame:
    """Load results CSV and filter to Overall rows if Bin column exists."""
    df = pd.read_csv(csv_path)
    # If new format with Bin column, keep only Overall rows
    if "Bin" in df.columns:
        df = df[df["Bin"] == "Overall"].copy()
    # Ensure Activation column exists (backward compat)
    if "Activation" not in df.columns:
        df["Activation"] = "REQU"  # legacy default
    # Ensure new metric columns exist so old CSVs plot gracefully (all NaN)
    for col in [
        "Winkler Score (mean)", "Winkler Score (std)",
        "Width-Error Corr (mean)", "Width-Error Corr (std)",
        "CCV (mean)", "CCV (std)",
    ]:
        if col not in df.columns:
            df[col] = np.nan
    print(f"Loaded {csv_path}: {len(df)} rows")
    print(f"  Datasets:    {df['Dataset'].unique().tolist()}")
    print(f"  Methods:     {df['Method'].unique().tolist()}")
    print(f"  Activations: {df['Activation'].unique().tolist()}")
    return df


# ============================================================================
# PLOTTING — adapted for pre-aggregated (mean/std) data
# ============================================================================

def plot_gauge_bars_on_ax(
    ax, df, dataset, mean_col, std_col=None,
    baseline=0.0,
    method_order=None,
    method_colors=None,
    target_line=None,
    ylim=None,
    show_error_bars=True,
    bar_width=0.55,
    show_n_d=True,
    n_attempts=10,
):
    """
    Plot gauge bars for one dataset on a given axis.

    Works with pre-aggregated data: one row per (Dataset, Method).
    Error bars show ±std (not min/max).
    """
    d = df[df["Dataset"] == dataset].copy()
    if mean_col not in d.columns or d[mean_col].isna().all():
        ax.set_axis_off()
        return

    # Order methods
    if method_order is not None:
        d["Method"] = pd.Categorical(d["Method"], categories=method_order, ordered=True)
        d = d.sort_values("Method").dropna(subset=["Method"]).reset_index(drop=True)
    else:
        d = d.sort_values("Method").reset_index(drop=True)

    methods = d["Method"].astype(str).tolist()
    means = d[mean_col].to_numpy(dtype=float)
    stds = d[std_col].to_numpy(dtype=float) if std_col and std_col in d.columns else np.zeros_like(means)

    # Convert std to 95% CI of the mean: 1.96 * std / sqrt(n)
    ci95 = 1.96 * stds / np.sqrt(n_attempts)

    # Baseline
    bl = baseline
    if bl == "auto":
        data_range = np.nanmax(means + ci95) - np.nanmin(means - ci95) + 1e-12
        bl = float(np.nanmin(means - ci95)) - 1.0 * data_range
    x = np.arange(len(methods))
    bottoms = np.minimum(bl, means)
    heights = np.abs(means - bl)

    # Colours
    colors = [method_colors.get(m, COLORS['bar']) for m in methods] if method_colors else [COLORS['bar']] * len(methods)

    # Bars
    ax.bar(x, heights, bottom=bottoms, width=bar_width,
           color=colors, alpha=0.85, edgecolor='white', linewidth=0.5, zorder=2)

    # Error bars (95% CI)
    if show_error_bars and np.any(ci95 > 0):
        ax.errorbar(x, means, yerr=ci95, fmt='none', ecolor=COLORS['error'],
                     elinewidth=1.5, capsize=4, capthick=1.2, alpha=0.7, zorder=3)

    # Mean diamonds
    ax.scatter(x, means, s=25, color=COLORS['mean'],
               marker='D', zorder=4, edgecolors='white', linewidths=0.5)

    # Baseline
    ax.axhline(bl, color=COLORS['baseline'], linewidth=1.0, linestyle='-', alpha=0.4, zorder=1)

    # Target line
    if target_line is not None:
        ax.axhline(target_line, color=COLORS['target'],
                   linestyle='--', linewidth=1.3, alpha=0.75, zorder=1)

    # Title: dataset name only
    ax.set_title(dataset, fontsize=10, pad=6, fontweight='medium')

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)
    ax.margins(x=0.15)

    # Y-limits
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        lo_vals = means - ci95
        hi_vals = means + ci95
        all_vals = np.r_[lo_vals, hi_vals, bl]
        pad = 0.1 * (np.nanmax(all_vals) - np.nanmin(all_vals) + 1e-12)
        ax.set_ylim(np.nanmin(all_vals) - pad, np.nanmax(all_vals) + pad)

    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#2c3e50')
        spine.set_alpha(0.3)


def plot_one_figure_per_metric(
    df, metric_key, metric_cfg,
    method_order=None,
    method_colors=None,
    datasets=None,
    ncols=4,
    figsize_per_ax=(2.8, 2.8),
    save_dir="figures_cqr",
    dpi=300,
    global_ylim=False,
    show_error_bars=True,
    bar_width=0.55,
    save_formats=('png',),  # pdf
    filename_suffix="",
):
    """Create one figure per metric with all datasets as subplots."""
    if datasets is None:
        datasets = sorted(df["Dataset"].dropna().unique().tolist())

    n = len(datasets)
    if n == 0:
        print(f"No datasets for metric '{metric_key}'")
        return

    ncols = min(ncols, n)
    nrows = ceil(n / ncols)

    fig_w = figsize_per_ax[0] * ncols
    fig_h = figsize_per_ax[1] * nrows

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    mean_col = metric_cfg["mean_col"]
    std_col = metric_cfg.get("std_col")

    # Optional global ylim
    ylim = None
    if global_ylim and mean_col in df.columns:
        vals = df[df["Dataset"].isin(datasets)][mean_col].dropna().astype(float)
        if len(vals):
            vmin, vmax = float(vals.min()), float(vals.max())
            pad = 0.1 * (vmax - vmin + 1e-12)
            ylim = (vmin - pad, vmax + pad)

    for i, ds in enumerate(datasets):
        plot_gauge_bars_on_ax(
            axes_flat[i], df, ds, mean_col,
            std_col=std_col,
            baseline=metric_cfg.get("baseline", 0.0),
            method_order=method_order,
            method_colors=method_colors,
            target_line=metric_cfg.get("target_line"),
            ylim=ylim,
            show_error_bars=show_error_bars,
            bar_width=bar_width,
        )

    for j in range(n, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # fig.suptitle(metric_cfg["display"], fontsize=11, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_key)
    for fmt in save_formats:
        fp = Path(save_dir) / f"{safe}{filename_suffix}.{fmt}"
        fig.savefig(fp, dpi=dpi, bbox_inches="tight", format=fmt, facecolor='white')
        print(f"Saved: {fp}")

    plt.show()
    plt.close(fig)


# ============================================================================
# GRID FIGURE — all metrics × all datasets
# ============================================================================

def plot_all_metrics_grid(
    df,
    metrics_config,
    method_order=None,
    method_colors=None,
    datasets=None,
    figsize_per_ax=(2.6, 2.6),
    save_dir="figures_cqr",
    save_formats=('png',) # pdf
):
    """One big figure: rows = metrics, cols = datasets."""
    if datasets is None:
        datasets = sorted(df["Dataset"].dropna().unique().tolist())

    metric_keys = list(metrics_config.keys())
    n_metrics = len(metric_keys)
    n_datasets = len(datasets)

    fig, axes = plt.subplots(
        n_metrics, n_datasets,
        figsize=(figsize_per_ax[0] * n_datasets, figsize_per_ax[1] * n_metrics),
        squeeze=False,
    )

    for i, mk in enumerate(metric_keys):
        cfg = metrics_config[mk]
        for j, ds in enumerate(datasets):
            show_title = (i == 0)
            ax = axes[i, j]

            plot_gauge_bars_on_ax(
                ax, df, ds, cfg["mean_col"],
                std_col=cfg.get("std_col"),
                baseline=cfg.get("baseline", 0.0),
                method_order=method_order,
                method_colors=method_colors,
                target_line=cfg.get("target_line"),
                show_n_d=show_title,
            )
            if not show_title:
                ax.set_title("")

        # Y-label on first column
        axes[i, 0].set_ylabel(cfg["display"], fontsize=9, fontweight='medium')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    """for fmt in save_formats:
        fp = Path(save_dir) / f"all_metrics_grid.{fmt}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", format=fmt, facecolor='white')
        print(f"Saved: {fp}")

    plt.show()
    plt.close(fig)"""


# ============================================================================
# SUMMARY TABLE
# ============================================================================

def print_summary_table(df, method_order, datasets):
    """Print summary statistics table to console."""
    print("\n" + "=" * 100)
    print("SUMMARY TABLE — Global CQR vs Localized CQR")
    print("=" * 100)

    metrics = [
        ("Coverage",         "Coverage (mean)", "Coverage (std)"),
        ("Avg Width (orig)", "Avg Width (orig)", None),
        ("Worst-Bin Cov",    "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)"),
    ]

    for label, mean_col, std_col in metrics:
        if mean_col not in df.columns:
            continue
        print(f"\n{label}:")
        print("-" * 90)
        for ds in datasets:
            row_str = f"  {ds:22s}"
            for method in method_order:
                sub = df[(df["Dataset"] == ds) & (df["Method"] == method)]
                if len(sub) == 0:
                    continue
                m = float(sub[mean_col].iloc[0])
                if std_col and std_col in sub.columns:
                    s = float(sub[std_col].iloc[0])
                    row_str += f"  {method}: {m:.3f}±{s:.3f}"
                else:
                    row_str += f"  {method}: {m:.3f}"
            print(row_str)
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot CQR results")
    parser.add_argument("--csv", nargs="+", default=["results_real_data.csv"],
                        help="Results CSV(s) — multiple files are concatenated")
    parser.add_argument("--save_dir", default="figures_cqr", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["pdf"], help="Save formats")
    args = parser.parse_args()

    # Load and concatenate all CSVs
    frames = [load_results(p) for p in args.csv]
    df = pd.concat(frames, ignore_index=True)
    datasets = ["bio", "community", "rf1", "scm1d", "scm20d"]  # df["Dataset"].unique().tolist()

    activations = sorted(df["Activation"].unique().tolist())

    for act in activations:
        df_act = df[df["Activation"] == act].copy()
        act_lower = act.lower()
        suffix = f"_{act_lower}" if len(activations) > 1 else ""

        # Print summary
        print_summary_table(df_act, METHOD_ORDER, datasets)

        # Individual metric figures
        for metric_key, metric_cfg in METRICS_CONFIG.items():
            plot_one_figure_per_metric(
                df_act, metric_key, metric_cfg,
                method_order=METHOD_ORDER,
                method_colors=METHOD_COLORS,
                datasets=datasets,
                ncols=len(datasets),
                figsize_per_ax=(2.8, 2.8),
                save_dir=args.save_dir,
                global_ylim=False,
                show_error_bars=True,
                bar_width=0.55,
                save_formats=args.formats,
                filename_suffix=suffix,
            )

        # Combined grid figure
        plot_all_metrics_grid(
            df_act,
            METRICS_CONFIG,
            method_order=METHOD_ORDER,
            method_colors=METHOD_COLORS,
            datasets=datasets,
            figsize_per_ax=(2.6, 2.6),
            save_dir=args.save_dir,
            save_formats=args.formats,
        )


if __name__ == "__main__":
    main()
