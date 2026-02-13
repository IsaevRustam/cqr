"""
Plot per-bin results (Coverage & Width) for Global CQR vs Localized CQR.

Datasets: bio, community, rf1, scm1d, scm20d

Usage:
    python plot_results_by_bin.py
    python plot_results_by_bin.py --csv results_real_data.csv --save_dir figures_cqr
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ============================================================================
# STYLE
# ============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.2,
})

METHOD_COLORS = {
    "Global CQR": "#3498db",
    "Localized CQR": "#e74c3c",
}
METHOD_HATCHES = {
    "Global CQR": "",
    "Localized CQR": "///",
}
METHODS = ["Global CQR", "Localized CQR"]
DATASETS = ["bio", "community", "rf1", "scm1d", "scm20d"]

DATASET_LABELS = {
    "bio": "Bio",
    "community": "Community",
    "rf1": "RF1",
    "scm1d": "SCM1D",
    "scm20d": "SCM20D",
}


# ============================================================================
# DATA
# ============================================================================

def load_bin_data(csv_path: str = "results_real_data.csv") -> pd.DataFrame:
    """Load CSV and keep only per-bin rows (exclude Overall)."""
    df = pd.read_csv(csv_path)
    df_bins = df[df["Bin"].str.startswith("Bin", na=False)].copy()
    # Extract bin number
    df_bins["Bin Number"] = df_bins["Bin"].str.extract(r"(\d+)").astype(int)
    # Sort by Bin Rank (prediction-interval width rank) for consistent ordering
    df_bins = df_bins.sort_values(["Dataset", "Method", "Bin Rank"])
    return df_bins


# ============================================================================
# FIGURE 1: Per-bin COVERAGE — one subplot per dataset
# ============================================================================

def plot_bin_coverage(df: pd.DataFrame, datasets: list, save_dir: str = "figures_cqr",
                      save_formats=("png",)):
    """Grouped bar chart of per-bin coverage for each dataset."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 3.5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    bar_width = 0.35

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds]
        for k, method in enumerate(METHODS):
            m_data = sub[sub["Method"] == method].sort_values("Bin Rank")
            if m_data.empty:
                continue
            bins = m_data["Bin Rank"].values
            x = np.arange(len(bins))
            offset = (k - 0.5) * bar_width
            means = m_data["Coverage (mean)"].values
            stds = m_data["Coverage (std)"].values

            ax.bar(
                x + offset, means, bar_width,
                label=method if ds == datasets[0] else None,
                color=METHOD_COLORS[method],
                hatch=METHOD_HATCHES[method],
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2,
            )
            ax.errorbar(
                x + offset, means, yerr=stds,
                fmt="none", ecolor="#2c3e50", elinewidth=1.0,
                capsize=3, capthick=0.8, alpha=0.6, zorder=3,
            )

        # 90 % target line
        ax.axhline(0.90, color="#27ae60", ls="--", lw=1.2, alpha=0.7, zorder=1,
                    label="Target 90 %" if ds == datasets[0] else None)

        n_bins = len(sub[sub["Method"] == METHODS[0]])
        ax.set_xticks(np.arange(n_bins))
        ax.set_xticklabels([f"Bin {i+1}" for i in range(n_bins)], rotation=30, ha="right")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontweight="medium")
        ax.grid(axis="y", alpha=0.2, ls="--")
        ax.set_axisbelow(True)

        # Tighten y range around 0.9
        all_cov = sub["Coverage (mean)"].values
        all_std = sub["Coverage (std)"].values
        lo = min(all_cov - all_std) - 0.02
        hi = max(all_cov + all_std) + 0.02
        ax.set_ylim(max(lo, 0.60), min(hi, 1.01))

    axes[0].set_ylabel("Coverage")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=3, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for fmt in save_formats:
        fp = Path(save_dir) / f"bin_coverage.{fmt}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", format=fmt, facecolor="white")
        print(f"Saved: {fp}")
    plt.show()
    plt.close(fig)


# ============================================================================
# FIGURE 2: Per-bin AVG WIDTH — one subplot per dataset
# ============================================================================

def plot_bin_width(df: pd.DataFrame, datasets: list, save_dir: str = "figures_cqr",
                   save_formats=("png",)):
    """Grouped bar chart of per-bin average width for each dataset."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(1, n_ds, figsize=(3.2 * n_ds, 3.5), sharey=False)
    if n_ds == 1:
        axes = [axes]

    bar_width = 0.35

    for ax, ds in zip(axes, datasets):
        sub = df[df["Dataset"] == ds]
        for k, method in enumerate(METHODS):
            m_data = sub[sub["Method"] == method].sort_values("Bin Rank")
            if m_data.empty:
                continue
            bins = m_data["Bin Rank"].values
            x = np.arange(len(bins))
            offset = (k - 0.5) * bar_width
            widths = m_data["Avg Width (mean)"].values

            ax.bar(
                x + offset, widths, bar_width,
                label=method if ds == datasets[0] else None,
                color=METHOD_COLORS[method],
                hatch=METHOD_HATCHES[method],
                edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2,
            )

        n_bins = len(sub[sub["Method"] == METHODS[0]])
        ax.set_xticks(np.arange(n_bins))
        ax.set_xticklabels([f"Bin {i+1}" for i in range(n_bins)], rotation=30, ha="right")
        ax.set_title(DATASET_LABELS.get(ds, ds), fontweight="medium")
        ax.grid(axis="y", alpha=0.2, ls="--")
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Average Interval Width")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=2, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 1.06))

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for fmt in save_formats:
        fp = Path(save_dir) / f"bin_width.{fmt}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", format=fmt, facecolor="white")
        print(f"Saved: {fp}")
    plt.show()
    plt.close(fig)


# ============================================================================
# FIGURE 3: Combined panel — Coverage (top) + Width (bottom) per dataset
# ============================================================================

def plot_bin_combined(df: pd.DataFrame, datasets: list, save_dir: str = "figures_cqr",
                      save_formats=("png",)):
    """Two-row panel: top = coverage per bin, bottom = width per bin."""
    n_ds = len(datasets)
    fig, axes = plt.subplots(2, n_ds, figsize=(3.2 * n_ds, 6.0), sharey="row")

    bar_width = 0.35

    for col, ds in enumerate(datasets):
        sub = df[df["Dataset"] == ds]

        for row, (metric, ylabel) in enumerate([
            ("Coverage (mean)", "Coverage"),
            ("Avg Width (mean)", "Avg Width"),
        ]):
            ax = axes[row, col]
            for k, method in enumerate(METHODS):
                m_data = sub[sub["Method"] == method].sort_values("Bin Rank")
                if m_data.empty:
                    continue
                x = np.arange(len(m_data))
                offset = (k - 0.5) * bar_width
                vals = m_data[metric].values

                ax.bar(
                    x + offset, vals, bar_width,
                    label=method if (col == 0 and row == 0) else None,
                    color=METHOD_COLORS[method],
                    hatch=METHOD_HATCHES[method],
                    edgecolor="white", linewidth=0.5, alpha=0.85, zorder=2,
                )

                # Error bars for coverage
                if row == 0 and "Coverage (std)" in m_data.columns:
                    stds = m_data["Coverage (std)"].values
                    ax.errorbar(
                        x + offset, vals, yerr=stds,
                        fmt="none", ecolor="#2c3e50", elinewidth=1.0,
                        capsize=3, capthick=0.8, alpha=0.6, zorder=3,
                    )

            # Target line for coverage
            if row == 0:
                ax.axhline(0.90, color="#27ae60", ls="--", lw=1.2, alpha=0.7, zorder=1,
                            label="Target 90 %" if col == 0 else None)
                all_cov = sub["Coverage (mean)"].values
                all_std = sub["Coverage (std)"].fillna(0).values
                lo = min(all_cov - all_std) - 0.02
                hi = max(all_cov + all_std) + 0.02
                ax.set_ylim(max(lo, 0.60), min(hi, 1.01))

            n_bins = len(sub[sub["Method"] == METHODS[0]])
            ax.set_xticks(np.arange(n_bins))
            if row == 1:
                ax.set_xticklabels([f"Bin {i+1}" for i in range(n_bins)], rotation=30, ha="right")
            else:
                ax.set_xticklabels([])

            if row == 0:
                ax.set_title(DATASET_LABELS.get(ds, ds), fontweight="medium")

            ax.grid(axis="y", alpha=0.2, ls="--")
            ax.set_axisbelow(True)

        # Y-labels on first column
    axes[0, 0].set_ylabel("Coverage")
    axes[1, 0].set_ylabel("Average Width")

    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                if li not in labels:
                    handles.append(hi)
                    labels.append(li)

    fig.legend(handles, labels, loc="upper center",
               ncol=3, frameon=True, fancybox=True, shadow=False,
               bbox_to_anchor=(0.5, 1.04))

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for fmt in save_formats:
        fp = Path(save_dir) / f"bin_combined.{fmt}"
        fig.savefig(fp, dpi=300, bbox_inches="tight", format=fmt, facecolor="white")
        print(f"Saved: {fp}")
    plt.show()
    plt.close(fig)


# ============================================================================
# SUMMARY TABLE (per bin)
# ============================================================================

def print_bin_summary(df: pd.DataFrame, datasets: list):
    """Print per-bin coverage and width table."""
    print("\n" + "=" * 110)
    print("PER-BIN SUMMARY — Global CQR vs Localized CQR")
    print("=" * 110)

    for ds in datasets:
        sub = df[df["Dataset"] == ds]
        if sub.empty:
            continue
        row0 = sub.iloc[0]
        print(f"\n{'─'*110}")
        print(f"  Dataset: {ds}   (n={int(row0['n'])}, d={int(row0['d'])})")
        print(f"{'─'*110}")
        print(f"  {'Bin':<8} {'Method':<18} {'Coverage':>12} {'Cov Std':>10} {'Avg Width':>12}")
        print(f"  {'─'*8} {'─'*18} {'─'*12} {'─'*10} {'─'*12}")

        for rank in sorted(sub["Bin Rank"].unique()):
            for method in METHODS:
                row = sub[(sub["Bin Rank"] == rank) & (sub["Method"] == method)]
                if row.empty:
                    continue
                r = row.iloc[0]
                print(f"  Bin {int(rank):<4} {method:<18} "
                      f"{r['Coverage (mean)']:>11.4f} {r['Coverage (std)']:>10.4f} "
                      f"{r['Avg Width (mean)']:>12.4f}")
    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Plot per-bin CQR results")
    parser.add_argument("--csv", default="results_real_data.csv", help="Results CSV")
    parser.add_argument("--save_dir", default="figures_cqr", help="Output directory")
    parser.add_argument("--formats", nargs="+", default=["pdf"], help="Save formats")
    args = parser.parse_args()

    df = load_bin_data(args.csv)
    datasets = [ds for ds in DATASETS if ds in df["Dataset"].unique()]

    print(f"Datasets found: {datasets}")
    print(f"Total bin rows: {len(df)}")

    # Console summary
    print_bin_summary(df, datasets)

    # Figures
    plot_bin_coverage(df, datasets, save_dir=args.save_dir, save_formats=tuple(args.formats))
    plot_bin_width(df, datasets, save_dir=args.save_dir, save_formats=tuple(args.formats))
    plot_bin_combined(df, datasets, save_dir=args.save_dir, save_formats=tuple(args.formats))


if __name__ == "__main__":
    main()
