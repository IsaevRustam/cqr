"""Build NeurIPS-style table + figures for the Global vs Localized CQR
real-data experiment.

Reads the CSV produced by ``evaluate_real_data.py`` and writes:

* ``table_summary.md`` and ``table_summary.tex`` — one row per dataset, with
  marginal coverage, avg width, conditional coverage range and worst-bin
  coverage for both methods (mean ± std over attempts). Bold marks the
  method that is strictly better on width and coverage range.
* ``fig_summary_metrics.pdf`` — three-panel grouped bar chart across all
  datasets.
* ``fig_efficiency_scatter.pdf`` — paired (Width_G, Width_L) scatter.

Real data has no oracle, so we report raw widths and the **Local/Global**
width ratio in lieu of the oracle-relative efficiency used for the
synthetic regimes.

Usage:
    python make_real_data_figures.py
        --csv figures_cqr/cqr_compare_real/results_real_relu.csv
        --out_dir figures_cqr/cqr_compare_real
        --alpha 0.1
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading + reshaping                                                          #
# --------------------------------------------------------------------------- #

def load_overall(csv_path: str) -> pd.DataFrame:
    """Return only the per-(dataset, method) Overall rows, indexed by dataset."""
    df = pd.read_csv(csv_path)
    df = df[df["Bin"] == "Overall"].copy()
    df["Method"] = df["Method"].replace({"Localized CQR": "Local", "Global CQR": "Global"})
    keep = [
        "Dataset", "n", "d", "Method", "Activation",
        "Coverage (mean)", "Coverage (std)",
        "Avg Width (mean)", "Avg Width (std)",
        "Avg Width (orig)",
        "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)",
        "Cov Range",
    ]
    return df[keep].reset_index(drop=True)


def pivot_metric(df: pd.DataFrame, col: str) -> pd.DataFrame:
    return df.pivot(index="Dataset", columns="Method", values=col)


# --------------------------------------------------------------------------- #
# Table                                                                        #
# --------------------------------------------------------------------------- #

def write_summary_table(df: pd.DataFrame, alpha: float, out_dir: str) -> None:
    cov = pivot_metric(df, "Coverage (mean)")
    cov_std = pivot_metric(df, "Coverage (std)")
    width = pivot_metric(df, "Avg Width (mean)")
    width_std = pivot_metric(df, "Avg Width (std)")
    cov_range = pivot_metric(df, "Cov Range")
    worst = pivot_metric(df, "Worst-Bin Cov (mean)")
    worst_std = pivot_metric(df, "Worst-Bin Cov (std)")
    n_lookup = df.drop_duplicates("Dataset").set_index("Dataset")["n"]
    d_lookup = df.drop_duplicates("Dataset").set_index("Dataset")["d"]

    datasets = sorted(cov.index, key=lambda s: (n_lookup[s], s))
    target = 1.0 - alpha

    md = [
        f"# Global vs Local CQR on real datasets — α={alpha} "
        f"(target coverage {target:.2f}, ReLU NN, 10 attempts)",
        "",
        "| Dataset | n | d | Cov_G | Cov_L | Width_G | Width_L | "
        "Width_L/G | CovRange_G | CovRange_L | WorstCov_G | WorstCov_L |",
        "|---|---:|---:|---|---|---|---|---|---|---|---|---|",
    ]
    tex = [
        r"\begin{tabular}{lrrcccccccccc}",
        r"\toprule",
        r"Dataset & n & d & Cov$_G$ & Cov$_L$ & Width$_G$ & Width$_L$ "
        r"& W$_L$/W$_G$ & CovRng$_G$ & CovRng$_L$ & WorstCov$_G$ & WorstCov$_L$ \\",
        r"\midrule",
    ]

    for ds in datasets:
        c_g, c_l = cov.loc[ds, "Global"], cov.loc[ds, "Local"]
        c_g_s, c_l_s = cov_std.loc[ds, "Global"], cov_std.loc[ds, "Local"]
        w_g, w_l = width.loc[ds, "Global"], width.loc[ds, "Local"]
        w_g_s, w_l_s = width_std.loc[ds, "Global"], width_std.loc[ds, "Local"]
        ratio = w_l / w_g if w_g else float("nan")
        cr_g, cr_l = cov_range.loc[ds, "Global"], cov_range.loc[ds, "Local"]
        wb_g, wb_l = worst.loc[ds, "Global"], worst.loc[ds, "Local"]
        wb_g_s, wb_l_s = worst_std.loc[ds, "Global"], worst_std.loc[ds, "Local"]

        # Bold the better-performing method on each metric where Local is preferred
        # only when it is *both* tighter on width *and* still covers within
        # 0.02 of the target (so we don't reward undercoverage).
        local_covers = abs(c_l - target) <= 0.02
        local_w_better = (w_l < w_g) and local_covers
        local_cr_better = cr_l < cr_g and local_covers
        local_wb_better = wb_l > wb_g and local_covers

        def md_pair(v, s, b):
            t = f"{v:.3f}±{s:.3f}"
            return f"**{t}**" if b else t

        def tex_pair(v, s, b):
            inner = f"{v:.3f}\\pm{s:.3f}"
            return r"$\mathbf{" + f"{v:.3f}" + r"}\pm" + f"{s:.3f}$" if b else f"${inner}$"

        md.append(
            f"| {ds} | {int(n_lookup[ds])} | {int(d_lookup[ds])} "
            f"| {c_g:.3f}±{c_g_s:.3f} | {c_l:.3f}±{c_l_s:.3f} "
            f"| {md_pair(w_g, w_g_s, not local_w_better)} "
            f"| {md_pair(w_l, w_l_s, local_w_better)} "
            f"| {ratio:.3f} "
            f"| {cr_g:.3f}{' '*(int(local_cr_better) and 0)} "
            f"| {cr_l:.3f} "
            f"| {wb_g:.3f}±{wb_g_s:.3f} "
            f"| {wb_l:.3f}±{wb_l_s:.3f} |"
        )
        tex.append(
            f"{ds} & {int(n_lookup[ds])} & {int(d_lookup[ds])} "
            f"& ${c_g:.3f}\\pm{c_g_s:.3f}$ & ${c_l:.3f}\\pm{c_l_s:.3f}$ "
            f"& {tex_pair(w_g, w_g_s, not local_w_better)} "
            f"& {tex_pair(w_l, w_l_s, local_w_better)} "
            f"& ${ratio:.3f}$ "
            f"& ${cr_g:.3f}$ & ${cr_l:.3f}$ "
            f"& ${wb_g:.3f}\\pm{wb_g_s:.3f}$ & ${wb_l:.3f}\\pm{wb_l_s:.3f}$ \\\\"
        )

    tex.append(r"\bottomrule")
    tex.append(r"\end{tabular}")

    md_path = os.path.join(out_dir, "table_summary.md")
    tex_path = os.path.join(out_dir, "table_summary.tex")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
        f.write(
            "\n\nLegend: Cov = marginal coverage; Width = mean prediction-interval "
            "width on standardized targets; W$_L$/W$_G$ = ratio Local/Global "
            "(<1 means Local is tighter); CovRng = max−min coverage across 5 PCA "
            "bins; WorstCov = lowest-coverage bin. Subscript G = Global CQR, "
            "L = Localized CQR. **Bold** = Local strictly better on the metric "
            "*and* still covers within 0.02 of the target.\n"
        )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex))
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def figure_summary_metrics(df: pd.DataFrame, alpha: float, out_dir: str) -> None:
    import matplotlib.pyplot as plt

    cov = pivot_metric(df, "Coverage (mean)")
    cov_std = pivot_metric(df, "Coverage (std)")
    width = pivot_metric(df, "Avg Width (mean)")
    width_std = pivot_metric(df, "Avg Width (std)")
    cov_range = pivot_metric(df, "Cov Range")
    n_lookup = df.drop_duplicates("Dataset").set_index("Dataset")["n"]

    target = 1.0 - alpha
    datasets = sorted(cov.index, key=lambda s: (n_lookup[s], s))
    x = np.arange(len(datasets))
    w = 0.4

    cov_gap_g = np.abs(cov.loc[datasets, "Global"].values - target)
    cov_gap_l = np.abs(cov.loc[datasets, "Local"].values - target)
    w_g = width.loc[datasets, "Global"].values
    w_l = width.loc[datasets, "Local"].values
    w_g_s = width_std.loc[datasets, "Global"].values
    w_l_s = width_std.loc[datasets, "Local"].values
    cr_g = cov_range.loc[datasets, "Global"].values
    cr_l = cov_range.loc[datasets, "Local"].values
    cov_g_s = cov_std.loc[datasets, "Global"].values
    cov_l_s = cov_std.loc[datasets, "Local"].values

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 9.0), sharex=True)

    # Marginal-coverage gap
    axes[0].bar(x - w / 2, cov_gap_g, w, yerr=cov_g_s, label="Global",
                color="#d62728", capsize=2)
    axes[0].bar(x + w / 2, cov_gap_l, w, yerr=cov_l_s, label="Local",
                color="#1f77b4", capsize=2)
    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel(r"$|\widehat{C} - (1-\alpha)|$")
    axes[0].set_title(
        f"Marginal-coverage gap (target {target:.2f}) — lower is better"
    )
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    # Width ratio Local/Global (single bar)
    ratio = w_l / np.where(w_g == 0, np.nan, w_g)
    bar_color = ["#1f77b4" if r < 1 else "#d62728" for r in ratio]
    axes[1].bar(x, ratio, w * 1.5, color=bar_color, edgecolor="black", linewidth=0.4)
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8, label="parity")
    axes[1].set_ylabel(r"$W_{\mathrm{Local}} / W_{\mathrm{Global}}$")
    axes[1].set_title(
        "Interval-width ratio Local/Global — below 1.0 means Local tighter"
    )

    # Conditional coverage range
    axes[2].bar(x - w / 2, cr_g, w, label="Global", color="#d62728")
    axes[2].bar(x + w / 2, cr_l, w, label="Local", color="#1f77b4")
    axes[2].set_ylabel("max − min bin coverage")
    axes[2].set_title("Conditional-coverage range across 5 PCA bins — lower is better")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(datasets, rotation=35, ha="right")

    fig.suptitle(
        f"Global vs Localized CQR on real datasets "
        f"(α={alpha}, ReLU NN, mean ± std over 10 attempts)",
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    for ext, dpi in [(".pdf", None), (".png", 180)]:
        path = os.path.join(out_dir, f"fig_summary_metrics{ext}")
        kw = {"bbox_inches": "tight"}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"Wrote {os.path.join(out_dir, 'fig_summary_metrics.pdf')}")


def figure_efficiency_scatter(df: pd.DataFrame, alpha: float, out_dir: str) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    width = pivot_metric(df, "Avg Width (mean)")
    cov = pivot_metric(df, "Coverage (mean)")
    n_lookup = df.drop_duplicates("Dataset").set_index("Dataset")["n"]
    d_lookup = df.drop_duplicates("Dataset").set_index("Dataset")["d"]
    datasets = sorted(width.index, key=lambda s: (n_lookup[s], s))

    target = 1.0 - alpha
    fig, ax = plt.subplots(figsize=(7.0, 6.5))

    g_w = width.loc[datasets, "Global"].values
    l_w = width.loc[datasets, "Local"].values
    g_c = cov.loc[datasets, "Global"].values
    l_c = cov.loc[datasets, "Local"].values

    lo = float(min(g_w.min(), l_w.min())) * 0.9
    hi = float(max(g_w.max(), l_w.max())) * 1.1
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8, label="parity")

    # Color-code by whether Local maintains coverage. Marker size by n.
    sizes = np.clip(np.log10(n_lookup.loc[datasets].values) * 35, 30, 220)
    for i, ds in enumerate(datasets):
        local_covers = abs(l_c[i] - target) <= 0.02
        color = "#1f77b4" if local_covers else "#d62728"
        ax.scatter(g_w[i], l_w[i], s=sizes[i], color=color,
                   edgecolor="black", linewidth=0.5, zorder=5)
        ax.annotate(
            f"{ds}\n(n={int(n_lookup[ds])}, d={int(d_lookup[ds])})",
            (g_w[i], l_w[i]), xytext=(5, 5), textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Global CQR  avg width")
    ax.set_ylabel(r"Local CQR   avg width")
    ax.set_title(
        f"Width comparison per dataset  "
        f"(below diagonal = Local tighter, α={alpha}, ReLU NN)"
    )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#1f77b4",
               markeredgecolor="black", markersize=8,
               label=f"Local covers within ±0.02 of {target:.2f}"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor="#d62728",
               markeredgecolor="black", markersize=8,
               label=f"Local undercovers"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True)
    fig.tight_layout()
    for ext, dpi in [(".pdf", None), (".png", 180)]:
        path = os.path.join(out_dir, f"fig_efficiency_scatter{ext}")
        kw = {"bbox_inches": "tight"}
        if dpi:
            kw["dpi"] = dpi
        fig.savefig(path, **kw)
    plt.close(fig)
    print(f"Wrote {os.path.join(out_dir, 'fig_efficiency_scatter.pdf')}")


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--csv", type=str,
        default="figures_cqr/cqr_compare_real/results_real_relu.csv",
    )
    p.add_argument(
        "--out_dir", type=str, default="figures_cqr/cqr_compare_real",
    )
    p.add_argument("--alpha", type=float, default=0.1)
    args = p.parse_args(argv)

    df = load_overall(args.csv)
    print(
        f"Loaded {args.csv}: {df['Dataset'].nunique()} datasets, "
        f"{len(df)} rows."
    )

    write_summary_table(df, alpha=args.alpha, out_dir=args.out_dir)
    figure_summary_metrics(df, alpha=args.alpha, out_dir=args.out_dir)
    figure_efficiency_scatter(df, alpha=args.alpha, out_dir=args.out_dir)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
