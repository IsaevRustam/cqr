"""Compare Global CQR vs Localized CQR (kernel-weighted) across all 26
synthetic regimes from :mod:`synthetic_regimes`.

Pipeline (per regime, per seed):

1. Sample independent train, calibration, and test splits from the regime.
2. Train two ReLU quantile-regression networks (lower & upper) on the train
   split with pinball loss.
3. Compute conformity scores on the calibration split.
4. Apply Global CQR (scalar offset) and Localized CQR (Epanechnikov-weighted,
   bandwidth ``h = m^{-1/3}``) to obtain prediction intervals on the test
   split.
5. Compute marginal coverage, mean width, conditional coverage (5 bins on x),
   and the width / oracle-width efficiency ratio.

The script writes a per-run CSV plus a paper-ready table and three figures
under ``figures_cqr/cqr_compare/``. CLI flags allow shrinking ``--n_seeds``
or selecting a regime subset for quick iteration.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from synthetic_regimes import sample, list_regimes
from cqr.training import train_quantile_models_unified
from cqr.calibration import (
    global_calibration,
    LocalConformalOptimizer,
    compute_bandwidth,
    compute_conformity_scores,
)
from cqr.metrics import evaluate_intervals


OUT_DIR = os.path.join("figures_cqr", "cqr_compare")
os.makedirs(OUT_DIR, exist_ok=True)


@dataclass
class Config:
    alpha: float = 0.10
    n_train: int = 2500
    n_cal: int = 2500
    n_test: int = 5000
    n_seeds: int = 10
    hidden_dim: int = 64
    n_layers: int = 2
    epochs: int = 300
    lr: float = 0.01
    activation: str = "relu"
    bandwidth_scale: float = 1.0


# --------------------------------------------------------------------------- #
# Per-(regime, seed) experiment                                               #
# --------------------------------------------------------------------------- #

def _to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(x, dtype=np.float32).reshape(-1, 1))


def run_one(regime: str, seed: int, cfg: Config) -> list[dict]:
    """Run a single (regime, seed) replicate; returns rows for Global and Local."""
    base = seed * 7919  # cheap deterministic offset to avoid overlap across seeds
    d_tr = sample(regime, n=cfg.n_train, alpha=cfg.alpha, seed=base + 0)
    d_ca = sample(regime, n=cfg.n_cal, alpha=cfg.alpha, seed=base + 1)
    d_te = sample(regime, n=cfg.n_test, alpha=cfg.alpha, seed=base + 2)

    X_tr = _to_tensor(d_tr["X"])
    Y_tr = _to_tensor(d_tr["Y"])
    X_ca = _to_tensor(d_ca["X"])
    X_te = _to_tensor(d_te["X"])

    m_lo, m_hi = train_quantile_models_unified(
        X_tr,
        Y_tr,
        tau_low=cfg.alpha / 2.0,
        tau_high=1.0 - cfg.alpha / 2.0,
        input_dim=1,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        epochs=cfg.epochs,
        lr=cfg.lr,
        activation=cfg.activation,
        seed=seed,
        verbose=False,
    )
    m_lo.eval()
    m_hi.eval()
    with torch.no_grad():
        pred_ca_lo = m_lo(X_ca).numpy().flatten()
        pred_ca_hi = m_hi(X_ca).numpy().flatten()
        pred_te_lo = m_lo(X_te).numpy().flatten()
        pred_te_hi = m_hi(X_te).numpy().flatten()

    # Conformal scores on calibration split
    scores = compute_conformity_scores(pred_ca_lo, pred_ca_hi, d_ca["Y"])

    # ---------- Global CQR ----------
    Q_g = global_calibration(scores, cfg.alpha)
    lo_g = pred_te_lo - Q_g
    hi_g = pred_te_hi + Q_g

    # ---------- Local CQR ----------
    h = compute_bandwidth(cfg.n_cal, d=1, gamma=1.0, scale=cfg.bandwidth_scale)
    lco = LocalConformalOptimizer(d_ca["X"], scores, h=h)
    Q_l = lco.predict_corrections(d_te["X"], cfg.alpha)
    lo_l = pred_te_lo - Q_l
    hi_l = pred_te_hi + Q_l

    # ---------- Metrics ----------
    oracle_width = float(np.mean(d_te["oracle_hi"] - d_te["oracle_lo"]))

    rows = []
    for tag, lo, hi, q_obj in [
        ("Global", lo_g, hi_g, Q_g),
        ("Local", lo_l, hi_l, Q_l),
    ]:
        m = evaluate_intervals(
            d_te["Y"], lo, hi, d_te["X"], alpha=cfg.alpha, n_bins=5
        )
        widths = hi - lo
        oracle_widths = d_te["oracle_hi"] - d_te["oracle_lo"]
        rows.append(
            dict(
                regime=regime,
                method=tag,
                seed=seed,
                coverage=m["coverage"],
                avg_width=m["avg_width"],
                median_width=m["median_width"],
                width_std=m["width_std"],
                width_ratio_to_oracle=float(np.mean(widths / oracle_widths)),
                width_ratio_marginal=float(m["avg_width"] / oracle_width),
                worst_bin_cov=m["worst_bin_cov"],
                best_bin_cov=m["best_bin_cov"],
                coverage_gap_mean=m["coverage_gap"],
                coverage_gap_max=m["coverage_gap_max"],
                coverage_range=m["coverage_range"],
                oracle_width=oracle_width,
                bandwidth_h=h,
                Q_global=Q_g if tag == "Global" else np.nan,
                Q_local_mean=float(np.mean(Q_l)) if tag == "Local" else np.nan,
                Q_local_std=float(np.std(Q_l)) if tag == "Local" else np.nan,
                winkler_score=m["winkler_score"],
                width_error_corr=m["width_error_corr"],
                ccv=m["ccv"],
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #

def run_all(cfg: Config, regimes: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    t0 = time.time()
    total = len(regimes) * cfg.n_seeds
    done = 0
    for r in regimes:
        for s in range(cfg.n_seeds):
            t1 = time.time()
            rows.extend(run_one(r, s, cfg))
            done += 1
            elapsed = time.time() - t0
            eta = elapsed / done * (total - done)
            print(
                f"[{done:>3}/{total}] {r} seed={s}  "
                f"({time.time()-t1:.1f}s)  elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m",
                flush=True,
            )
    df = pd.DataFrame(rows)
    return df


def aggregate(df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Mean +/- std over seeds per (regime, method)."""
    agg = (
        df.groupby(["regime", "method"])
        .agg(
            coverage_mean=("coverage", "mean"),
            coverage_std=("coverage", "std"),
            width_mean=("avg_width", "mean"),
            width_std=("avg_width", "std"),
            width_ratio_mean=("width_ratio_to_oracle", "mean"),
            width_ratio_std=("width_ratio_to_oracle", "std"),
            cov_range_mean=("coverage_range", "mean"),
            cov_range_std=("coverage_range", "std"),
            worst_cov_mean=("worst_bin_cov", "mean"),
            worst_cov_std=("worst_bin_cov", "std"),
            cov_gap_mean=("coverage_gap_mean", "mean"),
            cov_gap_std=("coverage_gap_mean", "std"),
            winkler_mean=("winkler_score", "mean"),
            winkler_std=("winkler_score", "std"),
            width_error_corr_mean=("width_error_corr", "mean"),
            width_error_corr_std=("width_error_corr", "std"),
            ccv_mean=("ccv", "mean"),
            ccv_std=("ccv", "std"),
        )
        .reset_index()
    )
    agg["target_coverage"] = 1.0 - alpha
    agg["coverage_abs_gap_mean"] = (agg["coverage_mean"] - agg["target_coverage"]).abs()
    return agg


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #

def write_summary_table(agg: pd.DataFrame, alpha: float, out_dir: str) -> None:
    """Wide-format table: rows = regime, columns = method × metric."""
    pivot_cov = agg.pivot(index="regime", columns="method", values="coverage_mean")
    pivot_cov_std = agg.pivot(index="regime", columns="method", values="coverage_std")
    pivot_wr = agg.pivot(index="regime", columns="method", values="width_ratio_mean")
    pivot_wr_std = agg.pivot(index="regime", columns="method", values="width_ratio_std")
    pivot_cr = agg.pivot(index="regime", columns="method", values="cov_range_mean")
    pivot_cr_std = agg.pivot(index="regime", columns="method", values="cov_range_std")
    pivot_wk = agg.pivot(index="regime", columns="method", values="winkler_mean")
    pivot_wk_std = agg.pivot(index="regime", columns="method", values="winkler_std")
    pivot_wec = agg.pivot(index="regime", columns="method", values="width_error_corr_mean")
    pivot_wec_std = agg.pivot(index="regime", columns="method", values="width_error_corr_std")
    pivot_ccv = agg.pivot(index="regime", columns="method", values="ccv_mean")
    pivot_ccv_std = agg.pivot(index="regime", columns="method", values="ccv_std")

    regimes = sorted(pivot_cov.index, key=_regime_sort_key)
    target = 1.0 - alpha

    md_lines: list[str] = []
    md_lines.append(f"# Global vs Local CQR — α={alpha} (target coverage {target:.2f})\n")
    md_lines.append(
        "| Regime | Cov_G | Cov_L | WR_G | WR_L | CovRange_G | CovRange_L"
        " | Winkler_G | Winkler_L | WEC_G | WEC_L | CCV_G | CCV_L |"
    )
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    tex_lines: list[str] = []
    tex_lines.append(r"\begin{tabular}{lcccccccccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r"Regime & Cov$_G$ & Cov$_L$ & WR$_G$ & WR$_L$ & "
        r"CovRange$_G$ & CovRange$_L$ & "
        r"Winkler$_G$ & Winkler$_L$ & WEC$_G$ & WEC$_L$ & CCV$_G$ & CCV$_L$ \\"
    )
    tex_lines.append(r"\midrule")

    for r in regimes:
        # bold the better width ratio and tighter coverage range
        wr_g, wr_l = pivot_wr.loc[r, "Global"], pivot_wr.loc[r, "Local"]
        cr_g, cr_l = pivot_cr.loc[r, "Global"], pivot_cr.loc[r, "Local"]
        wr_g_std, wr_l_std = pivot_wr_std.loc[r, "Global"], pivot_wr_std.loc[r, "Local"]
        cr_g_std, cr_l_std = pivot_cr_std.loc[r, "Global"], pivot_cr_std.loc[r, "Local"]
        c_g, c_l = pivot_cov.loc[r, "Global"], pivot_cov.loc[r, "Local"]
        c_g_std, c_l_std = pivot_cov_std.loc[r, "Global"], pivot_cov_std.loc[r, "Local"]
        wk_g, wk_l = pivot_wk.loc[r, "Global"], pivot_wk.loc[r, "Local"]
        wk_g_std, wk_l_std = pivot_wk_std.loc[r, "Global"], pivot_wk_std.loc[r, "Local"]
        wec_g, wec_l = pivot_wec.loc[r, "Global"], pivot_wec.loc[r, "Local"]
        wec_g_std, wec_l_std = pivot_wec_std.loc[r, "Global"], pivot_wec_std.loc[r, "Local"]
        ccv_g, ccv_l = pivot_ccv.loc[r, "Global"], pivot_ccv.loc[r, "Local"]
        ccv_g_std, ccv_l_std = pivot_ccv_std.loc[r, "Global"], pivot_ccv_std.loc[r, "Local"]

        # Local "wins" on width if its ratio is closer to 1 (smaller in absolute deviation).
        # For width-ratio we prefer values closer to 1; for coverage-range / Winkler / CCV
        # smaller is better; for Width-Error Correlation higher is better.
        local_wr_better = abs(wr_l - 1.0) < abs(wr_g - 1.0)
        local_cr_better = cr_l < cr_g
        local_wk_better = wk_l < wk_g
        local_wec_better = wec_l > wec_g
        local_ccv_better = ccv_l < ccv_g

        def _fmt(v, s, bold_md, bold_tex):
            md = f"{v:.3f}±{s:.3f}"
            tex = f"${v:.3f} \\pm {s:.3f}$"
            if bold_md:
                md = f"**{md}**"
            if bold_tex:
                tex = r"$\mathbf{" + f"{v:.3f}" + r"}\pm" + f"{s:.3f}$"
            return md, tex

        wr_g_md, wr_g_tex = _fmt(wr_g, wr_g_std, not local_wr_better, not local_wr_better)
        wr_l_md, wr_l_tex = _fmt(wr_l, wr_l_std, local_wr_better, local_wr_better)
        cr_g_md, cr_g_tex = _fmt(cr_g, cr_g_std, not local_cr_better, not local_cr_better)
        cr_l_md, cr_l_tex = _fmt(cr_l, cr_l_std, local_cr_better, local_cr_better)
        wk_g_md, wk_g_tex = _fmt(wk_g, wk_g_std, not local_wk_better, not local_wk_better)
        wk_l_md, wk_l_tex = _fmt(wk_l, wk_l_std, local_wk_better, local_wk_better)
        wec_g_md, wec_g_tex = _fmt(wec_g, wec_g_std, not local_wec_better, not local_wec_better)
        wec_l_md, wec_l_tex = _fmt(wec_l, wec_l_std, local_wec_better, local_wec_better)
        ccv_g_md, ccv_g_tex = _fmt(ccv_g, ccv_g_std, not local_ccv_better, not local_ccv_better)
        ccv_l_md, ccv_l_tex = _fmt(ccv_l, ccv_l_std, local_ccv_better, local_ccv_better)

        md_lines.append(
            f"| {r} | {c_g:.3f}±{c_g_std:.3f} | {c_l:.3f}±{c_l_std:.3f} "
            f"| {wr_g_md} | {wr_l_md} | {cr_g_md} | {cr_l_md} "
            f"| {wk_g_md} | {wk_l_md} | {wec_g_md} | {wec_l_md} "
            f"| {ccv_g_md} | {ccv_l_md} |"
        )
        tex_lines.append(
            f"{r} & ${c_g:.3f}\\pm{c_g_std:.3f}$ & ${c_l:.3f}\\pm{c_l_std:.3f}$ "
            f"& {wr_g_tex} & {wr_l_tex} & {cr_g_tex} & {cr_l_tex} "
            f"& {wk_g_tex} & {wk_l_tex} & {wec_g_tex} & {wec_l_tex} "
            f"& {ccv_g_tex} & {ccv_l_tex} \\\\"
        )

    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")

    md_path = os.path.join(out_dir, "table_summary.md")
    tex_path = os.path.join(out_dir, "table_summary.tex")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        f.write(
            "\n\nLegend: Cov = marginal coverage, WR = width ratio (mean of "
            "interval / oracle widths, target = 1.0), CovRange = max−min coverage "
            "across 5 PCA bins, Winkler = mean Winkler/interval score (lower is better), "
            "WEC = width–error Pearson correlation (higher is better), "
            "CCV = conditional coverage violation / mean absolute bin-coverage deviation "
            "from nominal (lower is better). "
            "Subscript G = Global CQR, L = Local CQR. Bold marks the better method "
            "per metric.\n"
        )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(tex_lines))
    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")


def _regime_sort_key(r: str) -> tuple:
    """Sort A1, B1..B6, C1..C5, D1..D4, E1..E5, F1..F3, H1..H2."""
    series = r[0]
    try:
        idx = int(r[1:])
    except ValueError:
        idx = 0
    series_order = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "H": 6}
    return (series_order.get(series, 99), idx)


# --------------------------------------------------------------------------- #
# Figures                                                                     #
# --------------------------------------------------------------------------- #

def _qualitative_panel(ax, regime: str, cfg: Config) -> None:
    """For one regime, draw scatter + Global / Local / Oracle bands using a single seed."""
    base = 0
    d_tr = sample(regime, n=cfg.n_train, alpha=cfg.alpha, seed=base + 0)
    d_ca = sample(regime, n=cfg.n_cal, alpha=cfg.alpha, seed=base + 1)
    d_te = sample(regime, n=cfg.n_test, alpha=cfg.alpha, seed=base + 2)
    X_tr = _to_tensor(d_tr["X"])
    Y_tr = _to_tensor(d_tr["Y"])
    X_ca = _to_tensor(d_ca["X"])

    m_lo, m_hi = train_quantile_models_unified(
        X_tr, Y_tr,
        tau_low=cfg.alpha / 2.0, tau_high=1.0 - cfg.alpha / 2.0,
        input_dim=1, hidden_dim=cfg.hidden_dim, n_layers=cfg.n_layers,
        epochs=cfg.epochs, lr=cfg.lr, activation=cfg.activation,
        seed=0, verbose=False,
    )
    m_lo.eval(); m_hi.eval()
    with torch.no_grad():
        pred_ca_lo = m_lo(X_ca).numpy().flatten()
        pred_ca_hi = m_hi(X_ca).numpy().flatten()

    scores = compute_conformity_scores(pred_ca_lo, pred_ca_hi, d_ca["Y"])
    Q_g = global_calibration(scores, cfg.alpha)
    h = compute_bandwidth(cfg.n_cal, d=1, gamma=1.0, scale=cfg.bandwidth_scale)
    lco = LocalConformalOptimizer(d_ca["X"], scores, h=h)

    # Plot on a sorted x-grid covering the test support.
    xg = np.linspace(d_te["X"].min(), d_te["X"].max(), 600)
    Xg = torch.tensor(xg.reshape(-1, 1), dtype=torch.float32)
    with torch.no_grad():
        pg_lo = m_lo(Xg).numpy().flatten()
        pg_hi = m_hi(Xg).numpy().flatten()
    Q_local_grid = lco.predict_corrections(xg, cfg.alpha)

    # Oracle on grid (re-sample very small grid through same regime mu/sigma path
    # by using a fresh n=600 sample at the grid x's is awkward — instead fit a
    # KNN-style oracle via the test set). Use the test set's oracle directly:
    order = np.argsort(d_te["X"])
    xs = d_te["X"][order]
    olo = d_te["oracle_lo"][order]
    ohi = d_te["oracle_hi"][order]

    ax.scatter(d_te["X"], d_te["Y"], s=4, alpha=0.18, color="0.4", label="data")
    ax.plot(xs, olo, color="black", lw=1.2, ls="--", label="oracle")
    ax.plot(xs, ohi, color="black", lw=1.2, ls="--")
    ax.plot(xg, pg_lo - Q_g, color="#d62728", lw=1.4, label="Global CQR")
    ax.plot(xg, pg_hi + Q_g, color="#d62728", lw=1.4)
    ax.plot(xg, pg_lo - Q_local_grid, color="#1f77b4", lw=1.4, label="Local CQR")
    ax.plot(xg, pg_hi + Q_local_grid, color="#1f77b4", lw=1.4)
    ax.set_title(f"{regime}: {d_te['meta']['sigma_desc']}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def figure_qualitative_bands(
    cfg: Config,
    out_dir: str,
    regimes: tuple[str, ...] = ("B2", "D3", "H1"),
    n_cols: int = 4,
    out_name: str = "fig_qualitative_bands",
) -> None:
    """Multi-panel qualitative figure: each panel shows scatter + Global / Local /
    Oracle bands for one regime. Lays out as a grid when len(regimes) > n_cols.
    """
    import matplotlib.pyplot as plt

    regimes = tuple(sorted(regimes, key=_regime_sort_key))
    n = len(regimes)
    if n <= n_cols:
        n_rows = 1
        n_cols_eff = n
    else:
        n_cols_eff = n_cols
        n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols_eff,
        figsize=(4.2 * n_cols_eff, 3.2 * n_rows),
        squeeze=False,
    )

    for i, r in enumerate(regimes):
        row, col = divmod(i, n_cols_eff)
        ax = axes[row][col]
        _qualitative_panel(ax, r, cfg)
        # Drop redundant axis labels except on the outer edges
        if row != n_rows - 1:
            ax.set_xlabel("")
        if col != 0:
            ax.set_ylabel("")

    # Hide leftover empty axes
    for j in range(n, n_rows * n_cols_eff):
        row, col = divmod(j, n_cols_eff)
        axes[row][col].axis("off")

    # One figure-level legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(
        f"Global vs Localized CQR — qualitative bands (α={cfg.alpha})",
        y=1.00,
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    out_pdf = os.path.join(out_dir, f"{out_name}.pdf")
    out_png = os.path.join(out_dir, f"{out_name}.png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Wrote {out_pdf}")


def figure_summary_metrics(agg: pd.DataFrame, alpha: float, out_dir: str) -> None:
    """3-row grouped bar chart over all regimes:
       (top)  marginal-coverage gap |C - (1-α)|
       (mid)  width / oracle-width
       (bot)  conditional coverage range across bins
    """
    import matplotlib.pyplot as plt

    regimes = sorted(agg["regime"].unique(), key=_regime_sort_key)
    target = 1.0 - alpha

    def _series(metric):
        g = agg[agg.method == "Global"].set_index("regime").loc[regimes, metric].values
        l = agg[agg.method == "Local"].set_index("regime").loc[regimes, metric].values
        return g, l

    cov_g, cov_l = _series("coverage_mean")
    cov_gap_g = np.abs(cov_g - target)
    cov_gap_l = np.abs(cov_l - target)
    wr_g, wr_l = _series("width_ratio_mean")
    cr_g, cr_l = _series("cov_range_mean")
    wr_g_std, wr_l_std = _series("width_ratio_std")
    cr_g_std, cr_l_std = _series("cov_range_std")
    cov_g_std, cov_l_std = _series("coverage_std")

    x = np.arange(len(regimes))
    w = 0.4

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 9.0), sharex=True)

    # --- Coverage gap ---
    axes[0].bar(x - w / 2, cov_gap_g, w, yerr=cov_g_std, label="Global",
                color="#d62728", capsize=2)
    axes[0].bar(x + w / 2, cov_gap_l, w, yerr=cov_l_std, label="Local",
                color="#1f77b4", capsize=2)
    axes[0].axhline(0.0, color="0.5", lw=0.8)
    axes[0].set_ylabel(r"$|\widehat{C} - (1-\alpha)|$")
    axes[0].set_title(
        f"Marginal-coverage gap (target {target:.2f}) — lower is better"
    )
    axes[0].legend(loc="upper right", frameon=False, ncol=2)

    # --- Width ratio to oracle ---
    axes[1].bar(x - w / 2, wr_g, w, yerr=wr_g_std, label="Global",
                color="#d62728", capsize=2)
    axes[1].bar(x + w / 2, wr_l, w, yerr=wr_l_std, label="Local",
                color="#1f77b4", capsize=2)
    axes[1].axhline(1.0, color="0.5", ls="--", lw=0.8, label="oracle")
    axes[1].set_ylabel(r"width / oracle-width")
    axes[1].set_title("Interval-width efficiency — closer to 1 is better")

    # --- Conditional coverage range ---
    axes[2].bar(x - w / 2, cr_g, w, yerr=cr_g_std, label="Global",
                color="#d62728", capsize=2)
    axes[2].bar(x + w / 2, cr_l, w, yerr=cr_l_std, label="Local",
                color="#1f77b4", capsize=2)
    axes[2].set_ylabel("max − min bin coverage")
    axes[2].set_title("Conditional-coverage range across 5 bins — lower is better")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(regimes, rotation=45, ha="right")

    fig.suptitle(
        f"Global vs Localized CQR across 26 synthetic regimes "
        f"(α={alpha}, mean ± std over seeds)",
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


def figure_efficiency_scatter(agg: pd.DataFrame, alpha: float, out_dir: str) -> None:
    """For each regime: (width_ratio_Global, width_ratio_Local). Below diagonal => Local wins."""
    import matplotlib.pyplot as plt

    regimes = sorted(agg["regime"].unique(), key=_regime_sort_key)
    g_x = []
    l_y = []
    series = []
    for r in regimes:
        sub = agg[agg.regime == r]
        g_x.append(float(sub[sub.method == "Global"]["width_ratio_mean"].iloc[0]))
        l_y.append(float(sub[sub.method == "Local"]["width_ratio_mean"].iloc[0]))
        series.append(r[0])

    series_colors = {
        "A": "#1f77b4", "B": "#2ca02c", "C": "#ff7f0e",
        "D": "#9467bd", "E": "#d62728", "F": "#8c564b", "H": "#17becf",
    }
    colors = [series_colors[s] for s in series]

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    g_x_arr = np.array(g_x)
    l_y_arr = np.array(l_y)
    lo = min(g_x_arr.min(), l_y_arr.min(), 0.95) - 0.05
    hi = max(g_x_arr.max(), l_y_arr.max(), 1.05) + 0.05
    ax.plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=0.8)
    ax.axhline(1.0, color="0.85", lw=0.6)
    ax.axvline(1.0, color="0.85", lw=0.6)

    for r, gx, ly, c in zip(regimes, g_x, l_y, colors):
        ax.scatter(gx, ly, s=42, color=c, edgecolor="black", linewidth=0.4, zorder=5)
        ax.annotate(r, (gx, ly), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Global CQR  width / oracle-width")
    ax.set_ylabel("Local CQR  width / oracle-width")
    ax.set_title(
        f"Efficiency: Local vs Global  (below diagonal = Local tighter; ideal = (1,1))\n"
        f"α={alpha}"
    )

    # Series legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=v, markeredgecolor="black",
               markersize=7, label=k)
        for k, v in series_colors.items()
    ]
    ax.legend(handles=handles, loc="lower right", title="Series", frameon=True)
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

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--n_train", type=int, default=2500)
    p.add_argument("--n_cal", type=int, default=2500)
    p.add_argument("--n_test", type=int, default=5000)
    p.add_argument("--n_seeds", type=int, default=10)
    p.add_argument(
        "--regimes", type=str, default="ALL",
        help="Comma-separated regime ids, or 'ALL' for every regime.",
    )
    p.add_argument(
        "--qualitative_regimes", type=str, default="B2,D3,H1",
        help="Comma-separated regimes to plot in the qualitative-bands figure.",
    )
    p.add_argument(
        "--qualitative_all", action="store_true",
        help="Also produce fig_qualitative_bands_all with every registered regime.",
    )
    p.add_argument(
        "--qualitative_only", action="store_true",
        help="Skip table & summary figures; produce only the qualitative band figures.",
    )
    p.add_argument(
        "--reuse_csv", action="store_true",
        help="Skip the experiment loop and rebuild figures/tables from results.csv.",
    )
    p.add_argument(
        "--out_dir", type=str, default=OUT_DIR,
        help="Output directory.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    cfg = Config(
        alpha=args.alpha,
        n_train=args.n_train,
        n_cal=args.n_cal,
        n_test=args.n_test,
        n_seeds=args.n_seeds,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "results.csv")

    regimes = (
        list_regimes() if args.regimes.upper() == "ALL"
        else [r.strip() for r in args.regimes.split(",") if r.strip()]
    )

    if args.reuse_csv and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded {csv_path} ({len(df)} rows)")
    else:
        df = run_all(cfg, regimes)
        df.to_csv(csv_path, index=False)
        print(f"Wrote {csv_path} ({len(df)} rows)")

    agg = aggregate(df, alpha=cfg.alpha)
    agg_path = os.path.join(args.out_dir, "results_agg.csv")
    agg.to_csv(agg_path, index=False)
    print(f"Wrote {agg_path}")

    if not args.qualitative_only:
        write_summary_table(agg, alpha=cfg.alpha, out_dir=args.out_dir)
        figure_summary_metrics(agg, alpha=cfg.alpha, out_dir=args.out_dir)
        figure_efficiency_scatter(agg, alpha=cfg.alpha, out_dir=args.out_dir)

    qual = [r.strip() for r in args.qualitative_regimes.split(",") if r.strip()]
    if qual:
        figure_qualitative_bands(cfg, out_dir=args.out_dir, regimes=tuple(qual))
    if args.qualitative_all:
        figure_qualitative_bands(
            cfg,
            out_dir=args.out_dir,
            regimes=tuple(list_regimes()),
            n_cols=4,
            out_name="fig_qualitative_bands_all",
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
