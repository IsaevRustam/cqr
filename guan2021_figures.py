"""
Reproduce Figure 2 from Guan (2021), "Localized Conformal Prediction":
Local CQR vs Global CQR on Example 4.1 (four heteroscedastic settings).

Settings (ρ defines heteroscedasticity):
    A: ρ(x) = sin(x)
    B: ρ(x) = cos(x)
    C: ρ(x) = sqrt(|x|)
    D: ρ(x) = 1  (homoscedastic baseline)

Model:  X ~ N(0,1),  Y = ρ(X)·ε,  ε ~ N(0,1) ⊥ X.

Usage:
    python guan2021_figures.py                        # 2×2 panel PDF
    python guan2021_figures.py --settings A B         # subset of settings
    python guan2021_figures.py --epochs 500           # longer training
    python guan2021_figures.py --output my_fig.pdf
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import norm

from cqr import (
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
    setup_plotting,
    generate_guan2021,
    guan2021_oracle,
)
from cqr.calibration import compute_bandwidth
from cqr.models import train_quantile_models


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def run_setting(
    setting: str,
    n_train: int = 1000,
    n_cal: int = 1000,
    n_test: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    bandwidth_scale: float = 1.0,
    gamma: float = 1.0,
    n_grid: int = 500,
    verbose: bool = True,
) -> dict:
    """Run the full Local vs Global CQR pipeline for one Guan setting.

    Returns a results dict ready for plotting.
    """
    if verbose:
        print(f"\n{'='*55}")
        print(f"  Setting {setting}")
        print(f"{'='*55}")

    # ---- Data ---------------------------------------------------------------
    data = generate_guan2021(
        setting, n_train=n_train, n_cal=n_cal, n_test=n_test,
        seed=seed, alpha=alpha,
    )
    X_train = data["X_train"].reshape(-1, 1).astype(np.float32)
    Y_train = data["Y_train"].reshape(-1, 1).astype(np.float32)
    X_cal   = data["X_cal"].reshape(-1, 1).astype(np.float32)
    Y_cal   = data["Y_cal"].astype(np.float32)

    # Evaluation grid: [-3.5, 3.5] covers >99.9% of N(0,1) mass
    X_grid = np.linspace(-3.5, 3.5, n_grid).reshape(-1, 1).astype(np.float32)

    # ---- Train quantile models ----------------------------------------------
    tau_lo, tau_hi = alpha / 2, 1.0 - alpha / 2
    if verbose:
        print(f"  Training quantile networks (τ={tau_lo:.3f}, {tau_hi:.3f}, "
              f"epochs={epochs}) ...")

    X_t = torch.from_numpy(X_train)
    Y_t = torch.from_numpy(Y_train)
    model_lo, model_hi = train_quantile_models(
        X_t, Y_t,
        tau_low=tau_lo, tau_high=tau_hi,
        input_dim=1, hidden_dim=hidden_dim, epochs=epochs, lr=lr,
    )
    if verbose:
        print("  Training complete.")

    # ---- Calibration scores -------------------------------------------------
    with torch.no_grad():
        X_cal_t = torch.from_numpy(X_cal)
        pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
        pred_cal_hi = model_hi(X_cal_t).numpy().flatten()

    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, Y_cal)

    # ---- Global (unweighted) calibration ------------------------------------
    Q_global = global_calibration(scores, alpha)
    if verbose:
        print(f"  Global Q̂ = {Q_global:.4f}")

    # ---- Local (kernel-weighted) calibration --------------------------------
    h = compute_bandwidth(n_cal, d=1, gamma=gamma, scale=bandwidth_scale)
    if verbose:
        print(f"  Bandwidth h = {h:.4f}")

    lcp = LocalConformalOptimizer(X_cal, scores, h=h)

    batch = 200
    Q_local_chunks = []
    for i in range(0, n_grid, batch):
        q = lcp.predict_corrections(X_grid[i : i + batch], alpha)
        Q_local_chunks.append(q)
    Q_local = np.concatenate(Q_local_chunks)

    # ---- Predict on grid ----------------------------------------------------
    with torch.no_grad():
        X_grid_t = torch.from_numpy(X_grid)
        pred_lo = model_lo(X_grid_t).numpy().flatten()
        pred_hi = model_hi(X_grid_t).numpy().flatten()

    # ---- Build intervals ----------------------------------------------------
    x_grid_1d = X_grid.flatten()
    oracle_lo, oracle_hi = guan2021_oracle(setting.upper(), x_grid_1d, alpha)
    results = {
        "setting": setting,
        "x_grid": x_grid_1d,
        # Local CQR
        "local_lo": pred_lo - Q_local,
        "local_hi": pred_hi + Q_local,
        # Global CQR
        "global_lo": pred_lo - Q_global,
        "global_hi": pred_hi + Q_global,
        # Oracle (computed directly on the sorted grid)
        "oracle_lo": oracle_lo,
        "oracle_hi": oracle_hi,
        # Scatter
        "X_test": data["X_test"],
        "Y_test": data["Y_test"],
        "alpha": alpha,
    }

    if verbose:
        w_local  = np.mean(results["local_hi"]  - results["local_lo"])
        w_global = np.mean(results["global_hi"] - results["global_lo"])
        w_oracle = np.mean(results["oracle_hi"] - results["oracle_lo"])
        print(f"  Mean width  — local: {w_local:.3f}  |  "
              f"global: {w_global:.3f}  |  oracle: {w_oracle:.3f}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

SETTING_LABELS = {
    # Original Guan (2021) Figure 2
    "A": r"$\rho(x) = \sin(x)$",
    "B": r"$\rho(x) = \cos(x)$",
    "C": r"$\rho(x) = \sqrt{|x|}$",
    "D": r"$\rho(x) = 1$",
    # Smoothness / regularity stress
    "E": r"$\rho(x) = |x|$",
    "F": r"$\rho(x) = 1 + 0.5\,\mathrm{sign}(x)$",
    "G": r"$\rho(x) = e^{x/2}$",
    # Multi-scale / oscillation
    "H": r"$\rho(x) = 1 + \sin^2(2\pi x)$",
    "I": r"$\rho(x) = 1 + 0.3\sin(5x)$",
    # Asymmetric / non-Gaussian conditional
    "J": r"$\rho{=}\sqrt{|x|},\;\varepsilon{\sim}\chi^2_1{-}1$",
    "K": r"$\mu{=}\sin(\pi x),\;\rho{=}\sqrt{|x|},\;\varepsilon{\sim}t_3$",
    # Mixture / regime switching
    "L": r"$\rho{=}|\cos x|{+}0.1\;(|x|{<}1.5),\;\text{else }1$",
    # A-like variants: oscillatory bands that collapse to zero
    "M": r"$\rho(x) = \sin^2(x)$",
    "N": r"$\rho(x) = |\sin(2x)|$",
    # C-like variant: sharper cusp at 0
    "O": r"$\rho(x) = |x|^{2/3}$",
}

_COLOR_LOCAL  = "#2ca02c"
_COLOR_GLOBAL = "#d62728"
_COLOR_ORACLE = "#1f77b4"
_COLOR_HIST   = "#aec7e8"


def _plot_one_setting(fig, outer_gs_cell, res: dict, show_xlabel: bool, show_legend: bool):
    """Fill one 2-row sub-panel (main + histogram) for a single setting."""
    inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs_cell,
                                   height_ratios=[3, 1], hspace=0.06)
    ax_main = fig.add_subplot(inner[0])
    ax_hist = fig.add_subplot(inner[1], sharex=ax_main)

    x   = res["x_grid"]
    X_s = res["X_test"]
    Y_s = res["Y_test"]

    # --- Scatter (sub-sampled for clarity) ---
    rng_plot = np.random.default_rng(7)
    idx = rng_plot.choice(len(X_s), size=min(5000, len(X_s)), replace=False)
    ax_main.scatter(X_s[idx], Y_s[idx], s=8, alpha=0.25, c="gray",
                    edgecolors="none", zorder=1, label="Test data")

    # --- Local CQR ---
    ax_main.fill_between(x, res["local_lo"], res["local_hi"],
                         alpha=0.25, color=_COLOR_LOCAL, zorder=2)
    ax_main.plot(x, res["local_lo"], color=_COLOR_LOCAL, lw=1.5,
                 label="Local CQR", zorder=3)
    ax_main.plot(x, res["local_hi"], color=_COLOR_LOCAL, lw=1.5, zorder=3)

    # --- Global CQR ---
    ax_main.plot(x, res["global_lo"], color=_COLOR_GLOBAL, lw=1.8,
                 ls="--", label="Global CQR", zorder=4)
    ax_main.plot(x, res["global_hi"], color=_COLOR_GLOBAL, lw=1.8,
                 ls="--", zorder=4)

    # --- Oracle ---
    ax_main.plot(x, res["oracle_lo"], color=_COLOR_ORACLE, lw=2,
                 ls=":", label="Oracle", zorder=5)
    ax_main.plot(x, res["oracle_hi"], color=_COLOR_ORACLE, lw=2,
                 ls=":", zorder=5)

    setting = res["setting"].upper()
    ax_main.set_title(f"Setting {setting}: {SETTING_LABELS[setting]}", fontsize=12)
    ax_main.set_ylabel(r"$Y$", fontsize=11)
    ax_main.set_xlim(-3.6, 3.6)
    ax_main.tick_params(labelbottom=False)
    ax_main.grid(True, alpha=0.3)

    if show_legend:
        ax_main.legend(loc="upper left", fontsize=9, framealpha=0.8)

    # --- Histogram + N(0,1) PDF ---
    ax_hist.hist(X_s, bins=60, density=True, color=_COLOR_HIST,
                 edgecolor="white", alpha=0.8, zorder=1)
    xp = np.linspace(-3.6, 3.6, 400)
    ax_hist.plot(xp, norm.pdf(xp), color="#333333", lw=1.5,
                 label=r"$\mathcal{N}(0,1)$", zorder=2)
    ax_hist.set_ylabel("Density", fontsize=9)
    ax_hist.legend(loc="upper right", fontsize=8)
    ax_hist.grid(True, alpha=0.3)

    if show_xlabel:
        ax_hist.set_xlabel(r"$X$", fontsize=11)
    else:
        ax_hist.tick_params(labelbottom=False)


def make_figure(all_results: list, output_path: str, show: bool = True):
    """Produce a 2×2 panel figure (or 1×k for fewer settings)."""
    setup_plotting()

    n = len(all_results)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(7 * ncols, 6 * nrows))
    outer = GridSpec(nrows, ncols, figure=fig, hspace=0.35, wspace=0.28)

    for idx, res in enumerate(all_results):
        row, col = divmod(idx, ncols)
        show_xlabel = (row == nrows - 1)
        show_legend = (idx == 0)
        _plot_one_setting(fig, outer[row, col], res,
                          show_xlabel=show_xlabel, show_legend=show_legend)

    fig.suptitle(
        "Local vs Global CQR \u2014 Guan (2021), Example 4.1 + Extended Settings\n"
        r"$X \sim \mathcal{N}(0,1),\quad Y = \mu(X) + \rho(X)\,\varepsilon$",
        fontsize=13, y=1.01,
    )

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Figure saved: {output_path}")

    # Also save as PNG (same stem, next to the PDF)
    stem, _ = os.path.splitext(output_path)
    png_path = stem + ".png"
    plt.savefig(png_path, bbox_inches="tight", dpi=150)
    print(f"Figure saved: {png_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Guan (2021) Figure 2 with Local vs Global CQR"
    )
    parser.add_argument("--settings", nargs="+", default=["A", "B", "C", "D"],
                        choices=["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O"],
                        help="Which settings to run (default: original A\u2013D)")
    parser.add_argument("--n_train", type=int, default=1000)
    parser.add_argument("--n_cal",   type=int, default=1000)
    parser.add_argument("--n_test",  type=int, default=5000)
    parser.add_argument("--alpha",   type=float, default=0.05)
    parser.add_argument("--epochs",  type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--bandwidth_scale", type=float, default=1.5)
    parser.add_argument("--seed",    type=int, default=0)
    parser.add_argument("--output",  type=str, default=None,
                        help="Output PDF/PNG path (default: figures_cqr/guan2021_fig2.pdf)")
    parser.add_argument("--no_show", action="store_true",
                        help="Do not open the figure interactively")
    args = parser.parse_args()

    output = args.output or os.path.join("figures_cqr", "guan2021_fig2.pdf")
    os.makedirs(os.path.dirname(output), exist_ok=True)

    all_results = []
    for s in args.settings:
        res = run_setting(
            setting=s,
            n_train=args.n_train,
            n_cal=args.n_cal,
            n_test=args.n_test,
            seed=args.seed,
            alpha=args.alpha,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            bandwidth_scale=args.bandwidth_scale,
        )
        all_results.append(res)

    make_figure(all_results, output_path=output, show=not args.no_show)


if __name__ == "__main__":
    main()
