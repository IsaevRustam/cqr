"""
Synthetic Adaptivity Experiments for Localized CQR
====================================================
Three experiments on *custom* heteroscedastic synthetic data where the oracle
conditional quantiles are known analytically.  Each experiment produces a
publication-quality figure comparing:
  - Localized (Kernel-weighted) CQR  →  green fill
  - Global (Unweighted) CQR          →  red dashed
  - Oracle conditional quantiles      →  blue dotted

Scenarios
---------
1. **Smooth Quadratic Heteroscedasticity (1D)** — σ(x) grows quadratically
   from a quiet center to loud edges, combined with non-uniform data density.
2. **Banana / Crescent (2D)** — data lives on a curved parabolic manifold
   with position-dependent noise.
3. **Mixture Clusters (1D)** — three Gaussian clusters, each with a
   drastically different noise level.

These setups are intentionally designed so that a single global correction
cannot adapt, making the advantage of localized CQR visually obvious.

Usage:
    python th_xps.py                          # run all three experiments
    python th_xps.py --config configs/default.yaml   # use custom config
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm

from cqr import (
    ExperimentConfig,
    load_config,
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
    setup_plotting,
    marginal_coverage,
    average_width,
    conditional_coverage,
)
from cqr.training import train_quantile_models_unified
from cqr.calibration import compute_bandwidth


# =============================================================================
# SYNTHETIC DATA GENERATORS
# =============================================================================

def _oracle_bounds(mu, sigma, alpha):
    """mu ± z_{1-alpha/2} * sigma  →  (lo, hi) each shape (n,)."""
    z = norm.ppf(1 - alpha / 2)
    return (mu - z * sigma), (mu + z * sigma)


# ---- Experiment 0: Step-Variance (1D) [legacy] -----------------------------

def generate_step_variance(n, seed=None):
    """
    X ~ Uniform[-1, 1],  Y = mu(x) + sigma(x) * eps,  eps ~ N(0,1).

    mu(x)    = 2 sin(3x)
    sigma(x) = { 0.3   if |x| < 0.35
               { 2.0   if 0.35 <= |x| < 0.7
               { 0.5   if |x| >= 0.7
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 1)).astype(np.float32)
    mu, sigma = _step_variance_truth(X)
    eps = rng.randn(n, 1).astype(np.float32)
    Y = (mu + sigma * eps).astype(np.float32)
    return X, Y


def _step_variance_truth(X):
    """Return (mu, sigma) each shape (n, 1)."""
    x = X.reshape(-1, 1).astype(np.float64)
    ax = np.abs(x)
    mu = 2.0 * np.sin(3.0 * x)
    sigma = np.where(ax < 0.35, 0.3,
                     np.where(ax < 0.7, 2.0, 0.5))
    return mu.astype(np.float32), sigma.astype(np.float32)


def oracle_step_variance(X, alpha):
    mu, sigma = _step_variance_truth(X)
    return _oracle_bounds(mu.ravel(), sigma.ravel(), alpha)


# ---- Experiment 1: Smooth Quadratic Heteroscedasticity (1D) ----------------

def generate_smooth_hetero(n, seed=None):
    """
    X ~ TruncatedNormal(0, 0.35) on [-1, 1]  (more data near center)
    Y = mu(x) + sigma(x) * eps,  eps ~ N(0,1).

    mu(x)    = 2 sin(3x)
    sigma(x) = 0.2 + 3.0 * x^2   (smooth quadratic growth)

    Variance is minimal (0.2) at center and maximal (3.2) at edges —
    a natural 16:1 ratio. Combined with non-uniform X density that
    concentrates data at the center, global CQR is biased toward the
    high-density low-noise region and over-corrects in the center
    while under-correcting at the edges.
    """
    rng = np.random.RandomState(seed)
    # Non-uniform X — truncated normal concentrates data at center
    X = rng.normal(0, 0.35, size=(n, 1)).clip(-1, 1).astype(np.float32)
    mu, sigma = _smooth_hetero_truth(X)
    eps = rng.randn(n, 1).astype(np.float32)
    Y = (mu + sigma * eps).astype(np.float32)
    return X, Y


def _smooth_hetero_truth(X):
    """Return (mu, sigma) each shape (n, 1)."""
    x = X.reshape(-1, 1).astype(np.float64)
    mu = 2.0 * np.sin(3.0 * x)
    sigma = 0.2 + 3.0 * (x ** 2)
    return mu.astype(np.float32), sigma.astype(np.float32)


def oracle_smooth_hetero(X, alpha):
    mu, sigma = _smooth_hetero_truth(X)
    return _oracle_bounds(mu.ravel(), sigma.ravel(), alpha)


# ---- Experiment 2: Banana / Crescent (2D) ----------------------------------

def generate_banana(n, seed=None):
    """
    X1 ~ Uniform[-1, 1]
    X2 | X1 ~ N(X1^2 - 0.5, 0.15),  clipped to [-1, 1]

    mu(x) = 2 sin(3 x1) + x2
    sigma(x) = 0.2 + 2.5 |x1|
    """
    rng = np.random.RandomState(seed)
    x1 = rng.uniform(-1, 1, size=n).astype(np.float64)
    x2 = rng.normal(x1 ** 2 - 0.5, 0.15).clip(-1, 1).astype(np.float64)
    X = np.column_stack([x1, x2]).astype(np.float32)
    mu, sigma = _banana_truth(X)
    eps = rng.randn(n, 1).astype(np.float32)
    Y = (mu + sigma * eps).astype(np.float32)
    return X, Y


def _banana_truth(X):
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 2)
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    mu = 2.0 * np.sin(3.0 * x1) + x2
    sigma = 0.2 + 2.5 * np.abs(x1)
    return mu.astype(np.float32), sigma.astype(np.float32)


def oracle_banana(X, alpha):
    mu, sigma = _banana_truth(X)
    return _oracle_bounds(mu.ravel(), sigma.ravel(), alpha)


def banana_density(X):
    """Approximate p(x1, x2) for contour overlays."""
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 2)
    x1, x2 = X[:, 0], X[:, 1]
    # p(x1) = 0.5 (uniform on [-1,1])
    p_x1 = 0.5 * np.ones_like(x1)
    # p(x2|x1) ~ N(x1^2 - 0.5, 0.15) truncated to [-1,1]
    from scipy.stats import truncnorm
    loc = x1 ** 2 - 0.5
    sc = 0.15
    a_trunc = (-1 - loc) / sc
    b_trunc = (1 - loc) / sc
    p_x2 = truncnorm.pdf(x2, a_trunc, b_trunc, loc=loc, scale=sc)
    return (p_x1 * p_x2).astype(np.float64)


# ---- Experiment 3: Mixture Clusters (1D) ------------------------------------

def generate_clusters(n, seed=None):
    """
    X ~ 0.4 N(-0.6, 0.1) + 0.3 N(0.0, 0.1) + 0.3 N(0.6, 0.1),  clipped [-1,1]

    mu(x)    = 3 x
    sigma(x) = soft-assignment weighted blend of per-cluster noise:
               sigma_left=4.0, sigma_center=0.15, sigma_right=0.8

    The 27:1 ratio between the noisy left cluster and the quiet center
    makes the global correction vastly too wide for the center cluster.
    """
    rng = np.random.RandomState(seed)
    centers = np.array([-0.6, 0.0, 0.6])
    scale = 0.1
    weights = np.array([0.4, 0.3, 0.3])

    # Draw mixture component
    comp = rng.choice(3, size=n, p=weights)
    X = rng.normal(centers[comp], scale).clip(-1, 1).astype(np.float64)
    X = X.reshape(-1, 1).astype(np.float32)

    mu, sigma = _cluster_truth(X)
    eps = rng.randn(n, 1).astype(np.float32)
    Y = (mu + sigma * eps).astype(np.float32)
    return X, Y


def _cluster_truth(X):
    x = np.asarray(X, dtype=np.float64).reshape(-1, 1)
    mu = 3.0 * x

    centers = np.array([-0.6, 0.0, 0.6]).reshape(1, -1)
    sigmas_k = np.array([4.0, 0.15, 0.8]).reshape(1, -1)
    scale = 0.1

    # Soft-assignment weights (Gaussian proximity)
    w = np.exp(-0.5 * ((x - centers) / scale) ** 2)
    w = w / w.sum(axis=1, keepdims=True)
    sigma = (w * sigmas_k).sum(axis=1, keepdims=True)

    return mu.astype(np.float32), sigma.astype(np.float32)


def oracle_clusters(X, alpha):
    mu, sigma = _cluster_truth(X)
    return _oracle_bounds(mu.ravel(), sigma.ravel(), alpha)


def cluster_density(x):
    """Mixture density for histogram overlay."""
    x = np.asarray(x, dtype=np.float64).ravel()
    centers = [-0.6, 0.0, 0.6]
    scales = [0.1, 0.1, 0.1]
    weights = [0.4, 0.3, 0.3]
    pdf = np.zeros_like(x)
    for c, s, w in zip(centers, scales, weights):
        pdf += w * norm.pdf(x, loc=c, scale=s)
    return pdf


# =============================================================================
# SHARED EXPERIMENT RUNNER
# =============================================================================

def run_experiment(
    generate_fn,
    oracle_fn,
    name: str,
    d: int,
    alpha: float = 0.05,
    n_train: int = 10_000,
    n_cal: int = 10_000,
    hidden_dim: int = 64,
    epochs: int = 300,
    lr: float = 0.01,
    bandwidth_scale: float = 6.0,
    seed: int = 42,
    activation: str = "relu",
):
    """
    Train quantile models, calibrate (global + local), return results dict.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    h = compute_bandwidth(n_cal, d, gamma=1.0, scale=bandwidth_scale)
    tau_lo = alpha / 2
    tau_hi = 1.0 - alpha / 2

    print(f"\n{'=' * 60}")
    print(f"Experiment: {name}")
    print(f"{'=' * 60}")
    print(f"  n_train={n_train}, n_cal={n_cal}, alpha={alpha}, d={d}")
    print(f"  activation={activation}")
    print(f"  bandwidth h={h:.4f}  (scale={bandwidth_scale})")

    # ---- generate data ----
    X_train, Y_train = generate_fn(n_train, seed=seed)
    X_cal, Y_cal = generate_fn(n_cal, seed=seed + 1)
    X_scatter, Y_scatter = generate_fn(750, seed=seed + 2)

    # ---- train quantile networks ----
    X_t = torch.from_numpy(X_train)
    Y_t = torch.from_numpy(Y_train)

    print(f"  Training quantile networks ({activation.upper()}) \u2026")
    model_lo, model_hi = train_quantile_models_unified(
        X_t, Y_t, tau_lo, tau_hi,
        input_dim=d, hidden_dim=hidden_dim,
        n_layers=2, epochs=epochs, lr=lr,
        batch_size=0, weight_decay=1e-5,
        grad_clip=1.0, activation=activation,
        verbose=False, seed=seed,
    )
    print("  Training complete.")

    # ---- calibration scores ----
    with torch.no_grad():
        pred_cal_lo = model_lo(torch.from_numpy(X_cal)).numpy().flatten()
        pred_cal_hi = model_hi(torch.from_numpy(X_cal)).numpy().flatten()
    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, Y_cal)

    Q_global = global_calibration(scores, alpha)
    print(f"  Global Q̂ = {Q_global:.4f}")

    lcp = LocalConformalOptimizer(X_cal, scores, h=h)

    # ---- evaluation grid ----
    if d == 1:
        X_grid = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)
    else:
        ng = 100
        x1 = np.linspace(-1, 1, ng)
        x2 = np.linspace(-1, 1, ng)
        X1g, X2g = np.meshgrid(x1, x2)
        X_grid = np.column_stack([X1g.ravel(), X2g.ravel()]).astype(np.float32)

    # ---- predict on grid (batched) ----
    batch = 1000
    Q_local_chunks = []
    for i in range(0, len(X_grid), batch):
        Q_local_chunks.append(
            lcp.predict_corrections(X_grid[i:i + batch], alpha)
        )
    Q_local = np.concatenate(Q_local_chunks)

    with torch.no_grad():
        Xg_t = torch.from_numpy(X_grid)
        pred_lo = model_lo(Xg_t).numpy().flatten()
        pred_hi = model_hi(Xg_t).numpy().flatten()

    int_lo_local = pred_lo - Q_local
    int_hi_local = pred_hi + Q_local
    int_lo_global = pred_lo - Q_global
    int_hi_global = pred_hi + Q_global

    oracle_lo, oracle_hi = oracle_fn(X_grid, alpha)

    # ---- marginal coverage on a fresh test set ----
    X_test, Y_test = generate_fn(5000, seed=seed + 99)
    with torch.no_grad():
        pl = model_lo(torch.from_numpy(X_test)).numpy().flatten()
        ph = model_hi(torch.from_numpy(X_test)).numpy().flatten()
    sc_test = compute_conformity_scores(pl, ph, Y_test)
    # local corrections for test set
    Q_test_chunks = []
    for i in range(0, len(X_test), batch):
        Q_test_chunks.append(
            lcp.predict_corrections(X_test[i:i + batch], alpha)
        )
    Q_test_local = np.concatenate(Q_test_chunks)

    cov_local = marginal_coverage(Y_test, pl - Q_test_local, ph + Q_test_local)
    cov_global = marginal_coverage(Y_test, pl - Q_global, ph + Q_global)
    w_local = average_width(pl - Q_test_local, ph + Q_test_local)
    w_global = average_width(pl - Q_global, ph + Q_global)

    # conditional coverage
    cc_local = conditional_coverage(
        Y_test, pl - Q_test_local, ph + Q_test_local, X_test, alpha, n_bins=5
    )
    cc_global = conditional_coverage(
        Y_test, pl - Q_global, ph + Q_global, X_test, alpha, n_bins=5
    )

    print(f"  Local  CQR  — coverage: {cov_local:.3f}, avg width: {w_local:.2f}, "
          f"worst-bin cov: {cc_local['worst_bin_coverage']:.3f}")
    print(f"  Global CQR  — coverage: {cov_global:.3f}, avg width: {w_global:.2f}, "
          f"worst-bin cov: {cc_global['worst_bin_coverage']:.3f}")

    res = dict(
        X_grid=X_grid, X_scatter=X_scatter, Y_scatter=Y_scatter, X_train=X_train,
        int_lo_local=int_lo_local, int_hi_local=int_hi_local,
        int_lo_global=int_lo_global, int_hi_global=int_hi_global,
        oracle_lo=oracle_lo, oracle_hi=oracle_hi,
        cov_local=cov_local, cov_global=cov_global,
        w_local=w_local, w_global=w_global,
        cc_local=cc_local, cc_global=cc_global,
    )
    if d == 2:
        res["X1g"] = X1g
        res["X2g"] = X2g
        res["ng"] = ng
    return res


# =============================================================================
# PLOTTING
# =============================================================================

def plot_step_variance(res, output_path, show=True):
    """
    Two-panel figure: intervals + sigma(x) step function.
    """
    setup_plotting()

    xg = res["X_grid"].ravel()
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

    # ---- Top: intervals ----
    ax = fig.add_subplot(gs[0])
    ax.scatter(res["X_scatter"].ravel(), res["Y_scatter"].ravel(),
               s=12, alpha=0.35, c="gray", edgecolors="none", label="Test data", zorder=1)

    ax.fill_between(xg, res["int_lo_local"], res["int_hi_local"],
                    alpha=0.30, color="#2ca02c", label="Localized CQR", zorder=2)
    ax.plot(xg, res["int_lo_local"], color="#2ca02c", lw=1.5, zorder=2)
    ax.plot(xg, res["int_hi_local"], color="#2ca02c", lw=1.5, zorder=2)

    ax.plot(xg, res["int_lo_global"], color="#d62728", ls="--", lw=2,
            label="Global CQR", zorder=3)
    ax.plot(xg, res["int_hi_global"], color="#d62728", ls="--", lw=2, zorder=3)

    ax.plot(xg, res["oracle_lo"], color="#1f77b4", ls=":", lw=2,
            label="Oracle", zorder=4)
    ax.plot(xg, res["oracle_hi"], color="#1f77b4", ls=":", lw=2, zorder=4)

    ax.set_ylabel(r"$Y$", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3)

    # ---- Bottom: sigma(x) step function ----
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    _, sigma_grid = _step_variance_truth(res["X_grid"])
    sigma_grid = sigma_grid.ravel()

    colors_map = {0.3: "#2ca02c", 2.0: "#d62728", 0.5: "#ff7f0e"}
    labels_map = {0.3: r"$\sigma=0.3$  (quiet center)",
                  2.0: r"$\sigma=2.0$  (noisy mid-zone)",
                  0.5: r"$\sigma=0.5$  (moderate edges)"}
    prev_s = sigma_grid[0]
    start = 0
    for i in range(1, len(sigma_grid)):
        if sigma_grid[i] != prev_s or i == len(sigma_grid) - 1:
            end = i if sigma_grid[i] != prev_s else i + 1
            c = colors_map.get(prev_s, "gray")
            lab = labels_map.pop(prev_s, None)
            ax2.fill_between(xg[start:end], 0, sigma_grid[start:end],
                             color=c, alpha=0.5, label=lab)
            start = i
            prev_s = sigma_grid[i]

    ax2.set_xlabel(r"$X$", fontsize=14)
    ax2.set_ylabel(r"$\sigma(x)$", fontsize=12)
    ax2.set_ylim(0, 2.5)
    ax2.legend(loc="upper right", fontsize=9, ncol=3)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Saved \u2192 {output_path}")


def plot_smooth_hetero(res, output_path, show=True):
    """
    Two-panel figure: intervals + smooth sigma(x) curve with data density.
    """
    setup_plotting()

    xg = res["X_grid"].ravel()
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

    # ---- Top: intervals ----
    ax = fig.add_subplot(gs[0])
    ax.scatter(res["X_scatter"].ravel(), res["Y_scatter"].ravel(),
               s=12, alpha=0.35, c="gray", edgecolors="none", label="Test data", zorder=1)

    ax.fill_between(xg, res["int_lo_local"], res["int_hi_local"],
                    alpha=0.30, color="#2ca02c", label="Localized CQR", zorder=2)
    ax.plot(xg, res["int_lo_local"], color="#2ca02c", lw=1.5, zorder=2)
    ax.plot(xg, res["int_hi_local"], color="#2ca02c", lw=1.5, zorder=2)

    ax.plot(xg, res["int_lo_global"], color="#d62728", ls="--", lw=2.5,
            label="Global CQR", zorder=3)
    ax.plot(xg, res["int_hi_global"], color="#d62728", ls="--", lw=2.5, zorder=3)

    ax.plot(xg, res["oracle_lo"], color="#1f77b4", ls=":", lw=2.5,
            label="Oracle", zorder=4)
    ax.plot(xg, res["oracle_hi"], color="#1f77b4", ls=":", lw=2.5, zorder=4)

    ax.set_ylabel(r"$Y$", fontsize=14)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3)

    # ---- Bottom: sigma(x) + data density ----
    ax2 = fig.add_subplot(gs[1], sharex=ax)
    
    # Histogram of training X (data density)
    ax2.hist(res["X_train"].ravel(), bins=100, density=True,
             alpha=0.4, color="#1f77b4", edgecolor="none", label="Data density")
    
    # Smooth sigma(x) curve
    _, sigma_grid = _smooth_hetero_truth(res["X_grid"])
    sigma_grid = sigma_grid.ravel()
    
    # Plot on twin axis
    ax3 = ax2.twinx()
    ax3.plot(xg, sigma_grid, color="#d62728", lw=3, alpha=0.8,
             label=r"$\sigma(x) = 0.2 + 3x^2$")
    ax3.fill_between(xg, 0, sigma_grid, color="#d62728", alpha=0.15)
    
    ax2.set_xlabel(r"$X$", fontsize=14)
    ax2.set_ylabel("Data density", fontsize=12, color="#1f77b4")
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    
    ax3.set_ylabel(r"$\sigma(x)$", fontsize=12, color="#d62728")
    ax3.tick_params(axis="y", labelcolor="#d62728")
    ax3.set_ylim(0, 3.5)
    
    # Merge legends
    h1, l1 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h3, l1 + l3, loc="upper center", fontsize=10, ncol=2)
    
    ax2.set_xlim(-1.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Saved → {output_path}")


def plot_banana(res, output_path, show=True):
    """
    Side-by-side heatmaps: local interval width vs global, with banana density
    contours and oracle width.
    """
    setup_plotting()

    X1g = res["X1g"]
    X2g = res["X2g"]
    ng = res["ng"]
    Xf = res["X_grid"]

    w_local = (res["int_hi_local"] - res["int_lo_local"]).reshape(ng, ng)
    w_global = (res["int_hi_global"] - res["int_lo_global"]).reshape(ng, ng)
    w_oracle = (res["oracle_hi"] - res["oracle_lo"]).reshape(ng, ng)

    # Density for contour overlay
    density = banana_density(Xf).reshape(ng, ng)

    vmin = min(w_local.min(), w_global.min(), w_oracle.min())
    vmax = max(w_local.max(), w_global.max(), w_oracle.max())
    levels = 50

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    titles = ["Localized CQR", "Global CQR", "Oracle"]
    grids = [w_local, w_global, w_oracle]

    for idx, (ax, title, wg) in enumerate(zip(axes, titles, grids)):
        cf = ax.contourf(X1g, X2g, wg, levels=levels, cmap="YlGnBu_r",
                         vmin=vmin, vmax=vmax)
        # Density contours (banana shape)
        d_max = density.max()
        prob_levels = [0.1, 0.25, 0.5, 0.75]
        cs = ax.contour(X1g, X2g, density,
                        levels=[p * d_max for p in prob_levels],
                        colors="white", linestyles="dashed", linewidths=1.2, alpha=0.8)
        fmt = {lev: f"{p:.0%}" for lev, p in
               zip([p * d_max for p in prob_levels], prob_levels)}
        ax.clabel(cs, inline=True, fontsize=8, fmt=fmt)

        ax.set_xlabel(r"$X_1$", fontsize=13)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")

        # Move y-axis to the right for the 3rd panel (Oracle)
        if idx == 2:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_ylabel(r"$X_2$", fontsize=13)
        else:
            ax.set_ylabel(r"$X_2$", fontsize=13)

    # fig.suptitle("Banana Data: Localized CQR tracks the oracle width along the arc", fontsize=15, y=1.02)

    plt.tight_layout()

    # Add colorbar as a separate axis outside the figure panels
    cb = fig.colorbar(cf, ax=axes.tolist(), shrink=0.82, pad=0.06,
                      location="right", aspect=25)
    cb.set_label(r"Interval length $|\hat{\mathcal{C}}(x)|$", fontsize=12,
                 rotation=270, labelpad=22)

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Saved → {output_path}")


def plot_clusters(res, output_path, show=True):
    """
    Two-panel figure: intervals + density / sigma overlay.
    """
    setup_plotting()

    xg = res["X_grid"].ravel()
    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.06)

    # ---- Top: intervals ----
    ax = fig.add_subplot(gs[0])
    ax.scatter(res["X_scatter"].ravel(), res["Y_scatter"].ravel(),
               s=12, alpha=0.35, c="gray", edgecolors="none", label="Test data", zorder=1)

    ax.fill_between(xg, res["int_lo_local"], res["int_hi_local"],
                    alpha=0.30, color="#2ca02c", label="Localized CQR", zorder=2)
    ax.plot(xg, res["int_lo_local"], color="#2ca02c", lw=1.5, zorder=2)
    ax.plot(xg, res["int_hi_local"], color="#2ca02c", lw=1.5, zorder=2)

    ax.plot(xg, res["int_lo_global"], color="#d62728", ls="--", lw=2,
            label="Global CQR", zorder=3)
    ax.plot(xg, res["int_hi_global"], color="#d62728", ls="--", lw=2, zorder=3)

    ax.plot(xg, res["oracle_lo"], color="#1f77b4", ls=":", lw=2,
            label="Oracle", zorder=4)
    ax.plot(xg, res["oracle_hi"], color="#1f77b4", ls=":", lw=2, zorder=4)

    ax.set_ylabel(r"$Y$", fontsize=14)
    # ax.set_title("Mixture Clusters: per-cluster noise demands local adaptation", fontsize=14)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_xlim(-1.05, 1.05)
    ax.tick_params(labelbottom=False)
    ax.grid(True, alpha=0.3)

    # ---- Bottom: density + sigma ----
    ax2 = fig.add_subplot(gs[1], sharex=ax)

    # Histogram of training X
    ax2.hist(res["X_train"].ravel(), bins=100, density=True,
             alpha=0.55, color="#1f77b4", edgecolor="white", label="Data density")

    # Theoretical density line
    x_pdf = np.linspace(-1, 1, 500)
    ax2.plot(x_pdf, cluster_density(x_pdf), color="#1f77b4", lw=2, alpha=0.7)

    ax2.set_xlabel(r"$X$", fontsize=14)
    ax2.set_ylabel("Density", fontsize=12)

    # sigma(x) on twin axis
    ax3 = ax2.twinx()
    _, sigma_grid = _cluster_truth(res["X_grid"])
    ax3.plot(xg, sigma_grid.ravel(), color="#d62728", lw=2, ls="-",
             label=r"$\sigma(x)$")
    ax3.set_ylabel(r"$\sigma(x)$", fontsize=12, color="#d62728")
    ax3.tick_params(axis="y", labelcolor="#d62728")

    # Merge legends
    h1, l1 = ax2.get_legend_handles_labels()
    h3, l3 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h3, l1 + l3, loc="upper right", fontsize=9, ncol=2)

    ax2.set_xlim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Saved → {output_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Localized CQR — Synthetic Adaptivity Experiments")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config (default: built-in defaults)")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figures without displaying them")
    parser.add_argument("--activation", type=str, default="relu",
                        choices=["relu", "requ"],
                        help="Activation function for quantile NN (default: relu)")
    args = parser.parse_args()

    # Load (or default) config
    if args.config:
        cfg = load_config(args.config)
    else:
        cfg = ExperimentConfig()

    alpha = cfg.alpha
    hidden = cfg.hidden_dim
    epochs = cfg.train_epochs
    lr = cfg.learning_rate
    bw_scale = cfg.bandwidth_scale
    seed = cfg.seed
    show = not args.no_show
    activation = args.activation
    act_suffix = f"_{activation}"  # e.g. "_relu" or "_requ"

    os.makedirs("figures_cqr", exist_ok=True)

    # Override defaults for these experiments:
    # - Smaller models (hidden=32, epochs=150) so conformity scores are
    #   non-trivial and vary across the domain.
    # - Tighter bandwidth (scale=2.0) so local CQR truly adapts locally
    #   instead of averaging over the whole domain.
    # - Less training data (3000) to keep model fits imperfect.
    common_1d = dict(alpha=alpha, hidden_dim=32, epochs=150, lr=lr,
                     bandwidth_scale=2.0, seed=seed, activation=activation)

    # ---------- Experiment 0: Step-Variance (1D) ----------
    res0 = run_experiment(
        generate_step_variance, oracle_step_variance,
        name="Step-Variance (1D)", d=1,
        n_train=3_000, n_cal=5_000, **common_1d,
    )
    plot_step_variance(res0, f"figures_cqr/step_variance_adaptivity{act_suffix}.pdf", show=show)

    # ---------- Experiment 1: Smooth Quadratic Heteroscedasticity (1D) ----------
    res1 = run_experiment(
        generate_smooth_hetero, oracle_smooth_hetero,
        name="Smooth Quadratic Heteroscedasticity (1D)", d=1,
        n_train=3_000, n_cal=5_000, **common_1d,
    )
    plot_smooth_hetero(res1, f"figures_cqr/smooth_hetero_adaptivity{act_suffix}.pdf", show=show)

    # ---------- Experiment 2: Banana (2D) ----------
    res2 = run_experiment(
        generate_banana, oracle_banana,
        name="Banana / Crescent (2D)", d=2,
        n_train=5_000, n_cal=8_000,
        hidden_dim=64, epochs=200,
        alpha=alpha, lr=lr, bandwidth_scale=3.0, seed=seed,
        activation=activation,
    )
    plot_banana(res2, f"figures_cqr/banana_adaptivity{act_suffix}.pdf", show=show)

    # ---------- Experiment 3: Mixture Clusters (1D) ----------
    res3 = run_experiment(
        generate_clusters, oracle_clusters,
        name="Mixture Clusters (1D)", d=1,
        n_train=3_000, n_cal=5_000, **common_1d,
    )
    plot_clusters(res3, f"figures_cqr/clusters_adaptivity{act_suffix}.pdf", show=show)

    print("\n" + "=" * 60)
    print("All experiments completed — figures saved in figures_cqr/")
    print("=" * 60)


if __name__ == "__main__":
    main()