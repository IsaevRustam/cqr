"""
Plotting utilities for CQR experiments.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import linregress, truncnorm
from matplotlib.gridspec import GridSpec
from typing import Dict, Any


def setup_plotting():
    """
    Configure matplotlib and seaborn for publication-quality plots.

    Uses serif fonts with math text for LaTeX-like appearance.
    """
    sns.set_context("paper", font_scale=1.4)
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )


def plot_convergence(
    df: pd.DataFrame,
    output_path: str = "convergence.pdf",
    title: str = "CQR Convergence",
    theory_slope: float = -1 / 3,
    d: int = 1,
    beta: float = 1.0,
    show: bool = True,
) -> None:
    """
    Create log-log convergence plot showing RMSE vs sample size.

    Args:
        df: DataFrame with columns ['N', 'rmse_mean', 'rmse_std']
        output_path: Path to save the figure
        title: Plot title
        theory_slope: Theoretical convergence rate exponent
        d: Input dimension (for annotation)
        beta: Hölder smoothness (for annotation)
        show: Whether to display the plot
    """
    setup_plotting()

    # Linear regression in log-log space
    log_N = np.log(df["N"].values)
    log_err = np.log(df["rmse_mean"].values)
    slope, intercept, r_value, _, _ = linregress(log_N, log_err)

    print(f"\nEmpirical convergence rate: N^{{{slope:.3f}}}")
    print(f"(R² = {r_value**2:.4f})")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Experiment points with error bars
    ax.errorbar(
        df["N"],
        df["rmse_mean"],
        yerr=df["rmse_std"],
        fmt="o",
        color="black",
        ecolor="gray",
        capsize=4,
        label="Experiment",
        zorder=3,
    )

    # Empirical fit line
    fit_y = np.exp(intercept + slope * log_N)
    ax.plot(
        df["N"],
        fit_y,
        color="#d62728",
        linewidth=2,
        label=rf"Empirical: $N^{{{slope:.2f}}}$",
        zorder=2,
    )

    # Theoretical rate line
    idx_mid = len(df) // 2
    C_theory = df["rmse_mean"].iloc[idx_mid] / (df["N"].iloc[idx_mid] ** theory_slope)
    theory_y = C_theory * 1.2 * (df["N"].values ** theory_slope)

    # Format theory slope as fraction
    if abs(theory_slope + 1 / 3) < 0.01:
        theory_label = r"Theory: $N^{-1/3}$"
    elif abs(theory_slope + 1 / 4) < 0.01:
        theory_label = r"Theory: $N^{-1/4}$"
    elif abs(theory_slope + 1 / 6) < 0.01:
        theory_label = r"Theory: $N^{-1/6}$"
    else:
        theory_label = rf"Theory: $N^{{{theory_slope:.2f}}}$"

    ax.plot(
        df["N"],
        theory_y,
        color="#1f77b4",
        linestyle="--",
        linewidth=2,
        label=theory_label,
        zorder=1,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sample Size $N$", fontsize=14)
    ax.set_ylabel(
        r"Excess Length RMSE $\||\hat{\mathcal{C}}| - |\mathcal{C}^*|\|_{L_2}$",
        fontsize=14,
    )
    ax.set_title(rf"{title} ($\beta={beta:.0f}$, $d={d}$)", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"\nSaved: {output_path}")


def plot_convergence_multiple_c(
    all_results: Dict[float, pd.DataFrame],
    output_path: str = "convergence_multiple_c.pdf",
    title: str = "CQR Convergence",
    theory_slope: float = -1 / 3,
    d: int = 1,
    beta: float = 1.0,
    show: bool = True,
) -> None:
    """
    Create log-log convergence plot showing RMSE vs sample size for multiple c values.

    Args:
        all_results: Dictionary mapping c values to DataFrames with columns ['N', 'rmse_mean', 'rmse_std']
        output_path: Path to save the figure
        title: Plot title
        theory_slope: Theoretical convergence rate exponent
        d: Input dimension (for annotation)
        beta: Hölder smoothness (for annotation)
        show: Whether to display the plot
    """
    setup_plotting()

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Colors and markers for different c values
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    
    # Plot each c value
    for idx, (c_val, df) in enumerate(sorted(all_results.items(), reverse=True)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        # Linear regression in log-log space
        log_N = np.log(df["N"].values)
        log_err = np.log(df["rmse_mean"].values)
        slope, intercept, r_value, _, _ = linregress(log_N, log_err)

        print(f"\nc = {c_val} (m = N^{c_val}):")
        print(f"  Empirical convergence rate: N^{{{slope:.3f}}}")
        print(f"  (R² = {r_value**2:.4f})")

        # Experiment points with error bars
        ax.errorbar(
            df["N"],
            df["rmse_mean"],
            yerr=df["rmse_std"],
            fmt=marker,
            color=color,
            ecolor=color,
            alpha=0.7,
            capsize=4,
            markersize=8,
            label=f"$m = N^{{{c_val}}}$ (slope: ${slope:.2f}$)",
            zorder=3 + idx,
        )

        # Empirical fit line
        fit_y = np.exp(intercept + slope * log_N)
        ax.plot(
            df["N"],
            fit_y,
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.6,
            zorder=2 + idx,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Sample Size $N$", fontsize=14)
    ax.set_ylabel(
        r"Excess Length RMSE $\||\hat{\mathcal{C}}| - |\mathcal{C}^*|\|_{L_2}$",
        fontsize=14,
    )
    ax.set_title(rf"{title} ($\beta={beta:.0f}$, $d={d}$)", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"\nSaved: {output_path}")


def plot_density_intervals(
    results: Dict[str, Any],
    output_path: str = "density_intervals.pdf",
    title: str = "Localized CQR: Density-Adaptive Intervals",
    loc: float = 0.0,
    scale: float = 0.5,
    show: bool = True,
    distribution: str = "truncated_normal",
) -> None:
    """
    Create visualization showing weighted vs unweighted prediction intervals.

    Main plot: Intervals vs X with scatter overlay
    Subplot: Histogram of X distribution

    Args:
        results: Dictionary with keys:
            - X_grid: Grid points for interval boundaries
            - X_test: Test point X values (for scatter)
            - Y_test: Test point Y values (for scatter)
            - X_train: Training X values (for histogram)
            - interval_lo, interval_hi: Localized (weighted) CQR interval bounds
            - interval_lo_global, interval_hi_global: Global (unweighted) CQR bounds (optional)
            - oracle_lo, oracle_hi: Oracle interval bounds
        output_path: Path to save the figure
        title: Plot title
        loc, scale: Parameters of truncated normal (for PDF overlay, only if distribution='truncated_normal')
        show: Whether to display the plot
        distribution: Type of X distribution for PDF overlay
    """
    setup_plotting()

    X_grid = results["X_grid"]
    X_test = results["X_test"]
    Y_test = results["Y_test"]
    X_train = results["X_train"]
    interval_lo = results["interval_lo"]
    interval_hi = results["interval_hi"]
    oracle_lo = results["oracle_lo"]
    oracle_hi = results["oracle_hi"]

    # Check for global (unweighted) intervals
    has_global = "interval_lo_global" in results and results["interval_lo_global"] is not None

    # Create figure with GridSpec
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

    # --- MAIN PLOT: INTERVALS ---
    ax_main = fig.add_subplot(gs[0])

    # Fill between for Localized (Weighted) CQR intervals - GREEN
    ax_main.fill_between(
        X_grid,
        interval_lo,
        interval_hi,
        alpha=0.3,
        color="#2ca02c",
        label="Weighted (Localized) CQR",
    )
    ax_main.plot(X_grid, interval_lo, color="#2ca02c", linewidth=1.5)
    ax_main.plot(X_grid, interval_hi, color="#2ca02c", linewidth=1.5)

    # Global (Unweighted) CQR intervals - RED DASHED
    if has_global:
        interval_lo_global = results["interval_lo_global"]
        interval_hi_global = results["interval_hi_global"]
        ax_main.plot(
            X_grid,
            interval_lo_global,
            color="#d62728",
            linestyle="--",
            linewidth=2,
            label="Unweighted (Global) CQR",
        )
        ax_main.plot(
            X_grid,
            interval_hi_global,
            color="#d62728",
            linestyle="--",
            linewidth=2,
        )

    # Oracle boundaries (blue dotted)
    ax_main.plot(
        X_grid, oracle_lo, color="#1f77b4", linestyle=":", linewidth=2, label="Oracle"
    )
    ax_main.plot(X_grid, oracle_hi, color="#1f77b4", linestyle=":", linewidth=2)

    # Scatter plot of test data
    ax_main.scatter(
        X_test, Y_test, s=15, alpha=0.4, c="gray", edgecolors="none", label="Test Data"
    )

    ax_main.set_ylabel(r"Target $Y$", fontsize=14)
    ax_main.set_title(title, fontsize=16)
    ax_main.legend(loc="upper left", fontsize=11)
    ax_main.set_xlim(-1.15, 1.15)
    ax_main.tick_params(labelbottom=False)  # Hide x labels for top plot
    ax_main.grid(True, alpha=0.3)

    # --- SUBPLOT: HISTOGRAM OF X DISTRIBUTION ---
    ax_hist = fig.add_subplot(gs[1], sharex=ax_main)

    # Only use first dimension for histogram in multi-d case
    if X_train.ndim > 1:
        X_hist = X_train[:, 0] if X_train.shape[1] > 1 else X_train.flatten()
    else:
        X_hist = X_train

    ax_hist.hist(
        X_hist, bins=100, density=True, alpha=0.7, color="#1f77b4", edgecolor="white",
        label="Data Density"
    )

    # Overlay theoretical PDF only for truncated normal
    if distribution == "truncated_normal":
        x_pdf = np.linspace(-1, 1, 1500)
        a, b = (-1 - loc) / scale, (1 - loc) / scale
        pdf_vals = truncnorm.pdf(x_pdf, a, b, loc=loc, scale=scale)
        ax_hist.plot(
            x_pdf, pdf_vals, color="#d62728", linewidth=2, label="Truncated Normal PDF"
        )

    ax_hist.set_xlabel(r"Feature $X$", fontsize=14)
    ax_hist.set_ylabel("Density", fontsize=12)
    ax_hist.legend(loc="upper right", fontsize=10)
    ax_hist.set_xlim(-1.15, 1.15)
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    print(f"\nSaved: {output_path}")


def plot_heatmap_d2(
    results: Dict[str, Any],
    output_path: str = "density_heatmap_d2.pdf",
    title: str = "Localized CQR: Interval Width (d=2)",
    show: bool = True,
    add_density_contours: bool = True,
) -> None:
    """
    Create a beautiful heatmap visualization for d=2 case showing interval width over the domain.
    
    Uses smooth contourf visualization with a blue colormap (YlGnBu_r) and optional
    white dashed density contours showing probability levels.
    
    If global (unweighted) intervals are provided, creates side-by-side comparison.

    Args:
        results: Dictionary with keys:
            - X1_grid: 2D array of X1 coordinates
            - X2_grid: 2D array of X2 coordinates
            - width_grid: 2D array of interval widths (weighted)
            - width_grid_global: 2D array of interval widths (unweighted, optional)
            - X_train: Training points (X1, X2) for overlay (optional, not plotted)
            - density_grid: 2D array of true density values (optional, for contours)
        output_path: Path to save the figure
        title: Plot title
        show: Whether to display the plot
        add_density_contours: Whether to add density contour overlays
    """
    setup_plotting()

    X1_grid = results["X1_grid"]
    X2_grid = results["X2_grid"]
    width_grid = results["width_grid"]
    
    has_global = "width_grid_global" in results and results["width_grid_global"] is not None
    has_density = "density_grid" in results and results["density_grid"] is not None

    if has_global:
        # Side-by-side comparison with beautiful styling
        width_grid_global = results["width_grid_global"]
        
        # Use same color scale for both
        vmin = min(width_grid.min(), width_grid_global.min())
        vmax = max(width_grid.max(), width_grid_global.max())
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Number of contour levels for smooth appearance
        levels = 50
        
        # LEFT: Weighted (Localized) CQR - Beautiful blue heatmap
        c1 = ax1.contourf(
            X1_grid, X2_grid, width_grid, 
            levels=levels, cmap="YlGnBu_r", 
            vmin=vmin, vmax=vmax
        )
        
        # Add density contours if available
        if has_density and add_density_contours:
            density_grid = results["density_grid"]
            # Normalize density to probabilities
            density_max = density_grid.max()
            prob_levels = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]
            contour_levels = [p * density_max for p in prob_levels]
            cs1 = ax1.contour(
                X1_grid, X2_grid, density_grid,
                levels=contour_levels, colors='white', 
                linestyles='dashed', linewidths=1.2, alpha=0.8
            )
            # Add labels
            fmt = {lev: f'p={prob:.2f}' for lev, prob in zip(contour_levels, prob_levels)}
            ax1.clabel(cs1, inline=True, fontsize=8, fmt=fmt)
        
        ax1.set_xlabel(r"$X_1$", fontsize=14)
        ax1.set_ylabel(r"$X_2$", fontsize=14)
        ax1.set_title("Weighted (Kernel) CQR", fontsize=14)
        ax1.set_xlim(-1, 1)
        ax1.set_ylim(-1, 1)
        ax1.set_aspect("equal")
        
        # RIGHT: Unweighted (Global) CQR - Beautiful blue heatmap
        c2 = ax2.contourf(
            X1_grid, X2_grid, width_grid_global, 
            levels=levels, cmap="YlGnBu_r", 
            vmin=vmin, vmax=vmax
        )
        
        # Add density contours if available
        if has_density and add_density_contours:
            cs2 = ax2.contour(
                X1_grid, X2_grid, density_grid,
                levels=contour_levels, colors='white', 
                linestyles='dashed', linewidths=1.2, alpha=0.8
            )
            ax2.clabel(cs2, inline=True, fontsize=8, fmt=fmt)
        
        ax2.set_xlabel(r"$X_1$", fontsize=14)
        ax2.set_ylabel(r"$X_2$", fontsize=14)
        ax2.set_title("Unweighted (Global) CQR", fontsize=14)
        ax2.set_xlim(-1, 1)
        ax2.set_ylim(-1, 1)
        ax2.set_aspect("equal")
        
        # Shared colorbar with nice styling
        cbar = fig.colorbar(c2, ax=[ax1, ax2], label=r"Interval Width $|\hat{\mathcal{C}}(x)|$", shrink=0.8)
        
        # Add density contour legend if present
        if has_density and add_density_contours:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='white', linestyle='--', 
                                      label='True Density Contours')]
            ax2.legend(handles=legend_elements, loc='upper right', 
                      facecolor='gray', framealpha=0.5, fontsize=10)
        
        fig.suptitle(title, fontsize=16, y=1.02)
    else:
        # Single beautiful blue heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Number of contour levels for smooth appearance
        levels = 50

        # Create smooth contourf heatmap with blue colormap
        c = ax.contourf(
            X1_grid, X2_grid, width_grid, 
            levels=levels, cmap="YlGnBu_r"
        )
        
        # Add density contours if available
        if has_density and add_density_contours:
            density_grid = results["density_grid"]
            # Normalize density to probabilities
            density_max = density_grid.max()
            prob_levels = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]
            contour_levels = [p * density_max for p in prob_levels]
            cs = ax.contour(
                X1_grid, X2_grid, density_grid,
                levels=contour_levels, colors='white', 
                linestyles='dashed', linewidths=1.5, alpha=0.9
            )
            # Add probability labels to contours
            fmt = {lev: f'p={prob:.2f}' for lev, prob in zip(contour_levels, prob_levels)}
            ax.clabel(cs, inline=True, fontsize=9, fmt=fmt)
            
            # Add legend for density contours
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color='white', linestyle='--', 
                                      linewidth=1.5, label='True Density Contours')]
            ax.legend(handles=legend_elements, loc='upper right', 
                     facecolor='gray', framealpha=0.5, fontsize=11)
        
        # Create colorbar with nice formatting
        cbar = fig.colorbar(c, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(r"Interval Width $|\hat{\mathcal{C}}(x)|$", fontsize=13, rotation=270, labelpad=25)
        
        ax.set_xlabel(r"$X_1$", fontsize=14)
        ax.set_ylabel(r"$X_2$", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=150)

    if show:
        plt.show()
    else:
        plt.close()

    print(f"\nSaved: {output_path}")

