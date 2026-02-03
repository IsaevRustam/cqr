"""
Combined Results Plotter for Global CQR Experiments
====================================================
Loads multiple CSV result files from separate experiment runs and creates
two separate convergence plots (like in the reference paper):
1. Length deviation vs Training size (n)
2. Length deviation vs Calibration size (m)

Usage:
    # Plot results from multiple c values
    python plot_combined_results.py results_global_cqr_d1_c1.0_seed42.csv \\
                                     results_global_cqr_d1_c0.66_seed42.csv \\
                                     results_global_cqr_d1_c0.33_seed42.csv
    
    # Use glob pattern
    python plot_combined_results.py results_*.csv
    
    # Specify custom output prefix
    python plot_combined_results.py results_*.csv --output my_experiment
    
    This will create: my_experiment_vs_n.pdf and my_experiment_vs_m.pdf
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

from cqr import setup_plotting


def parse_csv_metadata(filename: str) -> Tuple[int, float, int]:
    """
    Extract metadata from CSV filename.
    
    Expected format: results_global_cqr_d{d}_c{c}_seed{seed}.csv
    
    Args:
        filename: CSV filename
    
    Returns:
        Tuple of (d, c, seed)
    
    Raises:
        ValueError: If filename doesn't match expected format
    """
    pattern = r"results_global_cqr_d(\d+)_c([\d.]+)_seed(\d+)\.csv"
    match = re.search(pattern, filename)
    
    if not match:
        raise ValueError(
            f"Filename '{filename}' doesn't match expected format: "
            f"results_global_cqr_d{{d}}_c{{c}}_seed{{seed}}.csv"
        )
    
    d = int(match.group(1))
    c = float(match.group(2))
    seed = int(match.group(3))
    
    return d, c, seed


def load_and_group_results(csv_paths: List[str]) -> Dict[float, pd.DataFrame]:
    """
    Load CSV files and group by calibration exponent c.
    
    Args:
        csv_paths: List of paths to CSV result files
    
    Returns:
        Dictionary mapping c values to their corresponding DataFrames
    """
    results_by_c = {}
    d_values = set()
    
    print("Loading CSV files...")
    print("=" * 60)
    
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        
        try:
            d, c, seed = parse_csv_metadata(filename)
        except ValueError as e:
            print(f"Warning: Skipping {filename}: {e}")
            continue
        
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        required_cols = ['n_train', 'm', 'rmse_mean', 'rmse_std']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"Warning: Skipping {filename}: missing columns {missing_cols}")
            continue
        
        d_values.add(d)
        
        if c in results_by_c:
            print(f"  {filename}: d={d}, c={c}, seed={seed} (duplicate c, keeping last)")
        else:
            print(f"  {filename}: d={d}, c={c}, seed={seed}")
        
        results_by_c[c] = df
    
    print("=" * 60)
    
    if len(d_values) > 1:
        raise ValueError(
            f"CSVs have incompatible dimensions: {d_values}. "
            f"All CSVs must be from experiments with the same d value."
        )
    
    if not results_by_c:
        raise ValueError("No valid CSV files were loaded.")
    
    print(f"Loaded {len(results_by_c)} distinct c values: {sorted(results_by_c.keys())}")
    print(f"Dimension: d={d_values.pop()}")
    
    return results_by_c


def plot_convergence_vs_variable(
    all_results: Dict[float, pd.DataFrame],
    variable: str,
    output_path: str,
    title: str,
    d: int = 1,
    beta: float = 1.0,
) -> None:
    """
    Create convergence plot showing RMSE vs a single variable (n or m).
    
    Args:
        all_results: Dictionary mapping c values to DataFrames
        variable: Either 'n_train' or 'm'
        output_path: Path to save the figure
        title: Plot title
        d: Input dimension
        beta: Hölder smoothness
    """
    setup_plotting()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Colors for different c values
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    
    # Plot each c value
    for idx, (c_val, df) in enumerate(sorted(all_results.items(), reverse=True)):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        x_data = df[variable].values
        y_data = df["rmse_mean"].values
        yerr = df["rmse_std"].values
        
        # Linear regression in log-log space
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        slope, intercept, r_value, _, _ = linregress(log_x, log_y)
        
        var_label = "n" if variable == "n_train" else "m"
        print(f"\nc = {c_val} (m = N^{c_val}):")
        print(f"  Empirical slope vs {var_label}: {slope:.3f}")
        print(f"  (R² = {r_value**2:.4f})")
        
        # Plot points with error bars
        ax.errorbar(
            x_data,
            y_data,
            yerr=yerr,
            fmt=marker,
            color=color,
            ecolor=color,
            alpha=0.7,
            capsize=4,
            markersize=7,
            label=f"$c = {c_val}$ (slope: ${slope:.2f}$)",
            zorder=3 + idx,
        )
        
        # Fit line
        fit_y = np.exp(intercept + slope * log_x)
        ax.plot(
            x_data,
            fit_y,
            color=color,
            linewidth=2,
            linestyle="--",
            alpha=0.5,
            zorder=2 + idx,
        )
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    
    xlabel = "# Training $(n)$" if variable == "n_train" else "# Calibration $(m)$"
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(
        r"Excess Length RMSE $\|\hat{\mathcal{C}}| - |\mathcal{C}^*|\|_{L_2}$",
        fontsize=14,
    )
    ax.set_title(rf"{title} ($\beta={beta:.0f}$, $d={d}$)", fontsize=16)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, which="both", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create two convergence plots: vs n and vs m",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_combined_results.py results_*.csv
  python plot_combined_results.py results_*.csv --output my_experiment
  
This creates two plots:
  - {output}_vs_n.pdf (convergence vs training size)
  - {output}_vs_m.pdf (convergence vs calibration size)
        """
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV result files to combine"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file prefix (default: convergence_d{d})"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Hölder smoothness for plot annotation (default: 1.0)"
    )
    
    args = parser.parse_args()
    
    # Load and group results
    try:
        all_results = load_and_group_results(args.csv_files)
    except ValueError as e:
        print(f"\nError: {e}")
        return 1
    
    # Extract dimension
    try:
        d, _, _ = parse_csv_metadata(os.path.basename(args.csv_files[0]))
    except:
        d = 1
    
    # Determine output prefix
    if args.output:
        output_prefix = args.output
    else:
        output_prefix = f"convergence_d{d}"
    
    # Generate two plots
    print(f"\n{'='*60}")
    print("Generating plots...")
    print(f"{'='*60}")
    
    # Plot 1: vs n (training size)
    output_vs_n = f"{output_prefix}_vs_n.pdf"
    plot_convergence_vs_variable(
        all_results,
        variable="n_train",
        output_path=output_vs_n,
        title="Convergence vs Training Size",
        d=d,
        beta=args.beta,
    )
    
    # Plot 2: vs m (calibration size)
    output_vs_m = f"{output_prefix}_vs_m.pdf"
    plot_convergence_vs_variable(
        all_results,
        variable="m",
        output_path=output_vs_m,
        title="Convergence vs Calibration Size",
        d=d,
        beta=args.beta,
    )
    
    print(f"\n{'='*60}")
    print(f"Successfully created two plots:")
    print(f"  1. {output_vs_n}")
    print(f"  2. {output_vs_m}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
