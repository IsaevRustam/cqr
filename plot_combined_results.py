"""
Combined Results Plotter for Global CQR Experiments
====================================================
Loads multiple CSV result files from separate experiment runs and creates
a single figure with two side-by-side plots (like in the reference paper):
- Left: Length deviation vs Training size (n)
- Right: Length deviation vs Calibration size (m)

Usage:
    # Plot results from multiple c values
    python plot_combined_results.py results_global_cqr_d1_c1.0_seed42.csv \\
                                     results_global_cqr_d1_c0.66_seed42.csv \\
                                     results_global_cqr_d1_c0.33_seed42.csv
    
    # Use glob pattern
    python plot_combined_results.py results_*.csv
    
    # Specify custom output file
    python plot_combined_results.py results_*.csv --output my_experiment.pdf
    
    This creates a single PDF with two plots side-by-side.
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


def plot_dual_convergence(
    all_results: Dict[float, pd.DataFrame],
    output_path: str,
    d: int = 1,
    beta: float = 1.0,
) -> None:
    """
    Create a single figure with two side-by-side plots (like in the reference paper):
    - Left: Convergence vs training size (n)
    - Right: Convergence vs calibration size (m)
    
    Args:
        all_results: Dictionary mapping c values to DataFrames
        output_path: Path to save the figure
        d: Input dimension
        beta: Hölder smoothness
    """
    setup_plotting()
    
    # Create figure with two square subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Colors for different c values
    colors = ["#d62728", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "^"]
    
    # Plot both graphs
    for variable, ax, title_suffix in [
        ("n_train", ax1, "vs Training Size $(n)$"),
        ("m", ax2, "vs Calibration Size $(m)$")
    ]:
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
            if variable == "n_train":
                print(f"\nc = {c_val}:")
                print(f"  Slope vs {var_label}: {slope:.3f} (R² = {r_value**2:.4f})", end="")
            else:
                print(f", vs {var_label}: {slope:.3f} (R² = {r_value**2:.4f})")
            
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
                markersize=6,
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
        
        # Configure axes
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        xlabel = "# Training $(n)$" if variable == "n_train" else "# Calibration $(m)$"
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel(
            r"Excess Length RMSE",
            fontsize=13,
        )
        ax.set_title(title_suffix, fontsize=14)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, which="both", alpha=0.3)
    
    # Add overall title
    fig.suptitle(rf"Global CQR Convergence ($\beta={beta:.0f}$, $d={d}$)", 
                 fontsize=16, y=0.98)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"\nSaved combined plot: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create combined convergence plot with two subplots (vs n and vs m)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_combined_results.py results_*.csv
  python plot_combined_results.py results_*.csv --output my_experiment.pdf
  
This creates a single PDF with two side-by-side plots.
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
        help="Output PDF path (default: convergence_d{d}.pdf)"
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
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"convergence_d{d}.pdf"
    
    # Generate combined plot
    print(f"\n{'='*60}")
    print("Generating combined plot...")
    print(f"{'='*60}")
    
    plot_dual_convergence(
        all_results,
        output_path=output_path,
        d=d,
        beta=args.beta,
    )
    
    print(f"\n{'='*60}")
    print(f"Successfully created combined plot: {output_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
