"""
Combined Results Plotter for Global CQR Experiments
====================================================
Loads multiple CSV result files from separate experiment runs and combines them
into a single convergence plot.

This script enables the decoupled workflow:
1. Run experiments separately (possibly with different seeds or configs)
2. Save raw results to CSVs
3. Combine and plot them together

Usage:
    # Plot results from multiple c values
    python plot_combined_results.py results_global_cqr_d1_c1.0_seed42.csv \\
                                     results_global_cqr_d1_c0.66_seed42.csv \\
                                     results_global_cqr_d1_c0.33_seed42.csv
    
    # Use glob pattern
    python plot_combined_results.py results_*.csv
    
    # Specify custom output path
    python plot_combined_results.py results_*.csv --output my_plot.pdf
    
    # Custom title
    python plot_combined_results.py results_*.csv --title "My Custom Title"
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from cqr import setup_plotting, plot_convergence_multiple_c


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
    # Pattern: results_global_cqr_d{d}_c{c}_seed{seed}.csv
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
    
    Raises:
        ValueError: If CSVs have incompatible dimensions or required columns are missing
    """
    results_by_c = {}
    d_values = set()
    
    print("Loading CSV files...")
    print("=" * 60)
    
    for csv_path in csv_paths:
        filename = os.path.basename(csv_path)
        
        # Parse metadata
        try:
            d, c, seed = parse_csv_metadata(filename)
        except ValueError as e:
            print(f"Warning: Skipping {filename}: {e}")
            continue
        
        # Load DataFrame
        if not os.path.exists(csv_path):
            print(f"Warning: File not found: {csv_path}")
            continue
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['N', 'n_train', 'm', 'rmse_mean', 'rmse_std']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"Warning: Skipping {filename}: missing columns {missing_cols}")
            continue
        
        # Track dimension
        d_values.add(d)
        
        # Group by c value
        if c in results_by_c:
            print(f"  {filename}: d={d}, c={c}, seed={seed} (duplicate c value, will average)")
            # If multiple CSVs with same c, we could average them
            # For now, just keep the last one
            results_by_c[c] = df
        else:
            print(f"  {filename}: d={d}, c={c}, seed={seed}")
            results_by_c[c] = df
    
    print("=" * 60)
    
    # Validate all CSVs have same dimension
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


def main():
    parser = argparse.ArgumentParser(
        description="Combine and plot results from multiple Global CQR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_combined_results.py results_*.csv
  python plot_combined_results.py results_d1_c*.csv --output combined_plot.pdf
  python plot_combined_results.py file1.csv file2.csv file3.csv --title "My Experiment"
        """
    )
    parser.add_argument(
        "csv_files",
        nargs="+",
        help="CSV result files to combine (supports glob patterns)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: combined_convergence_d{d}.pdf)"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Global Split CQR Convergence",
        help="Plot title (default: 'Global Split CQR Convergence')"
    )
    parser.add_argument(
        "--theory-slope",
        type=float,
        default=-1/3,
        help="Theoretical convergence rate exponent (default: -1/3)"
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
    
    # Extract dimension from first result
    first_df = next(iter(all_results.values()))
    # We can infer d from the filename or just use a default
    # For now, parse it from the first CSV filename
    try:
        d, _, _ = parse_csv_metadata(os.path.basename(args.csv_files[0]))
    except:
        d = 1  # default fallback
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"combined_convergence_d{d}.pdf"
    
    # Generate plot
    print(f"\nGenerating combined plot...")
    setup_plotting()
    plot_convergence_multiple_c(
        all_results,
        output_path=output_path,
        title=args.title,
        theory_slope=args.theory_slope,
        d=d,
        beta=args.beta,
        show=False,
    )
    
    print(f"\n{'='*60}")
    print(f"Combined plot saved to: {output_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == "__main__":
    exit(main())
