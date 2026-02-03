"""
Experiment A: Global Split CQR (Theorem 3)
==========================================
Demonstrates convergence rate of standard Split Conformal Quantile Regression
with a single global calibration constant Q̂.

Reference: Theorem 3 from the article on non-asymptotic guarantees for CQR.

This script runs a single experiment and saves results to CSV.
Use plot_combined_results.py to combine and visualize multiple runs.

Usage:
    python global_cqr.py                               # default config (d=1, c=0.5)
    python global_cqr.py --config configs/d2.yaml      # d=2
"""
# %%
import argparse
import os
import numpy as np
import torch
import pandas as pd

from cqr import (
    ExperimentConfig,
    load_config,
    generate_uniform_data,
    get_oracle_interval_length,
    compute_conformity_scores,
    global_calibration,
)
from cqr.models import train_quantile_models


def run_global_cqr_experiment(config: ExperimentConfig, c: float = 0.5) -> pd.DataFrame:
    """
    Run Global Split CQR experiment to demonstrate convergence rate (Theorem 3).

    Key: Uses a SINGLE global calibration constant Q̂ = Quantile(S, β_m).

    The interval for any test point x is:
        Ĉ(x) = [f̂_{α/2}(x) - Q̂, f̂_{1-α/2}(x) + Q̂]

    Args:
        config: Experiment configuration
        c: Calibration set size exponent (m = N^c)

    Returns:
        DataFrame with columns ['N', 'rmse_mean', 'rmse_std']
    """
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Training sample size grid
    n_train_grid = np.geomspace(
        config.n_train_grid_start, config.n_train_grid_end, num=config.n_train_grid_num, dtype=int
    )

    results = []

    # Generate fixed test set for evaluation
    if config.d == 1:
        X_test = np.linspace(-1, 1, config.n_test).reshape(-1, 1).astype(np.float32)
    else:
        # For higher d, use random test points
        X_test = np.random.uniform(-1, 1, (config.n_test, config.d)).astype(np.float32)

    len_oracle = get_oracle_interval_length(X_test, config.alpha, config.beta)

    print("=" * 60)
    print("Experiment A: Global Split CQR (Theorem 3)")
    print("=" * 60)
    print(f"Configuration: alpha={config.alpha}, beta={config.beta}, d={config.d}")
    print(f"Calibration size: m = {config.calibration_scale_c} * n^{c}, Theory rate: N^{{{config.theory_rate:.3f}}}")
    print(f"Attempts: {config.n_attempts}")
    print("-" * 60)

    for n_train in n_train_grid:
        # Calibration set size: m = C * n_train^c
        m = int(config.calibration_scale_c * (n_train ** c))
        N = n_train + m

        batch_errors = []

        for attempt in range(config.n_attempts):
            # Generate fresh data for each attempt
            X_train, Y_train = generate_uniform_data(
                n_train, d=config.d, beta=config.beta
            )
            X_cal, Y_cal = generate_uniform_data(m, d=config.d, beta=config.beta)

            # Convert to tensors
            X_t = torch.from_numpy(X_train)
            Y_t = torch.from_numpy(Y_train)

            # Train quantile models
            model_lo, model_hi = train_quantile_models(
                X_t,
                Y_t,
                tau_low=config.tau_low,
                tau_high=config.tau_high,
                input_dim=config.d,
                hidden_dim=config.hidden_dim,
                epochs=config.train_epochs,
                lr=config.learning_rate,
            )

            # Compute calibration scores
            with torch.no_grad():
                X_cal_t = torch.from_numpy(X_cal)
                pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
                pred_cal_hi = model_hi(X_cal_t).numpy().flatten()

            scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, Y_cal)

            # Global calibration
            Q_hat = global_calibration(scores, config.alpha)

            # Predict on test set
            with torch.no_grad():
                X_test_t = torch.from_numpy(X_test)
                pred_test_lo = model_lo(X_test_t).numpy().flatten()
                pred_test_hi = model_hi(X_test_t).numpy().flatten()

                # Interval length: (q_hi - q_lo) + 2 * Q̂
                len_hat = (pred_test_hi - pred_test_lo) + 2 * Q_hat
                len_hat = np.maximum(len_hat, 0.0)

            # RMSE of excess length
            rmse = np.sqrt(np.mean((len_hat - len_oracle) ** 2))
            batch_errors.append(rmse)

        results.append(
            {
                "N": N,
                "n_train": n_train,
                "m": m,
                "rmse_mean": np.mean(batch_errors),
                "rmse_std": np.std(batch_errors),
            }
        )
        print(
            f"n={n_train:5d}, m={m:5d}, N={N:5d}: RMSE = {np.mean(batch_errors):.4f} ± {np.std(batch_errors):.4f}"
        )

    print("-" * 60)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Global Split CQR Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file (default: use built-in defaults)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Run single experiment with calibration exponent from config
    c = config.calibration_exponent
    print(f"\n{'='*60}")
    print(f"Running experiment with c = {c} (m = {config.calibration_scale_c} * n^{c})")
    print(f"{'='*60}")
    df_results = run_global_cqr_experiment(config, c=c)

    # Save results to CSV
    csv_filename = f"results_global_cqr_d{config.d}_c{c}_seed{config.seed}.csv"
    csv_path = os.path.join(config.output_dir, csv_filename)
    df_results.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")
    print(f"\nTo generate combined plot (two subplots side-by-side):")
    print(f"  python plot_combined_results.py {csv_filename}")
    print(f"\nOr combine multiple CSVs:")
    print(f"  python plot_combined_results.py results_*.csv")
    print(f"\nThis creates: convergence_d{config.d}.pdf")


if __name__ == "__main__":
    main()

# %%
