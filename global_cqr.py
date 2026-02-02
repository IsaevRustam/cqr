"""
Experiment A: Global Split CQR (Theorem 3)
==========================================
Demonstrates convergence rate of standard Split Conformal Quantile Regression
with a single global calibration constant Q̂.

Reference: Theorem 3 from the article on non-asymptotic guarantees for CQR.

Usage:
    python experiment_global_cqr.py                    # default config (d=1)
    python experiment_global_cqr.py --config configs/d2.yaml  # d=2
"""
# %%
import argparse
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
    setup_plotting,
    plot_convergence,
)
from cqr.models import train_quantile_models


def run_global_cqr_experiment(config: ExperimentConfig) -> pd.DataFrame:
    """
    Run Global Split CQR experiment to demonstrate convergence rate (Theorem 3).

    Key: Uses a SINGLE global calibration constant Q̂ = Quantile(S, β_m).

    The interval for any test point x is:
        Ĉ(x) = [f̂_{α/2}(x) - Q̂, f̂_{1-α/2}(x) + Q̂]

    Args:
        config: Experiment configuration

    Returns:
        DataFrame with columns ['N', 'rmse_mean', 'rmse_std']
    """
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Sample size grid
    n_grid = np.geomspace(
        config.n_grid_start, config.n_grid_end, num=config.n_grid_num, dtype=int
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
    print(f"Attempts: {config.n_attempts}, Theory rate: N^{{{config.theory_rate:.3f}}}")
    print("-" * 60)

    for N in n_grid:
        # Split: train and calibration
        n_train = N // 2
        m = N - n_train

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
                "rmse_mean": np.mean(batch_errors),
                "rmse_std": np.std(batch_errors),
            }
        )
        print(
            f"N={N:5d}: RMSE = {np.mean(batch_errors):.4f} ± {np.std(batch_errors):.4f}"
        )

    print("-" * 60)
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Global Split CQR Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: experiment_theorem_3_d{d}.pdf)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"experiment_theorem_3_d{config.d}.pdf"

    # Run experiment
    df_results = run_global_cqr_experiment(config)

    # Plot convergence
    setup_plotting()
    plot_convergence(
        df_results,
        output_path=output_path,
        title="Global Split CQR Convergence",
        theory_slope=config.theory_rate,
        d=config.d,
        beta=config.beta,
    )


if __name__ == "__main__":
    main()

# %%
