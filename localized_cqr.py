"""
Experiment B: Localized (Kernel) CQR vs Global CQR Comparison
==============================================================
Demonstrates density-adaptive prediction intervals using kernel-weighted
conformal calibration and compares against unweighted (global) CQR.

Key insight: Weighted CQR produces narrow intervals where data density is high,
and wider where density is low. Unweighted CQR produces constant-width intervals.

Reference: Theorem 5 from the article on non-asymptotic guarantees for CQR.

Usage:
    python localized_cqr.py                                        # truncated normal, d=1
    python localized_cqr.py --distribution beta                    # beta distribution, d=1
    python localized_cqr.py --distribution mixture                 # gaussian mixture, d=1
    python localized_cqr.py --config configs/d2.yaml               # truncated normal, d=2
    python localized_cqr.py --config configs/d2.yaml --distribution beta  # beta, d=2
"""

import argparse
import numpy as np
import torch
from typing import Dict, Any, Callable, Tuple

from cqr import (
    ExperimentConfig,
    load_config,
    generate_truncated_normal_data,
    generate_beta_data,
    generate_mixture_data,
    get_oracle_bounds,
    get_oracle_bounds_beta,
    get_oracle_bounds_mixture,
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
    setup_plotting,
    plot_density_intervals,
    plot_heatmap_d2,
)
from cqr.models import train_quantile_models
from cqr.calibration import compute_bandwidth


# =============================================================================
# DATA GENERATOR REGISTRY
# =============================================================================

def get_data_generator(distribution: str) -> Tuple[Callable, Callable, str]:
    """
    Get data generator and oracle bounds function for the specified distribution.
    
    Returns:
        (generate_func, oracle_func, display_name)
    """
    if distribution == "truncated_normal":
        return generate_truncated_normal_data, get_oracle_bounds, "Truncated Normal"
    elif distribution == "beta":
        return generate_beta_data, get_oracle_bounds_beta, "Beta(2, 5)"
    elif distribution == "mixture":
        return generate_mixture_data, get_oracle_bounds_mixture, "Gaussian Mixture"
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_localized_cqr_experiment(
    config: ExperimentConfig,
    distribution: str = "truncated_normal"
) -> Dict[str, Any]:
    """
    Run Localized CQR experiment with weighted vs unweighted comparison.

    Computes BOTH:
    1. Weighted (kernel) intervals - adaptive to local density
    2. Unweighted (global) intervals - constant Q̂ everywhere

    Args:
        config: Experiment configuration
        distribution: One of 'truncated_normal', 'beta', 'mixture'

    Returns:
        Dictionary with results for plotting
    """
    # Set seeds
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    N = config.n_fixed

    # Split
    n_train = N // 2
    m = N - n_train

    # Bandwidth: h ~ m^{-1/(2γ+d)}, where γ is Hölder constant of density
    gamma = 1.0  # Assume smooth density
    h = compute_bandwidth(m, config.d, gamma)

    # Get appropriate data generator
    generate_data, get_oracle, dist_name = get_data_generator(distribution)

    print("=" * 60)
    print("Experiment B: Weighted vs Unweighted CQR Comparison")
    print("=" * 60)
    print(f"Configuration: N={N}, alpha={config.alpha}, beta={config.beta}, d={config.d}")
    print(f"Distribution: {dist_name}")
    print(f"Bandwidth: h={h:.4f}")
    print("-" * 60)

    # Generate data
    X_train, Y_train = generate_data(n_train, d=config.d, beta=config.beta)
    X_cal, Y_cal = generate_data(m, d=config.d, beta=config.beta)

    # Test scatter points for visualization
    X_test_scatter, Y_test_scatter = generate_data(750, d=config.d, beta=config.beta)

    # Sorted grid for evaluation
    if config.d == 1:
        X_grid = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)
        n_grid = 1000
    elif config.d == 2:
        # For d=2, create a meshgrid for heatmap
        n_grid = 100
        x1 = np.linspace(-1, 1, n_grid)
        x2 = np.linspace(-1, 1, n_grid)
        X1_grid, X2_grid = np.meshgrid(x1, x2)

        # Flatten for model prediction -> shape (10000, 2)
        X_grid = np.column_stack([X1_grid.ravel(), X2_grid.ravel()]).astype(np.float32)
    else:
        # For other multi-d, project onto first axis
        X_grid = np.linspace(-1, 1, 1000).reshape(-1, 1).astype(np.float32)
        # Pad with zeros for other dimensions to evaluate at x_2=...=x_d=0
        X_grid = np.hstack(
            [X_grid] + [np.zeros((1000, 1), dtype=np.float32)] * (config.d - 1)
        )

    # Convert to tensors
    X_t = torch.from_numpy(X_train)
    Y_t = torch.from_numpy(Y_train)

    print("Training quantile networks...")
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
    print("Training complete.")

    # Compute calibration scores
    with torch.no_grad():
        X_cal_t = torch.from_numpy(X_cal)
        pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
        pred_cal_hi = model_hi(X_cal_t).numpy().flatten()

    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, Y_cal)

    # =========================================================================
    # GLOBAL (UNWEIGHTED) CALIBRATION
    # =========================================================================
    Q_hat_global = global_calibration(scores, config.alpha)
    print(f"Global Q̂ = {Q_hat_global:.4f}")

    # =========================================================================
    # LOCAL (WEIGHTED) CALIBRATION
    # =========================================================================
    lcp = LocalConformalOptimizer(X_cal, scores, h=h)

    print(f"Predicting local corrections on grid ({len(X_grid)} points)...")

    # Split X_grid into batches to avoid memory issues with distance matrix
    batch_size = 1000
    Q_hat_chunks = []

    num_batches = int(np.ceil(len(X_grid) / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_grid))

        X_grid_batch = X_grid[start_idx:end_idx]

        # Predict only for current chunk
        q_chunk = lcp.predict_corrections(X_grid_batch, config.alpha)
        Q_hat_chunks.append(q_chunk)

        if (i + 1) % 5 == 0:
            print(f"  Processed batch {i + 1}/{num_batches}")

    Q_hat_grid = np.concatenate(Q_hat_chunks)
    print("Prediction complete.")

    # Predict intervals on grid
    with torch.no_grad():
        X_grid_t = torch.from_numpy(X_grid)
        pred_grid_lo = model_lo(X_grid_t).numpy().flatten()
        pred_grid_hi = model_hi(X_grid_t).numpy().flatten()

    # =========================================================================
    # CONSTRUCT INTERVALS
    # =========================================================================
    
    # Weighted (Localized) CQR interval boundaries
    interval_lo = pred_grid_lo - Q_hat_grid
    interval_hi = pred_grid_hi + Q_hat_grid

    # Unweighted (Global) CQR interval boundaries - CONSTANT correction
    interval_lo_global = pred_grid_lo - Q_hat_global
    interval_hi_global = pred_grid_hi + Q_hat_global

    # Oracle boundaries
    oracle_lo, oracle_hi = get_oracle(X_grid, config.alpha, config.beta)

    print("-" * 60)
    
    # Compute width statistics
    width_weighted = interval_hi - interval_lo
    width_global = interval_hi_global - interval_lo_global
    
    print(f"Weighted interval width: mean={np.mean(width_weighted):.2f}, std={np.std(width_weighted):.2f}")
    print(f"Global interval width:   mean={np.mean(width_global):.2f}, std={np.std(width_global):.2f} (constant)")
    print("-" * 60)

    # For plotting, use first dimension only if d != 2
    res = {
        "X_grid": X_grid[:, 0] if config.d > 1 else X_grid.flatten(),
        "X_test": X_test_scatter[:, 0] if config.d > 1 else X_test_scatter.flatten(),
        "Y_test": Y_test_scatter.flatten(),
        "X_train": X_train[:, 0] if config.d > 1 else X_train.flatten(),
        "interval_lo": interval_lo,
        "interval_hi": interval_hi,
        "interval_lo_global": interval_lo_global,
        "interval_hi_global": interval_hi_global,
        "oracle_lo": oracle_lo,
        "oracle_hi": oracle_hi,
        "distribution": distribution,
    }

    if config.d == 2:
        # Add 2D data for heatmap
        width_grid = (interval_hi - interval_lo).reshape(n_grid, n_grid)
        width_grid_global = (interval_hi_global - interval_lo_global).reshape(n_grid, n_grid)
        res.update({
            "X1_grid": X1_grid,
            "X2_grid": X2_grid,
            "width_grid": width_grid,
            "width_grid_global": width_grid_global,
            "X_train": X_train,  # Full 2D X_train
        })

    return res


def main():
    parser = argparse.ArgumentParser(description="Localized CQR Experiment")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: use built-in defaults)",
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="truncated_normal",
        choices=["truncated_normal", "beta", "mixture"],
        help="X distribution: truncated_normal, beta, or mixture",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PDF path (default: comparison_d{d}_{distribution}.pdf)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"comparison_d{config.d}_{args.distribution}.pdf"

    # Run experiment
    results = run_localized_cqr_experiment(config, distribution=args.distribution)

    # Get distribution display name for title
    _, _, dist_name = get_data_generator(args.distribution)

    # Plot comparison
    setup_plotting()

    if config.d == 2:
        plot_heatmap_d2(
            results,
            output_path=output_path,
            title=f"Weighted vs Unweighted CQR (d={config.d}, {dist_name})",
        )
    else:
        plot_density_intervals(
            results,
            output_path=output_path,
            title=f"Weighted vs Unweighted CQR (d={config.d}, {dist_name})",
            distribution=args.distribution,
        )


if __name__ == "__main__":
    main()
