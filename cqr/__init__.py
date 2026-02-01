"""
CQR Module: Conformalized Quantile Regression utilities.

Provides modular components for running CQR experiments with support
for arbitrary input dimensions d ∈ {1, 2, ...}.
"""

from .config import ExperimentConfig, load_config
from .models import QuantileNN, quantile_loss
from .data import (
    get_ground_truth,
    get_oracle_bounds_generic,
    generate_uniform_data,
    generate_truncated_normal_data,
    generate_beta_data,
    generate_mixture_data,
    get_oracle_interval_length,
    get_oracle_bounds,
    get_oracle_bounds_beta,
    get_oracle_bounds_mixture,
)
from .calibration import (
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
)
from .plotting import setup_plotting, plot_convergence, plot_density_intervals, plot_heatmap_d2

__all__ = [
    # Config
    "ExperimentConfig",
    "load_config",
    # Models
    "QuantileNN",
    "quantile_loss",
    # Data
    "get_ground_truth",
    "get_oracle_bounds_generic",
    "generate_uniform_data",
    "generate_truncated_normal_data",
    "generate_beta_data",
    "generate_mixture_data",
    "get_oracle_interval_length",
    "get_oracle_bounds",
    "get_oracle_bounds_beta",
    "get_oracle_bounds_mixture",
    # Calibration
    "compute_conformity_scores",
    "global_calibration",
    "LocalConformalOptimizer",
    # Plotting
    "setup_plotting",
    "plot_convergence",
    "plot_density_intervals",
    "plot_heatmap_d2",
]
