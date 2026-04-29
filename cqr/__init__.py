"""
CQR Module: Conformalized Quantile Regression utilities.

Provides modular components for running CQR experiments with support
for arbitrary input dimensions d ∈ {1, 2, ...}.
"""

from .config import ExperimentConfig, load_config
from .models import QuantileNN, quantile_loss
from .models_requ import ReQU, QuantileReQUNN, train_requ_quantile_models
from .training import train_quantile_models_unified
from .real_data import load_dataset, list_datasets, DATASET_REGISTRY, DEFAULT_DATASETS
from .metrics import (
    marginal_coverage, average_width, median_width, width_std,
    conditional_coverage, evaluate_intervals,
)
from .preprocessing import prepare_data, inverse_transform_width
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
    compute_truncated_normal_density,
    compute_beta_density,
    compute_mixture_density,
    get_density_function,
    generate_guan2021,
    guan2021_oracle,
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
    # Models (ReLU)
    "QuantileNN",
    "quantile_loss",
    # Models (ReQU)
    "ReQU",
    "QuantileReQUNN",
    "train_requ_quantile_models",
    # Unified training
    "train_quantile_models_unified",
    # Real data
    "load_dataset",
    "list_datasets",
    "DATASET_REGISTRY",
    "DEFAULT_DATASETS",
    # Metrics
    "marginal_coverage",
    "average_width",
    "median_width",
    "width_std",
    "conditional_coverage",
    "evaluate_intervals",
    # Preprocessing
    "prepare_data",
    "inverse_transform_width",
    # Data (synthetic)
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
    "compute_truncated_normal_density",
    "compute_beta_density",
    "compute_mixture_density",
    "get_density_function",
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
