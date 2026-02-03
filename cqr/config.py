"""
Experiment configuration with YAML support.
"""

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Dict, Any
import yaml


@dataclass
class ExperimentConfig:
    """
    Configuration for CQR experiments.

    Attributes:
        alpha: Miscoverage level (default 0.05 for 95% coverage)
        beta: Hölder smoothness parameter (fixed at 1.0 for now)
        d: Input dimension (1, 2, 4, etc.)
        seed: Random seed for reproducibility
        n_attempts: Number of Monte Carlo repetitions
        hidden_dim: Hidden layer dimension for QuantileNN
        train_epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        n_train_grid: Training sample sizes for convergence experiment (geomspace)
        n_train_grid_start: Start of geomspace for n_train_grid
        n_train_grid_end: End of geomspace for n_train_grid
        n_train_grid_num: Number of points in n_train_grid
        n_test: Number of test points for evaluation
        output_dir: Directory for output files
        dist_params: Distribution-specific parameters dict with keys:
            - truncated_normal: {loc, scale}
            - beta: {a, b}
            - mixture: {centers, scales, weights}
    """

    # Core experiment parameters
    alpha: float = 0.05
    beta: float = 1.0
    d: int = 1
    seed: int = 42

    # Monte Carlo settings
    n_attempts: int = 50

    # Neural network settings
    hidden_dim: int = 64
    train_epochs: int = 300
    learning_rate: float = 0.01

    # Sample size grid (for convergence experiments)
    # Now based on training sample size n, calibration m = n^c
    n_train_grid_start: int = 100
    n_train_grid_end: int = 10000
    n_train_grid_num: int = 15

    # Fixed sample size (for localized experiments)
    n_fixed: int = 20000

    # Bandwidth scale factor for localized CQR
    bandwidth_scale: float = 6.0
    
    # Calibration size multiplier for global CQR (m = C * n^c)
    # Ensures sufficient samples for quantile estimation while maintaining rate
    calibration_scale_c: float = 5.0

    # Test set size
    n_test: int = 1000

    # Output
    output_dir: str = "."
    
    # Distribution parameters (configurable per distribution)
    dist_params: Dict[str, Any] = field(default_factory=lambda: {
        "truncated_normal": {"loc": 0.0, "scale": 0.5},
        "beta": {"a": 2.0, "b": 5.0},
        "mixture": {
            "centers": (-0.6, 0.6),
            "scales": (0.15, 0.15),
            "weights": (0.5, 0.5)
        }
    })

    @property
    def tau_low(self) -> float:
        """Lower quantile level."""
        return self.alpha / 2

    @property
    def tau_high(self) -> float:
        """Upper quantile level."""
        return 1 - self.alpha / 2

    @property
    def theory_rate(self) -> float:
        """Theoretical convergence rate exponent: -β/(2β+d)."""
        return -self.beta / (2 * self.beta + self.d)

    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        """Load configuration from YAML file."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """
    Load configuration from YAML file or return default config.

    Args:
        path: Path to YAML config file. If None, returns default config.

    Returns:
        ExperimentConfig instance
    """
    if path is None:
        return ExperimentConfig()
    return ExperimentConfig.from_yaml(Path(path))


# Default config for quick access
DEFAULT_CONFIG = ExperimentConfig()
