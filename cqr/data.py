"""
Data generation utilities with multi-dimensional support.

Supports dimensions d ∈ {1, 2, 3, 4, ...} using norm-based heteroscedasticity.

CRITICAL: All generators use the SAME ground truth function (get_ground_truth)
to ensure valid comparison across different X distributions.
"""

import numpy as np
from scipy.stats import truncnorm, norm
from typing import Tuple


# =============================================================================
# UNIFIED GROUND TRUTH FUNCTION
# =============================================================================

def get_ground_truth(
    X: np.ndarray,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the ground truth mean μ(x) and std σ(x) for the regression model.
    
    This is the SINGLE SOURCE OF TRUTH for the regression function.
    All data generators and oracle functions MUST use this.
    
    Y | X=x ~ N(μ(x), σ(x)²)
    
    where:
        μ(x) = 3 * sin(5 * ||x||) + mu_scale * ||x||^β
        σ(x) = sigma_base + sigma_scale * ||x||^β
    
    Args:
        X: Features of shape (n,) or (n, d)
        beta: Hölder smoothness parameter
        mu_scale: Scaling for the trend component
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise
    
    Returns:
        (mu_x, sigma_x): Mean and std, each of shape (n, 1)
    """
    # Ensure X is 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Compute norm
    if X.shape[1] == 1:
        norm_x = np.abs(X)
    else:
        norm_x = np.linalg.norm(X, axis=1, keepdims=True)
    
    # Ground truth functions - SAME FOR ALL DISTRIBUTIONS
    # sin(5*||x||) adds non-linearity, ||x||^β adds trend
    mu_x = 3 * np.sin(5 * norm_x) + mu_scale * np.power(norm_x, beta)
    sigma_x = sigma_base + sigma_scale * np.power(norm_x, beta)
    
    return mu_x.astype(np.float32), sigma_x.astype(np.float32)


def get_oracle_bounds_generic(
    X: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oracle quantile bounds using the unified ground truth.
    
    For Gaussian noise:
        q_{α/2}(x) = μ(x) + σ(x) * Φ^{-1}(α/2)
        q_{1-α/2}(x) = μ(x) + σ(x) * Φ^{-1}(1-α/2)
    
    Args:
        X: Features of shape (n, d)
        alpha: Miscoverage level
        beta: Hölder smoothness
        mu_scale, sigma_base, sigma_scale: Ground truth parameters
    
    Returns:
        (q_lo, q_hi): Oracle lower and upper bounds, each of shape (n,)
    """
    mu_x, sigma_x = get_ground_truth(X, beta, mu_scale, sigma_base, sigma_scale)
    
    z_lo = norm.ppf(alpha / 2)
    z_hi = norm.ppf(1 - alpha / 2)
    
    q_lo = (mu_x + sigma_x * z_lo).flatten()
    q_hi = (mu_x + sigma_x * z_hi).flatten()
    
    return q_lo, q_hi


# =============================================================================
# UNIFORM DISTRIBUTION DATA
# =============================================================================


def generate_uniform_data(
    n: int,
    d: int = 1,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heteroscedastic regression data with uniform X.

    X ~ Uniform[-1, 1]^d
    Y = μ(X) + σ(X) * ε, where ε ~ N(0, 1)

    Uses get_ground_truth() for μ(x) and σ(x).

    Args:
        n: Sample size
        d: Input dimension
        beta: Hölder smoothness parameter
        mu_scale: Scaling for the trend component
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise

    Returns:
        X: Features of shape (n, d)
        Y: Targets of shape (n, 1)
    """
    X = np.random.uniform(-1, 1, (n, d)).astype(np.float32)
    
    # Use unified ground truth
    mu_x, sigma_x = get_ground_truth(X, beta, mu_scale, sigma_base, sigma_scale)

    # Generate response
    epsilon = np.random.normal(0, 1, (n, 1)).astype(np.float32)
    Y = mu_x + sigma_x * epsilon

    return X, Y.astype(np.float32)


def get_oracle_interval_length(
    X: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> np.ndarray:
    """
    Compute the oracle (true) interval length |C*(x)|.

    For Gaussian noise: |C*(x)| = 2 * z_{1-α/2} * σ(x)

    Args:
        X: Features of shape (n, d)
        alpha: Miscoverage level
        beta: Hölder smoothness
        sigma_base: Base noise level (must match generate functions)
        sigma_scale: Heteroscedastic scale (must match generate functions)

    Returns:
        Oracle interval lengths of shape (n,)
    """
    _, sigma_x = get_ground_truth(X, beta, mu_scale=5.0, sigma_base=sigma_base, sigma_scale=sigma_scale)
    z_score = norm.ppf(1 - alpha / 2)

    return (2 * z_score * sigma_x).flatten()


# =============================================================================
# TRUNCATED NORMAL DISTRIBUTION DATA
# =============================================================================


def generate_truncated_normal_data(
    n: int,
    d: int = 1,
    beta: float = 1.0,
    loc: float = 0.0,
    scale: float = 0.5,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heteroscedastic regression data with TRUNCATED NORMAL X.

    X_i ~ TruncatedNormal(loc, scale) on [-1, 1] independently for each dimension.
    Y = μ(X) + σ(X) * ε, where ε ~ N(0, 1)

    Uses get_ground_truth() for μ(x) and σ(x) - SAME as uniform!

    Args:
        n: Sample size
        d: Input dimension
        beta: Hölder smoothness parameter
        loc: Mean of the underlying normal (each dimension)
        scale: Std of the underlying normal (each dimension)
        mu_scale: Scaling for the mean function
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise

    Returns:
        X: Features of shape (n, d)
        Y: Targets of shape (n, 1)
    """
    # Standardized bounds for truncnorm
    a = (-1 - loc) / scale
    b = (1 - loc) / scale

    # Generate each dimension independently
    X = truncnorm.rvs(a, b, loc=loc, scale=scale, size=(n, d)).astype(np.float32)

    # Use unified ground truth - SAME FUNCTION AS UNIFORM
    mu_x, sigma_x = get_ground_truth(X, beta, mu_scale, sigma_base, sigma_scale)

    # Generate response
    epsilon = np.random.normal(0, 1, (n, 1)).astype(np.float32)
    Y = mu_x + sigma_x * epsilon

    return X, Y.astype(np.float32)


def get_oracle_bounds(
    X: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oracle quantile bounds for truncated normal (or any) distribution.
    
    Uses the unified get_ground_truth() function.
    """
    return get_oracle_bounds_generic(X, alpha, beta, mu_scale, sigma_base, sigma_scale)


# =============================================================================
# TEST GRID GENERATION
# =============================================================================


def generate_test_grid(d: int, n_per_dim: int = 50) -> np.ndarray:
    """
    Generate a regular grid on [-1, 1]^d for evaluation.

    For d=1: linspace
    For d>1: meshgrid flattened

    Args:
        d: Dimension
        n_per_dim: Number of points per dimension

    Returns:
        X_grid of shape (n_per_dim^d, d) — warning: grows exponentially!
    """
    if d == 1:
        return np.linspace(-1, 1, n_per_dim).reshape(-1, 1).astype(np.float32)

    # For higher dimensions, use meshgrid
    axes = [np.linspace(-1, 1, n_per_dim) for _ in range(d)]
    grids = np.meshgrid(*axes, indexing="ij")
    X_grid = np.stack([g.flatten() for g in grids], axis=1).astype(np.float32)

    return X_grid


def generate_random_test_points(n: int, d: int) -> np.ndarray:
    """
    Generate random test points uniformly on [-1, 1]^d.

    Preferred over grid for high dimensions to avoid exponential blowup.

    Args:
        n: Number of test points
        d: Dimension

    Returns:
        X_test of shape (n, d)
    """
    return np.random.uniform(-1, 1, (n, d)).astype(np.float32)


# =============================================================================
# BETA DISTRIBUTION DATA (ASYMMETRIC DENSITY)
# =============================================================================


def generate_beta_data(
    n: int,
    d: int = 1,
    beta: float = 1.0,
    a: float = 2.0,
    b: float = 5.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heteroscedastic regression data with BETA distribution X.

    X_i ~ Beta(a, b) scaled to [-1, 1] independently for each dimension.
    Y = μ(X) + σ(X) * ε, where ε ~ N(0, 1)

    With a=2, b=5: High density on the LEFT side (-1), low on the right (+1).
    
    Uses get_ground_truth() for μ(x) and σ(x) - SAME AS UNIFORM!

    Args:
        n: Sample size
        d: Input dimension
        beta: Hölder smoothness parameter
        a, b: Beta distribution parameters
        mu_scale: Scaling for the mean function
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise

    Returns:
        X: Features of shape (n, d)
        Y: Targets of shape (n, 1)
    """
    from scipy.stats import beta as beta_dist

    # Generate Beta(a, b) on [0, 1] then scale to [-1, 1]
    X_01 = beta_dist.rvs(a, b, size=(n, d))
    X = (2 * X_01 - 1).astype(np.float32)  # Scale to [-1, 1]

    # Use unified ground truth - SAME FUNCTION AS UNIFORM
    mu_x, sigma_x = get_ground_truth(X, beta, mu_scale, sigma_base, sigma_scale)

    # Generate response
    epsilon = np.random.normal(0, 1, (n, 1)).astype(np.float32)
    Y = mu_x + sigma_x * epsilon

    return X, Y.astype(np.float32)


def get_oracle_bounds_beta(
    X: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oracle quantile bounds for Beta distribution data.
    Uses the unified get_ground_truth() function - SAME AS ALL OTHERS.
    """
    return get_oracle_bounds_generic(X, alpha, beta, mu_scale, sigma_base, sigma_scale)


# =============================================================================
# GAUSSIAN MIXTURE DISTRIBUTION DATA (BIMODAL DENSITY)
# =============================================================================


def generate_mixture_data(
    n: int,
    d: int = 1,
    beta: float = 1.0,
    centers: Tuple[float, float] = (-0.6, 0.6),
    scales: Tuple[float, float] = (0.15, 0.15),
    weights: Tuple[float, float] = (0.5, 0.5),
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate heteroscedastic regression data with GAUSSIAN MIXTURE X.

    X_i ~ sum_k w_k * TruncatedNormal(center_k, scale_k) on [-1, 1]
    Y = μ(X) + σ(X) * ε, where ε ~ N(0, 1)

    Default: Two modes at -0.6 and +0.6 with tight variance.
    This creates sparse regions around x=0 and edges, dense regions at modes.
    
    Uses get_ground_truth() for μ(x) and σ(x) - SAME AS UNIFORM!

    Args:
        n: Sample size
        d: Input dimension
        beta: Hölder smoothness parameter
        centers: Centers of the two mixture components
        scales: Scales (std) of the two components
        weights: Mixing weights (sum to 1)
        mu_scale: Scaling for the mean function
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise

    Returns:
        X: Features of shape (n, d)
        Y: Targets of shape (n, 1)
    """
    # Sample component assignments
    n1 = int(n * weights[0])
    n2 = n - n1

    X_parts = []
    for i, (center, scale, count) in enumerate(
        [(centers[0], scales[0], n1), (centers[1], scales[1], n2)]
    ):
        if count == 0:
            continue
        # Truncated normal on [-1, 1]
        a = (-1 - center) / scale
        b = (1 - center) / scale
        X_comp = truncnorm.rvs(a, b, loc=center, scale=scale, size=(count, d))
        X_parts.append(X_comp)

    X = np.vstack(X_parts).astype(np.float32)
    # Shuffle to mix components
    np.random.shuffle(X)

    # Use unified ground truth - SAME FUNCTION AS UNIFORM
    mu_x, sigma_x = get_ground_truth(X, beta, mu_scale, sigma_base, sigma_scale)

    # Generate response
    epsilon = np.random.normal(0, 1, (n, 1)).astype(np.float32)
    Y = mu_x + sigma_x * epsilon

    return X, Y.astype(np.float32)


def get_oracle_bounds_mixture(
    X: np.ndarray,
    alpha: float,
    beta: float = 1.0,
    mu_scale: float = 5.0,
    sigma_base: float = 3.0,
    sigma_scale: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute oracle quantile bounds for Mixture distribution data.
    Uses the unified get_ground_truth() function - SAME AS ALL OTHERS.
    """
    return get_oracle_bounds_generic(X, alpha, beta, mu_scale, sigma_base, sigma_scale)
