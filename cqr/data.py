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

    For EACH dimension independently:
        X_j ~ sum_k w_k * TruncatedNormal(center_k, scale_k) on [-1, 1]
    
    This creates 2^d modes for d dimensions (product of marginals).
    Y = μ(X) + σ(X) * ε, where ε ~ N(0, 1)

    Default: Two modes at -0.6 and +0.6 with tight variance per dimension.
    For d=2, this creates 4 modes at corners: (-0.6,-0.6), (-0.6,0.6), (0.6,-0.6), (0.6,0.6)
    
    Uses get_ground_truth() for μ(x) and σ(x) - SAME AS UNIFORM!

    Args:
        n: Sample size
        d: Input dimension
        beta: Hölder smoothness parameter
        centers: Centers of the two mixture components (per marginal)
        scales: Scales (std) of the two components (per marginal)
        weights: Mixing weights (sum to 1, per marginal)
        mu_scale: Scaling for the mean function
        sigma_base: Base noise level
        sigma_scale: Scaling for heteroscedastic noise

    Returns:
        X: Features of shape (n, d)
        Y: Targets of shape (n, 1)
    """
    # Allocate output array
    X = np.zeros((n, d), dtype=np.float32)
    
    # For each dimension INDEPENDENTLY, sample from the mixture
    for dim in range(d):
        # Sample component assignments independently for this dimension
        component = np.random.choice(len(centers), size=n, p=weights)
        
        for k, (center, scale) in enumerate(zip(centers, scales)):
            mask = (component == k)
            count = mask.sum()
            if count == 0:
                continue
            
            # Truncated normal on [-1, 1]
            a = (-1 - center) / scale
            b = (1 - center) / scale
            X[mask, dim] = truncnorm.rvs(a, b, loc=center, scale=scale, size=count)

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


# =============================================================================
# DENSITY COMPUTATION FUNCTIONS (FOR CONTOUR PLOTS)
# =============================================================================


def compute_truncated_normal_density(
    X: np.ndarray,
    loc: float = 0.0,
    scale: float = 0.5,
) -> np.ndarray:
    """
    Compute the true PDF of the truncated normal distribution at given points.
    
    For d > 1, computes the product of marginal densities (independent dimensions).
    
    Args:
        X: Points of shape (n, d) where to evaluate density
        loc: Mean of the underlying normal
        scale: Std of the underlying normal
        
    Returns:
        density: Array of shape (n,) with density values
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    a = (-1 - loc) / scale
    b = (1 - loc) / scale
    
    # Product of marginal densities
    density = np.ones(X.shape[0])
    for dim in range(X.shape[1]):
        density *= truncnorm.pdf(X[:, dim], a, b, loc=loc, scale=scale)
    
    return density


def compute_beta_density(
    X: np.ndarray,
    a: float = 2.0,
    b: float = 5.0,
) -> np.ndarray:
    """
    Compute the true PDF of the Beta distribution (scaled to [-1,1]) at given points.
    
    For d > 1, computes the product of marginal densities (independent dimensions).
    
    Args:
        X: Points of shape (n, d) in [-1, 1]
        a, b: Beta distribution parameters
        
    Returns:
        density: Array of shape (n,) with density values
    """
    from scipy.stats import beta as beta_dist
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Transform from [-1, 1] to [0, 1]
    X_01 = (X + 1) / 2
    
    # Product of marginal densities (with Jacobian 1/2 for each dimension)
    density = np.ones(X.shape[0])
    for dim in range(X.shape[1]):
        # PDF on [0,1] divided by 2 (Jacobian of x -> 2x - 1)
        density *= beta_dist.pdf(X_01[:, dim], a, b) / 2
    
    return density


def compute_mixture_density(
    X: np.ndarray,
    centers: Tuple[float, float] = (-0.6, 0.6),
    scales: Tuple[float, float] = (0.15, 0.15),
    weights: Tuple[float, float] = (0.5, 0.5),
) -> np.ndarray:
    """
    Compute the true PDF of the Gaussian mixture distribution at given points.
    
    For d > 1, computes the product of marginal densities (independent dimensions).
    Each marginal is a mixture of two truncated normals.
    
    Args:
        X: Points of shape (n, d) in [-1, 1]
        centers: Centers of the two mixture components
        scales: Scales (std) of the two components
        weights: Mixing weights (sum to 1)
        
    Returns:
        density: Array of shape (n,) with density values
    """
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Product of marginal densities
    density = np.ones(X.shape[0])
    
    for dim in range(X.shape[1]):
        x_dim = X[:, dim]
        marginal_density = np.zeros(X.shape[0])
        
        for center, scale, weight in zip(centers, scales, weights):
            a = (-1 - center) / scale
            b = (1 - center) / scale
            marginal_density += weight * truncnorm.pdf(x_dim, a, b, loc=center, scale=scale)
        
        density *= marginal_density
    
    return density


def get_density_function(distribution: str, dist_params: dict = None):
    """
    Get the density function for a given distribution type.
    
    Args:
        distribution: One of 'truncated_normal', 'beta', 'mixture'
        dist_params: Dict containing distribution-specific parameters (optional)
        
    Returns:
        density_func: Function that takes X and returns density values
    """
    from functools import partial
    
    if dist_params is None:
        dist_params = {}
    
    if distribution == "truncated_normal":
        params = dist_params.get("truncated_normal", {"loc": 0.0, "scale": 0.5})
        return partial(compute_truncated_normal_density, 
                      loc=params.get("loc", 0.0), 
                      scale=params.get("scale", 0.5))
    elif distribution == "beta":
        params = dist_params.get("beta", {"a": 2.0, "b": 5.0})
        return partial(compute_beta_density, 
                      a=params.get("a", 2.0), 
                      b=params.get("b", 5.0))
    elif distribution == "mixture":
        params = dist_params.get("mixture", {"centers": (-0.6, 0.6), "scales": (0.15, 0.15), "weights": (0.5, 0.5)})
        # Convert lists to tuples if needed (YAML loads as lists)
        centers = tuple(params.get("centers", (-0.6, 0.6)))
        scales = tuple(params.get("scales", (0.15, 0.15)))
        weights = tuple(params.get("weights", (0.5, 0.5)))
        return partial(compute_mixture_density, 
                      centers=centers, 
                      scales=scales, 
                      weights=weights)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


