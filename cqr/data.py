"""
Data generation utilities with multi-dimensional support.

Supports dimensions d ∈ {1, 2, 3, 4, ...} using norm-based heteroscedasticity.

CRITICAL: All generators use the SAME ground truth function (get_ground_truth)
to ensure valid comparison across different X distributions.
"""

import numpy as np
from scipy.stats import truncnorm, norm, chi2 as chi2_dist, t as t_dist
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


# =============================================================================
# GUAN (2021) EXAMPLE 4.1  +  EXTENDED STRESS-TEST SETTINGS
# =============================================================================

_GUAN_SETTINGS = {
    # ----- Original Guan (2021), Figure 2 -----
    "A": {"rho": lambda x: np.sin(x),            "noise": "gaussian", "mu": None},
    "B": {"rho": lambda x: np.cos(x),            "noise": "gaussian", "mu": None},
    "C": {"rho": lambda x: np.sqrt(np.abs(x)),   "noise": "gaussian", "mu": None},
    "D": {"rho": lambda x: np.ones_like(x),      "noise": "gaussian", "mu": None},
    # ----- Smoothness / regularity stress -----
    "E": {"rho": lambda x: np.abs(x),                            "noise": "gaussian", "mu": None},
    "F": {"rho": lambda x: 1.0 + 0.5 * np.sign(x),              "noise": "gaussian", "mu": None},
    "G": {"rho": lambda x: np.exp(x / 2),                        "noise": "gaussian", "mu": None},
    # ----- Multi-scale / oscillation -----
    "H": {"rho": lambda x: 1.0 + np.sin(2 * np.pi * x) ** 2,   "noise": "gaussian", "mu": None},
    "I": {"rho": lambda x: 1.0 + 0.3 * np.sin(5 * x),           "noise": "gaussian", "mu": None},
    # ----- Asymmetric / non-Gaussian conditional -----
    "J": {"rho": lambda x: np.sqrt(np.abs(x)),  "noise": "chi2_centered", "mu": None},
    "K": {
        "rho": lambda x: np.sqrt(np.abs(x)),
        "noise": "student_t_3",
        "mu":   lambda x: np.sin(np.pi * x),
    },
    # ----- Mixture / regime switching (Guan intro example, rescaled to N(0,1)) -----
    "L": {
        "rho": lambda x: np.where(
            np.abs(x) < 1.5,
            np.abs(np.cos(x)) + 0.1,
            np.ones_like(x),
        ),
        "noise": "gaussian",
        "mu": None,
    },
}


def guan2021_oracle(
    setting: str,
    x: np.ndarray,
    alpha: float = 0.05,
) -> tuple:
    """Compute oracle prediction bands for a Guan (2021) setting on array x.

    Dispatches over noise type:
        gaussian      : Y = \u03bc(x) + \u03c1(x)\u00b7\u03b5,  \u03b5 ~ N(0,1)
        chi2_centered : Y = \u03bc(x) + \u03c1(x)\u00b7(\u03c7\u00b2\u2081 \u2212 1)
        student_t_3   : Y = \u03bc(x) + \u03c1(x)\u00b7t\u2083

    Args:
        setting: One of 'A'\u2013'L'.
        x: Input array, shape (n,).
        alpha: Miscoverage level (coverage = 1 \u2212 alpha).

    Returns:
        (oracle_lo, oracle_hi): Each of shape (n,).
    """
    setting = setting.upper()
    if setting not in _GUAN_SETTINGS:
        raise ValueError(
            f"Unknown setting '{setting}'. Choose from {list(_GUAN_SETTINGS)}.")

    cfg   = _GUAN_SETTINGS[setting]
    rho_x = cfg["rho"](x)
    mu_x  = cfg["mu"](x) if cfg["mu"] is not None else np.zeros_like(x, dtype=float)
    noise = cfg["noise"]

    if noise == "gaussian":
        z = norm.ppf(1.0 - alpha / 2)
        oracle_lo = mu_x - z * np.abs(rho_x)
        oracle_hi = mu_x + z * np.abs(rho_x)
    elif noise == "chi2_centered":
        # Y = \u03c1(x)\u00b7(\u03c7\u00b2\u2081 \u2212 1);  \u03c1 \u2265 0 (sqrt(|x|))
        q_lo = chi2_dist.ppf(alpha / 2,       1) - 1.0
        q_hi = chi2_dist.ppf(1.0 - alpha / 2, 1) - 1.0
        oracle_lo = mu_x + rho_x * q_lo
        oracle_hi = mu_x + rho_x * q_hi
    elif noise == "student_t_3":
        # t\u2083 is symmetric;  \u03c1 \u2265 0 (sqrt(|x|))
        qt = t_dist.ppf(1.0 - alpha / 2, df=3)
        oracle_lo = mu_x - np.abs(rho_x) * qt
        oracle_hi = mu_x + np.abs(rho_x) * qt
    else:
        raise ValueError(f"Unknown noise type '{noise}'.")

    return oracle_lo, oracle_hi


def generate_guan2021(
    setting: str,
    n_train: int = 1000,
    n_cal: int = 1000,
    n_test: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Generate synthetic data for Guan (2021) Example 4.1 and extended settings.

    Model:
        X ~ N(0, 1)
        Y = \u03bc(X) + \u03c1(X) \u00b7 \u03b5,   \u03b5 \u22a5 X

    Noise types by setting:
        gaussian      (A\u2013I, L): \u03b5 ~ N(0, 1)
        chi2_centered (J):      \u03b5 ~ \u03c7\u00b2\u2081 \u2212 1  (zero-mean, right-skewed)
        student_t_3   (K):      \u03b5 ~ t\u2083        (symmetric, heavy-tailed)

    Settings:
        A: \u03c1(x) = sin(x)
        B: \u03c1(x) = cos(x)
        C: \u03c1(x) = sqrt(|x|)
        D: \u03c1(x) = 1  (homoscedastic)
        E: \u03c1(x) = |x|                          \u2014 kink at 0
        F: \u03c1(x) = 1 + 0.5\u00b7sign(x)             \u2014 jump discontinuity
        G: \u03c1(x) = exp(x/2)                     \u2014 exponential scale growth
        H: \u03c1(x) = 1 + sin(2\u03c0x)\u00b2             \u2014 bounded oscillation
        I: \u03c1(x) = 1 + 0.3\u00b7sin(5x)             \u2014 higher-frequency oscillation
        J: \u03c1(x) = sqrt(|x|), \u03b5 ~ \u03c7\u00b2\u2081\u22121       \u2014 skewed conditional
        K: \u03c1(x) = sqrt(|x|), \u03bc(x)=sin(\u03c0x), \u03b5 ~ t\u2083 \u2014 mean trend + heavy tails
        L: \u03c1(x) = |cos(x)|+0.1 if |x|<1.5 else 1  \u2014 regime switch

    Args:
        setting: One of 'A'\u2013'L'.
        n_train: Training set size.
        n_cal: Calibration set size.
        n_test: Test set size.
        seed: RNG seed (single np.random.default_rng).
        alpha: Miscoverage level; oracle bands at coverage 1\u2212alpha.

    Returns:
        dict with keys:
            X_train, Y_train : shape (n_train,)
            X_cal,   Y_cal   : shape (n_cal,)
            X_test,  Y_test  : shape (n_test,)
            oracle_lo, oracle_hi : oracle bands on X_test, shape (n_test,)
    """
    setting = setting.upper()
    if setting not in _GUAN_SETTINGS:
        raise ValueError(
            f"Unknown setting '{setting}'. Choose from {list(_GUAN_SETTINGS)}.")

    cfg   = _GUAN_SETTINGS[setting]
    rho   = cfg["rho"]
    mu_fn = cfg["mu"]
    noise = cfg["noise"]

    rng     = np.random.default_rng(seed)
    n_total = n_train + n_cal + n_test

    X_all  = rng.standard_normal(n_total)
    mu_all = mu_fn(X_all) if mu_fn is not None else np.zeros(n_total)

    if noise == "gaussian":
        eps = rng.standard_normal(n_total)
    elif noise == "chi2_centered":
        eps = rng.chisquare(1, n_total) - 1.0
    elif noise == "student_t_3":
        eps = rng.standard_t(3, n_total)
    else:
        raise ValueError(f"Unknown noise type '{noise}'.")

    Y_all = mu_all + rho(X_all) * eps

    X_train = X_all[:n_train]
    Y_train = Y_all[:n_train]
    X_cal   = X_all[n_train : n_train + n_cal]
    Y_cal   = Y_all[n_train : n_train + n_cal]
    X_test  = X_all[n_train + n_cal :]
    Y_test  = Y_all[n_train + n_cal :]

    oracle_lo, oracle_hi = guan2021_oracle(setting, X_test, alpha)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_cal":   X_cal,
        "Y_cal":   Y_cal,
        "X_test":  X_test,
        "Y_test":  Y_test,
        "oracle_lo": oracle_lo,
        "oracle_hi": oracle_hi,
    }


