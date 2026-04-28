"""Synthetic 1D regression regimes for evaluating quantile-regression / CQR methods.

Each regime defines an ``X`` distribution, a conditional mean ``mu(x)``, a scale
``sigma(x)``, and a noise distribution ``epsilon``. The single public entry
point :func:`sample` returns the sampled ``(X, Y)`` together with the *oracle*
conditional ``alpha/2`` and ``1 - alpha/2`` quantiles evaluated at the sampled
``X``.

For most regimes ``Y = mu(X) + sigma(X) * epsilon`` with ``epsilon`` independent
of ``X``, so the oracle is ``mu(x) + sigma(x) * F_eps^{-1}(p)``. A few regimes
deviate (E5 branches the noise on ``x``; H1/H2 contaminate the data) — for the
contamination regimes the oracle is the *clean* conditional quantile, by design,
since the contamination is exactly the deviation that downstream methods must
absorb.

CLI: ``python synthetic_regimes.py --regime B2 --n 2000 --alpha 0.05 --plot``
saves a diagnostic scatter + oracle band to ``regime_B2.png`` (override with
``--out``). Without ``--regime``, the module runs a smoke test over every
registered regime.

Dependencies: numpy, scipy.stats. Matplotlib is imported only inside ``_plot``.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
from typing import Callable, Dict, Tuple

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# X-distribution and noise helpers                                            #
# --------------------------------------------------------------------------- #

def _default_mu(x: np.ndarray) -> np.ndarray:
    return 2.0 * np.sin(3.0 * x)


def _uniform_x(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.uniform(-1.0, 1.0, size=n)


def _truncnorm_x(
    rng: np.random.Generator,
    n: int,
    mean: float = 0.0,
    sd: float = 0.35,
    lo: float = -1.0,
    hi: float = 1.0,
) -> np.ndarray:
    a, b = (lo - mean) / sd, (hi - mean) / sd
    return stats.truncnorm.rvs(a, b, loc=mean, scale=sd, size=n, random_state=rng)


def _bimodal_pm06(
    rng: np.random.Generator, n: int, sd: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """Bimodal mixture at +/-0.6, reject-resampled to [-1, 1]. Returns (X, cluster_id)."""
    cluster = rng.integers(0, 2, size=n)
    centers = np.where(cluster == 0, -0.6, 0.6)
    x = centers + sd * rng.standard_normal(n)
    bad = (x < -1.0) | (x > 1.0)
    while bad.any():
        n_bad = int(bad.sum())
        x[bad] = centers[bad] + sd * rng.standard_normal(n_bad)
        bad = (x < -1.0) | (x > 1.0)
    return x, cluster


def _three_cluster(
    rng: np.random.Generator,
    n: int,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    centers: Tuple[float, float, float] = (-0.6, 0.0, 0.6),
    sd: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray]:
    """3-component MoG. Returns (X, cluster_id in {0,1,2})."""
    cluster = rng.choice(len(weights), size=n, p=np.asarray(weights))
    centers_arr = np.asarray(centers)[cluster]
    x = centers_arr + sd * rng.standard_normal(n)
    return x, cluster


def _beta_x(
    rng: np.random.Generator, n: int, a: float = 0.5, b: float = 0.5
) -> np.ndarray:
    return 2.0 * rng.beta(a, b, size=n) - 1.0


def _truncated_t(
    rng: np.random.Generator, n: int, df: int = 3, bound: float = 3.0
) -> np.ndarray:
    """Sample t_df, rejection-truncated to [-bound, bound]."""
    out = np.empty(n)
    filled = 0
    while filled < n:
        cand = stats.t.rvs(df=df, size=n, random_state=rng)
        ok = np.abs(cand) <= bound
        n_ok = int(ok.sum())
        take = min(n_ok, n - filled)
        out[filled : filled + take] = cand[ok][:take]
        filled += take
    return out


# Mixture-of-Gaussians noise quantile (E4). Cached per (components, p).
@lru_cache(maxsize=64)
def _mog_ppf_scalar(components: Tuple[Tuple[float, float, float], ...], p: float) -> float:
    grid = np.linspace(-8.0, 8.0, 40001)
    cdf = np.zeros_like(grid)
    for w, m, s in components:
        cdf += w * stats.norm.cdf(grid, loc=m, scale=s)
    return float(np.interp(p, cdf, grid))


# --------------------------------------------------------------------------- #
# Regime registry                                                             #
# --------------------------------------------------------------------------- #

RegimeFn = Callable[
    [np.random.Generator, int, float, float],
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict],
]
_REGIMES: Dict[str, RegimeFn] = {}


def regime(name: str) -> Callable[[RegimeFn], RegimeFn]:
    def decorator(fn: RegimeFn) -> RegimeFn:
        _REGIMES[name] = fn
        return fn

    return decorator


def _gauss_oracle(
    mu: np.ndarray, sigma: np.ndarray, p_lo: float, p_hi: float
) -> Tuple[np.ndarray, np.ndarray]:
    z_lo = stats.norm.ppf(p_lo)
    z_hi = stats.norm.ppf(p_hi)
    return mu + sigma * z_lo, mu + sigma * z_hi


# --------------------------------------------------------------------------- #
# A series — baseline                                                         #
# --------------------------------------------------------------------------- #

@regime("A1")
def _a1(rng, n, p_lo, p_hi):
    """A1: defaults — X~U[-1,1], mu=2sin(3x), sigma=1, eps~N(0,1)."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = np.ones(n)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(x_dist="U[-1,1]", mu_desc="2 sin(3x)", sigma_desc="1", noise_desc="N(0,1)")
    return x, y, mu, sigma, lo, hi, meta


# --------------------------------------------------------------------------- #
# B series — heteroscedastic shapes                                           #
# --------------------------------------------------------------------------- #

def _b_template(rng, n, sigma_fn, sigma_desc, p_lo, p_hi):
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = sigma_fn(x)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(x_dist="U[-1,1]", mu_desc="2 sin(3x)", sigma_desc=sigma_desc, noise_desc="N(0,1)")
    return x, y, mu, sigma, lo, hi, meta


@regime("B1")
def _b1(rng, n, p_lo, p_hi):
    """B1: piecewise sigma {0.3, 2.0, 0.5} on |x|<0.35, 0.35<=|x|<0.7, |x|>=0.7."""
    def sigma_fn(x):
        ax = np.abs(x)
        out = np.full_like(x, 0.5)
        out[ax < 0.7] = 2.0
        out[ax < 0.35] = 0.3
        return out
    return _b_template(rng, n, sigma_fn, "piecewise(|x|; 0.3,2.0,0.5)", p_lo, p_hi)


@regime("B2")
def _b2(rng, n, p_lo, p_hi):
    """B2: sigma(x) = 0.2 + 3 x^2."""
    return _b_template(rng, n, lambda x: 0.2 + 3.0 * x ** 2, "0.2 + 3 x^2", p_lo, p_hi)


@regime("B3")
def _b3(rng, n, p_lo, p_hi):
    """B3: sigma(x) = 1 + 0.8 sin(2 pi x)."""
    return _b_template(
        rng, n, lambda x: 1.0 + 0.8 * np.sin(2.0 * np.pi * x), "1 + 0.8 sin(2 pi x)", p_lo, p_hi
    )


@regime("B4")
def _b4(rng, n, p_lo, p_hi):
    """B4: sigma(x) = 0.3 exp(1.5 x)."""
    return _b_template(
        rng, n, lambda x: 0.3 * np.exp(1.5 * x), "0.3 exp(1.5 x)", p_lo, p_hi
    )


@regime("B5")
def _b5(rng, n, p_lo, p_hi):
    """B5: sigma(x) = 0.2 + 2 exp(-5 x^2)."""
    return _b_template(
        rng, n, lambda x: 0.2 + 2.0 * np.exp(-5.0 * x ** 2), "0.2 + 2 exp(-5 x^2)", p_lo, p_hi
    )


@regime("B6")
def _b6(rng, n, p_lo, p_hi):
    """B6: sigma(x) = 2 - 1.7 exp(-5 x^2)."""
    return _b_template(
        rng, n, lambda x: 2.0 - 1.7 * np.exp(-5.0 * x ** 2), "2 - 1.7 exp(-5 x^2)", p_lo, p_hi
    )


# --------------------------------------------------------------------------- #
# C series — X-distribution coverage with sigma = 0.5 + |x|                    #
# --------------------------------------------------------------------------- #

def _c_template(rng, n, x, x_dist, p_lo, p_hi):
    mu = _default_mu(x)
    sigma = 0.5 + np.abs(x)
    y = mu + sigma * rng.standard_normal(len(x))
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(x_dist=x_dist, mu_desc="2 sin(3x)", sigma_desc="0.5 + |x|", noise_desc="N(0,1)")
    return x, y, mu, sigma, lo, hi, meta


@regime("C1")
def _c1(rng, n, p_lo, p_hi):
    """C1: X ~ TruncNormal(0, 0.35) on [-1, 1], sigma = 0.5 + |x|."""
    x = _truncnorm_x(rng, n)
    return _c_template(rng, n, x, "TruncN(0, 0.35) on [-1,1]", p_lo, p_hi)


@regime("C2")
def _c2(rng, n, p_lo, p_hi):
    """C2: X ~ 0.5 N(-0.6, 0.1^2) + 0.5 N(0.6, 0.1^2), reject-resampled to [-1,1]."""
    x, _ = _bimodal_pm06(rng, n, sd=0.1)
    return _c_template(rng, n, x, "MoG +/-0.6 (sd 0.1), trunc [-1,1]", p_lo, p_hi)


@regime("C3")
def _c3(rng, n, p_lo, p_hi):
    """C3: X ~ Beta(0.5, 0.5) rescaled to [-1, 1]."""
    x = _beta_x(rng, n)
    return _c_template(rng, n, x, "2*Beta(0.5,0.5) - 1", p_lo, p_hi)


@regime("C4")
def _c4(rng, n, p_lo, p_hi):
    """C4: X ~ t_3 truncated to [-3, 3]."""
    x = _truncated_t(rng, n, df=3, bound=3.0)
    return _c_template(rng, n, x, "t_3 truncated to [-3,3]", p_lo, p_hi)


@regime("C5")
def _c5(rng, n, p_lo, p_hi):
    """C5: X ~ 0.4 N(-0.6, 0.08^2) + 0.3 N(0, 0.08^2) + 0.3 N(0.6, 0.08^2)."""
    x, _ = _three_cluster(rng, n)
    return _c_template(rng, n, x, "3-cluster MoG (-0.6, 0, 0.6; sd 0.08)", p_lo, p_hi)


# --------------------------------------------------------------------------- #
# D series — density / sigma interaction                                       #
# --------------------------------------------------------------------------- #

@regime("D1")
def _d1(rng, n, p_lo, p_hi):
    """D1: X bimodal at +/-0.6 (as C2), sigma(x) = 0.2 + 3 exp(-5 x^2) (mass at low sigma)."""
    x, _ = _bimodal_pm06(rng, n, sd=0.1)
    mu = _default_mu(x)
    sigma = 0.2 + 3.0 * np.exp(-5.0 * x ** 2)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="MoG +/-0.6 (sd 0.1), trunc [-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.2 + 3 exp(-5 x^2)",
        noise_desc="N(0,1)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("D2")
def _d2(rng, n, p_lo, p_hi):
    """D2: X TruncN(0, 0.35), sigma(x) = 0.2 + 3 exp(-5 x^2) (mass at high sigma)."""
    x = _truncnorm_x(rng, n)
    mu = _default_mu(x)
    sigma = 0.2 + 3.0 * np.exp(-5.0 * x ** 2)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="TruncN(0, 0.35) on [-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.2 + 3 exp(-5 x^2)",
        noise_desc="N(0,1)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("D3")
def _d3(rng, n, p_lo, p_hi):
    """D3: X as C5, mu(x) = 3x, per-cluster sigma in {4.0, 0.15, 0.8} (cluster -0.6, 0, 0.6)."""
    x, cluster = _three_cluster(rng, n)
    sigmas_per_cluster = np.array([4.0, 0.15, 0.8])
    sigma = sigmas_per_cluster[cluster]
    mu = 3.0 * x
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="3-cluster MoG (-0.6, 0, 0.6; sd 0.08)",
        mu_desc="3x",
        sigma_desc="per-cluster {4.0, 0.15, 0.8}",
        noise_desc="N(0,1)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("D4")
def _d4(rng, n, p_lo, p_hi):
    """D4: X ~ U, sigma(x) = 1 + 0.5 sin(7x) (sigma frequency unrelated to density)."""
    return _b_template(
        rng, n, lambda xv: 1.0 + 0.5 * np.sin(7.0 * xv), "1 + 0.5 sin(7x)", p_lo, p_hi
    )


# --------------------------------------------------------------------------- #
# E series — non-Gaussian noise                                                #
# --------------------------------------------------------------------------- #

@regime("E1")
def _e1(rng, n, p_lo, p_hi):
    """E1: sigma=1, eps ~ t_3."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = np.ones(n)
    eps = stats.t.rvs(df=3, size=n, random_state=rng)
    y = mu + sigma * eps
    q_lo = stats.t.ppf(p_lo, df=3)
    q_hi = stats.t.ppf(p_hi, df=3)
    lo, hi = mu + sigma * q_lo, mu + sigma * q_hi
    meta = dict(x_dist="U[-1,1]", mu_desc="2 sin(3x)", sigma_desc="1", noise_desc="t_3")
    return x, y, mu, sigma, lo, hi, meta


@regime("E2")
def _e2(rng, n, p_lo, p_hi):
    """E2: defaults shape, eps = centered LogNormal(0, 0.5) (subtract mean exp(0.5^2/2))."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = np.ones(n)
    s = 0.5
    mean_ln = float(np.exp(s ** 2 / 2.0))
    eps = stats.lognorm.rvs(s=s, size=n, random_state=rng) - mean_ln
    y = mu + sigma * eps
    q_lo = stats.lognorm.ppf(p_lo, s=s) - mean_ln
    q_hi = stats.lognorm.ppf(p_hi, s=s) - mean_ln
    lo, hi = mu + sigma * q_lo, mu + sigma * q_hi
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="1",
        noise_desc="centered LogN(0, 0.5)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("E3")
def _e3(rng, n, p_lo, p_hi):
    """E3: sigma = 0.5 + |x|, eps ~ raw skew-normal with shape alpha(x) = 5*sign(x)."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = 0.5 + np.abs(x)
    shape = 5.0 * np.sign(x)
    eps = stats.skewnorm.rvs(a=shape, size=n, random_state=rng)
    y = mu + sigma * eps
    q_lo = stats.skewnorm.ppf(p_lo, a=shape)
    q_hi = stats.skewnorm.ppf(p_hi, a=shape)
    lo, hi = mu + sigma * q_lo, mu + sigma * q_hi
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.5 + |x|",
        noise_desc="skew-normal alpha=5*sign(x), uncentered",
    )
    return x, y, mu, sigma, lo, hi, meta


_E4_COMPONENTS: Tuple[Tuple[float, float, float], ...] = (
    (0.5, -1.0, 0.3),
    (0.5, 1.0, 0.3),
)


@regime("E4")
def _e4(rng, n, p_lo, p_hi):
    """E4: sigma=1, eps ~ 0.5 N(-1, 0.3^2) + 0.5 N(1, 0.3^2)."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = np.ones(n)
    cluster = rng.integers(0, 2, size=n)
    means = np.where(cluster == 0, -1.0, 1.0)
    eps = means + 0.3 * rng.standard_normal(n)
    y = mu + sigma * eps
    q_lo = _mog_ppf_scalar(_E4_COMPONENTS, p_lo)
    q_hi = _mog_ppf_scalar(_E4_COMPONENTS, p_hi)
    lo, hi = mu + sigma * q_lo, mu + sigma * q_hi
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="1",
        noise_desc="0.5 N(-1, 0.3^2) + 0.5 N(1, 0.3^2)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("E5")
def _e5(rng, n, p_lo, p_hi):
    """E5: sigma as B2 (0.2 + 3 x^2); eps ~ t_3 if |x|>0.5 else N(0,1)."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = 0.2 + 3.0 * x ** 2
    heavy = np.abs(x) > 0.5
    eps = np.empty(n)
    n_heavy = int(heavy.sum())
    n_light = n - n_heavy
    if n_heavy:
        eps[heavy] = stats.t.rvs(df=3, size=n_heavy, random_state=rng)
    if n_light:
        eps[~heavy] = rng.standard_normal(n_light)
    y = mu + sigma * eps
    q_lo = np.where(heavy, stats.t.ppf(p_lo, df=3), stats.norm.ppf(p_lo))
    q_hi = np.where(heavy, stats.t.ppf(p_hi, df=3), stats.norm.ppf(p_hi))
    lo, hi = mu + sigma * q_lo, mu + sigma * q_hi
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.2 + 3 x^2",
        noise_desc="t_3 if |x|>0.5 else N(0,1)",
    )
    return x, y, mu, sigma, lo, hi, meta


# --------------------------------------------------------------------------- #
# F series — mean-shape baselines                                              #
# --------------------------------------------------------------------------- #

@regime("F1")
def _f1(rng, n, p_lo, p_hi):
    """F1: sigma=0.5 (Holder-smooth mu, homoscedastic baseline)."""
    return _b_template(rng, n, lambda x: np.full_like(x, 0.5), "0.5", p_lo, p_hi)


@regime("F2")
def _f2(rng, n, p_lo, p_hi):
    """F2: mu(x) = 1.5 sign(x) + sin(3x), sigma=0.5."""
    x = _uniform_x(rng, n)
    mu = 1.5 * np.sign(x) + np.sin(3.0 * x)
    sigma = np.full(n, 0.5)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="1.5 sign(x) + sin(3x)",
        sigma_desc="0.5",
        noise_desc="N(0,1)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("F3")
def _f3(rng, n, p_lo, p_hi):
    """F3: mu(x) = sin(20x), sigma=0.3."""
    x = _uniform_x(rng, n)
    mu = np.sin(20.0 * x)
    sigma = np.full(n, 0.3)
    y = mu + sigma * rng.standard_normal(n)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="U[-1,1]", mu_desc="sin(20x)", sigma_desc="0.3", noise_desc="N(0,1)"
    )
    return x, y, mu, sigma, lo, hi, meta


# --------------------------------------------------------------------------- #
# H series — contamination (oracle = clean quantile)                          #
# --------------------------------------------------------------------------- #

@regime("H1")
def _h1(rng, n, p_lo, p_hi):
    """H1: B2 with prob 0.05 of replacing eps by N(0, 25). Oracle = clean B2 quantile."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = 0.2 + 3.0 * x ** 2
    eps_clean = rng.standard_normal(n)
    contam_mask = rng.random(n) < 0.05
    eps_contam = 5.0 * rng.standard_normal(n)  # N(0, 25) -> sd 5
    eps = np.where(contam_mask, eps_contam, eps_clean)
    y = mu + sigma * eps
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.2 + 3 x^2",
        noise_desc="N(0,1) contam 5%->N(0,25)",
        oracle_basis="clean Gaussian (B2)",
    )
    return x, y, mu, sigma, lo, hi, meta


@regime("H2")
def _h2(rng, n, p_lo, p_hi):
    """H2: B2 with prob 0.05 of replacing Y by U[-10, 10]. Oracle = clean B2 quantile."""
    x = _uniform_x(rng, n)
    mu = _default_mu(x)
    sigma = 0.2 + 3.0 * x ** 2
    y_clean = mu + sigma * rng.standard_normal(n)
    contam_mask = rng.random(n) < 0.05
    y_contam = rng.uniform(-10.0, 10.0, size=n)
    y = np.where(contam_mask, y_contam, y_clean)
    lo, hi = _gauss_oracle(mu, sigma, p_lo, p_hi)
    meta = dict(
        x_dist="U[-1,1]",
        mu_desc="2 sin(3x)",
        sigma_desc="0.2 + 3 x^2",
        noise_desc="N(0,1) contam 5%->Y~U[-10,10]",
        oracle_basis="clean Gaussian (B2)",
    )
    return x, y, mu, sigma, lo, hi, meta


# --------------------------------------------------------------------------- #
# Public API                                                                  #
# --------------------------------------------------------------------------- #

def sample(regime: str, n: int, alpha: float = 0.05, seed: int = 0) -> dict:
    """Sample ``n`` points from the named regime.

    Parameters
    ----------
    regime : str
        Regime id (e.g. ``"A1"``, ``"B2"``, ``"H1"``). See module docstring.
    n : int
        Sample size.
    alpha : float
        Miscoverage level. Oracle quantiles are at ``alpha/2`` and ``1 - alpha/2``.
    seed : int
        Seed for ``numpy.random.default_rng``.

    Returns
    -------
    dict with keys ``X``, ``Y``, ``mu``, ``sigma``, ``oracle_lo``, ``oracle_hi``,
    ``meta``. All array values have shape ``(n,)`` and dtype ``float64``.
    """
    if regime not in _REGIMES:
        raise ValueError(
            f"Unknown regime {regime!r}. Available: {sorted(_REGIMES.keys())}"
        )
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = np.random.default_rng(seed)
    p_lo, p_hi = alpha / 2.0, 1.0 - alpha / 2.0
    x, y, mu, sigma, lo, hi, meta = _REGIMES[regime](rng, int(n), p_lo, p_hi)
    meta = {"regime": regime, "alpha": alpha, "seed": seed, **meta}
    return dict(
        X=np.asarray(x, dtype=np.float64),
        Y=np.asarray(y, dtype=np.float64),
        mu=np.asarray(mu, dtype=np.float64),
        sigma=np.asarray(sigma, dtype=np.float64),
        oracle_lo=np.asarray(lo, dtype=np.float64),
        oracle_hi=np.asarray(hi, dtype=np.float64),
        meta=meta,
    )


def list_regimes() -> list[str]:
    return sorted(_REGIMES.keys())


# --------------------------------------------------------------------------- #
# Plotting + CLI                                                              #
# --------------------------------------------------------------------------- #

def _plot(out_path: str, result: dict) -> None:
    import matplotlib.pyplot as plt

    x = result["X"]
    y = result["Y"]
    order = np.argsort(x)
    xs = x[order]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.scatter(x, y, s=6, alpha=0.35, color="#1f77b4", label="samples")
    ax.plot(xs, result["mu"][order], color="black", lw=1.5, label="mu(x)")
    ax.plot(xs, result["oracle_lo"][order], color="crimson", lw=1.5, label="oracle low")
    ax.plot(xs, result["oracle_hi"][order], color="crimson", lw=1.5, label="oracle high")
    ax.fill_between(
        xs,
        result["oracle_lo"][order],
        result["oracle_hi"][order],
        color="crimson",
        alpha=0.08,
    )
    meta = result["meta"]
    ax.set_title(
        f"Regime {meta['regime']} — alpha={meta['alpha']}  "
        f"(sigma: {meta['sigma_desc']}, noise: {meta['noise_desc']})"
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _smoke_test() -> int:
    failures: list[str] = []
    for name in list_regimes():
        try:
            r = sample(name, n=500, alpha=0.05, seed=0)
            for key in ("X", "Y", "mu", "sigma", "oracle_lo", "oracle_hi"):
                assert r[key].shape == (500,), f"{name}: {key} shape {r[key].shape}"
            assert np.all(r["oracle_lo"] <= r["oracle_hi"]), f"{name}: oracle ordering violated"
            assert np.all(np.isfinite(r["X"])) and np.all(np.isfinite(r["Y"])), f"{name}: non-finite"
            print(f"OK {name}")
        except Exception as exc:  # surface but keep going
            failures.append(f"{name}: {exc}")
            print(f"FAIL {name}: {exc}")
    if failures:
        print(f"\n{len(failures)} regime(s) failed.")
        return 1
    print(f"\nAll {len(list_regimes())} regimes passed.")
    return 0


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Synthetic 1D regression regimes.")
    p.add_argument("--regime", type=str, default=None, help="Regime id (omit to run smoke test).")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--plot", action="store_true", help="Save diagnostic plot.")
    p.add_argument("--out", type=str, default=None, help="Plot output path.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    if args.regime is None:
        return _smoke_test()
    result = sample(args.regime, n=args.n, alpha=args.alpha, seed=args.seed)
    if args.plot:
        out = args.out or f"regime_{args.regime}.png"
        _plot(out, result)
        print(f"Wrote {out}")
    else:
        print(
            f"Regime {args.regime}: n={args.n}, alpha={args.alpha}, "
            f"oracle_width mean={float(np.mean(result['oracle_hi'] - result['oracle_lo'])):.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
