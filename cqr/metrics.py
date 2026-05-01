"""
Evaluation metrics for conformal prediction intervals.

Provides marginal coverage, average width, and conditional coverage analysis.
"""

import numpy as np
from scipy.stats import pearsonr
from typing import Dict, Any, Optional


def marginal_coverage(
    y_test: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
) -> float:
    """
    Marginal (unconditional) coverage: fraction of test points covered.

    Coverage = (1/n) * Σ 1{y_i ∈ [lo_i, hi_i]}

    Args:
        y_test: True labels, shape (n,)
        pred_lo: Lower interval bounds, shape (n,)
        pred_hi: Upper interval bounds, shape (n,)

    Returns:
        Empirical coverage in [0, 1]
    """
    y = np.asarray(y_test).flatten()
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    return float(np.mean((y >= lo) & (y <= hi)))


def average_width(
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
) -> float:
    """
    Average prediction interval width.

    Width = (1/n) * Σ (hi_i - lo_i)

    Args:
        pred_lo: Lower interval bounds, shape (n,)
        pred_hi: Upper interval bounds, shape (n,)

    Returns:
        Mean interval width
    """
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    widths = hi - lo
    return float(np.mean(widths))


def median_width(
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
) -> float:
    """Median prediction interval width."""
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    return float(np.median(hi - lo))


def width_std(
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
) -> float:
    """Standard deviation of prediction interval widths."""
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    return float(np.std(hi - lo))


def conditional_coverage(
    y_test: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    n_bins: int = 5,
    min_bin_size: int = 20,
    method: str = "pca",
) -> Dict[str, Any]:
    """
    Conditional coverage analysis: compute coverage across feature-space bins.

    Bins test points by a 1D projection of X_test (PCA or first feature),
    then computes per-bin coverage.  Bins with fewer than `min_bin_size`
    points are excluded from the aggregation to avoid noisy estimates.

    The returned ``ranked_bins`` list is sorted by coverage ascending
    (hardest-to-cover bin first).

    Args:
        y_test: True labels, shape (n,)
        pred_lo: Lower interval bounds, shape (n,)
        pred_hi: Upper interval bounds, shape (n,)
        X_test: Test features, shape (n, d)
        alpha: Nominal miscoverage level (for gap computation)
        n_bins: Number of bins for conditional coverage
        min_bin_size: Minimum number of points per bin to be considered
        method: Binning method — 'pca' (first PC) or 'first_feature'

    Returns:
        Dict with:
            bin_coverages, bin_counts, bin_avg_widths, bin_projection_ranges:
                raw per-bin lists (length n_bins)
            ranked_bins: list of dicts sorted by coverage ascending, each with
                rank, bin_id, coverage, count, avg_width, proj_lo, proj_hi
            worst_bin_coverage, best_bin_coverage,
            coverage_gap_max, coverage_gap_mean, coverage_range
    """
    y = np.asarray(y_test).flatten()
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    X = np.asarray(X_test)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Project to 1D for binning
    if method == "pca" and X.shape[1] > 1:
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=1)
            projection = pca.fit_transform(X).flatten()
        except Exception:
            projection = X[:, 0]
    else:
        projection = X[:, 0]

    # Create bins using quantiles of the projection
    bin_edges = np.percentile(projection, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    covered = (y >= lo) & (y <= hi)
    widths = hi - lo
    target_coverage = 1 - alpha

    bin_coverages = []
    bin_counts = []
    bin_avg_widths = []
    bin_proj_ranges = []

    for b in range(n_bins):
        mask = (projection >= bin_edges[b]) & (projection < bin_edges[b + 1])
        count = int(np.sum(mask))
        bin_counts.append(count)
        bin_proj_ranges.append((float(bin_edges[b]), float(bin_edges[b + 1])))
        if count > 0:
            bin_coverages.append(float(np.mean(covered[mask])))
            bin_avg_widths.append(float(np.mean(widths[mask])))
        else:
            bin_coverages.append(np.nan)
            bin_avg_widths.append(np.nan)

    # Build ranked list (hardest first), filtering by min_bin_size
    ranked_bins = []
    for b in range(n_bins):
        if np.isnan(bin_coverages[b]) or bin_counts[b] < min_bin_size:
            continue
        ranked_bins.append({
            "bin_id": b,
            "coverage": bin_coverages[b],
            "count": bin_counts[b],
            "avg_width": bin_avg_widths[b],
            "proj_lo": bin_proj_ranges[b][0],
            "proj_hi": bin_proj_ranges[b][1],
        })
    # Sort by coverage ascending (hardest to cover first)
    ranked_bins.sort(key=lambda d: d["coverage"])
    for rank, entry in enumerate(ranked_bins):
        entry["rank"] = rank + 1  # 1-based

    # Aggregate statistics from valid bins
    valid_coverages = [e["coverage"] for e in ranked_bins]

    if len(valid_coverages) == 0:
        return {
            "bin_coverages": bin_coverages,
            "bin_counts": bin_counts,
            "bin_avg_widths": bin_avg_widths,
            "bin_projection_ranges": bin_proj_ranges,
            "ranked_bins": [],
            "worst_bin_coverage": np.nan,
            "best_bin_coverage": np.nan,
            "coverage_gap_max": np.nan,
            "coverage_gap_mean": np.nan,
            "coverage_range": np.nan,
        }

    worst = min(valid_coverages)
    best = max(valid_coverages)
    abs_gaps = [abs(c - target_coverage) for c in valid_coverages]
    gap_max = max(abs_gaps)
    gap_mean = float(np.mean(abs_gaps))
    cov_range = best - worst

    return {
        "bin_coverages": bin_coverages,
        "bin_counts": bin_counts,
        "bin_avg_widths": bin_avg_widths,
        "bin_projection_ranges": bin_proj_ranges,
        "ranked_bins": ranked_bins,
        "worst_bin_coverage": worst,
        "best_bin_coverage": best,
        "coverage_gap_max": gap_max,
        "coverage_gap_mean": gap_mean,
        "coverage_range": cov_range,
    }


def winkler_score(
    y_test: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    alpha: float = 0.1,
) -> float:
    """
    Mean Winkler (interval) score.

    For each point: width + (2/alpha)*(lo - y)*I(y < lo) + (2/alpha)*(y - hi)*I(y > hi)

    Args:
        y_test: True labels, shape (n,)
        pred_lo: Lower interval bounds, shape (n,)
        pred_hi: Upper interval bounds, shape (n,)
        alpha: Nominal miscoverage level

    Returns:
        Mean Winkler score (lower is better)
    """
    y = np.asarray(y_test).flatten()
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    widths = hi - lo
    penalty = (2.0 / alpha) * (np.maximum(lo - y, 0.0) + np.maximum(y - hi, 0.0))
    return float(np.mean(widths + penalty))


def width_error_correlation(
    y_test: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
) -> float:
    """
    Pearson correlation between interval width and absolute prediction error.

    Measures whether intervals expand where the model makes larger errors.
    The prediction error is computed relative to the interval midpoint
    (lo + hi) / 2, which serves as an implicit point forecast.

    Args:
        y_test: True labels, shape (n,)
        pred_lo: Lower interval bounds, shape (n,)
        pred_hi: Upper interval bounds, shape (n,)

    Returns:
        Pearson r in [-1, 1], or np.nan if width or error is constant.
    """
    y = np.asarray(y_test).flatten()
    lo = np.asarray(pred_lo).flatten()
    hi = np.asarray(pred_hi).flatten()
    widths = hi - lo
    abs_error = np.abs(y - (lo + hi) / 2.0)
    if np.std(widths) < 1e-12 or np.std(abs_error) < 1e-12:
        return np.nan
    r, _ = pearsonr(widths, abs_error)
    return float(r)


def evaluate_intervals(
    y_test: np.ndarray,
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 0.1,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics for a set of prediction intervals.

    Args:
        y_test, pred_lo, pred_hi, X_test: As in other functions
        alpha: Nominal miscoverage level
        n_bins: Number of bins for conditional coverage

    Returns:
        Dict with all metrics:
            coverage, avg_width, median_width, width_std,
            worst_bin_cov, best_bin_cov, coverage_gap (mean),
            coverage_gap_max, coverage_range, bin_coverages, bin_counts,
            winkler_score, width_error_corr, ccv
    """
    cov = marginal_coverage(y_test, pred_lo, pred_hi)
    avg_w = average_width(pred_lo, pred_hi)
    med_w = median_width(pred_lo, pred_hi)
    w_std = width_std(pred_lo, pred_hi)
    cond = conditional_coverage(y_test, pred_lo, pred_hi, X_test, alpha=alpha, n_bins=n_bins)

    # Winkler Score
    w_score = winkler_score(y_test, pred_lo, pred_hi, alpha=alpha)

    # Width-Error Correlation
    w_err_corr = width_error_correlation(y_test, pred_lo, pred_hi)

    # CCV: mean absolute deviation of bin coverages from nominal.
    # Uses ranked_bins (filtered by min_bin_size) for consistency with
    # coverage_gap_mean — avoids noisy estimates from near-empty bins.
    ranked_covs = np.array([b["coverage"] for b in cond["ranked_bins"]], dtype=float)
    if len(ranked_covs) > 0:
        ccv = float(np.mean(np.abs(ranked_covs - (1.0 - alpha))))
    else:
        ccv = np.nan

    return {
        "coverage": cov,
        "avg_width": avg_w,
        "median_width": med_w,
        "width_std": w_std,
        "worst_bin_cov": cond["worst_bin_coverage"],
        "best_bin_cov": cond["best_bin_coverage"],
        "coverage_gap": cond["coverage_gap_mean"],
        "coverage_gap_max": cond["coverage_gap_max"],
        "coverage_range": cond["coverage_range"],
        "bin_coverages": cond["bin_coverages"],
        "bin_counts": cond["bin_counts"],
        "ranked_bins": cond["ranked_bins"],
        "winkler_score": w_score,
        "width_error_corr": w_err_corr,
        "ccv": ccv,
    }
