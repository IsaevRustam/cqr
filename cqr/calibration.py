"""
Conformal calibration methods: Global and Localized (Kernel-weighted).
"""

import numpy as np
from sklearn.metrics import pairwise_distances


def compute_conformity_scores(
    pred_lo: np.ndarray,
    pred_hi: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Compute conformity scores for CQR.

    S = max(q̂_lo - y, y - q̂_hi)

    Positive scores indicate that y is outside the predicted interval.

    Args:
        pred_lo: Predicted lower quantiles, shape (m,) or (m, 1)
        pred_hi: Predicted upper quantiles, shape (m,) or (m, 1)
        Y: True values, shape (m,) or (m, 1)

    Returns:
        Conformity scores of shape (m,)
    """
    pred_lo = np.asarray(pred_lo).flatten()
    pred_hi = np.asarray(pred_hi).flatten()
    Y = np.asarray(Y).flatten()

    return np.maximum(pred_lo - Y, Y - pred_hi)


def global_calibration(scores: np.ndarray, alpha: float) -> float:
    """
    Global calibration: compute single Q̂ for all test points.

    Q̂ = Quantile(S, β_m), where β_m = ceil((m+1)(1-α)) / m

    This ensures finite-sample coverage guarantee.

    Args:
        scores: Conformity scores from calibration set, shape (m,)
        alpha: Miscoverage level

    Returns:
        Q_hat: Global calibration constant (scalar)
    """
    m = len(scores)
    beta_m = np.ceil((m + 1) * (1 - alpha)) / m

    # Clip to [0, 1] in case of numerical issues
    beta_m = min(beta_m, 1.0)

    return float(np.quantile(scores, beta_m))


class LocalConformalOptimizer:
    """
    Kernel-weighted conformal calibration (Localized CQR).

    For each test point x, computes a local quantile Q̂_{x,h,1-α} from
    kernel-weighted calibration scores using Epanechnikov kernel.

    Attributes:
        X_cal: Calibration features, shape (m, d)
        scores: Conformity scores, shape (m,)
        h: Bandwidth parameter
    """

    def __init__(self, X_cal: np.ndarray, scores: np.ndarray, h: float):
        """
        Initialize the local calibrator.

        Args:
            X_cal: Calibration features, shape (m, d)
            scores: Conformity scores, shape (m,)
            h: Bandwidth parameter
        """
        self.X_cal = np.asarray(X_cal)
        if self.X_cal.ndim == 1:
            self.X_cal = self.X_cal.reshape(-1, 1)

        self.scores = np.asarray(scores).flatten()
        self.h = h

        # Pre-compute global quantile as fallback
        self._global_fallback = None

    def predict_corrections(
        self,
        X_test: np.ndarray,
        alpha: float,
        fallback_to_global: bool = True,
    ) -> np.ndarray:
        """
        Compute local conformity correction for each test point.

        Uses Epanechnikov kernel: K(u) = 0.75 * (1 - u²) for |u| < 1

        Args:
            X_test: Test features, shape (n_test, d)
            alpha: Miscoverage level
            fallback_to_global: Use global quantile when no neighbors found

        Returns:
            Q_hat_local: Local corrections of shape (n_test,)
        """
        X_test = np.asarray(X_test)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)

        # Pairwise distances: (n_test, m)
        D = pairwise_distances(X_test, self.X_cal, metric="euclidean")

        # Epanechnikov kernel
        u = D / self.h
        weights = 0.75 * (1 - u**2)
        weights[u >= 1.0] = 0.0
        weights = np.maximum(weights, 0.0)

        # Normalize weights
        weights_sum = np.sum(weights, axis=1, keepdims=True)
        no_neighbors = weights_sum.flatten() == 0
        weights_sum[no_neighbors] = 1.0  # Avoid division by zero
        weights = weights / weights_sum

        corrections = np.zeros(len(X_test))
        target_quantile = 1 - alpha

        # Global quantile as fallback
        if fallback_to_global:
            if self._global_fallback is None:
                self._global_fallback = np.quantile(self.scores, target_quantile)
            global_q = self._global_fallback
        else:
            global_q = np.max(self.scores)  # Conservative fallback

        for i in range(len(X_test)):
            if no_neighbors[i]:
                corrections[i] = global_q
                continue

            w_i = weights[i]
            mask = w_i > 0

            if not np.any(mask):
                corrections[i] = global_q
                continue

            s_masked = self.scores[mask]
            w_masked = w_i[mask]

            # Sort scores and compute weighted CDF
            sorter = np.argsort(s_masked)
            s_sorted = s_masked[sorter]
            w_sorted = w_masked[sorter]

            cdf = np.cumsum(w_sorted)
            cdf = cdf / cdf[-1]  # Normalize to [0, 1]

            idx = np.searchsorted(cdf, target_quantile)
            if idx < len(s_sorted):
                corrections[i] = s_sorted[idx]
            else:
                corrections[i] = s_sorted[-1]

        return corrections


def compute_bandwidth(m: int, d: int, gamma: float = 1.0) -> float:
    """
    Compute optimal bandwidth for localized CQR.

    h ~ m^{-1/(2γ+d)}, where γ is Hölder constant of density.

    Args:
        m: Calibration set size
        d: Input dimension
        gamma: Hölder smoothness of density (assume 1.0)

    Returns:
        Bandwidth h
    """
    return m ** (-1.0 / (2 * gamma + d))
