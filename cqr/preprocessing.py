"""
Data preprocessing for real-world CQR experiments.

Handles train/calibration/test splitting and feature/target standardization.
"""

import numpy as np
from typing import Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float = 0.3,
    cal_frac: float = 0.4,
    test_frac: float = 0.3,
    seed: int = 42,
    standardize_x: bool = True,
    standardize_y: bool = True,
    target_col: int = 0,
) -> Dict[str, Any]:
    """
    Split data into train/calibration/test and standardize.

    The three-way split is essential for CQR:
    - Train: fit quantile regression models
    - Calibration: compute conformity scores & calibrate
    - Test: evaluate coverage and interval width

    Args:
        X: Features, shape (n, d)
        y: Targets, shape (n,) or (n, 1) or (n, d_out) for multi-output
        train_frac: Fraction for training (default 0.4)
        cal_frac: Fraction for calibration (default 0.3)
        test_frac: Fraction for testing (default 0.3)
        seed: Random seed
        standardize_x: Whether to standardize features
        standardize_y: Whether to standardize target
        target_col: Which output column to use for multi-output datasets
                     (default 0 = first column). Ignored for scalar targets.

    Returns:
        Dict with keys:
            X_train, Y_train, X_cal, Y_cal, X_test, Y_test: np.float32 arrays
            scaler_x: fitted StandardScaler for X (or None)
            scaler_y: fitted StandardScaler for y (or None)
            n_train, n_cal, n_test: split sizes
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # Handle multi-output targets: select a single column
    if y.ndim == 2 and y.shape[1] > 1:
        assert 0 <= target_col < y.shape[1], \
            f"target_col={target_col} out of range for y with {y.shape[1]} outputs"
        y = y[:, target_col]
    y = y.flatten()

    assert abs(train_frac + cal_frac + test_frac - 1.0) < 1e-6, \
        f"Fractions must sum to 1, got {train_frac + cal_frac + test_frac}"

    n = len(X)

    # First split: train vs (cal + test)
    cal_test_frac = cal_frac + test_frac
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=cal_test_frac, random_state=seed
    )

    # Second split: cal vs test
    test_of_rest = test_frac / cal_test_frac
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_rest, y_rest, test_size=test_of_rest, random_state=seed
    )

    # Standardize features
    scaler_x = None
    if standardize_x:
        scaler_x = StandardScaler()
        X_train = scaler_x.fit_transform(X_train).astype(np.float32)
        X_cal = scaler_x.transform(X_cal).astype(np.float32)
        X_test = scaler_x.transform(X_test).astype(np.float32)

    # Standardize target
    scaler_y = None
    if standardize_y:
        scaler_y = StandardScaler()
        y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten().astype(np.float32)
        y_cal = scaler_y.transform(y_cal.reshape(-1, 1)).flatten().astype(np.float32)
        y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten().astype(np.float32)

    # Reshape targets to (n, 1) for PyTorch compatibility
    Y_train = y_train.reshape(-1, 1)
    Y_cal = y_cal.reshape(-1, 1)
    Y_test = y_test.reshape(-1, 1)

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_cal": X_cal,
        "Y_cal": Y_cal,
        "X_test": X_test,
        "Y_test": Y_test,
        "scaler_x": scaler_x,
        "scaler_y": scaler_y,
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
    }


def inverse_transform_width(
    width: float,
    scaler_y: StandardScaler,
) -> float:
    """
    Convert interval width from standardized scale back to original scale.

    Since standardization is affine: y_std = (y - mu) / sigma,
    an interval of width w in standardized space has width w * sigma in original space.

    Args:
        width: Width in standardized space
        scaler_y: Fitted StandardScaler for y

    Returns:
        Width in original scale
    """
    if scaler_y is None:
        return width
    return width * scaler_y.scale_[0]
