"""
Train-only bandwidth selection for localized CQR (protocol ``train_selected``).

Procedure (h is frozen before the real calibration set or test set is read):

1. Split the outer TRAINING split 70/15/15 into T-fit / T-cal / T-eval
   (``INNER_SPLIT_FRACS``).  T-cal is additionally capped at the real
   calibration-set SIZE when that is smaller; both sizes are logged.
2. Train the quantile regressors and the kernel-space model (VAE / PCA) on
   T-fit only.
3. For each candidate h in the fixed grid (the nine values 0.6..2.2 scaled by
   ``bandwidth_scale``): run localized conformal calibration on T-cal and
   score the resulting intervals on T-eval by mean Winkler score.
4. Freeze the numeric h with the lowest T-eval Winkler score.

Protocol v2 amendment: the data-driven candidates (silverman / scott / isj)
were removed from the grid — in the v1 runs they occasionally won the inner
selection with a degenerate small h (median test ESS < 10), losing coverage.
See TRAIN_SELECTED_PROTOCOL.md.

Leak-freedom contract: this module must never read calibration or test data.
Its only public entry point, ``select_bandwidth_on_train``, accepts the outer
training arrays plus scalars (config values, seed, resolved latent dim, and
the real calibration-set SIZE — a count, not data).  The contract is enforced
by ``rebuttal/verify_train_only_selection.py`` (source greps + runtime
asserts) and ``tests/test_h_selection_leakfree.py``.

The quantile trainers and the VAE trainer both call torch.manual_seed(seed)
internally, so running this selection before the outer pipeline does not
perturb the outer run's random state: the outer run stays numerically
identical to the published pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from rebuttal.protocol import INNER_SPLIT_FRACS

REPO_ROOT = Path(__file__).resolve().parent.parent


def _fix_crossings(lo: np.ndarray, hi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Same crossing fix as the outer pipeline (swap lo/hi where lo > hi)."""
    crossed = lo > hi
    if crossed.any():
        lo[crossed], hi[crossed] = hi[crossed].copy(), lo[crossed].copy()
    return lo, hi


def _inner_kernel_features(
    X_fit: np.ndarray,
    X_incal: np.ndarray,
    X_ineval: np.ndarray,
    mids: Tuple[np.ndarray, np.ndarray, np.ndarray],
    kernel_space: str,
    pca_components: int,
    vae_kwargs: Dict[str, Any],
    seed: int,
):
    """
    Kernel features for the inner splits — mirrors
    ``evaluate_real_data._build_kernel_features`` with T-fit in the role of
    the training set.  All fitting (PCA / VAE) happens on T-fit only.
    """
    if kernel_space == "yhat":
        mid_fit, mid_incal, mid_ineval = mids
        return (mid_fit.reshape(-1, 1), mid_incal.reshape(-1, 1),
                mid_ineval.reshape(-1, 1), 1)
    if kernel_space == "pca":
        from sklearn.decomposition import PCA
        k = min(int(pca_components), X_fit.shape[1])
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X_fit)
        return pca.transform(X_fit), pca.transform(X_incal), pca.transform(X_ineval), k
    if kernel_space == "vae":
        from cqr.vae import train_vae_encoder, encode_mean
        k = min(int(vae_kwargs["latent_dim"]), X_fit.shape[1])
        model = train_vae_encoder(
            X_fit,
            latent_dim=k,
            hidden_dim=int(vae_kwargs["hidden_dim"]),
            epochs=int(vae_kwargs["epochs"]),
            beta=float(vae_kwargs["beta"]),
            seed=int(seed),
            verbose=False,
        )
        return (encode_mean(model, X_fit), encode_mean(model, X_incal),
                encode_mean(model, X_ineval), k)
    # 'x'
    return X_fit, X_incal, X_ineval, X_fit.shape[1]


def select_bandwidth_on_train(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
    latent_dim: int,
    n_cal_real: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Select the localized-CQR bandwidth using training data only.

    Args:
        X_train, Y_train: The OUTER training split (already standardized).
        cfg: Resolved run config (same dict the outer pipeline uses).
        seed: Run seed — reused for the inner split and inner trainers.
        latent_dim: Resolved VAE latent dim (resolved from counts upstream so
            inner and outer kernel spaces share the same dimensionality).
        n_cal_real: SIZE of the real calibration set (count only, may be None).
            Used to cap T-cal and for logging.

    Returns:
        Dict with the frozen ``h_selected``, the winning candidate name,
        per-candidate Winkler scores on T-eval, and all inner split sizes.
    """
    import sys
    import torch
    from sklearn.model_selection import train_test_split

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from cqr.training import train_quantile_models_unified
    from cqr.calibration import compute_conformity_scores, LocalConformalOptimizer
    from cqr.metrics import winkler_score

    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(Y_train, dtype=np.float32).flatten()
    assert X.ndim == 2 and len(X) == len(y), "expected 2D X_train and matching Y_train"
    n_train, d = X.shape
    assert n_train >= 40, "training split too small for the inner 70/15/15 split"

    alpha = float(cfg["alpha"])
    assert 0.0 < alpha < 1.0

    f_fit, f_incal, f_ineval = INNER_SPLIT_FRACS
    assert abs(f_fit + f_incal + f_ineval - 1.0) < 1e-9
    n_ineval = max(int(round(f_ineval * n_train)), 2)
    n_incal_target = max(int(round(f_incal * n_train)), 2)
    n_incal = n_incal_target
    if n_cal_real is not None:
        n_incal = min(n_incal, int(n_cal_real))
    assert n_incal >= 2 and n_ineval >= 2
    assert n_incal + n_ineval < n_train, "inner split leaves no T-fit points"

    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=n_incal + n_ineval, random_state=seed)
    X_incal, X_ineval, y_incal, y_ineval = train_test_split(
        X_rest, y_rest, test_size=n_ineval, random_state=seed)
    assert len(X_fit) + len(X_incal) + len(X_ineval) == n_train
    assert len(X_incal) == n_incal and len(X_ineval) == n_ineval

    tau_low, tau_high = alpha / 2, 1 - alpha / 2
    model_lo, model_hi = train_quantile_models_unified(
        torch.from_numpy(X_fit), torch.from_numpy(y_fit.reshape(-1, 1)),
        tau_low=tau_low, tau_high=tau_high,
        input_dim=d,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        epochs=cfg["train_epochs"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        weight_decay=cfg["weight_decay"],
        grad_clip=cfg.get("grad_clip", 1.0),
        activation=cfg.get("activation", "requ"),
        verbose=False,
        seed=seed,
    )

    def _predict(X_part: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            t = torch.from_numpy(X_part)
            lo = model_lo(t).numpy().flatten()
            hi = model_hi(t).numpy().flatten()
        return _fix_crossings(lo, hi)

    fit_lo, fit_hi = _predict(X_fit)
    incal_lo, incal_hi = _predict(X_incal)
    ineval_lo, ineval_hi = _predict(X_ineval)

    scores = compute_conformity_scores(incal_lo, incal_hi, y_incal)

    kernel_space = str(cfg.get("kernel_space", "yhat"))
    vae_kwargs = {
        "latent_dim": int(latent_dim),
        "hidden_dim": int(cfg.get("vae_hidden_dim", 64)),
        "epochs": int(cfg.get("vae_epochs", 100)),
        "beta": float(cfg.get("vae_beta", 1.0)),
    }
    _feat_fit, feat_incal, feat_ineval, kernel_d = _inner_kernel_features(
        X_fit, X_incal, X_ineval,
        mids=((fit_lo + fit_hi) / 2, (incal_lo + incal_hi) / 2,
              (ineval_lo + ineval_hi) / 2),
        kernel_space=kernel_space,
        pca_components=int(cfg.get("pca_components", 3)),
        vae_kwargs=vae_kwargs,
        seed=seed,
    )

    bs = float(cfg["bandwidth_scale"])
    # v2: fixed grid only — no data-driven candidates (see module docstring).
    grid = list(cfg.get("fixed_bandwidth_grid") or [])
    assert grid, "train_selected requires a non-empty fixed_bandwidth_grid"
    candidates = [
        (f"fixed_{float(h):g}", max(bs * float(h), 1e-6)) for h in grid
    ]
    assert len(candidates) == len(grid)

    per_candidate: Dict[str, Dict[str, float]] = {}
    best_name: Optional[str] = None
    best_h = np.inf
    best_w = np.inf
    for name, h_val in candidates:
        assert np.isfinite(h_val) and h_val > 0
        lcp = LocalConformalOptimizer(feat_incal, scores, h=h_val)
        Q = lcp.predict_corrections(feat_ineval, alpha)
        w = float(winkler_score(y_ineval, ineval_lo - Q, ineval_hi + Q, alpha=alpha))
        per_candidate[name] = {"h": float(h_val), "winkler": w}
        if w < best_w:  # strict < keeps the earliest candidate on exact ties
            best_name, best_h, best_w = name, float(h_val), w

    assert best_name is not None and np.isfinite(best_h) and best_h > 0

    return {
        "h_selected": best_h,
        "candidate_selected": best_name,
        "winkler_selected": best_w,
        "candidates": per_candidate,
        "selection_metric": "winkler",
        "n_train": int(n_train),
        "n_fit": int(len(X_fit)),
        "n_incal": int(n_incal),
        "n_incal_target": int(n_incal_target),
        "n_ineval": int(n_ineval),
        "n_cal_real": None if n_cal_real is None else int(n_cal_real),
        "inner_split_fracs": [float(f) for f in INNER_SPLIT_FRACS],
        "kernel_space": kernel_space,
        "kernel_d": int(kernel_d),
        "latent_dim": int(latent_dim),
    }
