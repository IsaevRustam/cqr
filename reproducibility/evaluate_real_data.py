"""
Evaluate Global CQR vs Localized CQR on Real Datasets
======================================================
Compares the two conformal calibration strategies using ReQU neural networks
for quantile regression on a suite of real-world regression datasets.

Metrics reported per dataset and method:
  - Marginal coverage (should be ≈ 1 - α)
  - Average prediction interval width
  - Median width and width std
  - Conditional coverage: worst-bin coverage, coverage gap, coverage range

Usage:
    python evaluate_real_data.py                                        # all default datasets
    python evaluate_real_data.py --datasets diabetes concrete energy    # specific datasets
    python evaluate_real_data.py --config configs/real.yaml             # custom config
    python evaluate_real_data.py --alpha 0.05 --n_attempts 20          # override params
    python evaluate_real_data.py --output results.csv                  # save CSV
"""

import argparse
import time
import traceback
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional
from sklearn.decomposition import PCA

from cqr.real_data import load_dataset, DEFAULT_DATASETS, list_datasets
from cqr.preprocessing import prepare_data, inverse_transform_width
from cqr.training import train_quantile_models_unified
from cqr.calibration import (
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
)
from cqr.metrics import evaluate_intervals


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_real_config(path: Optional[str]) -> Dict[str, Any]:
    """Load config from YAML or return defaults for real-data experiments."""
    defaults = {
        "alpha": 0.1,
        "seed": 42,
        "n_attempts": 5,
        "hidden_dim": 128,
        "train_epochs": 100,
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 1e-5,
        "n_layers": 2,
        "bandwidth_scale": 1.0,
        "fixed_bandwidth": 1.5,            # h for the data-independent 'Fixed' rule (multiplied by bandwidth_scale)
        "fixed_bandwidth_grid": None,      # list of h values; when set, evaluates each as a separate Fixed method
        "n_cond_bins": 5,
        "activation": "requ",
        "clip_outliers_iqr": None,       # IQR factor for y-outlier removal
        "kernel_space": "yhat",            # 'yhat' | 'pca' | 'x' | 'vae'
        "pca_components": 3,               # number of PCA components (used when kernel_space='pca')
        "vae_latent_dim": "auto",          # int or 'auto' (auto = scaled to n_cal)
        "vae_hidden_dim": 64,              # encoder/decoder hidden width
        "vae_epochs": 100,
        "vae_beta": 1.0,                   # KL weight (1.0 = standard VAE)
        # Per-dataset overrides: key = dataset name, value = dict of overrides
        "dataset_overrides": {
            # rf1: 64 features, ~8757 samples → 128-unit NN overfits badly
            "rf1": {"hidden_dim": 32, "weight_decay": 1e-3},
        },
    }
    if path is not None:
        import yaml
        with open(path, "r") as f:
            overrides = yaml.safe_load(f)
        if overrides:
            defaults.update(overrides)
    return defaults


# =============================================================================
# SINGLE-RUN EVALUATION
# =============================================================================

def _build_method_specs(cfg: Dict[str, Any]) -> List[tuple]:
    """
    Return ordered (method_key, method_label, h_key) tuples for all methods
    evaluated in a single run.

    When ``fixed_bandwidth_grid`` is set in cfg, one Fixed entry is emitted
    per h value. Otherwise a single Fixed entry uses cfg['fixed_bandwidth'].
    """
    specs = [
        ("global",          "Global CQR",          None),
        ("local_silverman", "Local CQR (Silverman)", "h_silverman"),
        ("local_scott",     "Local CQR (Scott)",     "h_scott"),
        ("local_isj",       "Local CQR (ISJ)",       "h_isj"),
    ]
    grid = cfg.get("fixed_bandwidth_grid")
    if grid:
        for h in grid:
            tag = f"{float(h):g}"
            specs.append((
                f"local_fixed_{tag}",
                f"Local CQR (Fixed h={float(h):.2f})",
                f"h_fixed_{tag}",
            ))
    else:
        specs.append(("local_fixed", "Local CQR (Fixed)", "h_fixed"))
    return specs


def _compute_bandwidths(
    X_ref: np.ndarray,
    d: int,
    bandwidth_scale: float = 1.0,
    fixed_bandwidth: float = 1.5,
) -> Dict[str, float]:
    """
    Compute four kernel-regression bandwidths on the chosen kernel feature space.

    Using the predicted value ŷ = (q̂_lo + q̂_hi)/2 from the training set
    as the 1D reference avoids the curse of dimensionality (kernel is always
    applied in 1D space regardless of the original feature dimension d).
    Using training data (not calibration) keeps h independent of conformal
    calibration.

    Data-driven rules follow  h = bandwidth_scale * rate(m, d) * sigma_eff,
    where sigma_eff is the geometric mean of per-feature robust spread
    min(std_j, IQR_j / 1.349), keeping h << 1 on standardized features.

    Rules
    -----
    silverman : (4/(d+2))^{1/(d+4)} * m^{-1/(d+4)} * sigma_eff   (Silverman 1986)
    scott     : m^{-1/(d+4)}         * sigma_eff                   (Scott 1992)
    isj       : m^{-1/(d+4)}         * sigma_eff / sqrt(d)         (dim-penalized rate, Botev 2010)
    fixed     : bandwidth_scale * fixed_bandwidth                  (data-independent,
                useful for coverage-targeting sweeps; on standardized features a
                value around 1.0–2.0 spans almost the whole calibration cloud)
    """
    m = len(X_ref)
    stds = np.std(X_ref, axis=0, ddof=1)
    iqrs = (np.percentile(X_ref, 75, axis=0) - np.percentile(X_ref, 25, axis=0)) / 1.349
    spreads = np.maximum(np.minimum(stds, iqrs), 1e-8)
    sigma_eff = float(np.exp(np.mean(np.log(spreads))))

    rate_scott = m ** (-1.0 / (d + 4))
    rate_silverman = ((4.0 / (d + 2)) ** (1.0 / (d + 4))) * rate_scott

    return {
        "silverman": max(bandwidth_scale * rate_silverman * sigma_eff, 1e-6),
        "scott":     max(bandwidth_scale * rate_scott     * sigma_eff, 1e-6),
        "isj":       max(bandwidth_scale * rate_scott     * sigma_eff / np.sqrt(d), 1e-6),
        "fixed":     max(bandwidth_scale * float(fixed_bandwidth), 1e-6),
    }


def _auto_latent_dim(
    n_cal: int,
    n_features: int,
    target_neighbors: int = 30,
    min_dim: int = 2,
    max_dim: int = 8,
) -> int:
    """
    Pick a latent dim so an Epanechnikov ball at Silverman bandwidth holds
    roughly ``target_neighbors`` calibration points.

    With h ~ n_cal^{-1/(d+4)}, the asymptotic neighborhood count scales as
    n_cal^{4/(d+4)}. Requiring it >= target_neighbors gives
        d_max = floor(4 * log(n_cal) / log(target_neighbors) - 4).
    The result is clipped to [min_dim, min(max_dim, n_features)].
    """
    if n_cal <= 1 or n_features <= 1:
        return max(1, n_features)
    log_ratio = float(np.log(max(n_cal, 2)) / np.log(max(target_neighbors, 2)))
    raw = int(np.floor(4.0 * log_ratio - 4.0))
    return int(max(min_dim, min(raw, max_dim, n_features)))


def _resolve_latent_dim(value: Any, n_cal: int, n_features: int) -> int:
    """Resolve a user-supplied latent dim ('auto' / None / 0 / int) to int."""
    if value is None or value == 0 or (isinstance(value, str) and value.strip().lower() == "auto"):
        return _auto_latent_dim(n_cal, n_features)
    return int(value)


def _build_kernel_features(
    X_train: np.ndarray,
    X_cal: np.ndarray,
    X_test: np.ndarray,
    pred_train_lo: np.ndarray,
    pred_train_hi: np.ndarray,
    pred_cal_lo: np.ndarray,
    pred_cal_hi: np.ndarray,
    pred_test_lo: np.ndarray,
    pred_test_hi: np.ndarray,
    kernel_space: str,
    pca_components: int,
    vae_kwargs: Optional[Dict[str, Any]] = None,
    seed: int = 0,
):
    """
    Build 1D or low-d kernel features for LocalConformalOptimizer.

    kernel_space options:
      'yhat' : 1D midpoint of predicted interval (q̂_lo + q̂_hi) / 2
      'pca'  : top-k PCA components of X, fitted on X_train
      'x'    : raw X (no dimensionality reduction)
      'vae'  : posterior-mean latents from a VAE fitted on X_train.
               vae_kwargs supplies: latent_dim, hidden_dim, epochs, beta.

    Returns (feat_train, feat_cal, feat_test, kernel_d).
    """
    if kernel_space == "yhat":
        feat_train = ((pred_train_lo + pred_train_hi) / 2).reshape(-1, 1)
        feat_cal   = ((pred_cal_lo   + pred_cal_hi)   / 2).reshape(-1, 1)
        feat_test  = ((pred_test_lo  + pred_test_hi)  / 2).reshape(-1, 1)
        kernel_d   = 1
    elif kernel_space == "pca":
        k = min(pca_components, X_train.shape[1])
        pca = PCA(n_components=k, random_state=0)
        pca.fit(X_train)
        feat_train = pca.transform(X_train)
        feat_cal   = pca.transform(X_cal)
        feat_test  = pca.transform(X_test)
        kernel_d   = k
    elif kernel_space == "vae":
        from cqr.vae import train_vae_encoder, encode_mean
        kw = vae_kwargs or {}
        k = min(int(kw.get("latent_dim", 3)), X_train.shape[1])
        model = train_vae_encoder(
            X_train,
            latent_dim=k,
            hidden_dim=int(kw.get("hidden_dim", 64)),
            epochs=int(kw.get("epochs", 100)),
            beta=float(kw.get("beta", 1.0)),
            seed=int(seed),
            verbose=bool(kw.get("verbose", False)),
        )
        feat_train = encode_mean(model, X_train)
        feat_cal   = encode_mean(model, X_cal)
        feat_test  = encode_mean(model, X_test)
        kernel_d   = k
    else:  # 'x'
        feat_train = X_train
        feat_cal   = X_cal
        feat_test  = X_test
        kernel_d   = X_train.shape[1]
    return feat_train, feat_cal, feat_test, kernel_d


def evaluate_single_run(
    X: np.ndarray,
    y: np.ndarray,
    cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    """
    Single train-calibrate-test run for both Global and Localized CQR.

    Returns a dict with metrics for both methods.
    """
    alpha = cfg["alpha"]
    tau_low = alpha / 2
    tau_high = 1 - alpha / 2
    d = X.shape[1]

    # Split data — adaptive to dataset size:
    #   small (n < 2000): 0.4 / 0.3 / 0.3 — keeps test set large enough for
    #                      conditional-coverage bins to clear min_bin_size.
    #   large (n >= 2000): 0.3 / 0.5 / 0.2 — gives LocalConformalOptimizer
    #                      more calibration mass.
    if len(X) < 2000:
        train_frac, cal_frac, test_frac = 0.4, 0.3, 0.3
    else:
        train_frac, cal_frac, test_frac = 0.3, 0.5, 0.2

    data = prepare_data(
        X, y,
        train_frac=train_frac, cal_frac=cal_frac, test_frac=test_frac,
        seed=seed,
        clip_outliers_iqr=cfg.get("clip_outliers_iqr"),
        robust_scale_y=cfg.get("robust_scale_y", False),
    )

    X_train = torch.from_numpy(data["X_train"])
    Y_train = torch.from_numpy(data["Y_train"])

    # Train quantile models (activation from config: relu or requ)
    activation = cfg.get("activation", "requ")
    model_lo, model_hi = train_quantile_models_unified(
        X_train, Y_train,
        tau_low=tau_low, tau_high=tau_high,
        input_dim=d,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        epochs=cfg["train_epochs"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        weight_decay=cfg["weight_decay"],
        grad_clip=cfg.get("grad_clip", 1.0),
        activation=activation,
        verbose=cfg.get("verbose", True),
        seed=seed,
    )

    # Predict on calibration set
    with torch.no_grad():
        X_cal_t = torch.from_numpy(data["X_cal"])
        pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
        pred_cal_hi = model_hi(X_cal_t).numpy().flatten()

    # Fix quantile crossing: ensure lo ≤ hi
    crossed_cal = pred_cal_lo > pred_cal_hi
    if crossed_cal.any():
        pred_cal_lo[crossed_cal], pred_cal_hi[crossed_cal] = (
            pred_cal_hi[crossed_cal].copy(), pred_cal_lo[crossed_cal].copy()
        )

    # Conformity scores
    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, data["Y_cal"])

    # Predict on test set
    with torch.no_grad():
        X_test_t = torch.from_numpy(data["X_test"])
        pred_test_lo = model_lo(X_test_t).numpy().flatten()
        pred_test_hi = model_hi(X_test_t).numpy().flatten()

    # Fix quantile crossing: ensure lo ≤ hi
    crossed_test = pred_test_lo > pred_test_hi
    if crossed_test.any():
        n_crossed = int(crossed_test.sum())
        print(f"    [warning] {n_crossed}/{len(pred_test_lo)} test points had crossed quantiles (swapped)")
        pred_test_lo[crossed_test], pred_test_hi[crossed_test] = (
            pred_test_hi[crossed_test].copy(), pred_test_lo[crossed_test].copy()
        )

    # =========================================================================
    # GLOBAL CQR
    # =========================================================================
    Q_hat_global = global_calibration(scores, alpha)

    global_lo = pred_test_lo - Q_hat_global
    global_hi = pred_test_hi + Q_hat_global

    global_metrics = evaluate_intervals(
        data["Y_test"], global_lo, global_hi, data["X_test"],
        alpha=alpha, n_bins=cfg["n_cond_bins"],
    )

    # =========================================================================
    # LOCALIZED CQR — three SotA bandwidths (Silverman, Scott, ISJ)
    # =========================================================================
    # Build kernel features (yhat / pca / x) — fitted on train set only.
    kernel_space = str(cfg.get("kernel_space", "yhat"))
    pca_components = int(cfg.get("pca_components", 3))

    with torch.no_grad():
        X_train_t = torch.from_numpy(data["X_train"])
        pred_train_lo = model_lo(X_train_t).numpy().flatten()
        pred_train_hi = model_hi(X_train_t).numpy().flatten()

    resolved_latent_dim = _resolve_latent_dim(
        cfg.get("vae_latent_dim", "auto"),
        n_cal=int(data["n_cal"]),
        n_features=int(d),
    )
    if cfg.get("verbose", False) and kernel_space == "vae":
        print(f"  [vae] latent_dim={resolved_latent_dim} "
              f"(n_cal={data['n_cal']}, requested={cfg.get('vae_latent_dim', 'auto')})")
    vae_kwargs = {
        "latent_dim": resolved_latent_dim,
        "hidden_dim": int(cfg.get("vae_hidden_dim", 64)),
        "epochs":     int(cfg.get("vae_epochs", 100)),
        "beta":       float(cfg.get("vae_beta", 1.0)),
        "verbose":    bool(cfg.get("verbose", False)),
    }

    feat_train, feat_cal, feat_test, kernel_d = _build_kernel_features(
        data["X_train"], data["X_cal"], data["X_test"],
        pred_train_lo, pred_train_hi,
        pred_cal_lo, pred_cal_hi,
        pred_test_lo, pred_test_hi,
        kernel_space=kernel_space,
        pca_components=pca_components,
        vae_kwargs=vae_kwargs,
        seed=seed,
    )

    bandwidths = _compute_bandwidths(
        feat_train,
        d=int(kernel_d),
        bandwidth_scale=float(cfg["bandwidth_scale"]),
        fixed_bandwidth=float(cfg.get("fixed_bandwidth", 1.5)),
    )

    def _run_local(h_val):
        lcp = LocalConformalOptimizer(feat_cal, scores, h=h_val)
        parts = []
        for start in range(0, len(feat_test), 1000):
            end = min(start + 1000, len(feat_test))
            parts.append(lcp.predict_corrections(feat_test[start:end], alpha))
        Q_hat = np.concatenate(parts)
        return evaluate_intervals(
            data["Y_test"],
            pred_test_lo - Q_hat,
            pred_test_hi + Q_hat,
            data["X_test"],
            alpha=alpha, n_bins=cfg["n_cond_bins"],
        )

    local_silverman_metrics = _run_local(bandwidths["silverman"])
    local_scott_metrics     = _run_local(bandwidths["scott"])
    local_isj_metrics       = _run_local(bandwidths["isj"])

    result: Dict[str, Any] = {
        "global": global_metrics,
        "local_silverman": local_silverman_metrics,
        "local_scott": local_scott_metrics,
        "local_isj": local_isj_metrics,
        "h_silverman": bandwidths["silverman"],
        "h_scott": bandwidths["scott"],
        "h_isj": bandwidths["isj"],
        "Q_hat_global": Q_hat_global,
    }

    # Fixed-bandwidth method(s): single value or grid
    fixed_grid = cfg.get("fixed_bandwidth_grid")
    bs = float(cfg["bandwidth_scale"])
    if fixed_grid:
        for h in fixed_grid:
            tag = f"{float(h):g}"
            h_scaled = max(bs * float(h), 1e-6)
            result[f"local_fixed_{tag}"] = _run_local(h_scaled)
            result[f"h_fixed_{tag}"] = h_scaled
    else:
        result["local_fixed"] = _run_local(bandwidths["fixed"])
        result["h_fixed"] = bandwidths["fixed"]

    # Convert widths to original scale if target was standardized
    scaler_y = data["scaler_y"]
    method_keys = [s[0] for s in _build_method_specs(cfg)]
    for mkey in method_keys:
        m_dict = result[mkey]
        if scaler_y is not None:
            m_dict["avg_width_orig"] = inverse_transform_width(m_dict["avg_width"], scaler_y)
            m_dict["median_width_orig"] = inverse_transform_width(m_dict["median_width"], scaler_y)
        else:
            m_dict["avg_width_orig"] = m_dict["avg_width"]
            m_dict["median_width_orig"] = m_dict["median_width"]

    return result


# =============================================================================
# MULTI-RUN AGGREGATION
# =============================================================================

def run_dataset_evaluation(
    name: str,
    cfg: Dict[str, Any],
    verbose: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run multiple repetitions of Global vs Localized CQR on one dataset.

    Returns list of per-run result dicts.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Dataset: {name}")
        print(f"{'='*70}")

    try:
        X, y, info = load_dataset(name)
    except Exception as e:
        print(f"  SKIPPED — failed to load: {e}")
        return []

    if verbose:
        n_samples = int(info['n_samples'])
        _split = "0.4/0.3/0.3 (small)" if n_samples < 2000 else "0.3/0.5/0.2 (large)"
        print(f"  n={n_samples}, d={info['n_features']}, {info['description']}")
        print(f"  split (train/cal/test): {_split}")

    # Apply per-dataset config overrides (e.g. rf1 → clip_outliers_iqr)
    dataset_overrides = cfg.get("dataset_overrides", {})
    if name in dataset_overrides:
        cfg = {**cfg, **dataset_overrides[name]}
        if verbose:
            print(f"  [config] dataset overrides applied: {dataset_overrides[name]}")

    n_attempts = cfg["n_attempts"]
    base_seed = cfg["seed"]
    results = []

    method_specs = _build_method_specs(cfg)

    for attempt in range(n_attempts):
        seed = base_seed + attempt
        try:
            res = evaluate_single_run(X, y, cfg, seed)
            results.append(res)
            if verbose:
                parts = []
                for mkey, mlabel, hkey in method_specs:
                    m = res[mkey]
                    short = mlabel.replace("Local CQR (", "").rstrip(")") if mkey != "global" else "Global"
                    h_str = f"(h={res[hkey]:.3f})" if hkey is not None else ""
                    parts.append(
                        f"{short}{h_str} cov={m['coverage']:.3f} "
                        f"wgc={m['worst_bin_cov']:.3f} w={m['avg_width']:.3f}"
                    )
                print(f"  Run {attempt+1}/{n_attempts}: " + " | ".join(parts))
        except Exception as e:
            print(f"  Run {attempt+1}/{n_attempts}: FAILED — {e}")
            traceback.print_exc()

    # Print summary after all runs for this dataset
    if verbose and len(results) > 0:
        from scipy import stats as _stats
        from collections import defaultdict
        alpha = cfg["alpha"]
        n_runs = len(results)
        # 95% CI half-width: t_{0.025, n-1} * std / sqrt(n)
        t_val = _stats.t.ppf(0.975, df=max(n_runs - 1, 1))

        def ci(values):
            m, s = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0
            hw = t_val * s / np.sqrt(n_runs)
            return m, hw

        print(f"\n  --- Summary for {name} ({n_runs}/{n_attempts} successful runs, 95% CI) ---")
        print(f"  {'':28s} {'Coverage':>16s} {'Avg Width':>16s} {'Worst-Bin':>16s}")
        for method_key, method_label, _ in method_specs:
            covs = [r[method_key]["coverage"] for r in results]
            widths = [r[method_key]["avg_width"] for r in results]
            worst = [r[method_key]["worst_bin_cov"] for r in results]
            c_m, c_h = ci(covs)
            w_m, w_h = ci(widths)
            wb_m, wb_h = ci(worst)
            print(
                f"  {method_label:28s} "
                f"{c_m:.3f}±{c_h:.3f}  "
                f"{w_m:.3f}±{w_h:.3f}  "
                f"{wb_m:.3f}±{wb_h:.3f}"
            )
        # Width comparison vs Global
        g_w = np.mean([r["global"]["avg_width"] for r in results])
        for method_key, method_label, _ in method_specs:
            if method_key == "global":
                continue
            l_w = np.mean([r[method_key]["avg_width"] for r in results])
            diff_pct = (l_w - g_w) / g_w * 100
            sym = "narrower" if diff_pct < 0 else "wider"
            print(f"  {method_label} is {abs(diff_pct):.1f}% {sym} than Global (target cov={1-alpha:.0%})")

        # Per-bin breakdown for all methods
        for method_key, method_label, _ in method_specs:
            all_ranked = [r[method_key]["ranked_bins"] for r in results]
            bin_agg = defaultdict(lambda: {"coverages": [], "avg_widths": [], "counts": []})
            for run_ranked in all_ranked:
                for entry in run_ranked:
                    bid = entry["bin_id"]
                    bin_agg[bid]["coverages"].append(entry["coverage"])
                    bin_agg[bid]["avg_widths"].append(entry["avg_width"])
                    bin_agg[bid]["counts"].append(entry["count"])

            if len(bin_agg) == 0:
                continue

            bin_summaries = []
            for bid, agg in bin_agg.items():
                bin_summaries.append({
                    "bin_id": bid,
                    "mean_cov": np.mean(agg["coverages"]),
                    "std_cov": np.std(agg["coverages"]),
                    "mean_width": np.mean(agg["avg_widths"]),
                    "mean_count": np.mean(agg["counts"]),
                })
            bin_summaries.sort(key=lambda d: d["mean_cov"])

            print(f"\n  {method_label} -- Bins ranked by difficulty (hardest -> easiest):")
            print(f"  {'Rank':>4s}  {'Bin':>3s}  {'Count':>6s}  {'Coverage':>12s}  {'Avg Width':>10s}")
            for rank, bs in enumerate(bin_summaries, 1):
                print(
                    f"  {rank:4d}  {bs['bin_id']:3d}  {bs['mean_count']:6.0f}  "
                    f"{bs['mean_cov']:.3f}\u00b1{bs['std_cov']:.3f}  "
                    f"{bs['mean_width']:.3f}"
                )
        print()

    return results


def aggregate_results(
    dataset_name: str,
    n_samples: int,
    n_features: int,
    runs: List[Dict[str, Any]],
    alpha: float,
    activation: str = "requ",
    cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Aggregate multiple runs into summary rows (one per method) plus per-bin detail rows.

    Returns list of dicts suitable for DataFrame construction.
    Each method gets:
      - One "Overall" row with aggregate metrics
      - One row per bin with bin-specific metrics (ranked by difficulty)
    """
    if len(runs) == 0:
        return []

    rows = []
    method_specs = _build_method_specs(cfg or {})
    for method_key, method_label, _ in method_specs:
        coverages = [r[method_key]["coverage"] for r in runs]
        avg_widths = [r[method_key]["avg_width"] for r in runs]
        median_widths = [r[method_key]["median_width"] for r in runs]
        width_stds = [r[method_key]["width_std"] for r in runs]
        worst_bins = [r[method_key]["worst_bin_cov"] for r in runs]
        best_bins = [r[method_key]["best_bin_cov"] for r in runs]
        ranges = [r[method_key]["coverage_range"] for r in runs]
        avg_widths_orig = [r[method_key]["avg_width_orig"] for r in runs]
        winkler_scores = [r[method_key]["winkler_score"] for r in runs]
        width_error_corrs = [r[method_key]["width_error_corr"] for r in runs]
        ccvs = [r[method_key]["ccv"] for r in runs]

        # Overall summary row
        rows.append({
            "Dataset": dataset_name,
            "n": n_samples,
            "d": n_features,
            "Method": method_label,
            "Activation": activation.upper(),
            "Bin": "Overall",
            "Bin ID": "",
            "Bin Rank": "",
            "Target Coverage": f"{1-alpha:.0%}",
            "Coverage (mean)": np.mean(coverages),
            "Coverage (std)": np.std(coverages),
            "Avg Width (mean)": np.mean(avg_widths),
            "Avg Width (std)": np.std(avg_widths),
            "Avg Width (orig)": np.mean(avg_widths_orig),
            "Median Width": np.mean(median_widths),
            "Width Std": np.mean(width_stds),
            "Worst-Bin Cov (mean)": np.mean(worst_bins),
            "Worst-Bin Cov (std)": np.std(worst_bins),
            "Best-Bin Cov": np.mean(best_bins),
            "Cov Range": np.mean(ranges),
            "Bin Count (mean)": "",
            "Winkler Score (mean)": float(np.nanmean(winkler_scores)),
            "Winkler Score (std)": float(np.nanstd(winkler_scores)),
            "Width-Error Corr (mean)": float(np.nanmean(width_error_corrs)),
            "Width-Error Corr (std)": float(np.nanstd(width_error_corrs)),
            "CCV (mean)": float(np.nanmean(ccvs)),
            "CCV (std)": float(np.nanstd(ccvs)),
        })

        # Per-bin detail rows
        from collections import defaultdict
        all_ranked = [r[method_key]["ranked_bins"] for r in runs]
        bin_agg = defaultdict(lambda: {"coverages": [], "avg_widths": [], "counts": []})
        for run_ranked in all_ranked:
            for entry in run_ranked:
                bid = entry["bin_id"]
                bin_agg[bid]["coverages"].append(entry["coverage"])
                bin_agg[bid]["avg_widths"].append(entry["avg_width"])
                bin_agg[bid]["counts"].append(entry["count"])

        if len(bin_agg) > 0:
            # Build per-bin rows, sorted by mean coverage ascending
            bin_summaries = []
            for bid, agg in bin_agg.items():
                bin_summaries.append({
                    "bin_id": bid,
                    "mean_cov": np.mean(agg["coverages"]),
                    "std_cov": np.std(agg["coverages"]),
                    "mean_width": np.mean(agg["avg_widths"]),
                    "mean_count": np.mean(agg["counts"]),
                })
            bin_summaries.sort(key=lambda d: d["mean_cov"])

            for rank, bs in enumerate(bin_summaries, 1):
                rows.append({
                    "Dataset": dataset_name,
                    "n": n_samples,
                    "d": n_features,
                    "Method": method_label,
                    "Activation": activation.upper(),
                    "Bin": f"Bin {rank}",
                    "Bin ID": bs["bin_id"],
                    "Bin Rank": rank,
                    "Target Coverage": f"{1-alpha:.0%}",
                    "Coverage (mean)": bs["mean_cov"],
                    "Coverage (std)": bs["std_cov"],
                    "Avg Width (mean)": bs["mean_width"],
                    "Avg Width (std)": "",
                    "Avg Width (orig)": "",
                    "Median Width": "",
                    "Width Std": "",
                    "Worst-Bin Cov (mean)": "",
                    "Worst-Bin Cov (std)": "",
                    "Best-Bin Cov": "",
                    "Cov Range": "",
                    "Bin Count (mean)": bs["mean_count"],
                })

    return rows


# =============================================================================
# TABLE FORMATTING
# =============================================================================

DISPLAY_COLUMNS = [
    "Dataset", "n", "d", "Method", "Activation", "Bin", "Bin ID", "Bin Rank", "Target Coverage",
    "Coverage (mean)", "Coverage (std)",
    "Avg Width (mean)", "Avg Width (std)", "Avg Width (orig)",
    "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)",
    "Cov Range",
    "Winkler Score (mean)", "Winkler Score (std)",
    "Width-Error Corr (mean)", "Width-Error Corr (std)",
    "CCV (mean)", "CCV (std)",
    "Bin Count (mean)",
]


def print_results_table(df: pd.DataFrame) -> None:
    """Print a detailed comparison table (Overall rows only)."""
    # Filter to Overall rows for the display table
    display_df = df[df["Bin"] == "Overall"].copy()
    display_cols = [
        "Dataset", "n", "d", "Method", "Activation", "Target Coverage",
        "Coverage (mean)", "Coverage (std)",
        "Avg Width (mean)", "Avg Width (std)", "Avg Width (orig)",
        "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)",
        "Cov Range",
        "Winkler Score (mean)", "Winkler Score (std)",
        "Width-Error Corr (mean)", "Width-Error Corr (std)",
        "CCV (mean)", "CCV (std)",
    ]
    # Keep only columns that actually exist (backward compat with old CSVs)
    display_cols = [c for c in display_cols if c in display_df.columns]
    display = display_df[display_cols].copy()
    # Format numeric columns to 3 decimal places for readability
    float_cols = [c for c in display_cols if c not in ("Dataset", "n", "d", "Method", "Activation", "Target Coverage")]
    for c in float_cols:
        display[c] = display[c].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    print("\n" + "=" * 150)
    print("RESULTS: Global CQR vs Localized CQR (4 bandwidths) -- Overall Summary")
    print("=" * 150)
    print(display.to_string(index=False))
    print("=" * 150)

    # Print summary comparison
    print("\n--- SUMMARY ---")
    for dataset in display_df["Dataset"].unique():
        sub = display_df[display_df["Dataset"] == dataset]
        g_row = sub[sub["Method"] == "Global CQR"]
        if g_row.empty:
            continue
        g = g_row.iloc[0]
        local_labels = [m for m in sub["Method"].unique() if m != "Global CQR"]
        for local_label in local_labels:
            l_row = sub[sub["Method"] == local_label]
            if l_row.empty:
                continue
            l = l_row.iloc[0]
            width_diff = ((l["Avg Width (mean)"] - g["Avg Width (mean)"]) / g["Avg Width (mean)"]) * 100
            symbol = "-" if width_diff < 0 else "+"
            rule = local_label.split("(")[1].rstrip(")") if "(" in local_label else local_label
            print(
                f"  {dataset:20s} {rule:18s}: "
                f"Local width {symbol} {abs(width_diff):5.1f}% vs Global"
            )

    print("\n--- PER-BIN BREAKDOWN ---")
    print("Detailed per-bin results are included in the full CSV output.")
    print("Each dataset has bin-level rows ranked by difficulty (hardest -> easiest).")
    print()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Global vs Localized CQR on real datasets (ReQU NN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {', '.join(list_datasets())}",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help=f"Dataset names (default: {', '.join(DEFAULT_DATASETS)})",
    )
    parser.add_argument("--config", type=str, default="configs/real.yaml")
    parser.add_argument("--alpha", type=float, default=None, help="Miscoverage level")
    parser.add_argument("--n_attempts", type=int, default=None, help="Number of repetitions")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--train_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--bandwidth_scale", type=float, default=None)
    parser.add_argument("--fixed_bandwidth", type=float, default=None,
                        help="Bandwidth for the data-independent 'Fixed' rule (default: from config, default 1.5)")
    parser.add_argument("--fixed_bandwidth_grid", type=float, nargs="+", default=None,
                        help="Grid search over h: pass multiple values (e.g. --fixed_bandwidth_grid 0.6 0.8 1.0). "
                             "When set, one Fixed method is reported per h value.")
    parser.add_argument("--kernel_space", type=str, default=None,
                        choices=["yhat", "pca", "x", "vae"],
                        help="Kernel feature space: yhat (1D predicted midpoint), pca (top-k PCA of X), x (raw X), vae (VAE latents)")
    parser.add_argument("--pca_components", type=int, default=None,
                        help="Number of PCA components when --kernel_space pca (default: from config)")
    def _latent_dim_arg(s: str):
        if s.strip().lower() == "auto":
            return "auto"
        return int(s)
    parser.add_argument("--latent_dim", type=_latent_dim_arg, default=None,
                        help="VAE latent dimension when --kernel_space vae: int or 'auto' (default: from config, default 'auto')")
    parser.add_argument("--vae_epochs", type=int, default=None,
                        help="VAE training epochs when --kernel_space vae (default: from config)")
    parser.add_argument("--vae_beta", type=float, default=None,
                        help="VAE KL weight (beta-VAE) when --kernel_space vae (default: from config)")
    parser.add_argument("--activation", type=str, default=None,
                        choices=["relu", "requ"],
                        help="Activation function (default: from config, fallback requ)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output")
    args = parser.parse_args()

    # Load config
    config_path = args.config if Path(args.config).exists() else None
    cfg = load_real_config(config_path)

    # Apply CLI overrides
    for key in ["alpha", "n_attempts", "hidden_dim", "train_epochs", "batch_size", "bandwidth_scale", "fixed_bandwidth", "fixed_bandwidth_grid", "activation", "kernel_space", "pca_components", "vae_epochs", "vae_beta"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val
    # CLI --latent_dim → config key vae_latent_dim
    if getattr(args, "latent_dim", None) is not None:
        cfg["vae_latent_dim"] = args.latent_dim

    activation = cfg.get("activation", "requ")
    if args.output:
        output_path = args.output
    else:
        _ks = cfg.get("kernel_space", "yhat")
        if _ks == "pca":
            _kernel_suffix = f"PCA{int(cfg.get('pca_components', 3))}"
        elif _ks == "yhat":
            _kernel_suffix = "qdiff"
        elif _ks == "vae":
            _ld = cfg.get("vae_latent_dim", "auto")
            _kernel_suffix = f"VAE{_ld}" if isinstance(_ld, int) and _ld > 0 else "VAEauto"
        else:
            _kernel_suffix = "whole"
        _grid_suffix = "_hgrid" if cfg.get("fixed_bandwidth_grid") else ""
        output_path = f"results_real_data_{activation}_{_kernel_suffix}{_grid_suffix}.csv"

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    print("=" * 70)
    print("Global CQR vs Localized CQR -- Real Data Evaluation")
    print("=" * 70)
    print(f"Model: {activation.upper()} NN (hidden={cfg['hidden_dim']}, layers={cfg['n_layers']}, "
          f"epochs={cfg['train_epochs']}, lr={cfg['learning_rate']})")
    print(f"Alpha: {cfg['alpha']} (target coverage: {1-cfg['alpha']:.0%})")
    print(f"Repetitions: {cfg['n_attempts']}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Bandwidth scale: {cfg['bandwidth_scale']}")
    if cfg.get("fixed_bandwidth_grid"):
        _grid = [float(h) for h in cfg["fixed_bandwidth_grid"]]
        print(f"Fixed-h grid: {_grid}  ({len(_grid)} values per dataset)")
    else:
        print(f"Fixed bandwidth (single): {cfg.get('fixed_bandwidth', 1.5)}")
    _ks_print = cfg.get('kernel_space', 'yhat')
    if _ks_print == 'pca':
        _ks_extra = f" (k={cfg.get('pca_components', 3)})"
    elif _ks_print == 'vae':
        _ks_extra = (f" (latent={cfg.get('vae_latent_dim', 'auto')}, "
                     f"epochs={cfg.get('vae_epochs', 100)}, "
                     f"beta={cfg.get('vae_beta', 1.0)})")
    else:
        _ks_extra = ""
    print(f"Kernel space: {_ks_print}{_ks_extra}")
    print("-" * 70)

    all_rows = []
    t_start = time.time()

    for ds_name in datasets:
        t0 = time.time()
        runs = run_dataset_evaluation(ds_name, cfg, verbose=not args.quiet)
        dt = time.time() - t0

        if len(runs) == 0:
            continue

        # Get dataset info for the table
        try:
            _, _, info = load_dataset(ds_name)
        except Exception:
            info = {"n_samples": "?", "n_features": "?"}

        rows = aggregate_results(
            ds_name, info["n_samples"], info["n_features"],
            runs, cfg["alpha"], activation=activation, cfg=cfg,
        )
        all_rows.extend(rows)

        if not args.quiet:
            print(f"  Completed {ds_name} in {dt:.1f}s")

    if len(all_rows) == 0:
        print("No results to display.")
        return

    df = pd.DataFrame(all_rows)

    # Print formatted table
    print_results_table(df)

    # Save full results to CSV
    df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}")

    total_time = time.time() - t_start
    print(f"Total elapsed time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
