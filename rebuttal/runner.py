"""
Single (dataset, seed) job for the 20-seed rebuttal sweep.

Replicates ``evaluate_single_run`` from ``evaluate_real_data.py`` step for
step — same helper functions, same operation order, same config resolution —
but additionally captures per-test-point arrays (base interval, per-method
corrections, PC1 projection, kNN residual scale) so that Task 1/3/4 tables
can be computed offline without re-running anything.

All extra computations happen AFTER the original pipeline operations, and the
two torch trainers re-seed torch.manual_seed(seed) internally, so the captured
run is numerically identical to the published pipeline.

Paper configuration (verified bit-exact against
``results_real_data_relu_VAEauto_hgrid.csv``):
    configs/real.yaml + CLI overrides
    --kernel_space vae --latent_dim auto
    --fixed_bandwidth_grid 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2
Published seeds: 42..46 (cfg seed 42 + attempt).  Rebuttal sweep: 42..61.
"""

import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent

FIXED_GRID = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]

PAPER_DATASETS = [
    "diabetes", "california_housing", "concrete", "energy", "bio",
    "community", "blog_data", "rf1", "scm1d", "scm20d", "kin8nm",
]

SEEDS = list(range(42, 62))          # 42..61; 42..46 are the published runs
KNN_K = 50                           # neighbors for the residual-scale estimate

RAW_DIR = REPO_ROOT / "results" / "rebuttal" / "raw"


def paper_config() -> Dict[str, Any]:
    """Exact config of the published Table-1 sweep (README command)."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluate_real_data import load_real_config
    cfg = load_real_config(str(REPO_ROOT / "configs" / "real.yaml"))
    cfg["kernel_space"] = "vae"
    cfg["vae_latent_dim"] = "auto"
    cfg["fixed_bandwidth_grid"] = FIXED_GRID
    cfg["verbose"] = False
    return cfg


def _knn_residual_scale(X_cal, resid_cal, X_test, k=KNN_K):
    """
    Label-free local scale estimate for each test point: mean absolute
    calibration residual of the k nearest calibration neighbors in
    standardized X space.  Uses calibration labels only — never test labels.
    """
    from sklearn.neighbors import NearestNeighbors
    k = min(k, len(X_cal))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute")
    nn.fit(X_cal)
    scale = np.empty(len(X_test), dtype=np.float64)
    chunk = 2000
    for start in range(0, len(X_test), chunk):
        end = min(start + chunk, len(X_test))
        _, idx = nn.kneighbors(X_test[start:end])
        scale[start:end] = np.abs(resid_cal)[idx].mean(axis=1)
    return scale


def run_one(dataset: str, seed: int, out_dir: Path = RAW_DIR) -> Dict[str, Any]:
    """Run one (dataset, seed) pair; save npz + json checkpoint; return metrics."""
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from cqr.real_data import load_dataset
    from cqr.preprocessing import prepare_data, inverse_transform_width
    from cqr.training import train_quantile_models_unified
    from cqr.calibration import (
        compute_conformity_scores, global_calibration, LocalConformalOptimizer,
    )
    from cqr.metrics import evaluate_intervals
    from evaluate_real_data import (
        _build_method_specs, _compute_bandwidths, _build_kernel_features,
        _resolve_latent_dim,
    )

    t0 = time.time()
    cfg = paper_config()
    # Same per-dataset overrides as run_dataset_evaluation
    overrides = cfg.get("dataset_overrides", {})
    if dataset in overrides:
        cfg = {**cfg, **overrides[dataset]}

    X, y, info = load_dataset(dataset)

    # ---- begin: mirror of evaluate_single_run (identical order) ----
    alpha = cfg["alpha"]
    tau_low, tau_high = alpha / 2, 1 - alpha / 2
    d = X.shape[1]

    if len(X) < 2000:
        fracs = (0.4, 0.3, 0.3)
    else:
        fracs = (0.3, 0.5, 0.2)

    data = prepare_data(
        X, y, train_frac=fracs[0], cal_frac=fracs[1], test_frac=fracs[2],
        seed=seed,
        clip_outliers_iqr=cfg.get("clip_outliers_iqr"),
        robust_scale_y=cfg.get("robust_scale_y", False),
    )

    X_train_t = torch.from_numpy(data["X_train"])
    Y_train_t = torch.from_numpy(data["Y_train"])

    model_lo, model_hi = train_quantile_models_unified(
        X_train_t, Y_train_t,
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

    with torch.no_grad():
        X_cal_t = torch.from_numpy(data["X_cal"])
        pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
        pred_cal_hi = model_hi(X_cal_t).numpy().flatten()
    crossed = pred_cal_lo > pred_cal_hi
    if crossed.any():
        pred_cal_lo[crossed], pred_cal_hi[crossed] = (
            pred_cal_hi[crossed].copy(), pred_cal_lo[crossed].copy())

    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, data["Y_cal"])

    with torch.no_grad():
        X_test_t = torch.from_numpy(data["X_test"])
        pred_test_lo = model_lo(X_test_t).numpy().flatten()
        pred_test_hi = model_hi(X_test_t).numpy().flatten()
    crossed = pred_test_lo > pred_test_hi
    if crossed.any():
        pred_test_lo[crossed], pred_test_hi[crossed] = (
            pred_test_hi[crossed].copy(), pred_test_lo[crossed].copy())

    Q_hat_global = global_calibration(scores, alpha)
    global_metrics = evaluate_intervals(
        data["Y_test"], pred_test_lo - Q_hat_global, pred_test_hi + Q_hat_global,
        data["X_test"], alpha=alpha, n_bins=cfg["n_cond_bins"],
    )

    kernel_space = str(cfg.get("kernel_space", "yhat"))
    with torch.no_grad():
        pred_train_lo = model_lo(X_train_t).numpy().flatten()
        pred_train_hi = model_hi(X_train_t).numpy().flatten()

    resolved_latent_dim = _resolve_latent_dim(
        cfg.get("vae_latent_dim", "auto"),
        n_cal=int(data["n_cal"]), n_features=int(d),
    )
    vae_kwargs = {
        "latent_dim": resolved_latent_dim,
        "hidden_dim": int(cfg.get("vae_hidden_dim", 64)),
        "epochs": int(cfg.get("vae_epochs", 100)),
        "beta": float(cfg.get("vae_beta", 1.0)),
        "verbose": False,
    }
    feat_train, feat_cal, feat_test, kernel_d = _build_kernel_features(
        data["X_train"], data["X_cal"], data["X_test"],
        pred_train_lo, pred_train_hi, pred_cal_lo, pred_cal_hi,
        pred_test_lo, pred_test_hi,
        kernel_space=kernel_space,
        pca_components=int(cfg.get("pca_components", 3)),
        vae_kwargs=vae_kwargs, seed=seed,
    )
    bandwidths = _compute_bandwidths(
        feat_train, d=int(kernel_d),
        bandwidth_scale=float(cfg["bandwidth_scale"]),
        fixed_bandwidth=float(cfg.get("fixed_bandwidth", 1.5)),
    )

    qhat_vectors: Dict[str, np.ndarray] = {}

    def _run_local(h_val):
        lcp = LocalConformalOptimizer(feat_cal, scores, h=h_val)
        parts = []
        for start in range(0, len(feat_test), 1000):
            end = min(start + 1000, len(feat_test))
            parts.append(lcp.predict_corrections(feat_test[start:end], alpha))
        Q_hat = np.concatenate(parts)
        m = evaluate_intervals(
            data["Y_test"], pred_test_lo - Q_hat, pred_test_hi + Q_hat,
            data["X_test"], alpha=alpha, n_bins=cfg["n_cond_bins"],
        )
        return m, Q_hat

    result: Dict[str, Any] = {"global": global_metrics}
    h_used: Dict[str, float] = {}
    for key, h in (("local_silverman", bandwidths["silverman"]),
                   ("local_scott", bandwidths["scott"]),
                   ("local_isj", bandwidths["isj"])):
        result[key], qhat_vectors[key] = _run_local(h)
        h_used[key] = float(h)
    bs = float(cfg["bandwidth_scale"])
    for h in cfg["fixed_bandwidth_grid"]:
        tag = f"{float(h):g}"
        h_scaled = max(bs * float(h), 1e-6)
        key = f"local_fixed_{tag}"
        result[key], qhat_vectors[key] = _run_local(h_scaled)
        h_used[key] = float(h_scaled)

    scaler_y = data["scaler_y"]
    method_keys = [s[0] for s in _build_method_specs(cfg)]
    for mkey in method_keys:
        md = result[mkey]
        if scaler_y is not None:
            md["avg_width_orig"] = inverse_transform_width(md["avg_width"], scaler_y)
            md["median_width_orig"] = inverse_transform_width(md["median_width"], scaler_y)
        else:
            md["avg_width_orig"] = md["avg_width"]
            md["median_width_orig"] = md["median_width"]
    # ---- end: mirror of evaluate_single_run ----

    # ---- capture-only extras (no effect on the pipeline above) ----
    # PC1 projection of X_test — identical to the binning inside
    # cqr.metrics.conditional_coverage (PCA fitted on X_test).
    from sklearn.decomposition import PCA
    Xt = np.asarray(data["X_test"])
    if Xt.ndim == 1:
        Xt = Xt.reshape(-1, 1)
    if Xt.shape[1] > 1:
        pc1 = PCA(n_components=1).fit_transform(Xt).flatten()
    else:
        pc1 = Xt[:, 0]

    resid_cal = np.abs(np.asarray(data["Y_cal"]).flatten()
                       - (pred_cal_lo + pred_cal_hi) / 2.0)
    knn_scale = _knn_residual_scale(data["X_cal"], resid_cal, data["X_test"])

    out_ds = Path(out_dir) / dataset
    out_ds.mkdir(parents=True, exist_ok=True)
    npz_path = out_ds / f"seed_{seed}.npz"
    json_path = out_ds / f"seed_{seed}.json"

    np.savez_compressed(
        npz_path,
        y_test=np.asarray(data["Y_test"]).flatten().astype(np.float64),
        base_lo=pred_test_lo.astype(np.float64),
        base_hi=pred_test_hi.astype(np.float64),
        q_global=np.float64(Q_hat_global),
        pc1=pc1.astype(np.float64),
        knn_scale=knn_scale.astype(np.float64),
        **{f"qhat_{k}": v.astype(np.float64) for k, v in qhat_vectors.items()},
    )

    def _jsonable(md):
        out = {}
        for k, v in md.items():
            if k in ("ranked_bins",):
                out[k] = v
            elif isinstance(v, (list, tuple)):
                out[k] = [None if (isinstance(x, float) and np.isnan(x)) else x for x in v]
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, float) and np.isnan(v):
                out[k] = None
            else:
                out[k] = v
        return out

    meta = {
        "dataset": dataset,
        "seed": seed,
        "n_samples": int(info["n_samples"]),
        "n_features": int(info["n_features"]),
        "n_train": int(data["n_train"]),
        "n_cal": int(data["n_cal"]),
        "n_test": int(data["n_test"]),
        "alpha": alpha,
        "activation": cfg.get("activation"),
        "hidden_dim": cfg["hidden_dim"],
        "train_epochs": cfg["train_epochs"],
        "weight_decay": cfg["weight_decay"],
        "latent_dim": int(resolved_latent_dim),
        "kernel_d": int(kernel_d),
        "knn_k": int(min(KNN_K, data["n_cal"])),
        "scaler_y_scale": float(scaler_y.scale_[0]) if scaler_y is not None else None,
        "h": h_used,
        "elapsed_sec": time.time() - t0,
        "torch_threads": torch.get_num_threads(),
        "metrics": {k: _jsonable(result[k]) for k in method_keys},
    }
    with open(json_path, "w") as f:
        json.dump(meta, f)
    return meta
