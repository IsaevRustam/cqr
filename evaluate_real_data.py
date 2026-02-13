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
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from cqr.real_data import load_dataset, DEFAULT_DATASETS, list_datasets
from cqr.preprocessing import prepare_data, inverse_transform_width
from cqr.models_requ import train_requ_quantile_models
from cqr.calibration import (
    compute_conformity_scores,
    global_calibration,
    LocalConformalOptimizer,
    compute_bandwidth,
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
        "n_attempts": 10,
        "hidden_dim": 128,
        "train_epochs": 500,
        "learning_rate": 0.001,
        "batch_size": 256,
        "weight_decay": 1e-5,
        "n_layers": 2,
        "bandwidth_scale": 1.0,
        "n_cond_bins": 5,
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

    # Split data
    data = prepare_data(X, y, train_frac=0.4, cal_frac=0.3, test_frac=0.3, seed=seed)

    X_train = torch.from_numpy(data["X_train"])
    Y_train = torch.from_numpy(data["Y_train"])

    # Train ReQU quantile models
    model_lo, model_hi = train_requ_quantile_models(
        X_train, Y_train,
        tau_low=tau_low, tau_high=tau_high,
        input_dim=d,
        hidden_dim=cfg["hidden_dim"],
        n_layers=cfg["n_layers"],
        epochs=cfg["train_epochs"],
        lr=cfg["learning_rate"],
        batch_size=cfg["batch_size"],
        weight_decay=cfg["weight_decay"],
        verbose=cfg.get("verbose", True),
    )

    # Predict on calibration set
    with torch.no_grad():
        X_cal_t = torch.from_numpy(data["X_cal"])
        pred_cal_lo = model_lo(X_cal_t).numpy().flatten()
        pred_cal_hi = model_hi(X_cal_t).numpy().flatten()

    # Conformity scores
    scores = compute_conformity_scores(pred_cal_lo, pred_cal_hi, data["Y_cal"])

    # Predict on test set
    with torch.no_grad():
        X_test_t = torch.from_numpy(data["X_test"])
        pred_test_lo = model_lo(X_test_t).numpy().flatten()
        pred_test_hi = model_hi(X_test_t).numpy().flatten()

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
    # LOCALIZED CQR
    # =========================================================================
    m = data["n_cal"]

    # Compute bandwidth using Silverman's rule of thumb scaled by the
    # actual data spread (median pairwise distance).  This adapts to both
    # dimensionality d and calibration size m, avoiding the one-size-fits-all
    # pitfall of a fixed fraction of median_dist.
    #   h_silverman = (4/(d+2))^{1/(d+4)} * m^{-1/(d+4)}   (rate for density)
    #   h = bandwidth_scale * h_silverman * median_dist      (data scale)
    from sklearn.metrics import pairwise_distances as _pw_dist
    d = data["X_train"].shape[1]
    n_sub = min(m, 2000)
    rng_bw = np.random.RandomState(seed)
    idx_sub = rng_bw.choice(m, n_sub, replace=False) if m > n_sub else np.arange(m)
    D_sub = _pw_dist(data["X_cal"][idx_sub])
    median_dist = float(np.median(D_sub[np.triu_indices(n_sub, k=1)]))
    h_silverman = ((4.0 / (d + 2)) ** (1.0 / (d + 4))) * (m ** (-1.0 / (d + 4)))
    h = cfg["bandwidth_scale"] * h_silverman * median_dist
    h = max(h, 1e-6)

    lcp = LocalConformalOptimizer(data["X_cal"], scores, h=h)

    # Predict corrections in batches to manage memory
    batch_size_pred = 1000
    Q_hat_local_parts = []
    for start in range(0, len(data["X_test"]), batch_size_pred):
        end = min(start + batch_size_pred, len(data["X_test"]))
        X_batch = data["X_test"][start:end]
        q_part = lcp.predict_corrections(X_batch, alpha)
        Q_hat_local_parts.append(q_part)
    Q_hat_local = np.concatenate(Q_hat_local_parts)

    local_lo = pred_test_lo - Q_hat_local
    local_hi = pred_test_hi + Q_hat_local

    local_metrics = evaluate_intervals(
        data["Y_test"], local_lo, local_hi, data["X_test"],
        alpha=alpha, n_bins=cfg["n_cond_bins"],
    )

    # Convert widths to original scale if target was standardized
    scaler_y = data["scaler_y"]
    if scaler_y is not None:
        for m_dict in [global_metrics, local_metrics]:
            m_dict["avg_width_orig"] = inverse_transform_width(m_dict["avg_width"], scaler_y)
            m_dict["median_width_orig"] = inverse_transform_width(m_dict["median_width"], scaler_y)
    else:
        for m_dict in [global_metrics, local_metrics]:
            m_dict["avg_width_orig"] = m_dict["avg_width"]
            m_dict["median_width_orig"] = m_dict["median_width"]

    return {
        "global": global_metrics,
        "local": local_metrics,
        "h": h,
        "Q_hat_global": Q_hat_global,
    }


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
        print(f"  n={info['n_samples']}, d={info['n_features']}, {info['description']}")

    n_attempts = cfg["n_attempts"]
    base_seed = cfg["seed"]
    results = []

    for attempt in range(n_attempts):
        seed = base_seed + attempt
        try:
            res = evaluate_single_run(X, y, cfg, seed)
            results.append(res)
            if verbose:
                g = res["global"]
                l = res["local"]
                print(
                    f"  Run {attempt+1}/{n_attempts}: "
                    f"Global cov={g['coverage']:.3f} w={g['avg_width']:.3f} "
                    f"worst={g['worst_bin_cov']:.3f} | "
                    f"Local cov={l['coverage']:.3f} w={l['avg_width']:.3f} "
                    f"worst={l['worst_bin_cov']:.3f} "
                    f"(h={res['h']:.3f})"
                )
        except Exception as e:
            print(f"  Run {attempt+1}/{n_attempts}: FAILED — {e}")

    # Print summary after all runs for this dataset
    if verbose and len(results) > 0:
        from scipy import stats as _stats
        alpha = cfg["alpha"]
        n_runs = len(results)
        # 95% CI half-width: t_{0.025, n-1} * std / sqrt(n)
        t_val = _stats.t.ppf(0.975, df=max(n_runs - 1, 1))

        def ci(values):
            m, s = np.mean(values), np.std(values, ddof=1) if len(values) > 1 else 0.0
            hw = t_val * s / np.sqrt(n_runs)
            return m, hw

        print(f"\n  --- Summary for {name} ({n_runs}/{n_attempts} successful runs, 95% CI) ---")
        print(f"  {'':18s} {'Coverage':>16s} {'Avg Width':>16s} {'Worst-Bin':>16s}")
        for method_key, method_label in [("global", "Global CQR"), ("local", "Local CQR")]:
            covs = [r[method_key]["coverage"] for r in results]
            widths = [r[method_key]["avg_width"] for r in results]
            worst = [r[method_key]["worst_bin_cov"] for r in results]
            c_m, c_h = ci(covs)
            w_m, w_h = ci(widths)
            wb_m, wb_h = ci(worst)
            print(
                f"  {method_label:18s} "
                f"{c_m:.3f}±{c_h:.3f}  "
                f"{w_m:.3f}±{w_h:.3f}  "
                f"{wb_m:.3f}±{wb_h:.3f}"
            )
        # Width comparison
        g_w = np.mean([r["global"]["avg_width"] for r in results])
        l_w = np.mean([r["local"]["avg_width"] for r in results])
        diff_pct = (l_w - g_w) / g_w * 100
        sym = "narrower" if diff_pct < 0 else "wider"
        print(f"  Local is {abs(diff_pct):.1f}% {sym} than Global (target cov={1-alpha:.0%})")

        # Per-bin breakdown (averaged across runs, ranked by difficulty)
        from collections import defaultdict
        for method_key, method_label in [("global", "Global CQR"), ("local", "Local CQR")]:
            # Collect ranked_bins from all runs
            all_ranked = [r[method_key]["ranked_bins"] for r in results]
            # Aggregate per bin_id across runs
            bin_agg = defaultdict(lambda: {"coverages": [], "avg_widths": [], "counts": []})
            for run_ranked in all_ranked:
                for entry in run_ranked:
                    bid = entry["bin_id"]
                    bin_agg[bid]["coverages"].append(entry["coverage"])
                    bin_agg[bid]["avg_widths"].append(entry["avg_width"])
                    bin_agg[bid]["counts"].append(entry["count"])

            if len(bin_agg) == 0:
                continue

            # Build summary per bin, sort by mean coverage ascending
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

            print(f"\n  {method_label} — Bins ranked by difficulty (hardest \u2192 easiest):")
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
    for method_key, method_label in [("global", "Global CQR"), ("local", "Localized CQR")]:
        coverages = [r[method_key]["coverage"] for r in runs]
        avg_widths = [r[method_key]["avg_width"] for r in runs]
        median_widths = [r[method_key]["median_width"] for r in runs]
        width_stds = [r[method_key]["width_std"] for r in runs]
        worst_bins = [r[method_key]["worst_bin_cov"] for r in runs]
        best_bins = [r[method_key]["best_bin_cov"] for r in runs]
        ranges = [r[method_key]["coverage_range"] for r in runs]
        avg_widths_orig = [r[method_key]["avg_width_orig"] for r in runs]

        # Overall summary row
        rows.append({
            "Dataset": dataset_name,
            "n": n_samples,
            "d": n_features,
            "Method": method_label,
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
    "Dataset", "n", "d", "Method", "Bin", "Bin ID", "Bin Rank", "Target Coverage",
    "Coverage (mean)", "Coverage (std)",
    "Avg Width (mean)", "Avg Width (std)", "Avg Width (orig)",
    "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)",
    "Cov Range",
    "Bin Count (mean)",
]


def print_results_table(df: pd.DataFrame) -> None:
    """Print a detailed comparison table (Overall rows only)."""
    # Filter to Overall rows for the display table
    display_df = df[df["Bin"] == "Overall"].copy()
    display_cols = [
        "Dataset", "n", "d", "Method", "Target Coverage",
        "Coverage (mean)", "Coverage (std)",
        "Avg Width (mean)", "Avg Width (std)", "Avg Width (orig)",
        "Worst-Bin Cov (mean)", "Worst-Bin Cov (std)",
        "Cov Range",
    ]
    display = display_df[display_cols].copy()
    # Format numeric columns to 3 decimal places for readability
    float_cols = [c for c in display_cols if c not in ("Dataset", "n", "d", "Method", "Target Coverage")]
    for c in float_cols:
        display[c] = display[c].map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

    print("\n" + "=" * 150)
    print("RESULTS: Global CQR vs Localized CQR (ReQU Neural Network) — Overall Summary")
    print("=" * 150)
    print(display.to_string(index=False))
    print("=" * 150)

    # Print summary comparison
    print("\n--- SUMMARY ---")
    for dataset in display_df["Dataset"].unique():
        sub = display_df[display_df["Dataset"] == dataset]
        if len(sub) < 2:
            continue
        g = sub[sub["Method"] == "Global CQR"].iloc[0]
        l = sub[sub["Method"] == "Localized CQR"].iloc[0]

        width_diff = ((l["Avg Width (mean)"] - g["Avg Width (mean)"]) / g["Avg Width (mean)"]) * 100

        symbol = "↓" if width_diff < 0 else "↑"
        print(
            f"  {dataset:20s}: "
            f"Local width {symbol} {abs(width_diff):5.1f}% vs Global"
        )

    print("\n--- PER-BIN BREAKDOWN ---")
    print("Detailed per-bin results are included in the full CSV output.")
    print("Each dataset has bin-level rows ranked by difficulty (hardest → easiest).")
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
    parser.add_argument("--output", type=str, default="results_real_data.csv", help="Output CSV path")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-run output")
    args = parser.parse_args()

    # Load config
    config_path = args.config if Path(args.config).exists() else None
    cfg = load_real_config(config_path)

    # Apply CLI overrides
    for key in ["alpha", "n_attempts", "hidden_dim", "train_epochs", "batch_size", "bandwidth_scale"]:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS

    print("=" * 70)
    print("Global CQR vs Localized CQR — Real Data Evaluation")
    print("=" * 70)
    print(f"Model: ReQU NN (hidden={cfg['hidden_dim']}, layers={cfg['n_layers']}, "
          f"epochs={cfg['train_epochs']}, lr={cfg['learning_rate']})")
    print(f"Alpha: {cfg['alpha']} (target coverage: {1-cfg['alpha']:.0%})")
    print(f"Repetitions: {cfg['n_attempts']}")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Bandwidth scale: {cfg['bandwidth_scale']}")
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
            runs, cfg["alpha"],
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
    output_path = args.output
    df.to_csv(output_path, index=False)
    print(f"Full results saved to: {output_path}")

    total_time = time.time() - t_start
    print(f"Total elapsed time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
