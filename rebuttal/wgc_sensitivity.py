"""
Task 3 — WGC bin-sensitivity, computed offline from the captured per-point
arrays (results/rebuttal/raw/<ds>/seed_<s>.npz).  Nothing here re-runs
training or touches the paper's metric code.

Grouping rules evaluated (WGC = worst per-bin coverage, bins with < 20 test
points excluded, matching the paper's min_bin_size):

  pc1_K{3,5,10,20}   equal-frequency bins of the first principal component of
                     X_test — the rule actually implemented in
                     cqr/metrics.py (percentile edges, mirrored exactly;
                     pc1_K5 reproduces the paper's WGC).
  width_K{3,5,10,20} equal-frequency (rank-based) bins of the UNCALIBRATED
                     base interval width (base_hi - base_lo) on test points.
  knn_K5             quintiles of a kNN residual-scale estimate: mean
                     absolute calibration residual (|y_cal - midpoint|) of
                     the k=50 nearest calibration points in standardized X
                     space.  Label-free at test time: uses calibration labels
                     only, never test labels.

Output: results/rebuttal/wgc_sensitivity.md
Usage:  python -m rebuttal.wgc_sensitivity
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

from rebuttal.runner import PAPER_DATASETS, SEEDS, RAW_DIR
from rebuttal.aggregate import METHOD_LABELS, METHOD_ORDER, load_seed_metrics

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "rebuttal" / "wgc_sensitivity.md"

MIN_BIN_SIZE = 20  # same filter as cqr.metrics.conditional_coverage
K_GRID = [3, 5, 10, 20]

RULES: List[str] = (
    [f"pc1_K{k}" for k in K_GRID]
    + [f"width_K{k}" for k in K_GRID]
    + ["knn_K5"]
)


def _percentile_bins(proj: np.ndarray, k: int) -> List[np.ndarray]:
    """Mirror cqr.metrics.conditional_coverage binning (percentile edges)."""
    edges = np.percentile(proj, np.linspace(0, 100, k + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return [(proj >= edges[b]) & (proj < edges[b + 1]) for b in range(k)]


def _rank_bins(valsy: np.ndarray, k: int) -> List[np.ndarray]:
    """Equal-count bins by rank (ties broken by stable sort order)."""
    order = np.argsort(valsy, kind="stable")
    masks = []
    for chunk in np.array_split(order, k):
        m = np.zeros(len(valsy), dtype=bool)
        m[chunk] = True
        masks.append(m)
    return masks


def _bin_masks(z, rule: str) -> List[np.ndarray]:
    kind, k = rule.rsplit("_K", 1)
    k = int(k)
    if kind == "pc1":
        return _percentile_bins(z["pc1"], k)
    if kind == "width":
        return _rank_bins(z["base_hi"] - z["base_lo"], k)
    if kind == "knn":
        return _rank_bins(z["knn_scale"], k)
    raise ValueError(rule)


def _covered(z, method: str) -> np.ndarray:
    y, lo, hi = z["y_test"], z["base_lo"], z["base_hi"]
    q = float(z["q_global"]) if method == "global" else z[f"qhat_{method}"]
    return (y >= lo - q) & (y <= hi + q)


def wgc(z, method: str, rule: str) -> float:
    cov = _covered(z, method)
    vals = [float(np.mean(cov[m])) for m in _bin_masks(z, rule)
            if int(m.sum()) >= MIN_BIN_SIZE]
    return min(vals) if vals else np.nan


def bin_profile(z, method: str, rule: str) -> List[float]:
    """Per-bin coverage ordered by the grouping variable (low -> high)."""
    cov = _covered(z, method)
    return [float(np.mean(cov[m])) if m.sum() > 0 else np.nan
            for m in _bin_masks(z, rule)]


def _mean_ci(vals: np.ndarray):
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n == 0:
        return np.nan, np.nan, 0
    m = float(np.mean(vals))
    if n < 2:
        return m, 0.0, n
    s = float(np.std(vals, ddof=1))
    t = float(stats.t.ppf(0.975, df=n - 1))
    return m, t * s / np.sqrt(n), n


def main():
    lines: List[str] = []
    lines.append(
        "WGC bin-sensitivity over 20 seeds (42-61; 42-46 identical to the "
        "published runs). WGC = worst per-bin coverage; bins with < 20 test "
        "points are excluded (paper's min_bin_size), '-' = no bin qualifies. "
        "Grouping rules: pc1_K = equal-frequency bins of the first principal "
        "component of X_test (the paper's implemented rule; pc1_K5 = published "
        "WGC), width_K = equal-frequency bins of the uncalibrated base interval "
        "width, knn_K5 = quintiles of a kNN residual-scale estimate (mean "
        "|calibration residual| of the 50 nearest calibration points in "
        "standardized X space; label-free at test time). Paired diffs are "
        "mean(d) +/- t_{0.975,n-1} sd(d)/sqrt(n) with d_s = WGC(Local,s) - "
        "WGC(Global,s), and (k/n) = seeds where Local's WGC is higher.\n")

    # cache: {ds: {seed: npz-dict}}
    summary_counts: Dict[str, Dict[str, tuple]] = {}

    for ds in PAPER_DATASETS:
        metas = load_seed_metrics(ds)
        seeds = [s for s in SEEDS if s in metas]
        if not seeds:
            continue
        n = len(seeds)
        zs = {}
        for s in seeds:
            with np.load(RAW_DIR / ds / f"seed_{s}.npz") as f:
                zs[s] = {k: f[k] for k in f.files}

        m0 = metas[seeds[0]]
        note = "" if n == len(SEEDS) else f"  (INCOMPLETE: {n}/{len(SEEDS)} seeds)"
        lines.append(f"# {ds} (n={m0['n_samples']}, d={m0['n_features']}, "
                     f"n_seeds={n}){note}\n")

        # WGC per (method, rule): array over seeds
        W = {m: {r: np.array([wgc(zs[s], m, r) for s in seeds])
                 for r in RULES} for m in METHOD_ORDER}

        # sanity: pc1_K5 must reproduce the stored paper WGC
        stored = np.array([metas[s]["metrics"]["global"]["worst_bin_cov"]
                           for s in seeds])
        if np.nanmax(np.abs(W["global"]["pc1_K5"] - stored)) > 1e-9:
            lines.append("**WARNING: pc1_K5 does not reproduce the stored "
                         "paper WGC — investigate before posting.**\n")

        # ---- per-method WGC table ----
        lines.append("## Per-method WGC (mean ± 95% t-CI)\n")
        lines.append("| Method | " + " | ".join(RULES) + " |")
        lines.append("|" + "---|" * (len(RULES) + 1))
        for mkey in METHOD_ORDER:
            cells = []
            for r in RULES:
                m, hw, nn = _mean_ci(W[mkey][r])
                cells.append("-" if np.isnan(m) else f"{m:.3f} ± {hw:.3f}")
            lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
        lines.append("")

        # ---- paired diffs table ----
        lines.append("## Paired Local − Global WGC differences\n")
        lines.append("| Method (vs Global) | " + " | ".join(RULES) + " |")
        lines.append("|" + "---|" * (len(RULES) + 1))
        counts: Dict[str, tuple] = {}
        for mkey in METHOD_ORDER:
            if mkey == "global":
                continue
            cells = []
            n_rules_improved = 0
            n_rules_valid = 0
            for r in RULES:
                d = W[mkey][r] - W["global"][r]
                m, hw, nn = _mean_ci(d)
                if np.isnan(m):
                    cells.append("-")
                    continue
                k = int(np.nansum(d > 0))
                cells.append(f"{m:+.3f} ± {hw:.3f} ({k}/{nn})")
                n_rules_valid += 1
                if m > 0:
                    n_rules_improved += 1
            counts[mkey] = (n_rules_improved, n_rules_valid)
            lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
        lines.append("")
        summary_counts[ds] = counts

        # ---- K=5 coverage profiles ----
        lines.append("## Groupwise coverage profile, K=5 "
                     "(mean coverage per bin over seeds; bins ordered low → high "
                     "grouping variable)\n")
        for r in ["pc1_K5", "width_K5", "knn_K5"]:
            lines.append(f"### {r}\n")
            lines.append("| Method | " + " | ".join(f"Bin {b+1}" for b in range(5)) + " |")
            lines.append("|" + "---|" * 6)
            for mkey in METHOD_ORDER:
                P = np.array([bin_profile(zs[s], mkey, r) for s in seeds])
                cells = [f"{np.nanmean(P[:, b]):.3f}" for b in range(5)]
                lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
            lines.append("")

        # ---- summary lines ----
        lines.append("## Summary\n")
        for mkey, (ki, kv) in counts.items():
            lines.append(f"- {ds} / {METHOD_LABELS[mkey]}: improves WGC "
                         f"(mean paired diff > 0) under {ki}/{kv} grouping rules")
        lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
