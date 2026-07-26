"""
Aggregation for the rebuttal sweep.

Modes (from repo root):
    python -m rebuttal.aggregate --validate
        Compare seeds 42..46 aggregates against the published
        results_real_data_relu_VAEauto_hgrid.csv (Overall rows) and report
        the max absolute delta per metric.

    python -m rebuttal.aggregate --write
        Write results/rebuttal/table1_20seeds.md (per-method mean +/- 95% t-CI)
        and results/rebuttal/paired_diffs.md (paired Local-Global differences).
        Datasets with fewer than the full seed set are included and flagged.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from rebuttal.runner import PAPER_DATASETS, SEEDS, RAW_DIR

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "results" / "rebuttal"
PUBLISHED_CSV = REPO_ROOT / "results_real_data_relu_VAEauto_hgrid.csv"

METHOD_LABELS = {
    "global": "Global CQR",
    "local_silverman": "Local CQR (Silverman)",
    "local_scott": "Local CQR (Scott)",
    "local_isj": "Local CQR (ISJ)",
    **{f"local_fixed_{h:g}": f"Local CQR (Fixed h={h:.2f})"
       for h in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]},
}
METHOD_ORDER = list(METHOD_LABELS.keys())

# metric key in stored json -> (display name, improvement direction)
# direction: +1 = higher is better, -1 = lower is better, 0 = closer to target
METRICS = {
    "coverage":       ("Coverage", 0),
    "worst_bin_cov":  ("WGC", +1),
    "avg_width":      ("Avg Width", -1),
    "winkler_score":  ("Winkler", -1),
}
TARGET_COV = 0.90


def load_seed_metrics(dataset: str) -> Dict[int, dict]:
    """Load per-seed metrics json for one dataset. Returns {seed: meta}."""
    out = {}
    ds_dir = RAW_DIR / dataset
    if not ds_dir.exists():
        return out
    for p in sorted(ds_dir.glob("seed_*.json")):
        with open(p) as f:
            meta = json.load(f)
        out[meta["seed"]] = meta
    return out


def metric_matrix(seed_metas: Dict[int, dict], method: str, metric: str,
                  seeds: List[int]) -> np.ndarray:
    return np.array([seed_metas[s]["metrics"][method][metric] for s in seeds],
                    dtype=float)


# ---------------------------------------------------------------------------
# Validation vs published CSV (seeds 42..46)
# ---------------------------------------------------------------------------

def validate() -> bool:
    pub = pd.read_csv(PUBLISHED_CSV)
    pub = pub[pub["Bin"] == "Overall"]
    pub_cols = {
        "coverage": ("Coverage (mean)", "Coverage (std)"),
        "avg_width": ("Avg Width (mean)", "Avg Width (std)"),
        "worst_bin_cov": ("Worst-Bin Cov (mean)", "Worst-Bin Cov (std)"),
        "winkler_score": ("Winkler Score (mean)", "Winkler Score (std)"),
    }
    ok = True
    print(f"{'dataset':20s} {'seeds':6s} max|delta mean|  max|delta std|")
    overall_max = 0.0
    for ds in PAPER_DATASETS:
        metas = load_seed_metrics(ds)
        seeds = [s for s in range(42, 47) if s in metas]
        if len(seeds) < 5:
            print(f"{ds:20s} {len(seeds)}/5    (incomplete, skipped)")
            continue
        dmax_mean = dmax_std = 0.0
        for mkey in METHOD_ORDER:
            label = METHOD_LABELS[mkey]
            row = pub[(pub["Dataset"] == ds) & (pub["Method"] == label)]
            if row.empty:
                print(f"  !! {ds} / {label}: not in published CSV")
                ok = False
                continue
            row = row.iloc[0]
            for metric, (cm, cs) in pub_cols.items():
                vals = metric_matrix(metas, mkey, metric, seeds)
                dmax_mean = max(dmax_mean, abs(np.mean(vals) - row[cm]))
                dmax_std = max(dmax_std, abs(np.std(vals) - row[cs]))  # ddof=0, CSV convention
        overall_max = max(overall_max, dmax_mean, dmax_std)
        flag = "" if dmax_mean < 1e-6 else "  <-- CHECK"
        print(f"{ds:20s} 5/5    {dmax_mean:.3e}        {dmax_std:.3e}{flag}")
        if dmax_mean > 1e-6:
            ok = False
    print(f"\nOverall max |delta| = {overall_max:.3e} "
          f"({'PASS: within float-reduction noise' if ok else 'MISMATCH'})")
    return ok


# ---------------------------------------------------------------------------
# Task 1 tables
# ---------------------------------------------------------------------------

def _mean_ci(vals: np.ndarray):
    n = len(vals)
    m = float(np.mean(vals))
    if n < 2:
        return m, 0.0
    s = float(np.std(vals, ddof=1))
    t = float(stats.t.ppf(0.975, df=n - 1))
    return m, t * s / np.sqrt(n)


def _improves(diffs: np.ndarray, direction: int,
              g: np.ndarray, l: np.ndarray) -> int:
    if direction == +1:
        return int(np.sum(diffs > 0))
    if direction == -1:
        return int(np.sum(diffs < 0))
    return int(np.sum(np.abs(l - TARGET_COV) < np.abs(g - TARGET_COV)))


def write_tables(out_dir: Path = OUT_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    t1_lines: List[str] = []
    pd_lines: List[str] = []

    t1_lines.append(
        "Per-method results over all completed seeds (planned range 42-141; seeds "
        "42-46 are identical to the published 5-seed runs; per-dataset seed count "
        "n is stated in each header). Cells are mean +/- 95% t-CI half-width "
        "(t_{0.975,n-1} * sd/sqrt(n), sd with ddof=1; n per dataset header). "
        "Coverage target 0.90. WGC = worst-bin coverage over 5 equal-frequency "
        "bins of the first principal component of X_test (paper definition, "
        "unchanged). Width in standardized-y units. Winkler at alpha=0.1.\n")
    pd_lines.append(
        "Paired seed-wise differences d_s = metric(Local, s) - metric(Global, s) "
        "over all completed seeds (planned range 42-141; n stated per dataset), "
        "same split and base regressors shared per seed. "
        "Cells are mean(d) +/- t_{0.975,n-1} * sd(d)/sqrt(n) (ddof=1), followed by "
        "#seeds where Local improves on Global. Improvement directions: "
        "Coverage = |cov - 0.90| smaller; WGC = higher; Avg Width = lower; "
        "Winkler = lower. A CI excluding 0 indicates a significant paired "
        "difference at the 5% level.\n")

    for ds in PAPER_DATASETS:
        metas = load_seed_metrics(ds)
        seeds = sorted(metas)
        if not seeds:
            continue
        n = len(seeds)
        note = "" if n >= len(SEEDS) else f"  (PARTIAL: {n}/{len(SEEDS)} planned seeds)"
        m0 = metas[seeds[0]]
        hdr = (f"## {ds} (n={m0['n_samples']}, d={m0['n_features']}, "
               f"n_seeds={n}, seeds {seeds[0]}-{seeds[-1]}){note}\n")

        # ---- table1: per-method mean +/- CI ----
        t1_lines.append(hdr)
        cols = " | ".join(name for name, _ in METRICS.values())
        t1_lines.append(f"| Method | {cols} |")
        t1_lines.append("|" + "---|" * (len(METRICS) + 1))
        for mkey in METHOD_ORDER:
            cells = []
            for metric in METRICS:
                m, hw = _mean_ci(metric_matrix(metas, mkey, metric, seeds))
                cells.append(f"{m:.3f} ± {hw:.3f}")
            t1_lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
        t1_lines.append("")

        # ---- paired diffs vs Global ----
        pd_lines.append(hdr)
        pd_lines.append(f"| Method (vs Global) | {cols} |")
        pd_lines.append("|" + "---|" * (len(METRICS) + 1))
        for mkey in METHOD_ORDER:
            if mkey == "global":
                continue
            cells = []
            for metric, (_, direction) in METRICS.items():
                g = metric_matrix(metas, "global", metric, seeds)
                l = metric_matrix(metas, mkey, metric, seeds)
                d = l - g
                m, hw = _mean_ci(d)
                k = _improves(d, direction, g, l)
                cells.append(f"{m:+.3f} ± {hw:.3f} ({k}/{n})")
            pd_lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
        pd_lines.append("")

    (out_dir / "table1_20seeds.md").write_text("\n".join(t1_lines), encoding="utf-8")
    (out_dir / "paired_diffs.md").write_text("\n".join(pd_lines), encoding="utf-8")
    print(f"wrote {out_dir / 'table1_20seeds.md'}")
    print(f"wrote {out_dir / 'paired_diffs.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    if args.validate:
        validate()
    if args.write:
        write_tables()


if __name__ == "__main__":
    main()
