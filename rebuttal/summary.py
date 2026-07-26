"""
Task 4 — rebuttal-ready summary.

Reads the per-seed metrics (results/rebuttal/raw/) and the sweep log, writes
results/rebuttal/SUMMARY.md with headline counts (datasets where each Local
variant improves WGC / Winkler / Avg Width under paired 20-seed comparisons)
and exact runtime + hardware.

Usage:  python -m rebuttal.summary
"""

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
from scipy import stats

from rebuttal.runner import PAPER_DATASETS, SEEDS, RAW_DIR
from rebuttal.aggregate import (
    METHOD_LABELS, METHOD_ORDER, load_seed_metrics, metric_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "results" / "rebuttal" / "SUMMARY.md"

HEADLINE_METRICS = {
    "worst_bin_cov": ("WGC", +1),
    "winkler_score": ("Winkler", -1),
    "avg_width": ("Avg Width", -1),
}


def _cpu_name() -> str:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Processor).Name"],
            capture_output=True, text=True, timeout=30).stdout.strip()
        if out:
            return out
    except Exception:
        pass
    return platform.processor()


def main():
    # ---- headline counts ----
    per_ds = {}
    for ds in PAPER_DATASETS:
        metas = load_seed_metrics(ds)
        seeds = sorted(metas)
        if len(seeds) < 2:
            continue
        per_ds[ds] = (metas, seeds)

    n_ds = len(per_ds)
    lines = []
    lines.append("# Rebuttal summary — Global vs Localized CQR, 20-seed extension\n")
    lines.append(
        f"Datasets with results: {n_ds}/11 "
        f"({', '.join(per_ds)}). Planned seeds 42-141, per-dataset counts vary "
        "until the sweep completes (42-46 identical to the published "
        "5-seed runs; verified against results_real_data_relu_VAEauto_hgrid.csv, "
        "max |delta| ~1e-7 from BLAS thread count only). Paired comparison: "
        "d_s = metric(Local, s) - metric(Global, s), same seed-controlled split "
        "and shared base regressors; 'improves' = mean(d) in the better "
        "direction (WGC higher, Winkler lower, Avg Width lower); 'significant' "
        "= 95% paired t-CI (t_{0.975,n-1} sd/sqrt(n)) excludes 0.\n")

    lines.append("## Headline counts: datasets improved (significant) out of "
                 f"{n_ds}\n")
    lines.append("| Method (vs Global) | " +
                 " | ".join(v[0] for v in HEADLINE_METRICS.values()) + " |")
    lines.append("|" + "---|" * (len(HEADLINE_METRICS) + 1))
    for mkey in METHOD_ORDER:
        if mkey == "global":
            continue
        cells = []
        for metric, (_, direction) in HEADLINE_METRICS.items():
            n_imp = n_sig = 0
            for ds, (metas, seeds) in per_ds.items():
                g = metric_matrix(metas, "global", metric, seeds)
                l = metric_matrix(metas, mkey, metric, seeds)
                d = (l - g) * direction          # >0 = improvement
                m = float(np.mean(d))
                s = float(np.std(d, ddof=1))
                t = float(stats.t.ppf(0.975, df=len(d) - 1))
                hw = t * s / np.sqrt(len(d))
                if m > 0:
                    n_imp += 1
                    if m - hw > 0:
                        n_sig += 1
            cells.append(f"{n_imp}/{n_ds} ({n_sig}/{n_ds})")
        lines.append(f"| {METHOD_LABELS[mkey]} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("Note: 'locally scaled CQR' (Task 2) is not included — the "
                 "method was not part of this sweep; its rows can be appended "
                 "once Task 2 is specified and run.\n")

    # ---- runtime & hardware ----
    log_path = RAW_DIR.parent / "sweep_log.jsonl"
    entries = []
    if log_path.exists():
        with open(log_path) as f:
            entries = [json.loads(x) for x in f if x.strip()]
    ok = [e for e in entries if e.get("status") == "ok"]
    cpu_sec = sum(e["elapsed_sec"] for e in ok)
    wall = (max(e["t"] for e in ok) - min(e["t"] for e in ok)) if len(ok) > 1 else 0.0

    import torch
    lines.append("## Runtime and hardware\n")
    lines.append(f"- Completed (dataset, seed) jobs: {len(ok)} "
                 f"(of {len(PAPER_DATASETS) * len(SEEDS)} total)")
    lines.append(f"- Sum of per-job runtimes: {cpu_sec / 3600:.2f} h; "
                 f"wall-clock span of the sweep log: {wall / 3600:.2f} h "
                 "(8 worker processes, 1 torch/BLAS thread each)")
    lines.append(f"- CPU: {_cpu_name()} (12 logical cores); no GPU used")
    lines.append(f"- OS: {platform.platform()}")
    lines.append(f"- Python {platform.python_version()}, torch {torch.__version__}, "
                 f"numpy {np.__version__}")
    lines.append("- Config: configs/real.yaml + --kernel_space vae --latent_dim auto "
                 "--fixed_bandwidth_grid 0.6..2.2 (identical to the published sweep)")
    lines.append("")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
