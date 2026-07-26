"""
Checkpointed parallel driver for the 20-seed rebuttal sweep.

Usage (from repo root):
    python -m rebuttal.sweep                      # full sweep, 11 datasets x seeds 42..61
    python -m rebuttal.sweep --datasets diabetes  # subset
    python -m rebuttal.sweep --seeds 42 43        # subset
    python -m rebuttal.sweep --workers 8

Each (dataset, seed) job writes results/rebuttal/raw/<ds>/seed_<s>.npz + .json;
completed jobs are skipped on restart, so a crash resumes where it left off.
Datasets are processed smallest-first so intermediate tables can be produced
per completed dataset while the big ones are still running.

Workers pin torch/BLAS to 1 thread.  Thread count only affects float reduction
order (observed max |delta| ~1e-7 vs the published 10-thread run; coverage
indicators are unchanged).  Deltas vs the published CSV are reported by
rebuttal.aggregate --validate.
"""

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# BLAS/OMP threads must be pinned before numpy/torch import in the children.
# ProcessPoolExecutor(spawn) children inherit the parent's environment, so we
# set it here at module import time (this module is re-imported in children).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.runner import PAPER_DATASETS, SEEDS, RAW_DIR, run_one  # noqa: E402

# Smallest-first so early datasets finish quickly and can be reported.
DATASET_ORDER = [
    "diabetes", "energy", "concrete", "community", "kin8nm",
    "rf1", "scm1d", "scm20d", "california_housing", "bio", "blog_data",
]


def _job(dataset: str, seed: int) -> dict:
    import torch
    torch.set_num_threads(1)
    return run_one(dataset, seed)


def _done(dataset: str, seed: int) -> bool:
    j = RAW_DIR / dataset / f"seed_{seed}.json"
    n = RAW_DIR / dataset / f"seed_{seed}.npz"
    if not (j.exists() and n.exists()):
        return False
    try:
        with open(j) as f:
            json.load(f)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    datasets = [d for d in DATASET_ORDER if d in (args.datasets or PAPER_DATASETS)]
    seeds = args.seeds or SEEDS

    jobs = [(d, s) for d in datasets for s in seeds if not _done(d, s)]
    total = len(datasets) * len(seeds)
    print(f"[sweep] {total} jobs total, {total - len(jobs)} already done, "
          f"{len(jobs)} to run, {args.workers} workers, seeds {min(seeds)}..{max(seeds)}",
          flush=True)
    if not jobs:
        print("[sweep] nothing to do"); return

    log_path = RAW_DIR.parent / "sweep_log.jsonl"
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_fail = 0
    per_ds_remaining = {d: sum(1 for dd, _ in jobs if dd == d) for d in datasets}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, d, s): (d, s) for d, s in jobs}
        for fut in as_completed(futs):
            d, s = futs[fut]
            try:
                meta = fut.result()
                n_ok += 1
                per_ds_remaining[d] -= 1
                done_ds = " <-- DATASET COMPLETE" if per_ds_remaining[d] == 0 else ""
                print(f"[sweep] ok  {d} seed {s}  ({meta['elapsed_sec']:.1f}s)  "
                      f"[{n_ok + n_fail}/{len(jobs)}]{done_ds}", flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"dataset": d, "seed": s, "status": "ok",
                                        "elapsed_sec": meta["elapsed_sec"],
                                        "t": time.time()}) + "\n")
            except Exception as e:
                n_fail += 1
                print(f"[sweep] FAIL {d} seed {s}: {e}", flush=True)
                traceback.print_exc()
                with open(log_path, "a") as f:
                    f.write(json.dumps({"dataset": d, "seed": s, "status": "fail",
                                        "error": str(e), "t": time.time()}) + "\n")

    print(f"[sweep] finished: {n_ok} ok, {n_fail} failed, "
          f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
