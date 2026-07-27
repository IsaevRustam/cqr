"""
Checkpointed parallel driver for exploratory and confirmatory sweeps.

Usage (from repo root):
    python -m rebuttal.sweep                       # exploratory, seeds 42..141
    python -m rebuttal.sweep --protocol confirmatory --workers 2
    python -m rebuttal.sweep --datasets diabetes --seeds 42 43

Each job writes a checkpoint below results/rebuttal/. Completed jobs are
skipped on restart, so a crash resumes where it left off.
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

try:
    from tqdm.auto import tqdm
except ImportError:  # Keep the sweep usable in the original environment.
    tqdm = None

# BLAS/OMP threads must be pinned before numpy/torch import in the children.
# ProcessPoolExecutor(spawn) children inherit the parent's environment, so we
# set it here at module import time (this module is re-imported in children).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rebuttal.protocol import (  # noqa: E402
    CONFIRMATORY_PROTOCOL,
    CONFIRMATORY_SEEDS,
    CONFIRMATORY_VERSION,
    EXPLORATORY_PROTOCOL,
    PROTOCOL_CHOICES,
)
from rebuttal.runner import (  # noqa: E402
    PAPER_DATASETS,
    SEEDS,
    raw_dir_for_protocol,
    run_one,
)

# Smallest-first so early datasets finish quickly and can be reported.
DATASET_ORDER = [
    "diabetes", "energy", "concrete", "community", "kin8nm",
    "rf1", "scm1d", "scm20d", "california_housing", "bio", "blog_data",
]


def _job(dataset: str, seed: int, protocol: str) -> dict:
    import torch
    torch.set_num_threads(1)
    return run_one(dataset, seed, protocol=protocol)


def _done(dataset: str, seed: int, raw_dir: Path, protocol: str) -> bool:
    j = raw_dir / dataset / f"seed_{seed}.json"
    n = raw_dir / dataset / f"seed_{seed}.npz"
    if not (j.exists() and n.exists()):
        return False
    try:
        with open(j) as f:
            meta = json.load(f)
        if protocol == CONFIRMATORY_PROTOCOL:
            return (
                meta.get("protocol") == CONFIRMATORY_PROTOCOL
                and meta.get("protocol_version") == CONFIRMATORY_VERSION
            )
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--seed-range", type=int, nargs=2, metavar=("FIRST", "LAST"),
                    default=None, help="inclusive seed range, e.g. --seed-range 42 91")
    ap.add_argument("--protocol", choices=PROTOCOL_CHOICES,
                    default=EXPLORATORY_PROTOCOL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="print pending jobs without starting workers")
    args = ap.parse_args()

    if args.workers < 1:
        ap.error("--workers must be positive")

    datasets = [d for d in DATASET_ORDER if d in (args.datasets or PAPER_DATASETS)]
    if args.seed_range is not None:
        seeds = list(range(args.seed_range[0], args.seed_range[1] + 1))
    else:
        default_seeds = (
            CONFIRMATORY_SEEDS
            if args.protocol == CONFIRMATORY_PROTOCOL else SEEDS
        )
        seeds = args.seeds or default_seeds
    if not seeds:
        ap.error("seed list must be non-empty")
    if args.protocol == CONFIRMATORY_PROTOCOL:
        invalid = sorted(set(seeds) - set(CONFIRMATORY_SEEDS))
        if invalid:
            ap.error(
                "confirmatory seeds must be within the frozen range "
                f"{CONFIRMATORY_SEEDS[0]}..{CONFIRMATORY_SEEDS[-1]}; got {invalid}"
            )

    raw_dir = raw_dir_for_protocol(args.protocol)
    jobs = [
        (d, s) for d in datasets for s in seeds
        if not _done(d, s, raw_dir, args.protocol)
    ]
    total = len(datasets) * len(seeds)
    print(f"[sweep] protocol={args.protocol}; {total} jobs total, "
          f"{total - len(jobs)} already done, "
          f"{len(jobs)} to run, {args.workers} workers, seeds {min(seeds)}..{max(seeds)}",
          flush=True)
    if args.dry_run:
        for dataset, seed in jobs:
            print(f"[sweep] pending {dataset} seed {seed}")
        return
    if not jobs:
        print("[sweep] nothing to do"); return

    log_path = raw_dir.parent / "sweep_log.jsonl"
    raw_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_ok = n_fail = 0
    per_ds_remaining = {d: sum(1 for dd, _ in jobs if dd == d) for d in datasets}

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, d, s, args.protocol): (d, s) for d, s in jobs}
        completed = as_completed(futs)
        progress = (
            tqdm(completed, total=len(futs), desc="[sweep]", unit="job", dynamic_ncols=True)
            if tqdm is not None else completed
        )
        for fut in progress:
            d, s = futs[fut]
            try:
                meta = fut.result()
                n_ok += 1
                per_ds_remaining[d] -= 1
                done_ds = " <-- DATASET COMPLETE" if per_ds_remaining[d] == 0 else ""
                message = (f"[sweep] ok  {d} seed {s}  ({meta['elapsed_sec']:.1f}s)  "
                           f"[{n_ok + n_fail}/{len(jobs)}]{done_ds}")
                if tqdm is not None:
                    progress.write(message)
                else:
                    print(message, flush=True)
                with open(log_path, "a") as f:
                    f.write(json.dumps({"dataset": d, "seed": s, "status": "ok",
                                        "elapsed_sec": meta["elapsed_sec"],
                                        "t": time.time()}) + "\n")
            except Exception as e:
                n_fail += 1
                message = f"[sweep] FAIL {d} seed {s}: {e}"
                if tqdm is not None:
                    progress.write(message)
                else:
                    print(message, flush=True)
                traceback.print_exc()
                with open(log_path, "a") as f:
                    f.write(json.dumps({"dataset": d, "seed": s, "status": "fail",
                                        "error": str(e), "t": time.time()}) + "\n")
            if tqdm is not None:
                progress.set_postfix(ok=n_ok, fail=n_fail, last=f"{d}:{s}")

    print(f"[sweep] finished: {n_ok} ok, {n_fail} failed, "
          f"wall {time.time() - t0:.0f}s", flush=True)
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
