"""
Re-run selected (dataset, seed) jobs under NATIVE threading — no OMP/BLAS env
pinning, no torch.set_num_threads — i.e. exactly the conditions of the
original published sweep (single-machine defaults, 10 torch threads here).

Purpose: the published CSV is bit-exact only under native threading; the
1-thread parallel workers of rebuttal.sweep produce statistically equivalent
but not bit-identical runs (chaotic divergence of mini-batch training), and
the blog_data seed-79 VAE diverges to NaN only at 1 thread.

Default job list: all 11 paper datasets x seeds 42..46 (the published runs),
plus blog_data seed 79.  Existing checkpoints for these jobs are OVERWRITTEN.

Usage (from repo root):
    python -m rebuttal.rerun_published [--workers 2]

Afterwards run:  python -m rebuttal.aggregate --validate   (expect ~1e-15)
"""

import argparse
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

# NOTE: deliberately NO os.environ thread pinning here (contrast with sweep.py)

from rebuttal.runner import PAPER_DATASETS, run_one


def _job(dataset: str, seed: int) -> dict:
    # native threads: do not touch torch.set_num_threads
    return run_one(dataset, seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2,
                    help="processes; each uses native (default) torch threads, "
                         "so keep this small")
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    args = ap.parse_args()

    datasets = args.datasets or PAPER_DATASETS
    seeds = args.seeds or list(range(42, 47))
    jobs = [(d, s) for d in datasets for s in seeds]
    if args.datasets is None and args.seeds is None:
        jobs.append(("blog_data", 79))   # 1-thread NaN failure; fine natively

    print(f"[rerun] {len(jobs)} jobs under native threading, "
          f"{args.workers} workers (overwriting existing checkpoints)", flush=True)
    t0 = time.time()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, d, s): (d, s) for d, s in jobs}
        for fut in as_completed(futs):
            d, s = futs[fut]
            try:
                meta = fut.result()
                n_ok += 1
                print(f"[rerun] ok  {d} seed {s}  ({meta['elapsed_sec']:.1f}s, "
                      f"threads={meta['torch_threads']})  [{n_ok + n_fail}/{len(jobs)}]",
                      flush=True)
            except Exception as e:
                n_fail += 1
                print(f"[rerun] FAIL {d} seed {s}: {e}", flush=True)
                traceback.print_exc()
    print(f"[rerun] finished: {n_ok} ok, {n_fail} failed, "
          f"wall {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
