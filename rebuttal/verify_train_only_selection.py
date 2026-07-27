"""
Leakage verification for the ``train_selected`` protocol (grep + assert).

Static checks (source greps):
  1. rebuttal/h_selection.py never references calibration/test identifiers
     (X_cal, Y_cal, X_test, Y_test, feat_cal, feat_test, n_test), never
     re-splits raw data itself (prepare_data / load_dataset), and never
     indexes the runner's ``data`` dict.
  2. cqr/vae.py (VAE training) references no calibration/test identifiers:
     the VAE can only be fitted on whatever single array it is given.
  3. evaluate_real_data._build_kernel_features fits the VAE and the PCA on
     X_train only (calibration/test features are encoded, never fitted on).
  4. The runner call site passes only ``data["X_train"]``/``data["Y_train"]``
     (plus scalar counts and config) into select_bandwidth_on_train.

Runtime checks (asserts):
  5. select_bandwidth_on_train's signature has no calibration/test parameter.
  6. On synthetic data: the inner 70/15/15 split partitions the training set
     exactly, the candidate grid has len(fixed_bandwidth_grid) entries
     (9 for the paper config; protocol v2 has no data-driven candidates),
     and the frozen h is positive and finite.

Usage (from repo root):
    python -m rebuttal.verify_train_only_selection [--skip-runtime]
    python -m rebuttal.verify_train_only_selection --report
        # print selected h per (dataset, seed) from existing checkpoints
"""

import argparse
import inspect
import json
import re
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parent.parent

H_SELECTION_PY = REPO_ROOT / "rebuttal" / "h_selection.py"
VAE_PY = REPO_ROOT / "cqr" / "vae.py"
EVAL_PY = REPO_ROOT / "evaluate_real_data.py"
RUNNER_PY = REPO_ROOT / "rebuttal" / "runner.py"

# Identifier-shaped patterns; prose like "calibration set" in docstrings is
# allowed, variables/keys holding calibration or test DATA are not.
FORBIDDEN_IN_H_SELECTION = [
    r"\b[XxYy]_cal\b",
    r"\b[XxYy]_test\b",
    r"\bfeat_cal\b",
    r"\bfeat_test\b",
    r"\bn_test\b",
    r"[\"'][XY]_cal[\"']",
    r"[\"'][XY]_test[\"']",
    r"\bprepare_data\b",
    r"\bload_dataset\b",
    r"\bdata\[",
]

FORBIDDEN_IN_VAE = [
    r"\b[A-Za-z_]+_cal\b",
    r"\b[A-Za-z_]+_test\b",
]


def _grep(path: Path, patterns: List[str], label: str) -> List[str]:
    failures = []
    src = path.read_text(encoding="utf-8")
    for pat in patterns:
        for lineno, line in enumerate(src.splitlines(), 1):
            if re.search(pat, line):
                failures.append(
                    f"{label}: forbidden pattern {pat!r} at "
                    f"{path.name}:{lineno}: {line.strip()}"
                )
    return failures


def static_check_failures() -> List[str]:
    failures: List[str] = []

    # 1. h-selection module reads no calibration/test data.
    failures += _grep(H_SELECTION_PY, FORBIDDEN_IN_H_SELECTION, "check1")

    # 2. VAE training module has no calibration/test identifiers at all.
    failures += _grep(VAE_PY, FORBIDDEN_IN_VAE, "check2")

    # 3. Outer kernel features: VAE and PCA are fitted on X_train only.
    eval_src = EVAL_PY.read_text(encoding="utf-8")
    if not re.search(r"train_vae_encoder\(\s*X_train\s*,", eval_src):
        failures.append("check3: evaluate_real_data does not fit the VAE on X_train")
    if not re.search(r"pca\.fit\(X_train\)", eval_src):
        failures.append("check3: evaluate_real_data does not fit the PCA on X_train")
    for pat in (r"train_vae_encoder\(\s*X_cal", r"train_vae_encoder\(\s*X_test",
                r"pca\.fit\(X_cal", r"pca\.fit\(X_test"):
        if re.search(pat, eval_src):
            failures.append(f"check3: forbidden fit call matching {pat!r}")

    # 4. Runner call site passes only training arrays into the selection.
    runner_src = RUNNER_PY.read_text(encoding="utf-8")
    m = re.search(r"select_bandwidth_on_train\(", runner_src)
    if not m:
        failures.append("check4: runner never calls select_bandwidth_on_train")
    else:
        # Extract the full call expression by matching parentheses.
        depth, i = 0, m.end() - 1
        while i < len(runner_src):
            if runner_src[i] == "(":
                depth += 1
            elif runner_src[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        call_text = runner_src[m.start():i + 1]
        if re.search(r"(X|Y|feat)_(cal|test)", call_text):
            failures.append(
                "check4: selection call site references calibration/test arrays:\n"
                + call_text
            )
        if 'data["X_train"]' not in call_text or 'data["Y_train"]' not in call_text:
            failures.append(
                "check4: selection call site does not pass the training arrays:\n"
                + call_text
            )
    return failures


def runtime_check_failures() -> List[str]:
    failures: List[str] = []
    import numpy as np

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from rebuttal.h_selection import select_bandwidth_on_train

    # 5. Signature accepts nothing calibration- or test-shaped.
    params = list(inspect.signature(select_bandwidth_on_train).parameters)
    expected = ["X_train", "Y_train", "cfg", "seed", "latent_dim", "n_cal_real"]
    if params != expected:
        failures.append(f"check5: unexpected signature {params} != {expected}")
    for p in params:
        if re.search(r"(?<!n_)(cal|test)", p) and p != "n_cal_real":
            failures.append(f"check5: suspicious parameter name {p!r}")

    # 6. Tiny synthetic run (small epochs; a few seconds on CPU).
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d)).astype(np.float32)
    y = (X[:, 0] + 0.5 * np.abs(X[:, 1]) * rng.normal(size=n)).astype(np.float32)
    grid = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
    cfg = {
        "alpha": 0.1,
        "hidden_dim": 16, "n_layers": 2, "train_epochs": 2,
        "learning_rate": 1e-3, "batch_size": 128, "weight_decay": 1e-3,
        "activation": "relu",
        "kernel_space": "vae", "vae_hidden_dim": 16, "vae_epochs": 2,
        "vae_beta": 1.0,
        "bandwidth_scale": 1.0, "fixed_bandwidth": 1.5,
        "fixed_bandwidth_grid": grid,
    }
    res = select_bandwidth_on_train(
        X, y, cfg=cfg, seed=0, latent_dim=2, n_cal_real=300,
    )
    if res["n_fit"] + res["n_incal"] + res["n_ineval"] != n:
        failures.append("check6: inner split does not partition the training set")
    if len(res["candidates"]) != len(grid):
        failures.append(
            f"check6: expected {len(grid)} candidates, got {len(res['candidates'])}"
        )
    if any(not c.startswith("fixed_") for c in res["candidates"]):
        failures.append(
            "check6: v2+ protocol must have fixed-grid candidates only, got "
            + str(sorted(res["candidates"]))
        )
    if res.get("selection_metric") != "winkler_2fold_mean":
        failures.append(
            f"check6: expected symmetrized selection metric, got "
            f"{res.get('selection_metric')!r}"
        )
    for cand, c in res["candidates"].items():
        mean_w = 0.5 * (c["winkler_cal_to_eval"] + c["winkler_eval_to_cal"])
        if abs(c["winkler"] - mean_w) > 1e-12:
            failures.append(
                f"check6: candidate {cand}: winkler is not the mean of both "
                "directions"
            )
    if not (np.isfinite(res["h_selected"]) and res["h_selected"] > 0):
        failures.append(f"check6: bad h_selected {res['h_selected']!r}")
    if res["candidate_selected"] not in res["candidates"]:
        failures.append("check6: selected candidate missing from candidate log")
    if abs(res["candidates"][res["candidate_selected"]]["h"] - res["h_selected"]) > 0:
        failures.append("check6: h_selected does not match the winning candidate")
    return failures


def report_selected_h() -> None:
    """Print selected h per (dataset, seed) from existing checkpoints."""
    raw = REPO_ROOT / "results" / "rebuttal" / "train_selected" / "raw"
    rows = []
    for path in sorted(raw.glob("*/seed_*.json")):
        with open(path) as f:
            meta = json.load(f)
        sel = meta.get("h_selection") or {}
        rows.append((
            meta["dataset"], meta["seed"],
            sel.get("h_selected"), sel.get("candidate_selected"),
            sel.get("n_incal"), sel.get("n_cal_real"),
        ))
    if not rows:
        print(f"no train_selected checkpoints under {raw}")
        return
    print(f"{'dataset':<20}{'seed':>6}{'h_selected':>12}  "
          f"{'candidate':<12}{'n_incal':>8}{'n_cal_real':>11}")
    for ds, seed, h, cand, n_incal, n_cal_real in rows:
        h_str = "n/a" if h is None else f"{h:.4f}"
        print(f"{ds:<20}{seed:>6}{h_str:>12}  "
              f"{str(cand):<12}{str(n_incal):>8}{str(n_cal_real):>11}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-runtime", action="store_true",
                    help="run only the static grep checks")
    ap.add_argument("--report", action="store_true",
                    help="print selected h per (dataset, seed) and exit")
    args = ap.parse_args()

    if args.report:
        report_selected_h()
        return

    failures = static_check_failures()
    print(f"[verify] static checks: {'PASS' if not failures else 'FAIL'}")
    if not args.skip_runtime:
        rt = runtime_check_failures()
        print(f"[verify] runtime checks: {'PASS' if not rt else 'FAIL'}")
        failures += rt
    for f in failures:
        print(f"[verify] {f}")
    if failures:
        raise SystemExit(1)
    print("[verify] OK: h-selection and VAE training read training data only")


if __name__ == "__main__":
    main()
