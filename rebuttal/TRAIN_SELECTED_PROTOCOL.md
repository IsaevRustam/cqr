# Train-only bandwidth selection protocol (`train_selected`)

Frozen on 2026-07-27 as `2026-07-27-v1`; amended the same day to
`SELECTED_VERSION = 2026-07-27-v2` (see the amendment note below).

## Motivation

The exploratory grid reports every bandwidth, and the confirmatory protocol
fixes `h = 1.4` a priori. This protocol answers the remaining question: can a
*practical, data-driven* h be selected without ever consulting calibration or
test data? Selection happens entirely inside the training split; the frozen h
then flows through the untouched published pipeline.

## Selection procedure (training data only)

1. Split the outer TRAINING split 70/15/15 into **T-fit / T-cal / T-eval**
   (`INNER_SPLIT_FRACS`, same `train_test_split(random_state=seed)` mechanics
   as the outer split). T-cal is capped at the real calibration-set size when
   that is smaller; with the paper split fractions the real calibration set is
   always larger, so T-cal is 15% of train. Both sizes (`n_incal`,
   `n_cal_real`) are logged per (dataset, seed).
2. Train the quantile regressors and the VAE on **T-fit only** (same
   hyperparameters and seed as the outer pipeline).
3. For each candidate h in the fixed grid — the nine values
   `0.6, 0.8, ..., 2.2` scaled by `bandwidth_scale` (v2; see amendment) — run
   localized conformal calibration on T-cal and score the intervals on T-eval
   by **mean Winkler score**.
4. **Freeze the numeric h** with the lowest T-eval Winkler score (ties break
   toward the earliest candidate in grid order). The frozen h is a plain
   number fixed before any calibration or test data is read.

## Amendment v2 (2026-07-27): fixed grid only

v1 additionally included the three data-driven candidates (silverman, scott,
isj, computed on T-fit kernel features — a 12-grid). After the v1 runs on the
first six datasets (diabetes, energy, concrete, community, kin8nm, rf1;
seeds 142–161), the data-driven rules were removed because they occasionally
won the inner selection with a degenerate small h (e.g. isj h ≈ 0.14–0.27 in
the VAE space) whose neighborhoods hold almost no calibration points (median
test ESS < 10, sometimes 0), losing conditional and marginal coverage
(community: 3/20 seeds at coverage 0.72–0.76).

This amendment was made after observing v1 TEST outcomes, so it is a
protocol revision, not a pre-registered choice: v1 and v2 results must be
reported as such, side by side where relevant. The v1 checkpoints are
preserved under `results/rebuttal/train_selected/raw_v1/`. Note the failure
mode that motivated the amendment is flagged label-free by the
pre-registered ESS diagnostic; the fixed grid itself predates all
`train_selected` runs (it is the published exploratory grid).

## Final run (per dataset, seed)

- Retrain regressors + VAE on the **full training split** — bit-identical to
  the published pipeline (both trainers re-seed torch internally, so running
  the selection first does not perturb the outer run).
- Calibrate on the **real calibration set** with the frozen h.
- The **test set is touched once**: one local run at the frozen h, plus the
  h-free Global CQR comparator. No other bandwidth ever sees test data.
- Evaluation uses the same primary WGC as the confirmatory protocol (worst
  coverage over 5 rank-binned groups of uncalibrated base interval width) and
  the split-conformal order statistic.

## Leakage verification (grep + assert)

`python -m rebuttal.verify_train_only_selection` — also run as unit tests in
`tests/test_h_selection_leakfree.py`:

- greps `rebuttal/h_selection.py` for calibration/test identifiers
  (`X_cal`, `Y_cal`, `X_test`, `Y_test`, `feat_cal`, `feat_test`, `n_test`,
  `data[`, `prepare_data`, `load_dataset`) — none may appear;
- greps `cqr/vae.py` for any `*_cal` / `*_test` identifier — none may appear;
- greps `evaluate_real_data.py` to confirm the outer VAE and PCA are fitted on
  `X_train` only;
- greps the runner call site to confirm only `data["X_train"]` /
  `data["Y_train"]` (plus scalar counts and config) enter the selection;
- runtime asserts: the selection function's signature has no calibration/test
  parameter; on synthetic data the inner split partitions the training set
  exactly, the candidate grid has 9 fixed-grid entries (v2), and the frozen h
  is positive and finite. The runner additionally asserts the partition and h validity on
  every real job.

## Logging

Each checkpoint JSON (`results/rebuttal/train_selected/raw/<dataset>/
seed_<seed>.json`) records under `h_selection`: the frozen `h_selected`, the
winning candidate name, per-candidate T-eval Winkler scores, and all inner
split sizes next to `n_cal_real`.
`python -m rebuttal.verify_train_only_selection --report` tabulates the
selected h per (dataset, seed).

## Commands

```bash
# default: fresh seeds 142..161 (exploratory seeds 42..141 are excluded)
python -m rebuttal.sweep --protocol train_selected --workers 2

# leakage verification
python -m rebuttal.verify_train_only_selection
```
