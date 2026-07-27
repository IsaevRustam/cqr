# Confirmatory bandwidth protocol

Frozen on 2026-07-27, before running confirmatory seeds.

## Motivation

The existing sweep evaluates a dataset-specific grid of bandwidths. Any row
chosen after inspecting test-set coverage is exploratory: test labels have then
influenced the reported bandwidth. Those runs remain useful as sensitivity
analysis, but cannot support a confirmatory claim about practical bandwidth
selection.

## Primary analysis

- Compare Global CQR with Local CQR at one common bandwidth, `h = 1.4`, in the
  VAE latent space for every dataset. `1.4` is the midpoint of the pre-existing
  grid `0.6, 0.8, ..., 2.2`; it is fixed without using dataset outcomes.
- Use fresh seeds `142, ..., 161`. Seeds `42, ..., 141` belong to the
  exploratory development range and are excluded from confirmatory intervals
  and tests.
- Share split, trained quantile regressors, and latent representation between
  Global and Local CQR within each `(dataset, seed)` pair.
- Primary WGC is worst coverage over five equal-count groups ordered by the
  uncalibrated base interval width `q_hi(X) - q_lo(X)`. This grouping uses no
  test labels and is fixed before calibration-method comparison.
- Report every completed confirmatory seed. Do not choose datasets, seeds,
  methods, or bandwidths from confirmatory outcomes.

## Diagnostics and sensitivity

- Store Kish effective sample size
  `ESS(x) = (sum_i w_i(x))^2 / sum_i w_i(x)^2` for every test point. Report its
  minimum, 10th percentile, median, and fraction below `30`. The threshold is a
  descriptive warning, not a tuning rule: it does not select `h`, remove test
  points, or trigger fallback in the primary analysis.
- Report WGC sensitivity for `K = 3, 5, 10` groups defined by PC1, uncalibrated
  base width, and calibration-only local residual scale. These analyses are
  secondary and cannot replace the primary result.
- The old full bandwidth grid remains explicitly exploratory.

## Conformal quantile

Confirmatory runs use the split-conformal order statistic
`S_(ceil((m+1)(1-alpha)))`, clipped at rank `m`, rather than NumPy's linearly
interpolated sample quantile. Global CQR and Local CQR fallback use the same
implementation. Therefore both methods must be rerun on the fresh seeds.

## Commands

Two-worker timing pass, one fresh seed on every dataset:

```bash
python -m rebuttal.sweep --protocol confirmatory \
  --seed-range 142 142 --workers 2
```

Full confirmatory sweep:

```bash
python -m rebuttal.sweep --protocol confirmatory --workers 2
```

Outputs are checkpointed under
`results/rebuttal/confirmatory/raw/<dataset>/seed_<seed>.{json,npz}`. JSON files
contain scalar metrics and protocol metadata. NPZ files contain per-point data
for diagnostics and sensitivity analysis.
