# Reproducibility Package

This folder contains all code needed to reproduce the experiments in the paper.

## Requirements

```
pip install -r requirements.txt
```

## Experiments

### Experiment 1 — Real-data evaluation (Global vs Localized CQR)

Evaluates Global CQR and Localized CQR on 11 benchmark regression datasets using VAE latent features as the kernel space, with a fixed bandwidth grid.

```bash
python evaluate_real_data.py \
    --kernel_space vae \
    --latent_dim auto \
    --fixed_bandwidth_grid 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0 2.2
```

Results are saved to `results_real_data.csv` by default. Use `--output <file>` to change the output path.

Key hyperparameters (set in `configs/real.yaml`, overridable via CLI):
- `--alpha 0.1` — miscoverage level (90% coverage target)
- `--n_attempts 5` — number of random seeds
- `--epochs 100` — training epochs
- `--hidden_dim 128` — neural network hidden dimension (32 is used automatically for rf1)
- `--activation relu`

### Experiment 2 — Synthetic heteroscedastic settings

Compares Local CQR and Global CQR on heteroscedastic synthetic data across multiple noise-function settings.

```bash
python guan2021_figures.py --epochs 100 --settings A B H I L N
```

Figures are saved to `figures/` by default. Use `--output <path>` to override.

## File Structure

```
reproducibility/
├── evaluate_real_data.py   # Real-data experiment (Experiment 1)
├── guan2021_figures.py     # Synthetic experiment (Experiment 2)
├── configs/
│   └── real.yaml           # Default hyperparameter config
├── datasets/
│   └── rf1_train_clean.npz # Local cache for the RF1 dataset
└── cqr/                    # CQR package
    ├── __init__.py
    ├── calibration.py      # Global and localized conformal calibration
    ├── config.py           # ExperimentConfig dataclass
    ├── data.py             # Synthetic data generators
    ├── metrics.py          # Coverage and width metrics
    ├── models.py           # ReLU quantile network
    ├── models_requ.py      # ReQU quantile network
    ├── plotting.py         # Plotting utilities
    ├── preprocessing.py    # Data standardization and splitting
    ├── real_data.py        # Real-world dataset loaders
    ├── training.py         # Unified training loop
    └── vae.py              # VAE encoder for kernel features
```

## Datasets

The following 11 datasets are used in Experiment 1. Most are loaded automatically via scikit-learn or OpenML on first run:

| Dataset            | n       | d    | Source       |
|--------------------|---------|------|--------------|
| diabetes           | 442     | 10   | sklearn      |
| concrete           | 1030    | 8    | OpenML       |
| energy             | 768     | 8    | OpenML       |
| kin8nm             | 8192    | 8    | OpenML       |
| community          | 1994    | ~100 | OpenML       |
| california_housing | 20640   | 8    | sklearn      |
| bio (CASP)         | 45730   | 9    | OpenML       |
| blog_data          | 52397   | 280  | OpenML       |
| rf1                | ~73000  | 64   | local cache  |
| scm1d              | ~9000   | 280  | OpenML       |
| scm20d             | ~9000   | 61   | OpenML       |

The `rf1` dataset is provided as a local `.npz` cache in `datasets/rf1_train_clean.npz`.
