"""
Real-world dataset loaders for CQR evaluation.

Supports datasets from:
- Scikit-learn built-in (California Housing, Diabetes)
- OpenML (scm20d, scm1d, rf1, households)
- CSV downloads (MEPS, bio/CASP, blog_data, concrete, community, CalCOFI)

All loaders return (X: np.float32, y: np.float32, info: dict).
"""

import os
import numpy as np
import pandas as pd
import openml
from typing import Tuple, Dict, Any, Optional, Callable
from pathlib import Path

# Default cache directory for downloaded datasets
_CACHE_DIR = Path(__file__).parent.parent / "datasets"


def _ensure_cache_dir(cache_dir: Path = _CACHE_DIR) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _to_float32(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return X.astype(np.float32), y.astype(np.float32).flatten()


def _make_info(name: str, X: np.ndarray, description: str = "") -> Dict[str, Any]:
    return {
        "name": name,
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "description": description,
    }


# =============================================================================
# SCIKIT-LEARN BUILT-IN DATASETS
# =============================================================================


def load_california_housing(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """California Housing dataset (20640 samples, 8 features)."""
    from sklearn.datasets import fetch_california_housing
    data = fetch_california_housing()
    X, y = _to_float32(data.data, data.target)
    return X, y, _make_info("california_housing", X, "Median house value prediction")


def load_diabetes(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Diabetes dataset (442 samples, 10 features)."""
    from sklearn.datasets import load_diabetes as _load
    data = _load()
    X, y = _to_float32(data.data, data.target)
    return X, y, _make_info("diabetes", X, "Disease progression prediction")


# =============================================================================
# OPENML DATASETS
# =============================================================================


def _load_openml(data_id: int, name: str, target_col: Optional[str] = None,
                 description: str = "", max_samples: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Generic OpenML loader."""
    from sklearn.datasets import fetch_openml
    data = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    df = data.data
    target = data.target

    # For multi-output: select first target column
    if hasattr(target, "columns"):
        if target_col and target_col in target.columns:
            y_series = target[target_col]
        else:
            y_series = target.iloc[:, 0]
    else:
        y_series = target

    # Convert to numeric, drop rows with NaN
    df_numeric = df.apply(pd.to_numeric, errors="coerce")
    y_numeric = pd.to_numeric(y_series, errors="coerce")
    valid = df_numeric.notna().all(axis=1) & y_numeric.notna()
    df_numeric = df_numeric[valid]
    y_numeric = y_numeric[valid]

    X = df_numeric.values
    y = y_numeric.values

    if max_samples > 0 and len(X) > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), max_samples, replace=False)
        X, y = X[idx], y[idx]

    X, y = _to_float32(X, y)
    return X, y, _make_info(name, X, description)


def _load_openml_multitarget(
    data_id: int, name: str, n_targets: int,
    description: str = "", target_col: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a multi-target regression dataset from OpenML.
    Selects one target column (default: first) for scalar CQR.
    """
    dataset = openml.datasets.get_dataset(data_id)
    data, _, _, _ = dataset.get_data(dataset_format="dataframe", target=None)

    Y_all = data.iloc[:, -n_targets:].to_numpy(dtype=np.float32)
    X_df = data.iloc[:, :-n_targets]

    # Convert to numeric, clean NaN/Inf
    X_np = X_df.apply(pd.to_numeric, errors="coerce").values.astype(np.float32)
    y = Y_all[:, target_col]

    valid = np.isfinite(X_np).all(axis=1) & np.isfinite(y)
    if not valid.all():
        n_invalid = (~valid).sum()
        print(f"  [{name}] Removing {n_invalid}/{len(X_np)} samples with NaN/Inf")
        X_np = X_np[valid]
        y = y[valid]

    return X_np, y, _make_info(name, X_np, description)


def load_scm20d(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """SCM20D dataset (8966 samples, 61 features, 16 targets -> 1st used)."""
    return _load_openml_multitarget(
        data_id=41486, name="scm20d", n_targets=16,
        description="Supply chain management (20d target, 1st used)",
        target_col=kwargs.get("target_col", 0),
    )


def load_scm1d(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """SCM1D dataset (9803 samples, 280 features, 16 targets -> 1st used)."""
    return _load_openml_multitarget(
        data_id=41485, name="scm1d", n_targets=16,
        description="Supply chain management (1d target)",
        target_col=kwargs.get("target_col", 0),
    )


def load_rf2(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """RF2 river flow dataset (9125 samples, 576 features, 8 targets -> 1st used)."""
    return _load_openml_multitarget(
        data_id=41484, name="rf2", n_targets=8,
        description="River flow prediction (1st target)",
        target_col=kwargs.get("target_col", 0),
    )


def load_sgemm(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """SGEMM GPU kernel performance (241600 samples, 14 features, 4 targets -> 1st used)."""
    return _load_openml_multitarget(
        data_id=44069, name="sgemm", n_targets=4,
        description="SGEMM GPU kernel performance (1st target)",
        target_col=kwargs.get("target_col", 0),
    )


def load_rf1(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    RF1 river flow dataset from Mulan (train split only).
    ~5875 samples, 64 features, 8 targets -> 1st used by default.

    Uses only the Mulan *train* split to avoid temporal distribution mismatch
    (rf1 is time-series data — combining train+test and randomly re-splitting
    creates heterogeneous splits that destabilise training).

    Applies per-column IQR outlier removal on targets before caching.
    """
    import urllib.request, io

    cache_dir = _ensure_cache_dir()
    cache_file = cache_dir / "rf1_train_clean.npz"

    if cache_file.exists():
        loaded = np.load(cache_file, allow_pickle=False)
        X, Y_all = loaded["X"], loaded["Y"]
    else:
        MULAN_BASE = "https://sourceforge.net/projects/mulan/files/datasets/multi-target%20regression%20datasets"
        n_targets = 8

        url = f"{MULAN_BASE}/rf1-train.arff/download"
        print(f"  Downloading RF1 train set...")
        req = urllib.request.urlopen(url)
        raw = req.read().decode("utf-8")
        # Parse ARFF manually: skip everything before @data
        lines = raw.split("\n")
        data_start = None
        for i, line in enumerate(lines):
            if line.strip().upper() == "@DATA":
                data_start = i + 1
                break
        if data_start is None:
            raise ValueError("Could not find @DATA section in RF1 train ARFF")
        csv_text = "\n".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(csv_text), header=None)
        df = df.apply(pd.to_numeric, errors="coerce")

        Y_all = df.iloc[:, -n_targets:].to_numpy(dtype=np.float32)
        X = df.iloc[:, :-n_targets].to_numpy(dtype=np.float32)

        # --- Remove NaN / Inf ---
        valid = np.isfinite(X).all(axis=1) & np.isfinite(Y_all).all(axis=1)
        if not valid.all():
            n_bad = int((~valid).sum())
            print(f"  [rf1] Removing {n_bad}/{len(X)} samples with NaN/Inf")
            X, Y_all = X[valid], Y_all[valid]

        # --- IQR outlier removal (per target column, 5×IQR) ---
        iqr_factor = 5.0
        outlier_mask = np.zeros(len(Y_all), dtype=bool)
        print(f"  [rf1] Detecting outliers in {n_targets} target columns (IQR×{iqr_factor})...")
        for col in range(Y_all.shape[1]):
            yc = Y_all[:, col]
            q1, q3 = np.percentile(yc, [25, 75])
            iqr = q3 - q1
            lo, hi = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
            col_out = (yc < lo) | (yc > hi)
            if col_out.any():
                print(f"    target col {col}: {int(col_out.sum())} outliers "
                      f"outside [{lo:.2f}, {hi:.2f}]")
            outlier_mask |= col_out

        if outlier_mask.any():
            n_out = int(outlier_mask.sum())
            print(f"  [rf1] Removing {n_out}/{len(X)} outlier rows")
            X, Y_all = X[~outlier_mask], Y_all[~outlier_mask]

        np.savez(cache_file, X=X, Y=Y_all)
        print(f"  RF1 cached (train-only, clean): {X.shape[0]} samples, "
              f"{X.shape[1]} features, {Y_all.shape[1]} targets")

    target_col = kwargs.get("target_col", 0)
    y = Y_all[:, target_col]
    return X, y, _make_info("rf1", X, "River flow prediction (train split, 1st target)")


def load_households(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Individual household electric power consumption (subsampled)."""
    max_samples = kwargs.get("max_samples", 50000)
    return _load_openml(
        data_id=42792, name="households",
        description="Household electric power consumption",
        max_samples=max_samples,
    )


# =============================================================================
# CSV-BASED DATASETS (following Romano et al. CQR paper)
# =============================================================================


def _download_file(url: str, filepath: Path) -> None:
    """Download a file if it doesn't exist."""
    import urllib.request
    if not filepath.exists():
        print(f"  Downloading {filepath.name}...")
        urllib.request.urlretrieve(url, str(filepath))


def load_bio(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Bio / CASP dataset — protein tertiary structure (45730 samples, 9 features).
    Predicts RMSD from physicochemical properties.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    filepath = cache_dir / "CASP.csv"
    _download_file(url, filepath)

    df = pd.read_csv(filepath)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("bio", X, "Protein structure RMSD prediction (CASP)")


def load_blog_data(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Blog Feedback dataset (52397 samples, 280 features).
    Predicts number of comments in the next 24 hours.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://media.githubusercontent.com/media/xinbinhuang/feature-selection_blogfeedback/master/data/train/blogData_train.csv"
    filepath = cache_dir / "blogData_train.csv"
    _download_file(url, filepath)

    df = pd.read_csv(filepath, header=None)
    X = df.iloc[:, 0:280].values
    y = df.iloc[:, -1].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("blog_data", X, "Blog feedback comment count prediction")


def load_concrete(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Concrete Compressive Strength (1030 samples, 8 features).
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
    filepath = cache_dir / "Concrete_Data.xls"
    _download_file(url, filepath)

    df = pd.read_excel(filepath)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("concrete", X, "Concrete compressive strength prediction")


def load_community(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Communities and Crime dataset (1994 samples, ~100 features).
    Predicts per-capita violent crime rate.
    """
    from sklearn.impute import SimpleImputer
    cache_dir = _ensure_cache_dir(cache_dir)

    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.data"
    names_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/communities/communities.names"
    data_path = cache_dir / "communities.data"
    names_path = cache_dir / "communities.names"
    _download_file(data_url, data_path)
    _download_file(names_url, names_path)

    # Parse attribute names from .names file
    col_names = []
    with open(names_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    col_names.append(parts[1])

    if len(col_names) == 0:
        col_names = [f"col_{i}" for i in range(128)]

    df = pd.read_csv(data_path, names=col_names[:128] if len(col_names) >= 128 else col_names, header=None)

    # Drop non-predictive columns
    drop_cols = [c for c in ["state", "county", "community", "communityname", "fold"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    df = df.replace("?", np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, thresh=int(0.8 * len(df)))  # drop cols with >20% missing

    # Impute remaining missing values
    y = df.iloc[:, -1].values
    X = df.iloc[:, :-1].values

    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X)

    valid = ~np.isnan(y)
    X, y = X[valid], y[valid]
    X, y = _to_float32(X, y)
    return X, y, _make_info("community", X, "Communities & Crime per-capita violent crime")


def load_energy(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Energy Efficiency dataset (768 samples, 8 features).
    Predicts heating load.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
    filepath = cache_dir / "ENB2012_data.xlsx"
    _download_file(url, filepath)

    df = pd.read_excel(filepath)
    df = df.dropna()
    X = df.iloc[:, :8].values
    y = df.iloc[:, 8].values  # Heating load (Y1)
    X, y = _to_float32(X, y)
    return X, y, _make_info("energy", X, "Energy efficiency heating load prediction")


def load_house(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    House prices dataset — uses California Housing as a reliable proxy.
    """
    return load_california_housing(**kwargs)


# =============================================================================
# LOW-DIMENSIONAL BENCHMARK DATASETS
# =============================================================================


def load_kin8nm(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Kin8nm robot arm kinematics dataset (8192 samples, 8 features).
    Predicts the distance of the end-effector from a target.
    """
    return _load_openml(
        data_id=189, name="kin8nm",
        description="Robot arm kinematics prediction",
    )


def load_airfoil(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    NASA Airfoil Self-Noise dataset (1503 samples, 5 features).
    Predicts scaled sound pressure level from aerodynamic and geometric properties.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat"
    filepath = cache_dir / "airfoil_self_noise.dat"
    _download_file(url, filepath)
    df = pd.read_csv(filepath, sep="\t", header=None)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("airfoil", X, "Airfoil self-noise sound pressure prediction")


def load_wine_quality(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Wine Quality dataset — red wine (1599 samples, 11 features).
    Predicts wine quality score from physicochemical properties.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    filepath = cache_dir / "winequality-red.csv"
    _download_file(url, filepath)
    df = pd.read_csv(filepath, sep=";")
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("wine_quality", X, "Red wine quality score prediction")


def load_yacht(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Yacht Hydrodynamics dataset (308 samples, 6 features).
    Predicts residuary resistance of sailing yachts.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data"
    filepath = cache_dir / "yacht_hydrodynamics.data"
    _download_file(url, filepath)
    df = pd.read_csv(filepath, sep=r"\s+", header=None).dropna()
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X, y = _to_float32(X, y)
    return X, y, _make_info("yacht", X, "Yacht hydrodynamics residuary resistance")


def load_meps(year: int, cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    MEPS (Medical Expenditure Panel Survey) dataset.
    Downloads pre-processed CSVs from the Romano et al. CQR repository.
    """
    cache_dir = _ensure_cache_dir(cache_dir)
    filename = f"meps_{year}_reg.csv"
    filepath = cache_dir / filename

    url = f"https://raw.githubusercontent.com/yromano/cqr/master/datasets/{filename}"

    if not filepath.exists():
        # MEPS CSVs are NOT hosted in the repo; they must be generated locally.
        # See: https://github.com/yromano/cqr/blob/master/get_meps_data/README.md
        raise FileNotFoundError(
            f"{filename} not found in {cache_dir}. MEPS data cannot be auto-downloaded.\n"
            f"To use MEPS datasets:\n"
            f"  1. Clone https://github.com/yromano/cqr\n"
            f"  2. cd get_meps_data && Rscript download_data.R\n"
            f"  3. python main_clean_and_save_to_csv.py\n"
            f"  4. Copy {filename} to {cache_dir}"
        )

    df = pd.read_csv(filepath)
    response_name = "UTILIZATION_reg"

    col_names = [c for c in df.columns if c != response_name and c != "Unnamed: 0"]
    y = df[response_name].values
    X = df[col_names].values
    X, y = _to_float32(X, y)
    return X, y, _make_info(f"meps_{year}", X, f"MEPS {year} healthcare utilization")


def load_meps_19(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return load_meps(19, **kwargs)

def load_meps_20(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return load_meps(20, **kwargs)

def load_meps_21(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    return load_meps(21, **kwargs)


def load_calcofi(cache_dir: Path = _CACHE_DIR, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    CalCOFI oceanographic dataset — predict water temperature.
    Uses OpenML or a curated subset.
    """
    try:
        return _load_openml(
            data_id=42728, name="calcofi",
            description="CalCOFI — predict water temperature",
            max_samples=kwargs.get("max_samples", 50000),
        )
    except Exception:
        # Fallback: use a simplified version via sklearn
        print("  CalCOFI: OpenML unavailable, skipping.")
        raise


def load_taxi(**kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    NYC Taxi trip duration — from OpenML (subsampled).
    """
    max_samples = kwargs.get("max_samples", 50000)
    try:
        return _load_openml(
            data_id=42729, name="taxi",
            description="NYC Taxi trip duration prediction",
            max_samples=max_samples,
        )
    except Exception:
        print("  Taxi: OpenML unavailable, skipping.")
        raise


# =============================================================================
# DATASET REGISTRY
# =============================================================================

DATASET_REGISTRY: Dict[str, Callable] = {
    # Scikit-learn
    "california_housing": load_california_housing,
    "diabetes": load_diabetes,
    # OpenML multi-target (using correct IDs)
    "scm20d": load_scm20d,
    "scm1d": load_scm1d,
    "rf1": load_rf1,
    "rf2": load_rf2,
    "sgemm": load_sgemm,
    "households": load_households,
    # Low-dimensional benchmarks
    "kin8nm":       load_kin8nm,
    "airfoil":      load_airfoil,
    "wine_quality": load_wine_quality,
    "yacht":        load_yacht,
    # CSV / CQR paper
    "meps_19": load_meps_19,
    "meps_20": load_meps_20,
    "meps_21": load_meps_21,
    "bio": load_bio,
    "blog_data": load_blog_data,
    "concrete": load_concrete,
    "community": load_community,
    "energy": load_energy,
    "house": load_house,
    # Extra
    "calcofi": load_calcofi,
    "taxi": load_taxi,
}


# A curated subset that are fast and reliably loadable
DEFAULT_DATASETS = [
    "diabetes",           # d=10,  n=442
    "california_housing", # d=8,   n=20640
    "concrete",           # d=8,   n=1030
    "energy",             # d=8,   n=768
    "bio",                # d=9,   n=45730
    "community",          # d~100, n=1994
    "blog_data",          # d=280, n=52397
    "rf1",
    "rf2",
    "scm1d",
    "scm20d",
    # Low-dimensional benchmarks
    "kin8nm",             # d=8,  n=8192
    "airfoil",            # d=5,  n=1503
    "wine_quality",       # d=11, n=1599
]


def load_dataset(name: str, **kwargs) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load a real-world dataset by name.

    Args:
        name: Dataset name (see DATASET_REGISTRY keys)
        **kwargs: Passed to the underlying loader (e.g., cache_dir, max_samples)

    Returns:
        (X, y, info) where X is (n, d) float32, y is (n,) float32,
        and info is a dict with name, n_samples, n_features, description
    """
    if name not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}")

    return DATASET_REGISTRY[name](**kwargs)


def list_datasets() -> list:
    """Return sorted list of available dataset names."""
    return sorted(DATASET_REGISTRY.keys())
