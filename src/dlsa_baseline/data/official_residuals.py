from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np

from dlsa_baseline.config import ROOT
from dlsa_baseline.data.pca_residuals import ResidualDataset

OFFICIAL_RESIDUALS_DIR = ROOT / "references/dlsa-public/residuals"
OFFICIAL_START, OFFICIAL_END = "1998-01-02", "2016-12-30"

# File-name templates of the precomputed OOS residuals in gregzanotti/dlsa-public.
OFFICIAL_FILES = {
    "FF": "famafrench/DailyFamaFrench_OOSresiduals_{k}_factors_1998_initialOOSYear_"
    "60_rollingWindow_0.01_Cap.npy",
    "PCA": "pca/AvPCA_OOSresiduals_{k}_factors_1998_initialOOSYear_60_rollingWindow_"
    "252_covWindow_0.01_Cap.npy",
    "IPCA": "ipca_normalized/IPCA_DailyOOSresiduals_{k}_factors_420_initialMonths_240_window_"
    "12_reestimationFreq_0.01_cap.npy",
}
OFFICIAL_FACTORS = {
    "FF": (0, 1, 2, 3, 4, 5, 8),
    "PCA": (0, 1, 3, 5, 8, 10, 15),
    "IPCA": (0, 1, 3, 5, 8, 10, 15),
}


def official_trading_days() -> np.ndarray:
    """NYSE sessions 1998-01-02..2016-12-30: exactly the 4,781 rows of the official residuals."""
    import exchange_calendars as xcals

    sessions = xcals.get_calendar("XNYS", start=OFFICIAL_START).sessions_in_range(
        OFFICIAL_START, OFFICIAL_END
    )
    return sessions.to_numpy().astype("datetime64[D]")


def official_residual_path(
    family: str, n_factors: int, root: Path = OFFICIAL_RESIDUALS_DIR
) -> Path:
    if family not in OFFICIAL_FILES:
        raise ValueError(f"family must be one of {sorted(OFFICIAL_FILES)}")
    if n_factors not in OFFICIAL_FACTORS[family]:
        raise ValueError(f"{family} residuals exist only for K in {OFFICIAL_FACTORS[family]}")
    return root / OFFICIAL_FILES[family].format(k=n_factors)


def load_official_residuals(
    family: str, n_factors: int, root: Path = OFFICIAL_RESIDUALS_DIR
) -> ResidualDataset:
    """Load an official ``[4781, 9483]`` residual panel as float32.

    Reads the unpacked ``.npy`` when present, otherwise the shipped ``.npy.gz``. A zero entry
    means the asset is outside the cap-filtered universe on that date (about 91% of cells);
    consumers must treat zero as missing, as ``preprocess.py`` in the official code does.
    """
    path = official_residual_path(family, n_factors, root)
    if path.exists():
        residuals = np.load(path).astype(np.float32, copy=False)
    elif path.with_name(path.name + ".gz").exists():
        with gzip.open(path.with_name(path.name + ".gz")) as handle:
            residuals = np.load(handle).astype(np.float32)
    else:
        raise FileNotFoundError(
            f"{path}(.gz) not found; clone gregzanotti/dlsa-public into references/dlsa-public"
        )
    dates = official_trading_days()
    dataset = ResidualDataset(
        residuals=residuals,
        dates=dates,
        tickers=np.array([f"asset_{i}" for i in range(residuals.shape[1])]),
        metadata={
            "method": f"official_{family.lower()}",
            "n_factors": n_factors,
            "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "missing_value": 0.0,
        },
    )
    dataset.validate()
    return dataset
