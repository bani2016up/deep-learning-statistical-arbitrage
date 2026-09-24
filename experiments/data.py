"""Residual panels for different factor counts, built from the baseline's return panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dlsa_baseline.data.pca_residuals import (
    ResidualDataset,
    load_residual_dataset,
    rolling_pca_residuals,
    save_residual_dataset,
)
from dlsa_baseline.data.windows import WindowDataset, build_cumulative_windows

REPO = Path(__file__).resolve().parents[1]
RETURNS_PATH = REPO / "data/processed/daily_returns.parquet"
RESIDUAL_DIR = REPO / "data/residuals"
COVARIANCE_LOOKBACK = 252
LOADING_LOOKBACK = 60


def residual_path(n_factors: int) -> Path:
    return RESIDUAL_DIR / f"pca_k{n_factors}.npz"


def build_residuals(
    n_factors: int, returns: pd.DataFrame | None = None
) -> ResidualDataset:
    """K=0 is the paper's "zero-factor model": raw returns on the same dates as K>0 panels."""
    if returns is None:
        returns = pd.read_parquet(RETURNS_PATH)
    if n_factors == 0:
        values = returns.to_numpy(dtype=np.float64)[COVARIANCE_LOOKBACK:]
        dataset = ResidualDataset(
            residuals=values.astype(np.float32),
            dates=returns.index.to_numpy(dtype="datetime64[D]")[COVARIANCE_LOOKBACK:],
            tickers=returns.columns.to_numpy(dtype=str),
            metadata={"method": "raw_returns", "n_factors": 0},
        )
        dataset.validate()
        return dataset
    return rolling_pca_residuals(
        returns, n_factors, COVARIANCE_LOOKBACK, LOADING_LOOKBACK
    )


def load_residuals(n_factors: int) -> ResidualDataset:
    path = residual_path(n_factors)
    if not path.exists():
        save_residual_dataset(build_residuals(n_factors), path)
    return load_residual_dataset(path)


def load_windows(n_factors: int, lookback: int) -> WindowDataset:
    return build_cumulative_windows(load_residuals(n_factors), lookback)


def targets_on_dates(n_factors: int, dates: np.ndarray) -> np.ndarray:
    """Residuals [len(dates), assets] realized on ``dates`` (the dates a run traded)."""
    residuals = load_residuals(n_factors)
    dates = np.asarray(dates).astype("datetime64[D]")
    index = np.searchsorted(residuals.dates, dates)
    if index.max(initial=0) >= len(residuals.dates) or not np.array_equal(
        residuals.dates[index], dates
    ):
        raise ValueError("dates are not all present in the residual panel")
    return residuals.residuals[index].astype(np.float64)


def full_model_name(factor_model: str, n_factors: int) -> str:
    """Residual file of a full-data run: ``ff5``, ``pca{K}`` or ``official_{family}{K}``."""
    if factor_model == "ff5":
        return "ff5"
    if factor_model.startswith("official_"):  # authors' CRSP residuals, experiments.official
        return f"{factor_model}{n_factors}"
    return f"pca{n_factors}"


def load_run_data(config: dict) -> object:
    """Training data for a run config (dict or RunConfig.to_dict()): Yahoo sample or full panel."""
    if config.get("dataset", "yahoo") == "yahoo":
        return load_windows(config["n_factors"], config["lookback"])
    from experiments.data_full import load_full
    from experiments.panel import MaskedPanel

    full = load_full(
        full_model_name(config.get("factor_model", "pca"), config["n_factors"])
    )
    return MaskedPanel(
        full.residuals, full.eligible, full.dates, full.tickers, config["lookback"]
    )


def run_targets(config: dict, dates: np.ndarray) -> np.ndarray:
    """Realized residuals [len(dates), assets] a run traded (for decomposition/ensembles)."""
    if config.get("dataset", "yahoo") == "yahoo":
        return targets_on_dates(config["n_factors"], dates)
    from experiments.data_full import load_full

    full = load_full(
        full_model_name(config.get("factor_model", "pca"), config["n_factors"])
    )
    index = np.searchsorted(full.dates, np.asarray(dates).astype("datetime64[D]"))
    return np.nan_to_num(full.residuals[index].astype(np.float64))
