from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ResidualDataset:
    residuals: np.ndarray  # [dates, assets]
    dates: np.ndarray  # datetime64[D], aligned to residuals
    tickers: np.ndarray  # strings, aligned to assets
    metadata: dict[str, object]

    def validate(self) -> None:
        if self.residuals.shape != (len(self.dates), len(self.tickers)):
            raise ValueError("Residual, date, and ticker dimensions do not align")
        if not np.isfinite(self.residuals).all():
            raise ValueError("Residuals contain non-finite values")


def rolling_pca_residuals(
    returns: pd.DataFrame,
    n_factors: int = 5,
    covariance_lookback: int = 252,
    loading_lookback: int = 60,
) -> ResidualDataset:
    """Estimate date-t residuals using PCA/loadings fitted strictly through t-1.

    PCA eigenvectors use ``returns[t-covariance_lookback:t]``. Loadings use the
    last ``loading_lookback`` rows of that historical slice. Only the date-t
    factor realization uses date-t returns, as required for a contemporaneous
    return decomposition.
    """
    values = returns.to_numpy(dtype=np.float64)
    total_dates, n_assets = values.shape
    if n_factors <= 0 or n_factors >= n_assets:
        raise ValueError("n_factors must be between 1 and n_assets - 1")
    if loading_lookback > covariance_lookback:
        raise ValueError("loading_lookback cannot exceed covariance_lookback")
    if total_dates <= covariance_lookback:
        raise ValueError("Not enough dates for the covariance lookback")
    if not np.isfinite(values).all():
        raise ValueError("Returns must be a finite rectangular panel")

    residuals = np.empty((total_dates - covariance_lookback, n_assets), dtype=np.float64)
    explained = np.empty(total_dates - covariance_lookback, dtype=np.float64)
    for output_index, t in enumerate(range(covariance_lookback, total_dates)):
        history = values[t - covariance_lookback : t]
        scale = history.std(axis=0, ddof=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (history - history.mean(axis=0)) / scale
        covariance = standardized.T @ standardized / len(history)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1][:n_factors]
        components = eigenvectors[:, order]
        factor_history = (history / scale) @ components
        x = factor_history[-loading_lookback:]
        y = history[-loading_lookback:]
        loadings = np.linalg.lstsq(x, y, rcond=None)[0]  # [factors, assets]
        factor_today = (values[t] / scale) @ components
        residuals[output_index] = values[t] - factor_today @ loadings
        explained[output_index] = eigenvalues[order].sum() / eigenvalues.clip(min=0).sum()

    raw_corr = _mean_absolute_off_diagonal_correlation(values[covariance_lookback:])
    residual_corr = _mean_absolute_off_diagonal_correlation(residuals)
    metadata: dict[str, object] = {
        "method": "rolling_pca",
        "n_factors": n_factors,
        "covariance_lookback": covariance_lookback,
        "loading_lookback": loading_lookback,
        "pca_information_end": "t-1",
        "residual_mean": float(residuals.mean()),
        "residual_std": float(residuals.std()),
        "mean_explained_variance": float(explained.mean()),
        "raw_mean_abs_correlation": raw_corr,
        "residual_mean_abs_correlation": residual_corr,
    }
    dataset = ResidualDataset(
        residuals=residuals.astype(np.float32),
        dates=returns.index.to_numpy(dtype="datetime64[D]")[covariance_lookback:],
        tickers=returns.columns.to_numpy(dtype=str),
        metadata=metadata,
    )
    dataset.validate()
    return dataset


def _mean_absolute_off_diagonal_correlation(values: np.ndarray) -> float:
    correlation = np.corrcoef(values, rowvar=False)
    mask = ~np.eye(correlation.shape[0], dtype=bool)
    return float(np.nanmean(np.abs(correlation[mask])))


def save_residual_dataset(dataset: ResidualDataset, path: Path) -> None:
    dataset.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        residuals=dataset.residuals,
        dates=dataset.dates.astype("datetime64[D]").astype(str),
        tickers=dataset.tickers.astype(str),
        metadata=json.dumps(dataset.metadata),
    )


def load_residual_dataset(path: Path) -> ResidualDataset:
    with np.load(path, allow_pickle=False) as data:
        dataset = ResidualDataset(
            residuals=data["residuals"].astype(np.float32),
            dates=data["dates"].astype("datetime64[D]"),
            tickers=data["tickers"].astype(str),
            metadata=json.loads(str(data["metadata"])),
        )
    dataset.validate()
    return dataset
