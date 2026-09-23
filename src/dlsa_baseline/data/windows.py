from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dlsa_baseline.data.pca_residuals import ResidualDataset


@dataclass(frozen=True)
class WindowDataset:
    windows: np.ndarray  # [decision dates, assets, lookback]
    targets: np.ndarray  # [decision dates, assets], return realized on decision date
    dates: np.ndarray  # [decision dates]
    tickers: np.ndarray  # [assets]


def build_cumulative_windows(dataset: ResidualDataset, lookback: int = 30) -> WindowDataset:
    """Build sample t from residuals [t-lookback, ..., t-1], never residual t."""
    dataset.validate()
    residuals = dataset.residuals
    if not 1 <= lookback < len(residuals):
        raise ValueError("lookback must be positive and shorter than the dataset")
    windows = np.stack(
        [np.cumsum(residuals[t - lookback : t], axis=0).T for t in range(lookback, len(residuals))]
    ).astype(np.float32)
    targets = residuals[lookback:].copy()
    result = WindowDataset(windows, targets, dataset.dates[lookback:], dataset.tickers.copy())
    if result.windows.shape[:2] != result.targets.shape:
        raise AssertionError("Window and target dimensions do not align")
    return result
