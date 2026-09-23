from __future__ import annotations

import numpy as np
import pandas as pd

from dlsa_baseline.data.pca_residuals import (
    load_residual_dataset,
    rolling_pca_residuals,
    save_residual_dataset,
)
from dlsa_baseline.data.returns import compute_returns
from dlsa_baseline.data.windows import build_cumulative_windows


def panel(rows: int = 90, assets: int = 6, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(rows, 2))
    values = factors @ rng.normal(size=(2, assets)) + 0.2 * rng.normal(size=(rows, assets))
    return pd.DataFrame(
        values / 100,
        index=pd.date_range("2020-01-01", periods=rows),
        columns=list("ABCDEF")[:assets],
    )


def test_return_computation_does_not_fill_missing_prices() -> None:
    prices = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": [10.0, np.nan, 12.0]})
    result = compute_returns(prices, min_history_fraction=0.5)
    assert np.isclose(result["A"].iloc[0], 0.1)
    assert "B" not in result or result["B"].notna().all()


def test_pca_shapes_and_no_lookahead() -> None:
    returns = panel()
    first = rolling_pca_residuals(returns, 2, covariance_lookback=30, loading_lookback=15)
    changed = returns.copy()
    changed.iloc[40:] *= 100
    second = rolling_pca_residuals(changed, 2, covariance_lookback=30, loading_lookback=15)
    assert first.residuals.shape == (60, 6)
    # Output zero is date index 30 and cannot depend on modifications from index 40 onward.
    np.testing.assert_allclose(first.residuals[:10], second.residuals[:10], atol=1e-6)
    assert first.metadata["pca_information_end"] == "t-1"


def test_residual_round_trip(tmp_path) -> None:
    expected = rolling_pca_residuals(panel(), 2, 30, 15)
    path = tmp_path / "residuals.npz"
    save_residual_dataset(expected, path)
    actual = load_residual_dataset(path)
    np.testing.assert_array_equal(actual.residuals, expected.residuals)
    np.testing.assert_array_equal(actual.dates, expected.dates)
    np.testing.assert_array_equal(actual.tickers, expected.tickers)


def test_window_excludes_trade_date_return() -> None:
    dataset = rolling_pca_residuals(panel(), 2, 30, 15)
    windows = build_cumulative_windows(dataset, lookback=5)
    expected = np.cumsum(dataset.residuals[:5], axis=0).T
    np.testing.assert_allclose(windows.windows[0], expected)
    np.testing.assert_array_equal(windows.targets[0], dataset.residuals[5])
    assert windows.dates[0] == dataset.dates[5]
