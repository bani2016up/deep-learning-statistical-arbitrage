import numpy as np
import pandas as pd
import pytest

from experiments.compare import (
    TEST_START,
    block_bootstrap_diff,
    period_slice,
    residual_key,
    sharpe,
)


def test_sharpe_annualizes_and_handles_flat_series():
    returns = np.array([0.01, -0.01] * 50) + 0.001
    assert sharpe(returns) == pytest.approx(0.001 / 0.01 * np.sqrt(252))
    assert np.isnan(sharpe(np.zeros(10)))


def test_bootstrap_identical_series_is_centered_on_zero():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0005, 0.01, 1000)
    low, high, share = block_bootstrap_diff(returns, returns, draws=200)
    assert low == high == 0.0 and share == 1.0


def test_bootstrap_detects_a_clearly_better_series():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.01, 2000)
    better, worse = noise + 0.002, noise * 0.5 + rng.normal(0, 0.01, 2000)
    low, high, share = block_bootstrap_diff(better, worse, draws=300)
    assert low > 0 and high > low and share < 0.01


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=500), rng.normal(size=500)
    assert block_bootstrap_diff(a, b, draws=50) == block_bootstrap_diff(a, b, draws=50)


def test_periods_split_at_test_start_without_overlap():
    dates = pd.bdate_range("2005-12-20", "2006-01-10")
    frame = pd.DataFrame({"date": dates, "return": 0.0})
    validation, test = period_slice(frame, "validation"), period_slice(frame, "test")
    assert validation.date.max() < TEST_START <= test.date.min()
    assert len(validation) + len(test) == len(period_slice(frame, "full"))


def test_residual_key_distinguishes_factor_models():
    assert residual_key({"n_factors": 5}) == "pca5"
    assert residual_key({"factor_model": "pca", "n_factors": 8}) == "pca8"
    assert residual_key({"factor_model": "ff5", "n_factors": 5}) == "ff5"
