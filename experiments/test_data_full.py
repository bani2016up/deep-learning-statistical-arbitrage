import numpy as np
import pandas as pd

from experiments.data_full import (
    _complete_before,
    _recent_extreme,
    eligibility,
    ff5_residuals,
    monthly_universe,
    pca_residuals,
)


def _dates(n):
    return pd.bdate_range("2000-01-03", periods=n).to_numpy("datetime64[D]")


def test_universe_uses_only_pre_month_volume():
    rng = np.random.default_rng(0)
    dates = _dates(200)
    volume = rng.lognormal(size=(200, 20))
    base = monthly_universe(volume, dates, top=5, lookback=40)
    months = pd.DatetimeIndex(dates).to_period("M")
    month_start = int(np.flatnonzero(months == months[120])[0])
    changed = volume.copy()
    changed[month_start:] *= rng.lognormal(sigma=3, size=changed[month_start:].shape)
    np.testing.assert_array_equal(
        monthly_universe(changed, dates, 5, 40)[: month_start + 5],
        base[: month_start + 5],
    )
    assert (base.sum(axis=1)[month_start:] == 5).all()


def test_complete_before_and_extreme_windows():
    values = np.ones((10, 1))
    values[4] = np.nan
    complete = _complete_before(values, 3)[:, 0]
    assert complete.tolist() == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    ]
    returns = np.zeros((10, 1))
    returns[2] = 0.9
    assert (
        _recent_extreme(returns, 0.5, 3)[:, 0].tolist()
        == [False] * 3 + [True] * 3 + [False] * 4
    )


def test_ff5_residuals_vanish_for_exact_factor_returns_and_do_not_look_ahead():
    rng = np.random.default_rng(1)
    factors = rng.normal(scale=0.01, size=(150, 5))
    returns = factors @ rng.normal(size=(5, 8))
    residuals = ff5_residuals(returns, factors, window=60)
    assert np.nanmax(np.abs(residuals)) < 1e-10
    noisy = returns + rng.normal(scale=0.01, size=returns.shape)
    base = ff5_residuals(noisy, factors, window=60)
    changed = noisy.copy()
    changed[100:] = 5.0
    np.testing.assert_allclose(
        ff5_residuals(changed, factors, window=60)[:100], base[:100]
    )


def test_pca_residuals_do_not_look_ahead():
    rng = np.random.default_rng(2)
    returns = rng.normal(scale=0.01, size=(120, 12))
    universe = np.ones_like(returns, dtype=bool)
    base = pca_residuals(returns, universe, (2,), cov_window=60, loading_window=30)[2]
    changed = returns.copy()
    changed[90:] = rng.normal(scale=0.2, size=changed[90:].shape)
    np.testing.assert_allclose(
        pca_residuals(changed, universe, (2,), 60, 30)[2][:90], base[:90]
    )
    assert np.isfinite(base[60:]).all() and np.isnan(base[:60]).all()


def test_eligibility_requires_universe_window_and_no_recent_extreme():
    residuals = np.zeros((400, 3))
    residuals[100, 1] = np.nan
    returns = np.zeros((400, 3))
    returns[200, 2] = 3.0
    universe = np.ones((400, 3), dtype=bool)
    universe[:, 0] = False
    eligible = eligibility(residuals, returns, universe, window=30)
    assert not eligible[:, 0].any()
    assert not eligible[101:131, 1].any() and eligible[131, 1]
    assert not eligible[201:453, 2].any() and eligible[199, 2]
