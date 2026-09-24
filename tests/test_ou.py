from __future__ import annotations

import numpy as np
import pytest

from dlsa_baseline.analysis.statistics import load_predictions
from dlsa_baseline.data.pca_residuals import ResidualDataset
from dlsa_baseline.models.ou import hysteresis_positions, ou_signal, threshold_positions
from dlsa_baseline.training.ou_backtest import OURule, backtest_ou


def ou_level(b: float, mu: float, sd: float, length: int, assets: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = np.full((length, assets), mu)
    for t in range(1, length):
        level[t] = mu + b * (level[t - 1] - mu) + rng.normal(0, sd * np.sqrt(1 - b**2), assets)
    return level


def returns_of(level: np.ndarray) -> np.ndarray:
    return np.diff(np.vstack([np.zeros(level.shape[1]), level]), axis=0)


def dataset(residuals: np.ndarray) -> ResidualDataset:
    return ResidualDataset(
        residuals=residuals.astype(np.float32),
        dates=np.datetime64("2020-01-01") + np.arange(len(residuals)),
        tickers=np.array([f"a{i}" for i in range(residuals.shape[1])]),
        metadata={},
    )


def test_signal_recovers_ou_parameters_on_long_window() -> None:
    b, mu, sd = 0.9, 0.05, 0.02
    signal = ou_signal(returns_of(ou_level(b, mu, sd, 20_000, 3)))
    np.testing.assert_allclose(np.exp(-signal.kappa), b, atol=0.01)
    np.testing.assert_allclose(signal.mu, mu, atol=0.005)
    np.testing.assert_allclose(signal.sigma_eq, sd, rtol=0.05)
    np.testing.assert_allclose(signal.r2, b**2, atol=0.02)
    assert signal.valid.all()


def test_signal_matches_official_preprocess_formula() -> None:
    window = np.random.default_rng(3).normal(0, 0.02, (30, 200))
    level = np.cumsum(window, axis=0)
    x, y = level[:-1], level[1:]
    cov = ((x - x.mean(0)) * (y - y.mean(0))).mean(0)
    b = cov / x.var(0)
    c = y.mean(0) - b * x.mean(0)
    official_mu = c / (1 - b + 1e-6)
    official_sigma = np.sqrt((y - b * x - c).var(0) / np.abs(1 - b**2 + 1e-6))
    official = (official_mu - level[-1]) / official_sigma  # official sign: (mu - X_L) / sigma
    signal = ou_signal(window)
    keep = signal.valid & (b < 0.99)
    np.testing.assert_allclose(signal.r2, cov**2 / (x.var(0) * y.var(0)))
    np.testing.assert_array_equal(signal.valid, (b > 0) & (b < 1))
    np.testing.assert_allclose(-signal.s_score[keep], official[keep], rtol=1e-3, atol=1e-4)


def test_threshold_rule_is_contrarian() -> None:
    level = ou_level(0.7, 0.0, 0.02, 30, 1, seed=5)
    rich, cheap = level.copy(), level.copy()
    rich[-1], cheap[-1] = 0.2, -0.2
    assert threshold_positions(ou_signal(returns_of(rich)), c_crit=0.0)[0] == -1.0
    assert threshold_positions(ou_signal(returns_of(cheap)), c_crit=0.0)[0] == 1.0
    assert threshold_positions(ou_signal(returns_of(level)), c_thresh=1e9)[0] == 0.0


def test_hysteresis_holds_between_bands_and_closes_near_mean() -> None:
    signal = ou_signal(returns_of(ou_level(0.7, 0.0, 0.02, 30, 4, seed=2)))
    s = np.array([-0.9, -0.3, 0.9, 0.3])  # long holds, long closes, short holds, short closes
    signal = signal.__class__(
        **{**signal.__dict__, "s_score": s, "valid": np.ones(4, bool), "r2": np.ones(4)}
    )
    result = hysteresis_positions(signal, np.array([1.0, 1.0, -1.0, -1.0]))
    np.testing.assert_array_equal(result, [1.0, 0.0, -1.0, 0.0])
    assert threshold_positions(signal).tolist() == [0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("rule", [OURule(c_crit=0.0), OURule(kind="al", c_crit=0.0)])
def test_backtest_has_no_lookahead_and_unit_gross(rule: OURule) -> None:
    residuals = returns_of(ou_level(0.6, 0.0, 0.02, 120, 12, seed=7))
    first = backtest_ou(dataset(residuals), rule)
    changed = residuals.copy()
    changed[80:] *= -5
    second = backtest_ou(dataset(changed), rule)
    decision_80 = 80 - 30  # decision row for date index 80 uses residuals [50, 79]
    full = _full(first, residuals.shape[1])
    np.testing.assert_allclose(first.returns[:decision_80], second.returns[:decision_80])
    np.testing.assert_allclose(full[decision_80], _full(second, residuals.shape[1])[decision_80])
    gross = np.abs(full).sum(axis=1)
    assert np.all(np.isclose(gross, 1.0, atol=1e-6) | (gross == 0))
    np.testing.assert_allclose(first.returns, (full * residuals[30:]).sum(axis=1), rtol=1e-5)


def _full(result, n_assets: int) -> np.ndarray:
    full = np.zeros((len(result.dates), n_assets))
    full[:, [int(t[1:]) for t in result.tickers]] = result.weights
    return full


def test_zero_residual_means_missing_and_is_never_traded() -> None:
    residuals = returns_of(ou_level(0.6, 0.0, 0.02, 90, 5, seed=11))
    residuals[40, 2] = 0.0
    result = _full(backtest_ou(dataset(residuals), OURule(c_crit=0.0)), 5)
    assert np.all(result[41 - 30 : 71 - 30, 2] == 0.0)  # windows containing date index 40


def test_predictions_round_trip_through_analysis_loader(tmp_path) -> None:
    residuals = returns_of(ou_level(0.6, 0.0, 0.02, 80, 6, seed=13))
    result = backtest_ou(dataset(residuals), OURule(c_crit=0.0))
    result.save_predictions(tmp_path / "ou.npz")
    loaded = load_predictions(tmp_path / "ou.npz")
    np.testing.assert_allclose(loaded.returns, result.returns)
    assert loaded.weights.shape == (len(result.dates), len(result.tickers))
