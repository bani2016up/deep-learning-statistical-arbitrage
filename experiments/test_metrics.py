import numpy as np

from experiments.metrics import compute_metrics, weight_stats


def test_weight_stats_long_short_book():
    weights = np.array([[0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]])
    stats = weight_stats(weights)
    assert stats["turnover"] == 1.0  # (0 + 2) / 2, initial entry excluded
    assert stats["short_fraction"] == 0.5
    assert stats["effective_positions"] == 2.0


def test_compute_metrics_contract():
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 300)
    weights = rng.normal(size=(300, 4))
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    dates = np.arange("2020-01-01", 300, dtype="datetime64[D]")
    metrics = compute_metrics(returns, returns - 1e-4, weights, dates)
    for key in (
        "sharpe",
        "mean_return",
        "volatility",
        "max_drawdown",
        "turnover",
        "net_sharpe",
    ):
        assert np.isfinite(metrics[key])
    assert metrics["net_sharpe"] < metrics["sharpe"]
