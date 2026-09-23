from __future__ import annotations

import json

import numpy as np
import pandas as pd

from dlsa_baseline.analysis.reporting import create_plots, write_json_report, write_markdown_report
from dlsa_baseline.analysis.statistics import (
    PredictionData,
    analyze_predictions,
    benjamini_hochberg,
    circular_block_bootstrap,
    factor_regression,
    newey_west_mean_test,
    performance_metrics,
    probability_of_backtest_overfitting,
    transaction_cost_sensitivity,
)


def prediction_data(seed: int = 5) -> PredictionData:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0008, 0.01, 300)
    raw_weights = rng.normal(size=(300, 6))
    weights = raw_weights / np.abs(raw_weights).sum(axis=1, keepdims=True)
    return PredictionData(
        returns=returns,
        dates=np.arange("2020-01-01", "2020-10-27", dtype="datetime64[D]"),
        weights=weights,
        tickers=np.array([f"A{i}" for i in range(6)]),
    )


def test_performance_hac_and_bootstrap_are_finite() -> None:
    data = prediction_data()
    performance = performance_metrics(data.returns, data.dates)
    hac = newey_west_mean_test(data.returns)
    first = circular_block_bootstrap(data.returns, samples=200, block_size=10, seed=9)
    second = circular_block_bootstrap(data.returns, samples=200, block_size=10, seed=9)
    assert performance["maximum_drawdown"] <= 0
    assert np.isfinite(performance["annualized_sharpe"])
    assert np.isfinite(hac["t_statistic"])
    assert first["sharpe_confidence_interval_95"] == second["sharpe_confidence_interval_95"]


def test_costs_multiple_testing_and_pbo() -> None:
    data = prediction_data()
    costs = transaction_cost_sensitivity(data.returns, data.weights, [0.0, 10.0])
    assert costs["scenarios"]["10"]["annualized_mean"] < costs["scenarios"]["0"]["annualized_mean"]
    adjusted = benjamini_hochberg({"a": 0.01, "b": 0.04, "c": 0.2})
    assert 0 <= adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1
    strategies = np.column_stack([data.returns, -data.returns + 0.0001])
    pbo = probability_of_backtest_overfitting(strategies, ["a", "b"], blocks=6)
    assert 0 <= pbo["probability_of_backtest_overfitting"] <= 1
    assert pbo["combinations"] == 20


def test_factor_regression_recovers_positive_alpha() -> None:
    rng = np.random.default_rng(12)
    dates = np.arange("2022-01-01", "2022-10-28", dtype="datetime64[D]")
    market = rng.normal(0, 0.01, len(dates))
    returns = 0.001 + 0.5 * market + rng.normal(0, 0.001, len(dates))
    data = PredictionData(returns=returns, dates=dates)
    factors = pd.DataFrame({"market": market}, index=pd.to_datetime(dates))
    result = factor_regression(data, factors)
    assert result["annualized_alpha"] > 0.2
    assert result["alpha_hac_t_statistic"] > 5
    assert abs(result["factor_loadings"]["market"]["coefficient"] - 0.5) < 0.05


def test_full_analysis_writes_reports_and_plots(tmp_path) -> None:
    primary = prediction_data()
    benchmark = prediction_data(seed=8)
    report, plot_data = analyze_predictions(
        primary,
        benchmarks={"benchmark": benchmark},
        bootstrap_samples=200,
        block_size=10,
        pbo_blocks=6,
    )
    write_json_report(report, tmp_path / "report.json")
    write_markdown_report(report, tmp_path / "report.md")
    create_plots(primary, report, plot_data, tmp_path)
    loaded = json.loads((tmp_path / "report.json").read_text())
    assert "success_assessment" in loaded
    assert "pbo" in loaded["multiple_testing"]
    assert (tmp_path / "equity_and_drawdown.png").exists()
    assert (tmp_path / "turnover_cost_sensitivity.png").exists()
