"""Unified per-run metrics. Keys are the contract with analysis/ (see experiments/README.md)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dlsa_baseline.analysis.statistics import newey_west_mean_test, performance_metrics

ANNUALIZATION = 252


def _sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=0)
    return (
        float(returns.mean() / std * math.sqrt(ANNUALIZATION))
        if std > 0
        else float("nan")
    )


def weight_stats(weights: np.ndarray) -> dict[str, float]:
    previous = np.vstack([np.zeros_like(weights[:1]), weights[:-1]])
    turnover = np.abs(weights - previous).sum(axis=1)[
        1:
    ]  # exclude the initial entry trade
    gross = np.abs(weights).sum(axis=1)
    active = gross > 0
    squared = (weights[active] ** 2).sum(axis=1) / gross[active] ** 2
    return {
        "turnover": float(turnover.mean()) if len(turnover) else 0.0,
        "short_fraction": float(
            (np.clip(weights, None, 0).sum(axis=1) * -1)[active].mean()
        ),
        "net_exposure": float(weights.sum(axis=1).mean()),
        "effective_positions": float((1.0 / squared).mean()) if active.any() else 0.0,
        "active_days_fraction": float(active.mean()),
    }


def compute_metrics(
    returns: np.ndarray,
    net_returns: np.ndarray,
    weights: np.ndarray,
    dates: np.ndarray,
) -> dict[str, Any]:
    """Headline metrics on gross returns plus the paper's cost model (5bp turnover, 1bp short)."""
    returns = np.asarray(returns, dtype=float)
    net_returns = np.asarray(net_returns, dtype=float)
    base = performance_metrics(returns, dates, ANNUALIZATION)
    hac = newey_west_mean_test(returns)
    stats = weight_stats(weights)
    mean_daily = returns.mean()
    return {
        "observations": base["observations"],
        "start_date": base["start_date"],
        "end_date": base["end_date"],
        "sharpe": base["annualized_sharpe"],
        "mean_return": base["annualized_mean"],
        "volatility": base["annualized_volatility"],
        "sortino": base["annualized_sortino"],
        "calmar": base["calmar_ratio"],
        "cagr": base["cagr"],
        "total_return": base["total_return"],
        "max_drawdown": base["maximum_drawdown"],
        "max_drawdown_duration_days": base["maximum_drawdown_duration_days"],
        "hit_rate": base["positive_day_fraction"],
        "positive_month_fraction": base["positive_month_fraction"],
        "skewness": base["skewness"],
        "excess_kurtosis": base["excess_kurtosis"],
        "var_95": base["historical_var_95"],
        "cvar_95": base["historical_cvar_95"],
        "hac_t_stat": hac["t_statistic"],
        "hac_p_value": hac["two_sided_p_value"],
        **stats,
        "break_even_cost_bps": (
            float(mean_daily * 1e4 / stats["turnover"])
            if stats["turnover"] > 0
            else float("nan")
        ),
        "net_sharpe": _sharpe(net_returns),
        "net_mean_return": float(net_returns.mean() * ANNUALIZATION),
        "net_volatility": float(net_returns.std(ddof=0) * math.sqrt(ANNUALIZATION)),
        "yearly_returns": base["yearly_returns"],
    }
