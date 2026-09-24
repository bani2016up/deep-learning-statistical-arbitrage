"""Return, risk and cost metrics for a daily return series.

Turnover here is one-way (0.5 * ||w_t - w_{t-1}||_1); cost functions charge twice that, which
equals the paper's cost per unit of ||w_t - w_{t-1}||_1. ``one_way_turnover`` converts the
full-L1 ``turnover`` stored by ``experiments``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_cumulative_returns(returns: Union[np.ndarray, pd.Series]) -> pd.Series:
    """Computes compounding cumulative returns: prod(1 + r_t) - 1."""
    s = pd.Series(returns).dropna()
    return (1.0 + s).cumprod() - 1.0


def compute_drawdown_series(returns: Union[np.ndarray, pd.Series]) -> pd.Series:
    """Computes the drawdown series (0 at peak, negative during drawdown)."""
    s = pd.Series(returns).dropna()
    wealth = (1.0 + s).cumprod()
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return drawdown


def maximum_drawdown(returns: Union[np.ndarray, pd.Series]) -> float:
    """Computes the maximum drawdown as a positive fraction (e.g., 0.15 for 15%)."""
    dd = compute_drawdown_series(returns)
    if len(dd) == 0:
        return 0.0
    return float(-dd.min())


def annualized_return(returns: Union[np.ndarray, pd.Series], freq: int = 252) -> float:
    """Annualized mean return: mean(r) * freq."""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return 0.0
    return float(s.mean() * freq)


def annualized_volatility(
    returns: Union[np.ndarray, pd.Series], freq: int = 252
) -> float:
    """Annualized volatility: std(r, ddof=1) * sqrt(freq)."""
    s = pd.Series(returns).dropna()
    if len(s) <= 1:
        return 0.0
    return float(s.std(ddof=1) * np.sqrt(freq))


def annualized_sharpe(
    returns: Union[np.ndarray, pd.Series],
    risk_free: float = 0.0,
    freq: int = 252,
    eps: float = 1e-8,
) -> float:
    """Annualized Sharpe ratio: (mean(r - rf) / std(r)) * sqrt(freq)."""
    s = pd.Series(returns).dropna() - (risk_free / freq)
    if len(s) <= 1:
        return 0.0
    vol = s.std(ddof=1)
    if vol < eps:
        return 0.0
    return float((s.mean() / vol) * np.sqrt(freq))


def sortino_ratio(
    returns: Union[np.ndarray, pd.Series],
    target_return: float = 0.0,
    freq: int = 252,
    eps: float = 1e-8,
) -> float:
    """Annualized Sortino ratio: (mean(r) - target) / downside_std * sqrt(freq)."""
    s = pd.Series(returns).dropna()
    if len(s) <= 1:
        return 0.0
    downside_diff = np.minimum(0.0, s - target_return / freq)
    downside_variance = np.mean(downside_diff**2)
    downside_vol = np.sqrt(downside_variance)
    if downside_vol < eps:
        return 0.0
    return float(((s.mean() - target_return / freq) / downside_vol) * np.sqrt(freq))


def calmar_ratio(
    returns: Union[np.ndarray, pd.Series], freq: int = 252, eps: float = 1e-8
) -> float:
    """Annualized return divided by maximum drawdown."""
    ret = annualized_return(returns, freq=freq)
    mdd = maximum_drawdown(returns)
    if mdd < eps:
        return 0.0
    return float(ret / mdd)


def win_rate(returns: Union[np.ndarray, pd.Series]) -> float:
    """Percentage of periods with positive return."""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return 0.0
    return float((s > 0).mean())


def profit_factor(returns: Union[np.ndarray, pd.Series], eps: float = 1e-8) -> float:
    """Sum of positive returns divided by absolute sum of negative returns."""
    s = pd.Series(returns).dropna()
    pos_sum = s[s > 0].sum()
    neg_sum = np.abs(s[s < 0].sum())
    if neg_sum < eps:
        return float(np.inf if pos_sum > 0 else 0.0)
    return float(pos_sum / neg_sum)


def value_at_risk(returns: Union[np.ndarray, pd.Series], alpha: float = 0.05) -> float:
    """Historical Value at Risk at confidence 1 - alpha (as a positive loss fraction)."""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return 0.0
    q = np.percentile(s, alpha * 100.0)
    return float(-q)


def conditional_value_at_risk(
    returns: Union[np.ndarray, pd.Series], alpha: float = 0.05
) -> float:
    """Historical CVaR / Expected Shortfall at confidence 1 - alpha."""
    s = pd.Series(returns).dropna()
    if len(s) == 0:
        return 0.0
    q = np.percentile(s, alpha * 100.0)
    tail = s[s <= q]
    if len(tail) == 0:
        return float(-q)
    return float(-tail.mean())


def compute_turnover_from_weights(
    weights_df: pd.DataFrame, freq: int = 252
) -> Dict[str, float]:
    """Computes daily one-way turnover and annualized turnover from portfolio weights panel.

    weights_df: DataFrame of shape [dates, assets], weights sum(abs(w)) <= 1 or similar.
    One-way turnover at date t is: 0.5 * sum(|w_{i,t} - w_{i,t-1}|).
    """
    diff = weights_df.diff().iloc[1:]
    daily_turnover = 0.5 * diff.abs().sum(axis=1)
    mean_daily = float(daily_turnover.mean())
    return {
        "daily_turnover": mean_daily,
        "annualized_turnover": mean_daily * freq,
        "max_daily_turnover": float(daily_turnover.max()),
    }


def one_way_turnover(values: Mapping[str, Any]) -> float | None:
    """One-way daily turnover (this module's convention, 0.5 * ||w_t - w_{t-1}||_1).

    ``experiments`` stores ``turnover`` as the full ||w_t - w_{t-1}||_1, so it is halved.
    """
    value = values.get("daily_turnover")
    if value is not None and not pd.isnull(value):
        return float(value)
    value = values.get("turnover")
    if value is None or pd.isnull(value):
        return None
    return float(value) / 2.0


def evaluate_cost_impact(
    ann_return: float,
    ann_vol: float,
    daily_turnover: float,
    cost_bps: float,
    freq: int = 252,
) -> Dict[str, float]:
    """Evaluates performance net of proportional transaction costs in basis points.

    annual_cost = 2 * daily_turnover * (cost_bps / 10,000) * freq
    (one-way turnover is doubled to account for full round-trip or two-sided volume).
    """
    cost_factor = 2.0 * daily_turnover * (cost_bps / 10000.0) * freq
    net_return = ann_return - cost_factor
    net_sharpe = net_return / ann_vol if ann_vol > 1e-8 else 0.0
    return {
        f"net_return_{int(cost_bps)}bps": float(net_return),
        f"net_sharpe_{int(cost_bps)}bps": float(net_sharpe),
    }


def breakeven_cost_bps(
    ann_return: float, daily_turnover: float, freq: int = 252
) -> float:
    """Calculates the breakeven transaction cost in basis points where net return is zero.

    ann_return - 2 * daily_turnover * (cost_bps / 10000) * freq = 0
    => cost_bps = 10000 * ann_return / (2 * daily_turnover * freq)
    """
    denominator = 2.0 * daily_turnover * freq
    if denominator <= 1e-8 or ann_return <= 0:
        return 0.0
    return float(10000.0 * ann_return / denominator)


def compute_run_metrics(
    returns: Union[np.ndarray, pd.Series],
    daily_turnover: Optional[float] = None,
    freq: int = 252,
    cost_bps_list: Tuple[float, ...] = (5.0, 10.0, 20.0),
) -> Dict[str, Any]:
    """Computes a full dictionary of performance, risk, and friction metrics."""
    s = pd.Series(returns).dropna()
    n_obs = len(s)
    if n_obs == 0:
        return {}

    ann_ret = annualized_return(s, freq=freq)
    ann_vol = annualized_volatility(s, freq=freq)
    sharpe = annualized_sharpe(s, freq=freq)
    mdd = maximum_drawdown(s)
    sortino = sortino_ratio(s, freq=freq)
    calmar = calmar_ratio(s, freq=freq)
    win_r = win_rate(s)
    pf = profit_factor(s)
    var95 = value_at_risk(s, alpha=0.05)
    cvar95 = conditional_value_at_risk(s, alpha=0.05)
    skew = float(s.skew())
    kurt = float(s.kurtosis())

    result: Dict[str, Any] = {
        "n_observations": n_obs,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "annualized_sharpe": sharpe,
        "max_drawdown": mdd,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "win_rate": win_r,
        "profit_factor": pf,
        "var_95": var95,
        "cvar_95": cvar95,
        "skewness": skew,
        "kurtosis": kurt,
    }

    if daily_turnover is not None and daily_turnover > 0:
        result["daily_turnover"] = daily_turnover
        result["annualized_turnover"] = daily_turnover * freq
        result["breakeven_cost_bps"] = breakeven_cost_bps(
            ann_ret, daily_turnover, freq=freq
        )
        for cost in cost_bps_list:
            impact = evaluate_cost_impact(
                ann_ret, ann_vol, daily_turnover, cost_bps=cost, freq=freq
            )
            result.update(impact)

    return result
