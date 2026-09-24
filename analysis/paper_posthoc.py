"""Post-hoc analyses from Guijarro-Ordonez, Pelger, Zanotti (2022) on saved run weights.

- III.K: sparse portfolios (keep the top p% of |w|)
- III.M: holding-period persistence (overlapping weights, B = 1..30)
- Table I: Sharpe by model and number of factors K
- net Sharpe as a function of the cost in bps
- mean and sd across seeds
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from analysis.metrics import (
    annualized_return,
    annualized_sharpe,
    annualized_volatility,
    compute_drawdown_series,
    maximum_drawdown,
    one_way_turnover,
)


def evaluate_sparse_portfolio(
    weights: np.ndarray,
    targets: np.ndarray,
    top_fractions: Sequence[float] = (0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00),
    freq: int = 252,
) -> pd.DataFrame:
    """Evaluates sparse portfolio policies keeping top p% absolute weights (Paper Section III.K).

    Args:
        weights: [T, N] portfolio weights
        targets: [T, N] asset returns realized on decision dates
        top_fractions: fractions of assets to retain (sorted by |w|)
        freq: annualization factor
    """
    T, N = weights.shape
    rows = []

    for frac in top_fractions:
        k = max(1, int(np.round(N * frac)))
        sparse_w = np.zeros_like(weights)

        for t in range(T):
            row_w = weights[t]
            abs_w = np.abs(row_w)
            top_idx = np.argpartition(abs_w, -k)[-k:]
            sparse_w[t, top_idx] = row_w[top_idx]
            l1 = np.sum(np.abs(sparse_w[t]))
            if l1 > 1e-8:
                sparse_w[t] /= l1

        port_ret = np.sum(sparse_w * targets, axis=1)

        diff = np.abs(sparse_w[1:] - sparse_w[:-1]).sum(axis=1)
        daily_to = float(0.5 * diff.mean()) if len(diff) > 0 else 0.0

        sr = annualized_sharpe(port_ret, freq=freq)
        ret = annualized_return(port_ret, freq=freq)
        vol = annualized_volatility(port_ret, freq=freq)
        mdd = maximum_drawdown(port_ret)

        rows.append(
            {
                "top_fraction": frac,
                "n_assets_kept": k,
                "annualized_sharpe": sr,
                "annualized_return": ret,
                "annualized_volatility": vol,
                "maximum_drawdown": mdd,
                "daily_turnover": daily_to,
            }
        )

    return pd.DataFrame(rows)


def evaluate_holding_periods(
    weights: np.ndarray,
    targets: np.ndarray,
    horizons: Sequence[int] = (1, 2, 3, 5, 10, 15, 20, 30),
    freq: int = 252,
) -> pd.DataFrame:
    """Evaluates holding period persistence with overlapping weights (Paper Section III.M, Figure 12).

    Holding period B averages the last B target weights:
    w_{B, t} = (1 / B) * sum_{l=0}^{B-1} w_{t-l}
    """
    T, N = weights.shape
    rows = []

    for B in horizons:
        if B < 1:
            continue
        w_B = np.zeros_like(weights)
        for t in range(T):
            start_l = max(0, t - B + 1)
            window = weights[start_l : t + 1]
            avg_w = window.mean(axis=0)
            l1 = np.sum(np.abs(avg_w))
            if l1 > 1e-8:
                avg_w /= l1
            w_B[t] = avg_w

        port_ret = np.sum(w_B * targets, axis=1)

        diff = np.abs(w_B[1:] - w_B[:-1]).sum(axis=1)
        daily_to = float(0.5 * diff.mean()) if len(diff) > 0 else 0.0

        sr = annualized_sharpe(port_ret, freq=freq)
        ret = annualized_return(port_ret, freq=freq)
        vol = annualized_volatility(port_ret, freq=freq)
        mdd = maximum_drawdown(port_ret)

        cost_5bps = 2.0 * daily_to * 0.0005 * freq
        net_ret_5bps = ret - cost_5bps
        net_sr_5bps = net_ret_5bps / vol if vol > 1e-8 else 0.0

        rows.append(
            {
                "holding_days": B,
                "annualized_sharpe": sr,
                "annualized_return": ret,
                "annualized_volatility": vol,
                "maximum_drawdown": mdd,
                "daily_turnover": daily_to,
                "net_sharpe_5bps": net_sr_5bps,
            }
        )

    return pd.DataFrame(rows)


def compute_cost_sensitivity_curves(
    runs_summary: pd.DataFrame,
    bps_range: Sequence[float] = np.linspace(0, 30, 61),
    freq: int = 252,
) -> pd.DataFrame:
    """Computes Net Sharpe as a function of transaction costs in bps for each run in summary."""
    records = []
    for _, row in runs_summary.iterrows():
        name = row.get("run", row.get("run_name", "unknown"))
        ret = row.get("annualized_return", row.get("mean_return", 0.0))
        vol = row.get("annualized_volatility", row.get("volatility", 0.0))
        to = one_way_turnover(row) or 0.0

        if pd.isnull(ret) or pd.isnull(vol) or vol <= 1e-8:
            continue

        for bps in bps_range:
            cost = 2.0 * to * (bps / 10000.0) * freq
            net_ret = ret - cost
            net_sr = net_ret / vol
            records.append(
                {
                    "run": name,
                    "cost_bps": bps,
                    "net_return": net_ret,
                    "net_sharpe": net_sr,
                }
            )

    return pd.DataFrame(records)


def aggregate_seed_dispersion(
    summary_df: pd.DataFrame,
    groupby: Sequence[str] = ("suite", "variant"),
    metrics: Sequence[str] = (
        "sharpe",
        "mean_return",
        "volatility",
        "turnover",
        "net_sharpe",
    ),
) -> pd.DataFrame:
    """Aggregates multi-seed runs by (suite, variant), calculating mean and std."""
    available_group = [c for c in groupby if c in summary_df.columns]
    if not available_group:
        return summary_df

    available_metrics = [c for c in metrics if c in summary_df.columns]
    agg_funcs = {m: ["mean", "std", "count"] for m in available_metrics}
    grouped = summary_df.groupby(available_group).agg(agg_funcs)
    grouped.columns = [f"{col}_{stat}" for col, stat in grouped.columns]
    return grouped.reset_index()


def generate_table1_pivot(
    summary_df: pd.DataFrame,
    metric: str = "sharpe",
    model_col: str = "cfg_model",
    factor_col: str = "cfg_n_factors",
) -> pd.DataFrame:
    """Generates Table-I style pivot: metric across Model (rows) and K Factors (cols)."""
    if model_col not in summary_df.columns:
        model_col = next((c for c in summary_df.columns if "model" in c), "model")
    if factor_col not in summary_df.columns:
        factor_col = next((c for c in summary_df.columns if "factor" in c), "n_factors")

    if (
        model_col not in summary_df.columns
        or factor_col not in summary_df.columns
        or metric not in summary_df.columns
    ):
        return pd.DataFrame()

    pivot = summary_df.pivot_table(
        index=model_col,
        columns=factor_col,
        values=metric,
        aggfunc="mean",
    )
    return pivot


def compute_drift_decomposition(
    weights: np.ndarray,
    targets: np.ndarray,
    freq: int = 252,
) -> Dict[str, float]:
    """Decomposes portfolio returns into residual drift (net_exposure * EW-residual) and neutral remainder."""
    total_ret = np.sum(weights * targets, axis=1)
    net_exposure = np.sum(weights, axis=1)
    ew_residual = np.mean(targets, axis=1)

    drift_ret = net_exposure * ew_residual
    neutral_ret = total_ret - drift_ret

    return {
        "total_sharpe": annualized_sharpe(total_ret, freq=freq),
        "total_return": annualized_return(total_ret, freq=freq),
        "total_volatility": annualized_volatility(total_ret, freq=freq),
        "drift_sharpe": annualized_sharpe(drift_ret, freq=freq),
        "drift_return": annualized_return(drift_ret, freq=freq),
        "drift_volatility": annualized_volatility(drift_ret, freq=freq),
        "neutral_sharpe": annualized_sharpe(neutral_ret, freq=freq),
        "neutral_return": annualized_return(neutral_ret, freq=freq),
        "neutral_volatility": annualized_volatility(neutral_ret, freq=freq),
        "mean_net_exposure": float(np.mean(net_exposure)),
        "mean_abs_net_exposure": float(np.mean(np.abs(net_exposure))),
    }
