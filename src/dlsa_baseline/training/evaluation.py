from __future__ import annotations

import math

import numpy as np
import torch

from dlsa_baseline.models.allocation import normalize_l1


def scores_to_portfolio_returns(
    scores: torch.Tensor, next_returns: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply scores made from t-1 information to aligned date-t residual returns."""
    if scores.shape != next_returns.shape or scores.ndim != 2:
        raise ValueError("scores and next_returns must both be [dates, assets]")
    weights = normalize_l1(scores)
    portfolio_returns = (weights * next_returns).sum(dim=1)
    return portfolio_returns, weights


def metrics(returns: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    returns = np.asarray(returns, dtype=float)
    annual_mean = float(returns.mean() * 252)
    annual_volatility = float(returns.std(ddof=0) * math.sqrt(252))
    result = {
        "annualized_mean": annual_mean,
        "annualized_volatility": annual_volatility,
        "annualized_sharpe": annual_mean / max(annual_volatility, 1e-12),
        "observations": float(len(returns)),
    }
    if weights is not None:
        changes = np.abs(np.diff(weights, axis=0)).sum(axis=1)
        result["average_turnover"] = float(changes.mean()) if len(changes) else 0.0
        result["number_of_trades"] = float(
            np.count_nonzero(np.abs(np.diff(weights, axis=0)) > 1e-8)
        )
    return result
