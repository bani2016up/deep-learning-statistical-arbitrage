from __future__ import annotations

import torch


def sharpe_ratio(returns: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    if returns.ndim != 1 or returns.numel() < 2:
        raise ValueError("Sharpe requires at least two one-dimensional returns")
    if not torch.isfinite(returns).all():
        raise ValueError("Portfolio returns contain NaN or infinity")
    value = returns.mean() / returns.std(unbiased=False).clamp_min(epsilon)
    if not torch.isfinite(value):
        raise FloatingPointError("Non-finite Sharpe ratio")
    return value


def negative_sharpe(returns: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    return -sharpe_ratio(returns, epsilon)


def negative_mean_variance(returns: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    return -(returns.mean() - gamma * returns.var(unbiased=False))
