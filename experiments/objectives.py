"""Training objectives, including the paper's trading-friction cost model (Section III.J)."""

from __future__ import annotations

import math

import torch

from dlsa_baseline.models.allocation import normalize_l1

ANNUALIZATION = 252
OBJECTIVES = ("sharpe", "mean_variance", "sharpe_costs")


def trading_costs(
    weights: torch.Tensor, turnover_bps: float = 5.0, short_bps: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """cost_t = tc*||w_t - w_{t-1}||_1 + short*||min(w_t, 0)||_1 for [dates, assets] weights.

    The first date has no previous allocation inside the block and is charged as a fresh
    entry from zero, which matches evaluation on a contiguous out-of-sample path.
    """
    previous = torch.cat([torch.zeros_like(weights[:1]), weights[:-1]], dim=0)
    turnover = (weights - previous).abs().sum(dim=1)
    short = weights.clamp(max=0).abs().sum(dim=1)
    costs = turnover * turnover_bps / 1e4 + short * short_bps / 1e4
    return costs, turnover, short


def allocate(
    scores: torch.Tensor, neutralize: bool = False, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """[dates, assets] scores -> unit-gross weights over tradeable names only.

    ``neutralize`` demeans over the tradeable names of each date (dollar-neutral book);
    names outside ``mask`` always get weight 0.
    """
    if mask is not None:
        scores = torch.where(mask, scores, torch.zeros_like(scores))
    if neutralize:
        if mask is None:
            scores = scores - scores.mean(dim=-1, keepdim=True)
        else:
            count = mask.sum(dim=-1, keepdim=True).clamp_min(1)
            mean = scores.sum(dim=-1, keepdim=True) / count
            scores = torch.where(mask, scores - mean, torch.zeros_like(scores))
    return normalize_l1(scores)


def score(
    model: torch.nn.Module, windows: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Run the per-asset policy on tradeable (date, asset) pairs only → [dates, assets]."""
    dates, assets, lookback = windows.shape
    if mask is None or bool(mask.all()):
        return model(windows.reshape(dates * assets, lookback)).reshape(dates, assets)
    scores = torch.zeros((dates, assets), dtype=windows.dtype, device=windows.device)
    if bool(mask.any()):
        scores = scores.index_put((mask,), model(windows[mask]))
    return scores


def policy_returns(
    model: torch.nn.Module,
    windows: torch.Tensor,
    targets: torch.Tensor,
    neutralize: bool = False,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = allocate(score(model, windows, mask), neutralize, mask)
    return (weights * targets).sum(dim=1), weights


def loss(
    name: str,
    returns: torch.Tensor,
    weights: torch.Tensor,
    gamma: float = 1.0,
    turnover_bps: float = 5.0,
    short_bps: float = 1.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if name == "sharpe":
        return -returns.mean() / returns.std(unbiased=False).clamp_min(epsilon)
    if name == "mean_variance":
        # Annualized mean minus gamma * annualized variance (paper eq. 4, gamma = 1).
        return -(
            ANNUALIZATION * returns.mean()
            - gamma * ANNUALIZATION * returns.var(unbiased=False)
        )
    if name == "sharpe_costs":
        net = returns - trading_costs(weights, turnover_bps, short_bps)[0]
        return -net.mean() / net.std(unbiased=False).clamp_min(epsilon)
    raise ValueError(f"Unknown objective {name!r}; expected one of {OBJECTIVES}")


def annualized_sharpe(returns: torch.Tensor) -> float:
    std = float(returns.std(unbiased=False))
    return (
        float(returns.mean()) / std * math.sqrt(ANNUALIZATION)
        if std > 0
        else float("nan")
    )
