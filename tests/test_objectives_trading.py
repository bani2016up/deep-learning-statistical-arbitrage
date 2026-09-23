from __future__ import annotations

import torch

from dlsa_baseline.models.allocation import normalize_l1
from dlsa_baseline.training.evaluation import scores_to_portfolio_returns
from dlsa_baseline.training.objectives import negative_sharpe


def test_l1_normalization() -> None:
    weights = normalize_l1(torch.tensor([[1.0, -2.0, 1.0], [-3.0, 0.0, 1.0]]))
    torch.testing.assert_close(weights.abs().sum(dim=1), torch.ones(2))


def test_sharpe_is_finite_and_differentiable() -> None:
    returns = torch.tensor([0.01, -0.005, 0.02, 0.001], requires_grad=True)
    loss = negative_sharpe(returns)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert returns.grad is not None and torch.isfinite(returns.grad).all()


def test_portfolio_return_alignment() -> None:
    scores = torch.tensor([[1.0, -1.0], [2.0, 0.0]])
    returns_t = torch.tensor([[0.02, -0.01], [-0.03, 0.04]])
    portfolio, weights = scores_to_portfolio_returns(scores, returns_t)
    torch.testing.assert_close(portfolio, torch.tensor([0.015, -0.03]))
    torch.testing.assert_close(weights.abs().sum(dim=1), torch.ones(2))
