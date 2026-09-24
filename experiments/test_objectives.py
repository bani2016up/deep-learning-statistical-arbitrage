import pytest
import torch

from experiments.objectives import loss, trading_costs


def test_trading_costs_match_paper_formula():
    weights = torch.tensor([[0.5, -0.5], [0.25, -0.75]])
    costs, turnover, short = trading_costs(weights, turnover_bps=5, short_bps=1)
    assert turnover.tolist() == [1.0, 0.5]
    assert short.tolist() == [0.5, 0.75]
    assert costs[1].item() == pytest.approx(0.5 * 5e-4 + 0.75 * 1e-4)


def test_cost_aware_sharpe_is_lower_than_gross():
    returns = torch.tensor([0.01, 0.02, -0.005, 0.015])
    weights = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, -1.0]])
    assert loss("sharpe_costs", returns, weights) > loss("sharpe", returns, weights)


def test_unknown_objective_raises():
    with pytest.raises(ValueError):
        loss("nope", torch.zeros(3), torch.zeros(3, 2))


def test_neutralized_allocation_is_dollar_neutral_unit_gross():
    from experiments.objectives import allocate

    weights = allocate(torch.tensor([[3.0, 1.0, 1.0, -1.0]]), neutralize=True)
    assert weights.sum().item() == pytest.approx(0.0, abs=1e-7)
    assert weights.abs().sum().item() == pytest.approx(1.0)
