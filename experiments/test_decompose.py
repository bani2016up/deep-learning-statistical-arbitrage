import numpy as np
import pytest

from experiments.decompose import decompose


def test_drift_plus_neutral_reconstructs_returns():
    rng = np.random.default_rng(0)
    targets = rng.normal(0.0005, 0.01, size=(400, 8))
    weights = rng.normal(0.2, 1.0, size=(400, 8))
    weights /= np.abs(weights).sum(axis=1, keepdims=True)
    parts = decompose(weights, targets)
    total = (weights * targets).sum(axis=1)
    assert parts["mean_drift"] + parts["mean_neutral"] == pytest.approx(
        total.mean() * 252
    )


def test_dollar_neutral_book_has_no_drift():
    targets = np.random.default_rng(1).normal(0.001, 0.01, size=(300, 4))
    weights = np.tile([0.25, 0.25, -0.25, -0.25], (300, 1))
    parts = decompose(weights, targets)
    assert parts["net_exposure"] == pytest.approx(0.0)
    assert parts["mean_drift"] == pytest.approx(0.0)
