import numpy as np
import pytest
import torch

from experiments.models import (
    TRAINABLE,
    STATIC,
    build_model,
    fourier_features,
    ou_signals,
)


@pytest.mark.parametrize("name", sorted(TRAINABLE | STATIC))
def test_models_map_windows_to_scores(name):
    model = build_model(name, lookback=30, dropout=0.0)
    scores = model(torch.randn(64, 30).cumsum(-1))
    assert scores.shape == (64,)
    assert torch.isfinite(scores).all()


def test_fourier_features_are_invertible_length():
    x = torch.randn(5, 30)
    assert fourier_features(x).shape == (5, 30)


def test_ou_signals_recover_ar1_coefficient():
    rng = np.random.default_rng(0)
    b_true, n = 0.8, 5000
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.1 + b_true * x[t - 1] + rng.normal(scale=0.1)
    signals = ou_signals(torch.tensor(x[None]))
    assert signals["b"].item() == pytest.approx(b_true, abs=0.02)
    assert signals["mu"].item() == pytest.approx(0.1 / (1 - b_true), abs=0.05)


def test_ou_threshold_is_contrarian_on_stretched_residual():
    x = torch.sin(torch.linspace(0, 12, 30)) * 0.1
    x[-1] = 0.5  # far above the fitted mean
    score = build_model("ou_threshold", 30, 0.0)(x[None])
    assert score.item() in (-1.0, 0.0)
