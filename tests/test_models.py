from __future__ import annotations

import torch

from dlsa_baseline.models.baselines import RawFFN
from dlsa_baseline.models.cnn import CausalCNN
from dlsa_baseline.models.cnn_transformer import CNNTransformer
from dlsa_baseline.models.transformer import TemporalTransformer


def test_component_and_full_model_shapes() -> None:
    x = torch.randn(7, 30)
    cnn_output = CausalCNN()(x)
    assert cnn_output.shape == (7, 30, 8)
    transformer_output = TemporalTransformer(dropout=0.0)(cnn_output)
    assert transformer_output.shape == (7, 30, 8)
    output = CNNTransformer(dropout=0.0)(x)
    assert output.shape == (7,)
    assert torch.isfinite(output).all()


def test_models_backward() -> None:
    for model in (CNNTransformer(dropout=0.0), RawFFN()):
        output = model(torch.randn(8, 30)).sum()
        output.backward()
        assert any(parameter.grad is not None for parameter in model.parameters())
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        )
