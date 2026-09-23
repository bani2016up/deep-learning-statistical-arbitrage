from __future__ import annotations

import numpy as np
import torch

from dlsa_baseline.data.pca_residuals import ResidualDataset
from dlsa_baseline.data.windows import build_cumulative_windows
from dlsa_baseline.models.cnn_transformer import CNNTransformer
from dlsa_baseline.training.trainer import train_model


def test_tiny_end_to_end_training_epoch() -> None:
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 0.01, size=(90, 5)).astype(np.float32)
    dataset = ResidualDataset(
        residuals,
        np.arange("2021-01-01", "2021-04-01", dtype="datetime64[D]"),
        np.array(list("ABCDE")),
        {"method": "test"},
    )
    windows = build_cumulative_windows(dataset, lookback=10)
    result = train_model(
        CNNTransformer(dropout=0.0), windows, torch.device("cpu"), epochs=1, batch_dates=24
    )
    assert len(result["history"]) == 1
    assert np.isfinite(result["test_returns"]).all()
    assert np.isfinite(result["history"][0]["train_loss"])
