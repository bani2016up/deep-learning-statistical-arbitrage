from __future__ import annotations

import json

import numpy as np

from dlsa_baseline.config import ROOT
from dlsa_baseline.data.pca_residuals import ResidualDataset, load_residual_dataset
from dlsa_baseline.data.windows import build_cumulative_windows
from dlsa_baseline.models.cnn_transformer import CNNTransformer
from dlsa_baseline.training.trainer import train_model
from dlsa_baseline.utils.runtime import device_description, seed_everything, select_device


def synthetic_residuals(seed: int = 7) -> ResidualDataset:
    rng = np.random.default_rng(seed)
    values = np.zeros((360, 12), dtype=np.float32)
    shocks = rng.normal(0, 0.01, values.shape).astype(np.float32)
    for t in range(1, len(values)):
        values[t] = -0.65 * values[t - 1] + shocks[t]
    return ResidualDataset(
        values,
        np.arange("2020-01-01", "2020-12-26", dtype="datetime64[D]")[: len(values)],
        np.array([f"S{i:02d}" for i in range(values.shape[1])]),
        {"method": "synthetic_mean_reverting"},
    )


def run_one(dataset: ResidualDataset, epochs: int) -> dict[str, float]:
    windows = build_cumulative_windows(dataset, 30)
    model = CNNTransformer(dropout=0.0)
    result = train_model(model, windows, select_device(), epochs=epochs, batch_dates=64)
    return dict(result["test_metrics"])


def main() -> None:
    seed_everything(42)
    print(f"Selected device: {device_description(select_device())}")
    synthetic = run_one(synthetic_residuals(), epochs=4)
    print("Synthetic overfit smoke metrics:")
    print(json.dumps(synthetic, indent=2))
    public_path = ROOT / "data/processed/pca_residuals.npz"
    if public_path.exists():
        public = run_one(load_residual_dataset(public_path), epochs=2)
        print("Public-data pipeline smoke metrics (2 epochs):")
        print(json.dumps(public, indent=2))
    else:
        print(
            "Public residual file absent; run download_sample_data.py and build_pca_residuals.py."
        )


if __name__ == "__main__":
    main()
