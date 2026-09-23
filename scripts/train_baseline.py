from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from dlsa_baseline.config import ROOT, DataConfig, ModelConfig, TrainConfig
from dlsa_baseline.data.pca_residuals import load_residual_dataset
from dlsa_baseline.data.windows import build_cumulative_windows
from dlsa_baseline.models import CNNTransformer, RawFFN
from dlsa_baseline.training.trainer import train_model
from dlsa_baseline.utils.runtime import device_description, seed_everything, select_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", choices=["cnn_transformer", "raw_ffn"], default="cnn_transformer"
    )
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    data_config, model_config, train_config = DataConfig(), ModelConfig(), TrainConfig()
    seed_everything(train_config.seed)
    device = select_device() if args.device == "auto" else torch.device(args.device)
    print(f"Selected device: {device_description(device)}")

    residuals = load_residual_dataset(ROOT / "data/processed/pca_residuals.npz")
    windows = build_cumulative_windows(residuals, data_config.lookback)
    if args.model == "cnn_transformer":
        model = CNNTransformer(**model_config.__dict__)
    else:
        model = RawFFN(data_config.lookback, model_config.dropout)
    checkpoint = ROOT / f"outputs/{args.model}.pt"
    result = train_model(
        model,
        windows,
        device,
        epochs=args.epochs,
        batch_dates=train_config.batch_dates,
        learning_rate=train_config.learning_rate,
        train_fraction=train_config.train_fraction,
        validation_fraction=train_config.validation_fraction,
        epsilon=train_config.epsilon,
        checkpoint_path=checkpoint,
    )
    history = result["history"]
    plt.figure(figsize=(7, 4))
    plt.plot([row["epoch"] for row in history], [row["train_loss"] for row in history])
    plt.xlabel("Epoch")
    plt.ylabel("Negative batch Sharpe")
    plt.tight_layout()
    plt.savefig(ROOT / "outputs/training_loss.png", dpi=150)
    plt.close()

    test_returns = np.asarray(result["test_returns"])
    plt.figure(figsize=(8, 4))
    plt.plot(result["test_dates"], np.cumprod(1 + test_returns))
    plt.xlabel("Date")
    plt.ylabel("Cumulative wealth")
    plt.tight_layout()
    plt.savefig(ROOT / "outputs/cumulative_test_return.png", dpi=150)
    plt.close()
    np.savez_compressed(
        ROOT / "outputs/test_predictions.npz",
        returns=test_returns,
        weights=result["test_weights"],
        dates=np.asarray(result["test_dates"]).astype(str),
        tickers=windows.tickers,
    )
    test_start = result["split"].validation_end
    reversal_scores = -windows.windows[test_start:, :, -1]
    reversal_weights = reversal_scores / np.maximum(
        np.abs(reversal_scores).sum(axis=1, keepdims=True), 1e-8
    )
    reversal_returns = (reversal_weights * windows.targets[test_start:]).sum(axis=1)
    np.savez_compressed(
        ROOT / "outputs/reversal_predictions.npz",
        returns=reversal_returns,
        weights=reversal_weights,
        dates=np.asarray(result["test_dates"]).astype(str),
        tickers=windows.tickers,
    )
    metrics_path = ROOT / "outputs/test_metrics.json"
    metrics_path.write_text(json.dumps(result["test_metrics"], indent=2))
    print(json.dumps(result["test_metrics"], indent=2))
    print(f"Runtime seconds: {result['runtime_seconds']:.2f}")
    if device.type == "cuda":
        print(f"Peak allocated VRAM bytes: {torch.cuda.max_memory_allocated(device)}")


if __name__ == "__main__":
    main()
