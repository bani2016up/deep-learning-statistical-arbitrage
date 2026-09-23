from __future__ import annotations

import argparse
import json

import pandas as pd

from dlsa_baseline.config import ROOT, DataConfig
from dlsa_baseline.data.pca_residuals import rolling_pca_residuals, save_residual_dataset
from dlsa_baseline.data.returns import compute_returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factors", type=int, default=5)
    parser.add_argument("--covariance-lookback", type=int, default=252)
    parser.add_argument("--loading-lookback", type=int, default=60)
    args = parser.parse_args()
    config = DataConfig()
    long = pd.read_parquet(ROOT / "data/raw/adjusted_prices.parquet")
    prices = long.pivot(index="date", columns="ticker", values="adjusted_price")
    returns = compute_returns(prices, config.min_history_fraction)
    returns.to_parquet(ROOT / "data/processed/daily_returns.parquet")
    dataset = rolling_pca_residuals(
        returns, args.factors, args.covariance_lookback, args.loading_lookback
    )
    output = ROOT / "data/processed/pca_residuals.npz"
    save_residual_dataset(dataset, output)
    print(f"Saved {dataset.residuals.shape} residual panel to {output}")
    print(json.dumps(dataset.metadata, indent=2))


if __name__ == "__main__":
    main()
