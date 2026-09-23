from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class DataConfig:
    start: str = "2015-01-01"
    end: str = "2026-01-01"
    covariance_lookback: int = 252
    loading_lookback: int = 60
    n_factors: int = 5
    lookback: int = 30
    min_history_fraction: float = 0.95


@dataclass(frozen=True)
class ModelConfig:
    features: int = 8
    cnn_layers: int = 2
    kernel_size: int = 2
    attention_heads: int = 4
    transformer_ff: int = 16
    dropout: float = 0.25


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 12
    batch_dates: int = 125
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    seed: int = 42
    epsilon: float = 1e-6


def config_dict(*configs: object) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for config in configs:
        result.update(asdict(config))
    return result


SAMPLE_TICKERS = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "NVDA",
    "BRK-B",
    "JPM",
    "V",
    "UNH",
    "XOM",
    "JNJ",
    "WMT",
    "MA",
    "PG",
    "LLY",
    "HD",
    "AVGO",
    "CVX",
    "MRK",
    "ABBV",
    "PEP",
    "KO",
    "COST",
    "ADBE",
    "MCD",
    "CSCO",
    "ACN",
    "TMO",
    "CRM",
    "ABT",
    "NFLX",
    "LIN",
    "AMD",
    "ORCL",
    "DHR",
    "TXN",
    "PM",
    "NEE",
    "UPS",
    "RTX",
    "LOW",
    "HON",
    "QCOM",
    "INTC",
    "IBM",
    "CAT",
    "GS",
    "AMGN",
    "SBUX",
)
