"""Training protocols: paper-style rolling retraining, constant model, and the baseline split."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from dlsa_baseline.utils.runtime import seed_everything
from experiments import objectives
from experiments.models import STATIC, build_model
from experiments.panel import DensePanel, MaskedPanel, as_panel

PROTOCOLS = ("rolling", "constant", "fixed")


@dataclass(frozen=True)
class RunConfig:
    name: str
    model: str = "cnn_transformer"
    n_factors: int = 5
    lookback: int = 30
    objective: str = "sharpe"
    gamma: float = 1.0
    turnover_bps: float = 5.0  # paper cost model, used by sharpe_costs and net metrics
    short_bps: float = 1.0
    protocol: str = "rolling"
    train_window: int = 1000  # rolling / constant: days in the training window
    test_start: int = (
        1000  # rolling / constant: first OOS date index, shared across runs
    )
    retrain_every: int = 125
    epochs: int = 30
    batch_dates: int = 125
    learning_rate: float = 1e-3
    dropout: float = 0.25
    features: int = 8
    attention_heads: int = 4
    transformer_ff: int = 16
    seed: int = 0
    device: str = "cpu"
    neutralize: bool = False  # demean scores per date -> dollar-neutral residual book
    cost_weight: float = (
        1.0  # scales the cost penalty in the loss only; evaluation is unscaled
    )
    dataset: str = "yahoo"  # "yahoo" (49-stock sample) | "full" (team WIKI/FF5 panel)
    factor_model: str = "pca"  # full dataset only: "pca" (uses n_factors) | "ff5" | "official_{ff,pca,ipca}"
    max_blocks: int = (
        0  # rolling only: train/test just the first N blocks (0 = all); probes
    )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _model(config: RunConfig) -> torch.nn.Module:
    return build_model(
        config.model,
        config.lookback,
        config.dropout,
        features=config.features,
        attention_heads=config.attention_heads,
        transformer_ff=config.transformer_ff,
    )


def predict_weights(
    model: torch.nn.Module,
    panel: DensePanel | MaskedPanel,
    start: int,
    end: int,
    device: torch.device,
    neutralize: bool = False,
    chunk: int = 125,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for batch_start in range(start, end, chunk):
            windows, _, mask = panel.batch(
                batch_start, min(batch_start + chunk, end), device
            )
            scores = objectives.score(model, windows, mask)
            parts.append(objectives.allocate(scores, neutralize, mask).cpu().numpy())
    return np.concatenate(parts)


def fit(
    model: torch.nn.Module,
    panel: DensePanel | MaskedPanel,
    start: int,
    end: int,
    config: RunConfig,
    device: torch.device,
    validation: tuple[int, int] | None = None,
) -> list[dict[str, float]]:
    """Adam over consecutive non-overlapping date batches (paper Appendix D).

    With ``validation`` the best-validation-Sharpe epoch is restored (baseline protocol);
    without it the final epoch is kept (paper protocol, fixed epoch budget).
    """
    model.to(device)
    if config.model in STATIC:
        return []
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []
    best_state, best_score = None, -float("inf")
    for epoch in range(config.epochs):
        model.train()
        losses = []
        for batch_start in range(start, end, config.batch_dates):
            batch_end = min(batch_start + config.batch_dates, end)
            if batch_end - batch_start < 2:
                continue
            windows, targets, mask = panel.batch(batch_start, batch_end, device)
            returns, weights = objectives.policy_returns(
                model, windows, targets, config.neutralize, mask
            )
            value = objectives.loss(
                config.objective,
                returns,
                weights,
                config.gamma,
                config.turnover_bps * config.cost_weight,
                config.short_bps * config.cost_weight,
            )
            optimizer.zero_grad(set_to_none=True)
            value.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            losses.append(float(value.detach()))
        row = {"epoch": epoch + 1, "train_loss": float(np.mean(losses))}
        if validation is not None:
            weights = predict_weights(
                model, panel, *validation, device, config.neutralize
            )
            returns = (weights * panel.targets(*validation)).sum(axis=1)
            row["validation_sharpe"] = float(
                returns.mean() / max(returns.std(), 1e-12) * 252**0.5
            )
            if row["validation_sharpe"] > best_score:
                best_score = row["validation_sharpe"]
                best_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
        history.append(row)
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def _blocks(
    config: RunConfig, n_dates: int
) -> tuple[list[tuple[int, int, int, int]], tuple[int, int] | None]:
    """(train_start, train_end, test_start, test_end) per retraining block, plus validation."""
    if config.protocol not in PROTOCOLS:
        raise ValueError(f"Unknown protocol {config.protocol!r}")
    if config.protocol == "rolling":
        if config.train_window > config.test_start:
            raise ValueError("train_window cannot exceed test_start")
        blocks = [
            (
                test_start - config.train_window,
                test_start,
                test_start,
                min(test_start + config.retrain_every, n_dates),
            )
            for test_start in range(config.test_start, n_dates, config.retrain_every)
        ]
        return (blocks[: config.max_blocks] if config.max_blocks else blocks), None
    if config.protocol == "constant":
        start = config.test_start
        return [(start - config.train_window, start, start, n_dates)], None
    # baseline 60/20/20 chronological split with validation checkpoint selection
    train_end = int(n_dates * 0.6)
    validation_end = train_end + int(n_dates * 0.2)
    return [(0, train_end, validation_end, n_dates)], (train_end, validation_end)


def run_protocol(
    config: RunConfig,
    data: object,
    checkpoint_dir: Path | None = None,
    on_block: Callable[[int, dict], None] | None = None,
) -> dict[str, object]:
    """Return out-of-sample weights [dates, assets] with the retraining block of each date.

    With ``checkpoint_dir`` each finished block is saved, and a restarted run (e.g. after a
    Kaggle session limit) resumes from the first missing block.
    """
    panel = as_panel(data)
    device = torch.device(config.device)
    started = time.perf_counter()
    blocks, validation = _blocks(config, len(panel))
    history: list[dict[str, float]] = []
    all_weights, block_ids = [], []
    for block, (train_start, train_end, test_start, test_end) in enumerate(blocks):
        path = checkpoint_dir / f"block_{block:03d}.npz" if checkpoint_dir else None
        if path is not None and path.exists():
            with np.load(path) as saved:
                weights = saved["weights"]
                rows = json.loads(str(saved["history"]))
        else:
            seed_everything(config.seed * 10_007 + block)
            model = _model(config)
            rows = [
                {"block": block, **row}
                for row in fit(
                    model, panel, train_start, train_end, config, device, validation
                )
            ]
            weights = predict_weights(
                model, panel, test_start, test_end, device, config.neutralize
            )
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    path, weights=weights.astype(np.float32), history=json.dumps(rows)
                )
        history.extend(rows)
        all_weights.append(weights)
        if (
            on_block is not None
        ):  # e.g. W&B: last-epoch loss and this block's OOS Sharpe
            block_returns = (weights * panel.targets(test_start, test_end)).sum(axis=1)
            std = block_returns.std()
            on_block(
                block,
                {
                    "train_loss": rows[-1]["train_loss"] if rows else float("nan"),
                    "block_oos_sharpe": float(block_returns.mean() / std * 252**0.5)
                    if std > 0
                    else 0.0,
                    "block_start": str(panel.dates[test_start]),
                },
            )
        block_ids.append(np.full(test_end - test_start, block))
    first_test, last_test = blocks[0][2], blocks[-1][3]
    return {
        "weights": np.concatenate(all_weights).astype(np.float64),
        "block": np.concatenate(block_ids),
        "dates": panel.dates[first_test:last_test],
        "targets": panel.targets(first_test, last_test),
        "tickers": panel.tickers,
        "history": history,
        "runtime_seconds": time.perf_counter() - started,
    }
