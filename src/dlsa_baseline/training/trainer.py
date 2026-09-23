from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dlsa_baseline.data.windows import WindowDataset
from dlsa_baseline.models.allocation import normalize_l1
from dlsa_baseline.training.evaluation import metrics
from dlsa_baseline.training.objectives import negative_sharpe, sharpe_ratio


@dataclass(frozen=True)
class SplitIndices:
    train_end: int
    validation_end: int


def chronological_split(
    n_dates: int, train_fraction: float = 0.6, validation_fraction: float = 0.2
) -> SplitIndices:
    train_end = int(n_dates * train_fraction)
    validation_end = train_end + int(n_dates * validation_fraction)
    if train_end < 2 or validation_end >= n_dates:
        raise ValueError("Dataset is too short for requested chronological split")
    return SplitIndices(train_end, validation_end)


def _policy_returns(
    model: nn.Module, windows: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    dates, assets, lookback = windows.shape
    scores = model(windows.reshape(dates * assets, lookback)).reshape(dates, assets)
    weights = normalize_l1(scores)
    return (weights * targets).sum(dim=1), weights


def predict(
    model: nn.Module,
    windows: np.ndarray,
    targets: np.ndarray,
    device: torch.device,
    batch_dates: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_returns: list[np.ndarray] = []
    all_weights: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_dates):
            x = torch.as_tensor(windows[start : start + batch_dates], device=device)
            y = torch.as_tensor(targets[start : start + batch_dates], device=device)
            batch_returns, batch_weights = _policy_returns(model, x, y)
            all_returns.append(batch_returns.cpu().numpy())
            all_weights.append(batch_weights.cpu().numpy())
    return np.concatenate(all_returns), np.concatenate(all_weights)


def train_model(
    model: nn.Module,
    dataset: WindowDataset,
    device: torch.device,
    epochs: int = 12,
    batch_dates: int = 125,
    learning_rate: float = 1e-3,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    epsilon: float = 1e-6,
    checkpoint_path: Path | None = None,
) -> dict[str, object]:
    """Train chronologically; each optimization batch consists of complete dates."""
    split = chronological_split(len(dataset.dates), train_fraction, validation_fraction)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_validation = -float("inf")
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        batch_losses: list[float] = []
        # Chronological batches are intentional. Dates are never shuffled across splits.
        for start in range(0, split.train_end, batch_dates):
            end = min(start + batch_dates, split.train_end)
            if end - start < 2:
                continue
            x = torch.as_tensor(dataset.windows[start:end], device=device)
            y = torch.as_tensor(dataset.targets[start:end], device=device)
            portfolio_returns, _ = _policy_returns(model, x, y)
            loss = negative_sharpe(portfolio_returns, epsilon)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Non-finite model gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))

        validation_returns, _ = predict(
            model,
            dataset.windows[split.train_end : split.validation_end],
            dataset.targets[split.train_end : split.validation_end],
            device,
            batch_dates,
        )
        validation_sharpe = float(
            sharpe_ratio(torch.from_numpy(validation_returns), epsilon) * np.sqrt(252)
        )
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": float(np.mean(batch_losses)),
                "validation_sharpe": validation_sharpe,
            }
        )
        if validation_sharpe > best_validation:
            best_validation = validation_sharpe
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_returns, test_weights = predict(
        model,
        dataset.windows[split.validation_end :],
        dataset.targets[split.validation_end :],
        device,
        batch_dates,
    )
    result: dict[str, object] = {
        "history": history,
        "split": split,
        "test_returns": test_returns,
        "test_weights": test_weights,
        "test_dates": dataset.dates[split.validation_end :],
        "test_metrics": metrics(test_returns, test_weights),
        "runtime_seconds": time.perf_counter() - started,
    }
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "history": history}, checkpoint_path)
    return result
