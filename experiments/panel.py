"""Batch access to decision dates: windows, targets and a tradeability mask.

Decision index ``i`` trades on ``dates[i]`` using the ``lookback`` residuals before it.
``DensePanel`` wraps the rectangular sample (every name always tradeable); ``MaskedPanel``
serves the full dynamic-universe data and builds cumulative windows per batch on the
device, so the [T, N, lookback] tensor never exists in memory.
"""

from __future__ import annotations

import numpy as np
import torch

from dlsa_baseline.data.windows import WindowDataset


class DensePanel:
    def __init__(self, data: WindowDataset) -> None:
        self.data = data
        self.dates = data.dates
        self.tickers = data.tickers

    def __len__(self) -> int:
        return len(self.dates)

    def batch(
        self, start: int, end: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        windows = torch.as_tensor(
            self.data.windows[start:end], dtype=torch.float32, device=device
        )
        targets = torch.as_tensor(
            self.data.targets[start:end], dtype=torch.float32, device=device
        )
        return (
            windows,
            targets,
            torch.ones(targets.shape, dtype=torch.bool, device=device),
        )

    def targets(self, start: int, end: int) -> np.ndarray:
        return self.data.targets[start:end].astype(np.float64)


class MaskedPanel:
    """Residuals [T, N] (NaN = missing) plus eligibility [T, N] → per-batch windows."""

    def __init__(
        self,
        residuals: np.ndarray,
        eligible: np.ndarray,
        dates: np.ndarray,
        tickers: np.ndarray,
        lookback: int,
    ) -> None:
        if residuals.shape != eligible.shape:
            raise ValueError("residuals and eligible must have the same shape")
        # Drop leading dates before anything is tradeable, so decision index 0 is the
        # first tradeable date and the default rolling test_start works unchanged.
        first = (
            max(int(np.argmax(eligible.any(axis=1))) - lookback, 0)
            if eligible.any()
            else 0
        )
        self.lookback = lookback
        self._residuals = np.nan_to_num(
            residuals[first:].astype(np.float32)
        )  # missing → 0 return
        self._eligible = eligible[first:].astype(bool)
        self.dates = np.asarray(dates)[first + lookback :]
        self.tickers = tickers
        self._device_cache: dict[str, torch.Tensor] = {}

    def __len__(self) -> int:
        return len(self.dates)

    def _on(self, device: torch.device) -> torch.Tensor:
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = torch.as_tensor(self._residuals, device=device)
        return self._device_cache[key]

    def batch(
        self, start: int, end: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residuals = self._on(device)
        rows = residuals[start : end + self.lookback - 1]  # rows i .. i+L-1 for each i
        windows = rows.unfold(0, self.lookback, 1).cumsum(dim=-1)  # [B, N, L]
        targets = residuals[start + self.lookback : end + self.lookback]
        mask = torch.as_tensor(
            self._eligible[start + self.lookback : end + self.lookback], device=device
        )
        return windows, targets, mask

    def targets(self, start: int, end: int) -> np.ndarray:
        return self._residuals[start + self.lookback : end + self.lookback].astype(
            np.float64
        )


def as_panel(data: object) -> DensePanel | MaskedPanel:
    if isinstance(data, (DensePanel, MaskedPanel)):
        return data
    if isinstance(data, WindowDataset):
        return DensePanel(data)
    raise TypeError(f"Unsupported data type {type(data).__name__}")
