from __future__ import annotations

import torch
from torch import nn


class RawFFN(nn.Module):
    """Paper-style raw FFN mapping [batch, 30] cumulative windows to scores."""

    def __init__(self, lookback: int = 30, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(lookback, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def reversal_scores(windows: torch.Tensor) -> torch.Tensor:
    """Non-trainable contrarian score: short positive cumulative deviations."""
    return -windows[..., -1]
