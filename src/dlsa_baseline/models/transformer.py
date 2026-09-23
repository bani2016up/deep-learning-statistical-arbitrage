from __future__ import annotations

import torch
from torch import nn


class TemporalTransformer(nn.Module):
    """One small self-attention block: [batch, time, features] -> same shape."""

    def __init__(
        self, features: int = 8, heads: int = 4, feedforward: int = 16, dropout: float = 0.25
    ) -> None:
        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=features,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("TemporalTransformer expects [batch, time, features]")
        return self.encoder(x)
