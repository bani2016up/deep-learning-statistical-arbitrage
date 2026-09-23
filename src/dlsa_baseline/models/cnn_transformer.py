from __future__ import annotations

import torch
from torch import nn

from dlsa_baseline.models.allocation import AllocationHead
from dlsa_baseline.models.cnn import CausalCNN
from dlsa_baseline.models.transformer import TemporalTransformer


class CNNTransformer(nn.Module):
    """Per-asset policy mapping cumulative windows [B, L] to scores [B]."""

    def __init__(
        self,
        features: int = 8,
        cnn_layers: int = 2,
        kernel_size: int = 2,
        attention_heads: int = 4,
        transformer_ff: int = 16,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.cnn = CausalCNN(features, kernel_size, cnn_layers)
        self.transformer = TemporalTransformer(features, attention_heads, transformer_ff, dropout)
        self.allocation = AllocationHead(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.allocation(self.transformer(self.cnn(x)))
