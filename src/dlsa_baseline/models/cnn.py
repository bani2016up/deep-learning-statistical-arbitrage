from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalCNN(nn.Module):
    """Two-layer causal CNN: [batch, time] -> [batch, time, features]."""

    def __init__(self, features: int = 8, kernel_size: int = 2, layers: int = 2) -> None:
        super().__init__()
        if layers != 2:
            raise ValueError("This baseline intentionally implements exactly two CNN layers")
        self.kernel_size = kernel_size
        self.input_norm = nn.InstanceNorm1d(1)
        self.hidden_norm = nn.InstanceNorm1d(features)
        self.conv1 = nn.Conv1d(1, features, kernel_size)
        self.conv2 = nn.Conv1d(features, features, kernel_size)

    def _causal(self, x: torch.Tensor, convolution: nn.Conv1d) -> torch.Tensor:
        return convolution(F.pad(x, (self.kernel_size - 1, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError("CausalCNN expects [batch, time]")
        residual = x.unsqueeze(1).repeat(1, self.conv1.out_channels, 1)
        hidden = torch.relu(self._causal(self.input_norm(x.unsqueeze(1)), self.conv1))
        hidden = torch.relu(self._causal(self.hidden_norm(hidden), self.conv2))
        return (hidden + residual).transpose(1, 2)
