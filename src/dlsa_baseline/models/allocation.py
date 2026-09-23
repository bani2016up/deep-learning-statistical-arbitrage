from __future__ import annotations

import torch
from torch import nn


class AllocationHead(nn.Module):
    """Map the final temporal representation to one unnormalized asset score."""

    def __init__(self, features: int = 8) -> None:
        super().__init__()
        self.linear = nn.Linear(features, 1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.linear(sequence[:, -1]).squeeze(-1)


def normalize_l1(scores: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """Normalize the last (asset) dimension to unit gross exposure."""
    denominator = scores.abs().sum(dim=-1, keepdim=True).clamp_min(epsilon)
    return scores / denominator
