from __future__ import annotations

import torch
from torch import nn

# CUDA's memory-efficient attention rejects more than 65,535 sequences per call when attention
# dropout is on (training). A full-data batch is dates × eligible names, e.g. 125 × 863 on the
# authors' CRSP residuals, so larger batches go through the encoder in chunks. Every sequence
# is attended independently, so chunking does not change the result.
MAX_SEQUENCES = 65_535


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
        if x.shape[0] <= MAX_SEQUENCES:
            return self.encoder(x)
        return torch.cat([self.encoder(chunk) for chunk in x.split(MAX_SEQUENCES)])
