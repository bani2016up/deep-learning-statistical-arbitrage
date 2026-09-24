"""Paper benchmark policies missing from the baseline: Fourier+FFN, OU+FFN, OU+Threshold.

Every policy maps cumulative residual windows [batch, lookback] to unnormalized scores [batch];
cross-sectional L1 normalization happens in the objective, exactly as for the baseline models.
"""

from __future__ import annotations

import torch
from torch import nn

from dlsa_baseline.models import CNNTransformer, RawFFN

EPS = 1e-8


def allocation_ffn(inputs: int, dropout: float) -> nn.Sequential:
    """Same 16-8-4-1 allocation network as the baseline RawFFN, with a configurable input."""
    return nn.Sequential(
        nn.Linear(inputs, 16),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )


def fourier_features(x: torch.Tensor) -> torch.Tensor:
    """Real trigonometric coefficients (a_0..a_{L/2}, b_1..b_{L/2-1}): an invertible L-vector."""
    lookback = x.shape[-1]
    spectrum = torch.fft.rfft(x, dim=-1) / lookback
    imaginary = spectrum.imag[..., 1 : (lookback + 1) // 2]
    return torch.cat([spectrum.real, imaginary], dim=-1)


def ou_signals(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """AR(1) fit X_{l+1} = a + b X_l + e per window (Avellaneda-Lee / paper Appendix B.B)."""
    previous, following = x[..., :-1], x[..., 1:]
    mean_prev, mean_next = (
        previous.mean(-1, keepdim=True),
        following.mean(-1, keepdim=True),
    )
    centered_prev = previous - mean_prev
    variance = (centered_prev**2).mean(-1).clamp_min(EPS)
    b = (centered_prev * (following - mean_next)).mean(-1) / variance
    a = mean_next.squeeze(-1) - b * mean_prev.squeeze(-1)
    errors = following - (a.unsqueeze(-1) + b.unsqueeze(-1) * previous)
    total = ((following - mean_next) ** 2).sum(-1).clamp_min(EPS)
    r_squared = 1.0 - (errors**2).sum(-1) / total
    b_clamped = b.clamp(EPS, 1 - 1e-4)
    mu = a / (1 - b_clamped)
    sigma_eq = torch.sqrt(
        errors.var(-1, unbiased=False) / (1 - b_clamped**2)
    ).clamp_min(EPS)
    return {
        "b": b,
        "kappa": -torch.log(b_clamped),
        "mu": mu,
        "sigma_eq": sigma_eq,
        "last": x[..., -1],
        "r_squared": r_squared,
        "s_score": (x[..., -1] - mu) / sigma_eq,
    }


class FourierFFN(nn.Module):
    def __init__(self, lookback: int = 30, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = allocation_ffn(lookback, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Per-window scale normalization, analogous to the CNN's input instance norm.
        scale = x.std(-1, keepdim=True).clamp_min(EPS)
        return self.network(fourier_features(x / scale)).squeeze(-1)


class OUFFN(nn.Module):
    """FFN allocation on the 4-d OU signal (mu, sigma_eq, X_L, R^2), scaled per window."""

    def __init__(self, lookback: int = 30, dropout: float = 0.25) -> None:
        super().__init__()
        self.network = allocation_ffn(4, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signals = ou_signals(x)
        scale = x.std(-1).clamp_min(EPS)
        features = torch.stack(
            [
                (signals["mu"] / scale).clamp(-10, 10),
                (signals["sigma_eq"] / scale).clamp(0, 10),
                signals["last"] / scale,
                signals["r_squared"],
            ],
            dim=-1,
        )
        return self.network(features).squeeze(-1)


class OUThreshold(nn.Module):
    """Parametric benchmark: short if s > c, long if s < -c, only for mean-reverting good fits."""

    def __init__(self, threshold: float = 1.25, min_r_squared: float = 0.25) -> None:
        super().__init__()
        self.threshold = threshold
        self.min_r_squared = min_r_squared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        signals = ou_signals(x)
        valid = (
            (signals["r_squared"] > self.min_r_squared)
            & (signals["b"] > 0)
            & (signals["b"] < 1)
        )
        s = signals["s_score"]
        scores = torch.where(
            s > self.threshold, -1.0, torch.where(s < -self.threshold, 1.0, 0.0)
        )
        return scores * valid


class Reversal(nn.Module):
    """Non-trainable contrarian score, identical to the baseline's reversal benchmark."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return -x[..., -1]


TRAINABLE = {"cnn_transformer", "raw_ffn", "fourier_ffn", "ou_ffn"}
STATIC = {"ou_threshold", "reversal"}


def build_model(
    name: str, lookback: int, dropout: float, **cnn_kwargs: int
) -> nn.Module:
    if name == "cnn_transformer":
        return CNNTransformer(dropout=dropout, **cnn_kwargs)
    if name == "raw_ffn":
        return RawFFN(lookback, dropout)
    if name == "fourier_ffn":
        return FourierFFN(lookback, dropout)
    if name == "ou_ffn":
        return OUFFN(lookback, dropout)
    if name == "ou_threshold":
        return OUThreshold()
    if name == "reversal":
        return Reversal()
    raise ValueError(f"Unknown model {name!r}")
