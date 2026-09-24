"""Parametric Ornstein-Uhlenbeck benchmark (paper Section II.D.1 and Appendix B.B).

The official repository only ships OU+FFN; the OU+Threshold benchmark of Table I is
implemented here. Conventions follow ``preprocess.py:preprocess_ou`` of the official code:

* the input is a window of ``L`` residual returns; ``X = cumsum`` is the residual "price";
* ``X_{l+1} = a + b X_l + e`` is fit by OLS on the ``L - 1`` consecutive pairs;
* only ``0 < b < 1`` is valid (mean reversion, ``kappa = -log b`` defined);
* ``R^2`` is the squared centered correlation of ``X_l`` and ``X_{l+1}``;
* ``sigma / sqrt(2 kappa) = sqrt(Var(e) / (1 - b^2))`` with population variance.

The official code adds ``1e-6`` to ``1 - b``; that only changes s-scores of windows with
``b`` within ~1e-4 of one, whose ``|s|`` is far above any threshold, so it is omitted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OUSignal:
    """Per-asset OU signal ``theta^OU`` plus the s-score used by threshold rules."""

    kappa: np.ndarray  # mean-reversion speed per day
    mu: np.ndarray  # long-run level of X
    sigma_eq: np.ndarray  # sigma / sqrt(2 kappa), equilibrium std of X
    last: np.ndarray  # X_L
    r2: np.ndarray  # AR(1) goodness of fit
    s_score: np.ndarray  # (X_L - mu) / sigma_eq
    valid: np.ndarray  # 0 < b < 1


def ou_signal(window: np.ndarray) -> OUSignal:
    """Estimate the OU signal from residual returns ``window`` of shape ``[lookback, assets]``."""
    window = np.asarray(window, dtype=np.float64)
    if window.ndim != 2 or window.shape[0] < 3:
        raise ValueError("window must be [lookback >= 3, assets]")
    level = np.cumsum(window, axis=0)
    x, y = level[:-1], level[1:]
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    covariance = ((x - x_mean) * (y - y_mean)).mean(axis=0)
    var_x, var_y = x.var(axis=0), y.var(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = covariance / var_x
        r2 = covariance**2 / (var_x * var_y)
    valid = np.isfinite(b) & (b > 0) & (b < 1)
    b_safe = np.where(valid, b, 0.5)  # placeholder; invalid assets are never traded
    a = y_mean - b_safe * x_mean
    errors = y - (a + b_safe * x)
    mu = a / (1 - b_safe)
    sigma_eq = np.sqrt(errors.var(axis=0) / (1 - b_safe**2))
    with np.errstate(divide="ignore", invalid="ignore"):
        s_score = np.where(valid, (level[-1] - mu) / sigma_eq, 0.0)
    return OUSignal(
        kappa=-np.log(b_safe),
        mu=mu,
        sigma_eq=sigma_eq,
        last=level[-1],
        r2=np.nan_to_num(r2),
        s_score=np.nan_to_num(s_score),
        valid=valid & np.isfinite(s_score),
    )


def tradable(signal: OUSignal, c_crit: float = 0.25, b_max: float = 1.0) -> np.ndarray:
    """Validity mask: mean-reverting fit, ``R^2 > c_crit`` and optionally ``b < b_max``.

    ``b_max = exp(-2 / L)`` is the Avellaneda-Lee speed filter (reversion time below ``L / 2``).
    """
    return signal.valid & (signal.r2 > c_crit) & (np.exp(-signal.kappa) < b_max)


def threshold_positions(
    signal: OUSignal, c_thresh: float = 1.25, c_crit: float = 0.25, b_max: float = 1.0
) -> np.ndarray:
    """Paper rule (Section II.D.1): memoryless ``{-1, 0, +1}`` from the current s-score."""
    ok = tradable(signal, c_crit, b_max)
    s = signal.s_score
    return np.where(ok & (s > c_thresh), -1.0, np.where(ok & (s < -c_thresh), 1.0, 0.0))


def hysteresis_positions(
    signal: OUSignal,
    previous: np.ndarray,
    s_open: float = 1.25,
    s_close_long: float = 0.50,
    s_close_short: float = 0.75,
    c_crit: float = 0.25,
    b_max: float = 1.0,
) -> np.ndarray:
    """Avellaneda-Lee (2010) rule with memory: open beyond ``s_open``, close near the mean.

    Long is opened at ``s < -s_open`` and closed at ``s > -s_close_long``; short is opened at
    ``s > s_open`` and closed at ``s < s_close_short``. Invalid signals force a close.
    """
    s = signal.s_score
    positions = np.asarray(previous, dtype=np.float64).copy()
    positions[(positions > 0) & (s > -s_close_long)] = 0.0
    positions[(positions < 0) & (s < s_close_short)] = 0.0
    flat = positions == 0
    positions[flat & (s < -s_open)] = 1.0
    positions[flat & (s > s_open)] = -1.0
    positions[~tradable(signal, c_crit, b_max)] = 0.0
    return positions
