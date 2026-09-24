"""The two OU+Threshold implementations must agree.

``experiments.models.OUThreshold`` (torch, on cumulative windows, used by the rolling
harness) and ``dlsa_baseline.models.ou`` (numpy, on residual windows, used on the official
residuals in ``docs/OU_BASELINE.md``) were written independently. On official IPCA-5
residuals they give identical positions on 15,639 of 15,639 asset-days.
"""

import numpy as np
import torch

from dlsa_baseline.models.ou import ou_signal, threshold_positions
from experiments.models import OUThreshold, ou_signals


def residual_windows(assets: int = 2000, lookback: int = 30, seed: int = 0) -> np.ndarray:
    """[lookback, assets] residual returns: random walks plus OU noise of varied speed."""
    rng = np.random.default_rng(seed)
    b = rng.uniform(0.0, 0.98, assets)
    level = np.zeros((lookback, assets))
    for t in range(1, lookback):
        level[t] = b * level[t - 1] + rng.normal(0, 0.02, assets)
    level += np.cumsum(rng.normal(0, 0.01, (lookback, assets)), axis=0)
    return np.diff(np.vstack([np.zeros(assets), level]), axis=0)


def test_ou_threshold_implementations_agree() -> None:
    window = residual_windows()
    cumulative = torch.tensor(np.cumsum(window, axis=0).T, dtype=torch.float64)
    ours = threshold_positions(ou_signal(window))
    harness = OUThreshold()(cumulative).numpy()
    np.testing.assert_array_equal(ours, harness)
    assert np.abs(ours).sum() > 100  # the comparison is not vacuous


def test_ou_signals_agree_away_from_unit_root() -> None:
    window = residual_windows(seed=1)
    ours = ou_signal(window)
    harness = ou_signals(torch.tensor(np.cumsum(window, axis=0).T, dtype=torch.float64))
    keep = ours.valid & (harness["b"].numpy() < 0.99)
    np.testing.assert_allclose(ours.s_score[keep], harness["s_score"].numpy()[keep], atol=1e-9)
    np.testing.assert_allclose(ours.r2, harness["r_squared"].numpy(), atol=1e-9)
