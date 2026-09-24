import numpy as np
import pytest

from experiments.postprocess import combine, smooth


def test_smooth_is_causal_trailing_mean():
    weights = np.arange(10, dtype=float)[:, None]
    smoothed = smooth(weights, 3)
    assert smoothed[:, 0].tolist()[:4] == [0.0, 0.5, 1.0, 2.0]
    changed = weights.copy()
    changed[6:] = 99.0
    np.testing.assert_allclose(
        smooth(changed, 3)[:6], smoothed[:6]
    )  # future cannot leak back


def test_smoothing_reduces_turnover():
    rng = np.random.default_rng(0)
    weights = rng.normal(size=(500, 10))
    turnover = lambda w: np.abs(np.diff(w, axis=0)).sum(1).mean()
    assert turnover(combine([weights], 5, True)) < turnover(combine([weights], 1, True))


def test_combine_neutral_unit_gross():
    rng = np.random.default_rng(1)
    members = [rng.normal(size=(50, 6)) for _ in range(3)]
    weights = combine(members, 3, neutralize=True)
    np.testing.assert_allclose(weights.sum(1), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.abs(weights).sum(1), 1.0)


def test_combine_keeps_empty_days_empty():
    weights = combine([np.zeros((4, 3)), np.zeros((4, 3))], 1, neutralize=False)
    assert not np.isnan(weights).any() and weights.sum() == pytest.approx(0.0)
