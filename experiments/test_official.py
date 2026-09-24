import numpy as np
import pytest

from experiments.official import convert, parse_model


def test_parse_model_checks_family_and_published_factors():
    assert parse_model("ipca5") == ("ipca", 5)
    assert parse_model("ff8") == ("ff", 8)
    with pytest.raises(ValueError):
        parse_model("ff15")
    with pytest.raises(ValueError):
        parse_model("ipca")


def test_convert_marks_missing_and_needs_a_full_window_before_t():
    raw = np.array(
        [
            [0.01, 0.0, 0.0],
            [0.02, 0.03, 0.0],
            [-0.01, 0.01, 0.0],
            [0.0, 0.02, 0.0],
            [0.01, -0.02, 0.0],
        ],
        dtype=np.float32,
    )
    residuals, eligible, columns = convert(raw, lookback=2)
    # Asset 2 is never observed, so it is dropped.
    assert columns.tolist() == [0, 1]
    assert np.isnan(residuals[3, 0]) and residuals[3, 1] == pytest.approx(0.02)
    # Eligible at t needs observations on t-2 and t-1, not on t itself.
    assert eligible[:, 0].tolist() == [False, False, True, True, False]
    assert eligible[:, 1].tolist() == [False, False, False, True, True]
