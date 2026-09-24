import numpy as np
import pandas as pd

from experiments.deflated import FAMILIES, deflate, trial_returns


def _frame(returns):
    dates = pd.bdate_range("2006-01-02", periods=len(returns))
    return pd.DataFrame({"date": dates, "return": returns, "net_return": returns - 1e-4})


def test_strong_signal_survives_deflation_and_noise_does_not():
    rng = np.random.default_rng(0)
    trials = {f"noise_{i}": _frame(rng.normal(0, 0.01, 2500)) for i in range(30)}
    trials["signal"] = _frame(rng.normal(0.002, 0.01, 2500))  # SR ≈ 3.2
    frame = deflate(trials).set_index("run")
    assert frame.loc["signal", "gross_dsr"] > 0.99
    assert frame.drop("signal").gross_dsr.max() < 0.95
    assert frame.attrs["trials"] == 31


def test_more_trials_raise_the_bar():
    rng = np.random.default_rng(1)
    base = {f"t{i}": _frame(rng.normal(0.0003, 0.01, 2500)) for i in range(5)}
    more = base | {f"u{i}": _frame(rng.normal(0.0003, 0.01, 2500)) for i in range(50)}
    few, many = deflate(base), deflate(more)
    assert many.gross_expected_max_sharpe.iloc[0] > few.gross_expected_max_sharpe.iloc[0]


def test_families_select_their_own_runs(tmp_path):
    for name in (
        "full_paper__pca5__s0",
        "ensemble__full_paper-pca5_b5__ens",
        "official_paper__ipca5__s0",
    ):
        (tmp_path / name).mkdir()
        _frame(np.zeros(3)).to_csv(tmp_path / name / "returns.csv", index=False)
        (tmp_path / name / "metrics.json").write_text("{}")
    assert set(trial_returns(tmp_path)) == {
        "full_paper__pca5__s0",
        "ensemble__full_paper-pca5_b5__ens",
    }
    assert set(trial_returns(tmp_path, FAMILIES["official"])) == {"official_paper__ipca5__s0"}
