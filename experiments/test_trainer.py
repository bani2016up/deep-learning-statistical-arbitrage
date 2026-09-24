import numpy as np

from dlsa_baseline.data.windows import WindowDataset
from experiments.trainer import RunConfig, run_protocol


def _dataset(n_dates=60, assets=6, lookback=10, seed=0):
    rng = np.random.default_rng(seed)
    residuals = rng.normal(scale=0.01, size=(n_dates + lookback, assets))
    windows = np.stack(
        [
            residuals[t - lookback : t].cumsum(0).T
            for t in range(lookback, len(residuals))
        ]
    )
    return WindowDataset(
        windows.astype(np.float32),
        residuals[lookback:].astype(np.float32),
        np.arange("2020-01-01", n_dates, dtype="datetime64[D]"),
        np.array([f"A{i}" for i in range(assets)]),
    )


def _config(**overrides):
    base = dict(
        name="t",
        model="raw_ffn",
        lookback=10,
        train_window=20,
        test_start=20,
        retrain_every=10,
        epochs=2,
        batch_dates=10,
    )
    return RunConfig(**{**base, **overrides})


def test_rolling_blocks_cover_oos_period():
    output = run_protocol(_config(), _dataset())
    assert output["weights"].shape == (40, 6)
    assert sorted(set(output["block"])) == [0, 1, 2, 3]
    np.testing.assert_allclose(np.abs(output["weights"]).sum(1), 1.0, atol=1e-5)


def test_rolling_has_no_lookahead():
    """Changing data after block 0's test window must not change block 0's weights."""
    data = _dataset()
    first = run_protocol(_config(), data)["weights"][:10]
    data.windows[30:] = 123.0
    data.targets[30:] = -1.0
    second = run_protocol(_config(), data)["weights"][:10]
    np.testing.assert_allclose(first, second)


def test_train_window_shorter_than_test_start_keeps_same_oos():
    output = run_protocol(_config(train_window=10), _dataset())
    assert output["weights"].shape == (40, 6)


def test_neutralize_produces_dollar_neutral_weights():
    output = run_protocol(_config(neutralize=True), _dataset())
    # float32 cancellation when an untrained net emits near-constant scores -> ~1e-4 residue
    np.testing.assert_allclose(output["weights"].sum(1), 0.0, atol=1e-3)


def test_identical_config_is_reused(tmp_path, monkeypatch):
    import experiments.run as run_module

    monkeypatch.setattr(
        run_module, "load_run_data", lambda cfg: _dataset(lookback=cfg["lookback"])
    )
    first = run_module.run_experiment(_config(name="a"), tmp_path)
    monkeypatch.setattr(
        run_module, "run_protocol", lambda *a: (_ for _ in ()).throw(AssertionError)
    )
    second = run_module.run_experiment(_config(name="b"), tmp_path)
    assert first == second
    assert (tmp_path / "b" / "predictions.npz").exists()


def test_block_checkpoints_resume_without_retraining(tmp_path, monkeypatch):
    import experiments.trainer as trainer_module

    data = _dataset()
    first = run_protocol(_config(), data, checkpoint_dir=tmp_path)
    assert len(list(tmp_path.glob("block_*.npz"))) == 4
    monkeypatch.setattr(
        trainer_module, "fit", lambda *a, **k: (_ for _ in ()).throw(AssertionError)
    )
    resumed = run_protocol(_config(), data, checkpoint_dir=tmp_path)
    np.testing.assert_allclose(resumed["weights"], first["weights"], atol=1e-6)


def test_max_blocks_truncates_weights_dates_and_targets_consistently():
    output = run_protocol(_config(max_blocks=2), _dataset())
    assert (
        output["weights"].shape[0]
        == len(output["dates"])
        == len(output["targets"])
        == 20
    )


def test_deadline_guard(monkeypatch):
    import time as time_module

    from experiments.run_suite import _out_of_time

    monkeypatch.delenv("DLSA_DEADLINE", raising=False)
    assert not _out_of_time(10_000)
    monkeypatch.setenv("DLSA_DEADLINE", str(time_module.time() + 100))
    assert not _out_of_time(50) and _out_of_time(90)
