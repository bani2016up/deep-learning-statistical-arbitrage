import numpy as np
import pytest
import torch

from dlsa_baseline.data.pca_residuals import ResidualDataset
from dlsa_baseline.data.windows import build_cumulative_windows
from experiments.objectives import allocate, score
from experiments.panel import DensePanel, MaskedPanel


def test_masked_panel_matches_dense_windows_when_everything_is_eligible():
    rng = np.random.default_rng(0)
    residuals = rng.normal(scale=0.01, size=(80, 5)).astype(np.float32)
    dates = np.arange("2020-01-01", 80, dtype="datetime64[D]")
    tickers = np.array(list("ABCDE"))
    dense = DensePanel(
        build_cumulative_windows(ResidualDataset(residuals, dates, tickers, {}), 10)
    )
    masked = MaskedPanel(
        residuals, np.ones_like(residuals, dtype=bool), dates, tickers, 10
    )
    assert len(masked) == len(dense)
    w_dense, y_dense, _ = dense.batch(5, 40, torch.device("cpu"))
    w_mask, y_mask, mask = masked.batch(5, 40, torch.device("cpu"))
    torch.testing.assert_close(w_mask, w_dense)
    torch.testing.assert_close(y_mask, y_dense)
    assert mask.all()


def test_masked_panel_trims_untradeable_start_and_zero_fills_missing_targets():
    residuals = np.ones((60, 3), dtype=np.float32)
    residuals[45, 1] = np.nan
    eligible = np.zeros((60, 3), dtype=bool)
    eligible[20:] = True
    dates = np.arange("2020-01-01", 60, dtype="datetime64[D]")
    panel = MaskedPanel(residuals, eligible, dates, np.array(list("ABC")), lookback=5)
    assert panel.dates[0] == dates[20] and len(panel) == 40
    _, targets, mask = panel.batch(0, len(panel), torch.device("cpu"))
    assert mask.all() and targets[45 - 20, 1] == 0.0


def test_masked_allocation_ignores_ineligible_names():
    scores = torch.tensor([[5.0, 1.0, -1.0, 100.0]])
    mask = torch.tensor([[True, True, True, False]])
    weights = allocate(scores, neutralize=True, mask=mask)
    assert weights[0, 3] == 0
    assert weights.sum().item() == pytest.approx(0.0, abs=1e-7)
    assert weights.abs().sum().item() == pytest.approx(1.0)


def test_score_runs_model_only_on_eligible_pairs():
    calls = []

    class Recorder(torch.nn.Module):
        def forward(self, x):
            calls.append(x.shape[0])
            return x[:, -1]

    windows = torch.ones(2, 3, 4)
    mask = torch.tensor([[True, False, True], [False, False, True]])
    out = score(Recorder(), windows, mask)
    assert calls == [3] and out[0, 1] == 0 and out[1, 2] == 1
