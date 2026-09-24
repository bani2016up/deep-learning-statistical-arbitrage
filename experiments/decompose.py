"""Split every run's return into residual drift vs a dollar-neutral part.

    r_t = n_t * m_t + sum_i (w_{t,i} - n_t / N) * eps_{t,i}
          `------'   `---------------------------------'
           drift       neutral (cross-sectional timing)

with n_t = sum_i w_{t,i} (net exposure) and m_t = mean_i eps_{t,i} (equal-weight residual).
Writes results/decomposition.csv. CPU only; safe to run while suites train.

    uv run python -m experiments.decompose
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from experiments.data import run_targets
from experiments.run import RESULTS_DIR


def _sharpe(returns: np.ndarray) -> float:
    std = returns.std()
    return float(returns.mean() / std * math.sqrt(252)) if std > 0 else float("nan")


def decompose(weights: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    net = weights.sum(axis=1)
    equal_weight = targets.mean(axis=1)
    drift = net * equal_weight
    neutral = ((weights - net[:, None] / weights.shape[1]) * targets).sum(axis=1)
    return {
        "sharpe_total": _sharpe(drift + neutral),
        "sharpe_drift": _sharpe(drift),
        "sharpe_neutral": _sharpe(neutral),
        "mean_drift": float(drift.mean() * 252),
        "mean_neutral": float(neutral.mean() * 252),
        "net_exposure": float(net.mean()),
        "corr_equal_weight": float(np.corrcoef(drift + neutral, equal_weight)[0, 1]),
        "equal_weight_sharpe": _sharpe(equal_weight),
    }


def main() -> None:
    rows = []
    for config_path in sorted(RESULTS_DIR.glob("*/config.json")):
        run_dir = config_path.parent
        # metrics.json is written last, so its presence marks a complete run.
        if run_dir.name.startswith("_") or not (run_dir / "metrics.json").exists():
            continue
        config = json.loads(config_path.read_text())
        with np.load(run_dir / "predictions.npz") as data:
            weights = data["weights"].astype(np.float64)
            dates = data["dates"].astype("datetime64[D]")
            returns = data["returns"]
        targets = run_targets(config, dates)
        if not np.allclose((weights * targets).sum(axis=1), returns, atol=1e-5):
            raise ValueError(
                f"{run_dir.name}: stored returns do not match weights x residuals"
            )
        parts = run_dir.name.split("__")
        rows.append(
            {
                "run": run_dir.name,
                "suite": parts[0],
                "variant": parts[1] if len(parts) > 1 else run_dir.name,
                **decompose(weights, targets),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "decomposition.csv", index=False)
    summary = frame.groupby(["suite", "variant"])[
        ["sharpe_total", "sharpe_drift", "sharpe_neutral", "net_exposure"]
    ].mean()
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
