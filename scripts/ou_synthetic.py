"""OU+Threshold on synthetic residuals with a known data-generating process.

Each scenario is a signal component plus a random walk (2% daily vol, the median of real
IPCA-5 residuals). Signal strength sd = 2% gives lag-1 return autocorrelation near -0.013
(real residuals: about -0.02). Each point is a portfolio of N independent assets with
sum |w| = 1, so absolute Sharpe ratios are inflated; compare the shapes of the curves.

    uv run python scripts/ou_synthetic.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlsa_baseline.config import ROOT
from dlsa_baseline.data.pca_residuals import ResidualDataset
from dlsa_baseline.models.ou import ou_signal
from dlsa_baseline.training.ou_backtest import OURule, annualized, backtest_ou

LOOKBACK, DAYS, ASSETS, VOL = 30, 2000, 300, 0.02
RULES = {
    "paper 1.25 / R²>0.25": OURule(),
    "paper 1.25 / R²>0.75": OURule(c_crit=0.75),
    "AL + kappa filter": OURule(kind="al", c_crit=0.0, b_max=float(np.exp(-2 / LOOKBACK))),
}
rng = np.random.default_rng(42)


def ou_paths(half_life: float, n: int = ASSETS, t: int = DAYS + LOOKBACK, sd: float = VOL):
    """Exact AR(1) discretization of an OU level with stationary standard deviation ``sd``."""
    b = 0.5 ** (1 / half_life)
    shocks = rng.normal(0, sd * np.sqrt(1 - b**2), (t, n))
    level = np.zeros((t, n))
    level[0] = rng.normal(0, sd, n)
    for i in range(1, t):
        level[i] = b * level[i - 1] + shocks[i]
    return level


def random_walk(n: int = ASSETS, t: int = DAYS + LOOKBACK) -> np.ndarray:
    return np.cumsum(rng.normal(0, VOL, (t, n)), axis=0)


def momentum(phi: float, n: int = ASSETS, t: int = DAYS + LOOKBACK) -> np.ndarray:
    """OU (half-life 5d) plus a level whose returns are AR(1) with coefficient ``phi``."""
    shocks = rng.normal(0, VOL * np.sqrt(1 - phi**2), (t, n))
    returns = np.zeros((t, n))
    for i in range(1, t):
        returns[i] = phi * returns[i - 1] + shocks[i]
    return ou_paths(5, n, t) + np.cumsum(returns, axis=0)


def sine(period: float) -> np.ndarray:
    time = np.arange(DAYS + LOOKBACK)[:, None]
    phase = rng.uniform(0, 2 * np.pi, ASSETS)
    return VOL * np.sqrt(2) * np.sin(2 * np.pi * time / period + phase) + random_walk()


def mixture(slow_share: float) -> np.ndarray:
    return (
        np.sqrt(1 - slow_share) * ou_paths(2) + np.sqrt(slow_share) * ou_paths(60) + random_walk()
    )


def as_dataset(level: np.ndarray) -> ResidualDataset:
    returns = np.diff(level, axis=0)
    returns[returns == 0] = 1e-12  # zero means "missing" to the backtest
    return ResidualDataset(
        residuals=returns.astype(np.float32),
        dates=np.datetime64("2000-01-03") + np.arange(len(returns)),
        tickers=np.array([f"sim_{i}" for i in range(returns.shape[1])]),
        metadata={"method": "synthetic"},
    )


def sweep(name: str, grid: list[float], make_level) -> pd.DataFrame:
    rows = []
    for value in grid:
        dataset = as_dataset(make_level(value))
        row = {name: value}
        for label, rule in RULES.items():
            returns = backtest_ou(dataset, rule, LOOKBACK).returns
            row[label] = annualized(returns)["SR"] if returns.std() > 0 else 0.0
        rows.append(row)
        print({k: round(v, 2) for k, v in row.items()}, flush=True)
    return pd.DataFrame(rows)


def estimation_bias() -> dict[float, dict[str, np.ndarray]]:
    """AR(1) estimates on a single 30-day window, including a pure random walk (no reversion)."""
    result = {}
    for half_life in (2.0, 5.0, 10.0, 20.0, np.inf):
        if np.isinf(half_life):
            level = random_walk(4000, LOOKBACK)
        else:
            level = ou_paths(half_life, 4000, LOOKBACK, sd=3 * VOL)
        signal = ou_signal(np.diff(np.vstack([np.zeros(4000), level]), axis=0))
        b = np.exp(-signal.kappa[signal.valid])
        trade = signal.valid & (signal.r2 > 0.25) & (np.abs(signal.s_score) > 1.25)
        result[half_life] = {"b": b, "r2": signal.r2[signal.valid]}
        print(
            f"true half-life {half_life:>4}: median estimated half-life "
            f"{np.median(np.log(0.5) / np.log(b)):.1f}d, windows triggering a trade {trade.mean():.2f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ou")
    out = parser.parse_args().output_dir
    out.mkdir(parents=True, exist_ok=True)

    sweeps = {
        "half_life": sweep(
            "half_life", [1, 2, 3, 5, 10, 20, 40, 80], lambda h: ou_paths(h) + random_walk()
        ),
        "momentum_phi": sweep("momentum_phi", [-0.1, 0, 0.05, 0.1, 0.15, 0.2, 0.3], momentum),
        "period": sweep("period", [3, 5, 8, 12, 20, 30, 45, 60, 120], sine),
        "slow_share": sweep("slow_share", [0, 0.25, 0.5, 0.75, 0.9, 1.0], mixture),
    }
    for name, frame in sweeps.items():
        frame.to_csv(out / f"ou_synthetic_{name}.csv", index=False)
    bias = estimation_bias()

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    specs = [
        (axes[0, 1], "half_life", "(b) pure OU: SR vs half-life", "half-life, days", True),
        (axes[0, 2], "momentum_phi", "(c) OU + momentum: SR vs return autocorr φ", "φ", False),
        (
            axes[1, 0],
            "period",
            "(d) sine + noise: SR vs period (cf. Fig. A.4)",
            "period, days",
            True,
        ),
        (
            axes[1, 1],
            "slow_share",
            "(e) fast OU (2d) + slow OU (60d)",
            "slow variance share",
            False,
        ),
    ]
    for ax, name, title, xlabel, logx in specs:
        frame = sweeps[name]
        for label in RULES:
            ax.plot(frame[name], frame[label], marker="o", ms=4, label=label)
        ax.axhline(0, color="0.5", lw=0.8)
        if logx:
            ax.axvline(LOOKBACK, color="0.6", ls="--", lw=0.8)
            ax.set_xscale("log")
            ax.set_xticks(frame[name], frame[name])
        ax.set(title=title, xlabel=xlabel, ylabel="annualized SR")
    axes[0, 1].legend(fontsize=7)
    examples = axes[0, 0]
    t = np.arange(120)
    examples.plot(t, ou_paths(5, 1, 120, sd=3 * VOL)[:, 0], label="OU, half-life 5d")
    examples.plot(t, momentum(0.3, 1, 120)[:, 0], label="OU + momentum φ=0.3")
    examples.axvspan(0, LOOKBACK, color="0.93")
    examples.set_title("(a) example cumulative residuals (grey: lookback L=30)")
    examples.legend(fontsize=7)
    ax = axes[1, 2]
    for half_life, values in bias.items():
        label = "random walk" if np.isinf(half_life) else f"true half-life {half_life:g}d"
        ax.scatter(values["b"] ** 2, values["r2"], s=2, alpha=0.25, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for cutoff in (0.25, 0.75):
        ax.axhline(cutoff, color="C3", lw=0.8, ls=":")
    ax.set(
        xlabel=r"$\hat b^2$ on a 30-day window",
        ylabel=r"$R^2$",
        title=r"(f) $R^2 \approx \hat b^2$: the R² filter is a minimum half-life filter",
    )
    ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout()
    fig.savefig(out / "ou_synthetic.png", dpi=150)


if __name__ == "__main__":
    main()
