"""OU benchmark experiments on the official residuals (see docs/OU_BASELINE.md).

uv run python scripts/ou_experiments.py all-k     # Table I OU row, K = 0..15, incl. K=0 check
uv run python scripts/ou_experiments.py grid      # c_thresh x c_crit sensitivity, K=5
uv run python scripts/ou_experiments.py rules     # paper rule vs Avellaneda-Lee, with costs
uv run python scripts/ou_experiments.py by-year   # SR by year / sub-period, rolling SR
uv run python scripts/ou_experiments.py example   # Fig. A.2-style single-residual example
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlsa_baseline.config import ROOT
from dlsa_baseline.data.official_residuals import OFFICIAL_FACTORS, load_official_residuals
from dlsa_baseline.models.ou import ou_signal, threshold_positions
from dlsa_baseline.training.ou_backtest import OURule, annualized, backtest_ou

OOS_START = "2002-01-01"
FAMILIES = ("FF", "PCA", "IPCA")
B_KAPPA = float(np.exp(-2 / 30))
PAPER_TABLE_I = {  # OU+Thresh, Sharpe ratios of Table I
    "FF": {0: -0.18, 1: 0.16, 3: 0.54, 5: 0.38, 8: 1.16},
    "PCA": {0: -0.18, 1: 0.21, 3: 0.77, 5: 0.73, 8: 0.87, 10: 0.63, 15: 0.62},
    "IPCA": {0: -0.18, 1: 0.60, 3: 0.88, 5: 0.97, 8: 0.91, 10: 0.86, 15: 0.93},
}
RULES = {
    "paper (no memory)": OURule(),
    "AL 1.25/0.50/0.75": OURule(kind="al"),
    "AL + kappa filter": OURule(kind="al", b_max=B_KAPPA),
    "AL exit at mean": OURule(kind="al", s_close_long=0.0, s_close_short=0.0),
    "AL pure (kappa, no R2)": OURule(kind="al", c_crit=0.0, b_max=B_KAPPA),
}


def sharpe(x: np.ndarray) -> float:
    return annualized(np.asarray(x))["SR"]


def all_k(out: Path) -> None:
    rows = []
    for family in FAMILIES:
        for k in OFFICIAL_FACTORS[family]:
            if k not in PAPER_TABLE_I[family]:
                continue
            result = backtest_ou(load_official_residuals(family, k), start=OOS_START)
            perf = annualized(result.returns)
            rows.append({"family": family, "K": k, **perf, "SR_paper": PAPER_TABLE_I[family][k]})
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ou_all_k.csv", index=False)
    print(frame.pivot(index="K", columns="family", values=["SR", "SR_paper"]).round(2))


def grid(out: Path) -> None:
    rows = []
    for family in FAMILIES:
        dataset = load_official_residuals(family, 5)
        for c_thresh, c_crit in itertools.product([1.0, 1.25, 1.5], [0.25, 0.5, 0.75]):
            result = backtest_ou(dataset, OURule(c_thresh=c_thresh, c_crit=c_crit))  # 1998-2016
            oos = result.dates >= np.datetime64(OOS_START)
            rows.append(
                {
                    "family": family,
                    "c_thresh": c_thresh,
                    "c_crit": c_crit,
                    "SR_validation_1998_2001": sharpe(result.returns[~oos]),
                    "SR": sharpe(result.returns[oos]),
                    "SR_net": sharpe(result.net_returns()[oos]),
                    "turnover": result.turnover[oos].mean(),
                }
            )
            print(rows[-1], flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "ou_grid_k5.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, family in zip(axes, FAMILIES):
        table = frame[frame.family == family].pivot(index="c_thresh", columns="c_crit", values="SR")
        image = ax.imshow(table.values, cmap="RdYlGn", vmin=-1.2, vmax=1.2)
        for (i, j), value in np.ndenumerate(table.values):
            bold = (table.index[i], table.columns[j]) == (1.25, 0.25)
            ax.text(
                j, i, f"{value:.2f}", ha="center", va="center", fontweight="bold" if bold else None
            )
        ax.set_xticks(range(3), table.columns)
        ax.set_yticks(range(3), table.index)
        ax.set(
            xlabel="c_crit (R² cutoff)", ylabel="c_thresh", title=f"{family}-5: OOS SR 2002-2016"
        )
    fig.colorbar(image, ax=axes, shrink=0.8)
    fig.savefig(out / "ou_grid_sr_k5.png", dpi=150, bbox_inches="tight")


def rules(out: Path) -> None:
    rows, curves = [], {}
    for family in FAMILIES:
        dataset = load_official_residuals(family, 5)
        for name, rule in RULES.items():
            result = backtest_ou(dataset, rule, start=OOS_START)
            net = result.net_returns()
            curves[(family, name)] = (result.dates, result.returns.cumsum(), net.cumsum())
            gross = annualized(result.returns)
            rows.append(
                {
                    "family": family,
                    "rule": name,
                    **gross,
                    "SR_net": sharpe(net),
                    "turnover": result.turnover.mean(),
                    "holding_days": result.holding_days.mean() if rule.kind == "al" else np.nan,
                    # cost per unit turnover at which the net mean return is zero
                    "breakeven_trade_bp": 1e4
                    * (gross["mu"] - 252 * 1e-4 * result.short.mean())
                    / (252 * result.turnover.mean()),
                }
            )
            print(
                {k: round(v, 3) if isinstance(v, float) else v for k, v in rows[-1].items()},
                flush=True,
            )
    pd.DataFrame(rows).to_csv(out / "ou_rules_k5.csv", index=False)
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for col, family in enumerate(FAMILIES):
        for name in RULES:
            dates, gross, net = curves[(family, name)]
            axes[0, col].plot(dates, gross, lw=1, label=name)
            axes[1, col].plot(dates, net, lw=1, label=name)
        axes[0, col].set_title(f"{family}-5: cumulative return, gross")
        axes[1, col].set_title(f"{family}-5: net of 5bp trade + 1bp short")
        for ax in axes[:, col]:
            ax.axhline(0, color="0.6", lw=0.7)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "ou_rules_k5.png", dpi=150)


def by_year(out: Path) -> None:
    series = {}
    for family in FAMILIES:
        dataset = load_official_residuals(family, 5)
        for name in ("paper (no memory)", "AL pure (kappa, no R2)"):
            result = backtest_ou(dataset, RULES[name], start=OOS_START)
            series[f"{family} | {name}"] = pd.Series(
                result.returns, index=pd.DatetimeIndex(result.dates)
            )
    frame = pd.DataFrame(series)
    yearly = frame.groupby(frame.index.year).agg(sharpe).round(2)
    yearly.to_csv(out / "ou_by_year_k5.csv")
    periods = {
        "2002-2007": frame.loc["2002":"2007"],
        "2008-2009": frame.loc["2008":"2009"],
        "2010-2016": frame.loc["2010":"2016"],
        "all excl. 2008-09": frame.drop(frame.loc["2008":"2009"].index),
    }
    sub = pd.DataFrame({k: v.agg(sharpe) for k, v in periods.items()}).T
    sub.loc["PnL share 2008-09"] = frame.loc["2008":"2009"].sum() / frame.sum()
    sub.round(2).to_csv(out / "ou_subperiods_k5.csv")
    print(yearly.to_string(), "\n", sub.round(2).to_string())
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, name in zip(axes, ("paper (no memory)", "AL pure (kappa, no R2)")):
        for family in FAMILIES:
            column = frame[f"{family} | {name}"]
            ax.plot(column.rolling(252).apply(sharpe, raw=True), lw=1.2, label=family)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.axvspan(pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-30"), color="C3", alpha=0.08)
        ax.set_title(f"Rolling 252-day SR, K=5 - {name}")
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "ou_by_year_k5.png", dpi=150)


def example(out: Path, lookback: int = 30, horizon: int = 30, seed: int = 1) -> None:
    """Single IPCA-5 residual traded alone with w in {-1, 0, 1}: a trend failure and a success."""
    dataset = load_official_residuals("IPCA", 5)
    residuals, dates = dataset.residuals.astype(np.float64), dataset.dates
    first = int(np.searchsorted(dates, np.datetime64(OOS_START))) + lookback
    rng = np.random.default_rng(seed)

    def trade(asset: int, t0: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, weights = np.zeros(horizon), np.zeros(horizon)
        for h in range(horizon):
            signal = ou_signal(residuals[t0 + h - lookback : t0 + h, [asset]])
            scores[h], weights[h] = signal.s_score[0], threshold_positions(signal)[0]
        return scores, weights, weights * residuals[t0 : t0 + horizon, asset]

    trend, revert = [], []
    for _ in range(3000):
        t0 = int(rng.integers(first, len(dates) - horizon))
        live = np.flatnonzero(~np.any(residuals[t0 - lookback : t0 + horizon] == 0, axis=0))
        asset = int(rng.choice(live))
        _, weights, pnl = trade(asset, t0)
        path = residuals[t0 : t0 + horizon, asset]
        drift = path.sum() / (path.std() * np.sqrt(horizon))
        if np.abs(weights).mean() > 0.4 and abs(drift) > 1.5:
            trend.append((pnl.sum() - 0.02 * abs(drift), asset, t0))
        if abs(drift) < 0.7 and np.abs(np.diff(weights)).sum() >= 3:
            revert.append((-pnl.sum(), asset, t0))
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for row, (title, (_, asset, t0)) in enumerate(
        [("Trend: OU fails", min(trend)), ("Mean reversion: OU works", min(revert))]
    ):
        scores, weights, pnl = trade(asset, t0)
        span, oos = dates[t0 - lookback : t0 + horizon], dates[t0 : t0 + horizon]
        axes[row, 0].plot(span, np.cumsum(residuals[t0 - lookback : t0 + horizon, asset]), "k")
        axes[row, 0].axvspan(oos[0], oos[-1], color="0.93")
        axes[row, 0].set_title(f"{title}: cumulative residual")
        axes[row, 1].plot(oos, scores)
        axes[row, 1].fill_between(oos, -1.25, 1.25, color="0.9")
        axes[row, 1].set_title("s-score (X_L - mu) / (sigma / sqrt(2 kappa))")
        axes[row, 2].step(oos, weights, where="post", color="C3")
        axes[row, 2].set(ylim=(-1.3, 1.3), yticks=[-1, 0, 1], title="position {-1, 0, +1}")
        axes[row, 3].plot(oos, np.cumsum(pnl), color="C2" if pnl.sum() > 0 else "C3")
        axes[row, 3].axhline(0, color="0.6", lw=0.8)
        axes[row, 3].set_title(f"cumulative PnL = {pnl.sum():+.1%}")
        for ax in axes[row]:
            ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        print(f"{title}: asset_{asset}, start {oos[0]}, PnL {pnl.sum():+.3f}")
    fig.suptitle("OU+Threshold on one IPCA-5 residual (c_thresh=1.25, c_crit=0.25, L=30)")
    fig.tight_layout()
    fig.savefig(out / "ou_example_ipca5.png", dpi=150)


def main() -> None:
    commands = {
        "all-k": all_k,
        "grid": grid,
        "rules": rules,
        "by-year": by_year,
        "example": example,
    }
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=[*commands, "all"])
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ou")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, command in commands.items():
        if args.command in (name, "all"):
            print(f"== {name} ==")
            command(args.output_dir)


if __name__ == "__main__":
    main()
