"""Paired comparison on the full dataset: paper replication vs our recipe.

``full_paper`` (Sharpe loss, unconstrained book) and ``full_recipe`` (dollar-neutral book,
cost-aware loss) share residuals, dates and seeds. For every paper group and each recipe group
on the same residuals this ensembles the **common seeds** of both (so neither side gets more
seeds), applies the same causal B-day smoothing, and compares daily returns per period:

    full        : 2003-02 … 2016-12
    validation  : 2003-02 … 2005-12   (pre-registered selection span, docs/kaggle_plan.md)
    test        : 2006-01 … 2016-12   (reported, never tuned on)

ΔSR = recipe − paper, with a 95% moving-block bootstrap interval on the paired days.
Writes results/paper_vs_recipe.csv and results/paper_vs_recipe_yearly.csv.

    uv run python -m experiments.compare
    uv run python -m experiments.compare --days 1 5 10 --bootstrap 5000
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import run_targets
from experiments.postprocess import combine
from experiments.run import RESULTS_DIR, evaluate
from experiments.trainer import RunConfig

PAPER_SUITE = "full_paper"
RECIPE_SUITE = "full_recipe"
TEST_START = pd.Timestamp("2006-01-01")
PERIODS = ("full", "validation", "test")
ANNUALIZATION = 252


def sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=0)
    return float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else float("nan")


def block_bootstrap_diff(
    ours: np.ndarray,
    theirs: np.ndarray,
    block: int = 21,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap of sharpe(ours) - sharpe(theirs) on paired days.

    Returns the 2.5% and 97.5% quantiles and the share of draws with difference <= 0.
    """
    rng = np.random.default_rng(seed)
    days = len(ours)
    starts = rng.integers(0, days - block + 1, size=(draws, days // block + 1))
    index = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :days]
    diffs = np.array([sharpe(ours[i]) - sharpe(theirs[i]) for i in index])
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high), float((diffs <= 0).mean())


def period_slice(frame: pd.DataFrame, period: str) -> pd.DataFrame:
    if period == "validation":
        return frame[frame.date < TEST_START]
    if period == "test":
        return frame[frame.date >= TEST_START]
    return frame


def seeds_of(group: str, results_dir: Path) -> set[int]:
    return {
        int(path.parent.name.rsplit("__s", 1)[1])
        for path in results_dir.glob(f"{group}__s*/metrics.json")
        if path.parent.name.rsplit("__s", 1)[1].isdigit()
    }


def residual_key(config: dict) -> str:
    """Residual family of a run: ``pca5``, ``pca8`` or ``ff5``."""
    model = config.get("factor_model", "pca")
    return f"pca{config['n_factors']}" if model == "pca" else model


def ensemble(
    group: str,
    seeds: list[int],
    days: int,
    results_dir: Path,
    targets_cache: dict,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Seed-ensembled, smoothed daily frame (standard ``returns.csv`` columns) and weights."""
    fields = {field.name for field in dataclasses.fields(RunConfig)}
    configs, weights, dates = [], [], None
    for seed in seeds:
        path = results_dir / f"{group}__s{seed}"
        configs.append(json.loads((path / "config.json").read_text()))
        with np.load(path / "predictions.npz") as data:
            weights.append(data["weights"].astype(np.float64))
            member_dates = data["dates"].astype("datetime64[D]")
        if dates is not None and not np.array_equal(dates, member_dates):
            raise ValueError(f"{group}: seeds traded on different dates")
        dates = member_dates
    base = RunConfig(**{k: v for k, v in configs[0].items() if k in fields})
    key = residual_key(configs[0])
    if key not in targets_cache:
        targets_cache[key] = run_targets(base.to_dict(), dates)
    combined = combine(weights, days, base.neutralize)
    output = {
        "weights": combined,
        "targets": targets_cache[key],
        "dates": dates,
        "block": np.zeros(len(dates), dtype=int),
        "history": [],
        "runtime_seconds": 0.0,
    }
    frame, _ = evaluate(base, output)
    return frame, combined


def side_metrics(frame: pd.DataFrame, weights: np.ndarray, prefix: str) -> dict:
    turnover = float(frame["turnover"].iloc[1:].mean())
    wealth = (1 + frame["return"]).cumprod()
    return {
        f"{prefix}_sharpe": sharpe(frame["return"].to_numpy()),
        f"{prefix}_net_sharpe": sharpe(frame["net_return"].to_numpy()),
        f"{prefix}_mean": float(frame["return"].mean() * ANNUALIZATION),
        f"{prefix}_turnover": turnover,
        f"{prefix}_break_even_bps": float(frame["return"].mean() * 1e4 / turnover),
        f"{prefix}_net_exposure": float(weights.sum(axis=1).mean()),
        f"{prefix}_max_drawdown": float((wealth / wealth.cummax() - 1).min()),
    }


def pairs(results_dir: Path) -> list[tuple[str, str]]:
    """(paper group, recipe group) with the same residuals and at least one common seed."""

    def groups(suite: str) -> dict[str, str]:
        found = {}
        for path in results_dir.glob(f"{suite}__*__s*/config.json"):
            group = path.parent.name.rsplit("__s", 1)[0]
            found[group] = residual_key(json.loads(path.read_text()))
        return found

    paper, recipe = groups(PAPER_SUITE), groups(RECIPE_SUITE)
    return sorted(
        (p, r)
        for p, p_key in paper.items()
        for r, r_key in recipe.items()
        if p_key == r_key and seeds_of(p, results_dir) & seeds_of(r, results_dir)
    )


def compare(
    days_list: tuple[int, ...] = (1, 5),
    draws: int = 2000,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yearly = [], []
    targets_cache: dict = {}
    for paper, recipe in pairs(results_dir):
        seeds = sorted(seeds_of(paper, results_dir) & seeds_of(recipe, results_dir))
        for days in days_list:
            paper_frame, paper_w = ensemble(paper, seeds, days, results_dir, targets_cache)
            recipe_frame, recipe_w = ensemble(recipe, seeds, days, results_dir, targets_cache)
            for period in PERIODS:
                p = period_slice(paper_frame, period)
                r = period_slice(recipe_frame, period)
                p_w, r_w = paper_w[p.index], recipe_w[r.index]
                gross = block_bootstrap_diff(
                    r["return"].to_numpy(), p["return"].to_numpy(), draws=draws
                )
                net = block_bootstrap_diff(
                    r["net_return"].to_numpy(), p["net_return"].to_numpy(), draws=draws
                )
                row = {
                    "paper": paper,
                    "recipe": recipe,
                    "seeds": " ".join(map(str, seeds)),
                    "smoothing_days": days,
                    "period": period,
                    "start": p.date.min().date(),
                    "end": p.date.max().date(),
                    **side_metrics(p, p_w, "paper"),
                    **side_metrics(r, r_w, "recipe"),
                    "corr_gross": float(np.corrcoef(p["return"], r["return"])[0, 1]),
                }
                row["delta_sharpe"] = row["recipe_sharpe"] - row["paper_sharpe"]
                row["delta_sharpe_lo"], row["delta_sharpe_hi"], row["p_delta_le_0"] = gross
                row["delta_net_sharpe"] = row["recipe_net_sharpe"] - row["paper_net_sharpe"]
                (
                    row["delta_net_sharpe_lo"],
                    row["delta_net_sharpe_hi"],
                    row["p_delta_net_le_0"],
                ) = net
                rows.append(row)
            by_year = pd.DataFrame(
                {
                    "year": paper_frame.date.dt.year,
                    "paper": paper_frame["return"],
                    "recipe": recipe_frame["return"],
                }
            ).groupby("year")
            for year, group in by_year:
                yearly.append(
                    {
                        "paper": paper,
                        "recipe": recipe,
                        "smoothing_days": days,
                        "year": year,
                        "paper_sharpe": sharpe(group["paper"].to_numpy()),
                        "recipe_sharpe": sharpe(group["recipe"].to_numpy()),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    frame, yearly = compare(tuple(args.days), args.bootstrap)
    if frame.empty:
        raise SystemExit(f"no {PAPER_SUITE}/{RECIPE_SUITE} pairs in {RESULTS_DIR}")
    frame.to_csv(RESULTS_DIR / "paper_vs_recipe.csv", index=False)
    yearly.to_csv(RESULTS_DIR / "paper_vs_recipe_yearly.csv", index=False)
    columns = [
        "recipe",
        "smoothing_days",
        "period",
        "paper_sharpe",
        "recipe_sharpe",
        "delta_sharpe",
        "delta_sharpe_lo",
        "delta_sharpe_hi",
        "paper_net_sharpe",
        "recipe_net_sharpe",
        "delta_net_sharpe",
        "delta_net_sharpe_lo",
        "delta_net_sharpe_hi",
        "paper_turnover",
        "recipe_turnover",
        "corr_gross",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for paper, group in frame.groupby("paper", sort=False):
            print(f"\n{paper} (seeds {group.seeds.iloc[0]}) vs:")
            print(group[columns].round(2).to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'paper_vs_recipe.csv'}")


if __name__ == "__main__":
    main()
