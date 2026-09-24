"""Paired comparison of full-dataset suites: paper replication, paper with costs, our recipe.

    full_paper        Sharpe loss, unconstrained book (paper Table I)
    full_paper_costs  Sharpe net of the full 5 bp + 1 bp costs, unconstrained book (paper III.J)
    full_recipe       Sharpe net of cost_weight × costs, dollar-neutral book (docs/final_model.md)
    official_*        the same three arms on the authors' CRSP residuals (experiments.official)

All suites share residuals and dates. For a (base, other) pair of suites, every base group is
matched with each other group on the same residuals. Both are ensembled over their **common
seeds** (so neither side gets more seeds), smoothed with the same causal B-day average, and
their daily returns are compared per period:

    full        : 2003-02 … 2016-12   (official_*: from 2002-02)
    validation  : 2003-02 … 2005-12   (pre-registered selection span, docs/kaggle_plan.md)
    test        : 2006-01 … 2016-12   (reported, never tuned on)

ΔSR = other − base, with a 95% moving-block bootstrap interval on the paired days.
Writes results/full_comparison.csv and results/full_comparison_yearly.csv.

    uv run python -m experiments.compare
    uv run python -m experiments.compare --pairs full_paper:full_recipe --days 1 5 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import full_model_name, run_targets
from experiments.postprocess import combine
from experiments.run import RESULTS_DIR, evaluate
from experiments.trainer import RunConfig

DEFAULT_PAIRS = (
    ("full_paper", "full_paper_costs"),
    ("full_paper", "full_recipe"),
    ("full_paper_costs", "full_recipe"),
    ("official_paper", "official_paper_costs"),
    ("official_paper", "official_recipe"),
    ("official_paper_costs", "official_recipe"),
)
TEST_START = pd.Timestamp("2006-01-01")
PERIODS = ("full", "validation", "test")
ANNUALIZATION = 252


def sharpe(returns: np.ndarray) -> float:
    std = returns.std(ddof=0)
    return float(returns.mean() / std * math.sqrt(ANNUALIZATION)) if std > 0 else float("nan")


def block_bootstrap_diff(
    other: np.ndarray,
    base: np.ndarray,
    block: int = 21,
    draws: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Moving-block bootstrap of sharpe(other) - sharpe(base) on paired days.

    Returns the 2.5% and 97.5% quantiles and the share of draws with difference <= 0.
    """
    rng = np.random.default_rng(seed)
    days = len(other)
    starts = rng.integers(0, days - block + 1, size=(draws, days // block + 1))
    index = (starts[:, :, None] + np.arange(block)).reshape(draws, -1)[:, :days]
    diffs = np.array([sharpe(other[i]) - sharpe(base[i]) for i in index])
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
    """Residuals a run traded: ``pca5``, ``pca8``, ``ff5``, ``official_ipca5``, ..."""
    return full_model_name(config.get("factor_model", "pca"), config["n_factors"])


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


def pairs(base_suite: str, other_suite: str, results_dir: Path) -> list[tuple[str, str]]:
    """(base group, other group) with the same residuals and at least one common seed."""

    def groups(suite: str) -> dict[str, str]:
        found = {}
        for path in results_dir.glob(f"{suite}__*__s*/config.json"):
            group = path.parent.name.rsplit("__s", 1)[0]
            found[group] = residual_key(json.loads(path.read_text()))
        return found

    base, other = groups(base_suite), groups(other_suite)
    return sorted(
        (b, o)
        for b, b_key in base.items()
        for o, o_key in other.items()
        if b_key == o_key and seeds_of(b, results_dir) & seeds_of(o, results_dir)
    )


def compare(
    suite_pairs: tuple[tuple[str, str], ...] = DEFAULT_PAIRS,
    days_list: tuple[int, ...] = (1, 5),
    draws: int = 2000,
    results_dir: Path = RESULTS_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, yearly = [], []
    targets_cache: dict = {}
    for base_suite, other_suite in suite_pairs:
        for base, other in pairs(base_suite, other_suite, results_dir):
            seeds = sorted(seeds_of(base, results_dir) & seeds_of(other, results_dir))
            for days in days_list:
                base_frame, base_w = ensemble(base, seeds, days, results_dir, targets_cache)
                other_frame, other_w = ensemble(other, seeds, days, results_dir, targets_cache)
                for period in PERIODS:
                    b = period_slice(base_frame, period)
                    o = period_slice(other_frame, period)
                    gross = block_bootstrap_diff(
                        o["return"].to_numpy(), b["return"].to_numpy(), draws=draws
                    )
                    net = block_bootstrap_diff(
                        o["net_return"].to_numpy(), b["net_return"].to_numpy(), draws=draws
                    )
                    row = {
                        "base": base,
                        "other": other,
                        "seeds": " ".join(map(str, seeds)),
                        "smoothing_days": days,
                        "period": period,
                        "start": b.date.min().date(),
                        "end": b.date.max().date(),
                        **side_metrics(b, base_w[b.index], "base"),
                        **side_metrics(o, other_w[o.index], "other"),
                        "corr_gross": float(np.corrcoef(b["return"], o["return"])[0, 1]),
                    }
                    row["delta_sharpe"] = row["other_sharpe"] - row["base_sharpe"]
                    row["delta_sharpe_lo"], row["delta_sharpe_hi"], row["p_delta_le_0"] = gross
                    row["delta_net_sharpe"] = row["other_net_sharpe"] - row["base_net_sharpe"]
                    (
                        row["delta_net_sharpe_lo"],
                        row["delta_net_sharpe_hi"],
                        row["p_delta_net_le_0"],
                    ) = net
                    rows.append(row)
                by_year = pd.DataFrame(
                    {
                        "year": base_frame.date.dt.year,
                        "base": base_frame["return"],
                        "other": other_frame["return"],
                    }
                ).groupby("year")
                for year, group in by_year:
                    yearly.append(
                        {
                            "base": base,
                            "other": other,
                            "smoothing_days": days,
                            "year": year,
                            "base_sharpe": sharpe(group["base"].to_numpy()),
                            "other_sharpe": sharpe(group["other"].to_numpy()),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(yearly)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{b}:{o}" for b, o in DEFAULT_PAIRS],
        help="base_suite:other_suite",
    )
    parser.add_argument("--days", nargs="+", type=int, default=[1, 5])
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    suite_pairs = tuple(tuple(pair.split(":", 1)) for pair in args.pairs)
    frame, yearly = compare(suite_pairs, tuple(args.days), args.bootstrap)
    if frame.empty:
        raise SystemExit(f"no matching groups for {args.pairs} in {RESULTS_DIR}")
    frame.to_csv(RESULTS_DIR / "full_comparison.csv", index=False)
    yearly.to_csv(RESULTS_DIR / "full_comparison_yearly.csv", index=False)
    columns = [
        "other",
        "smoothing_days",
        "period",
        "base_sharpe",
        "other_sharpe",
        "delta_sharpe",
        "delta_sharpe_lo",
        "delta_sharpe_hi",
        "base_net_sharpe",
        "other_net_sharpe",
        "delta_net_sharpe",
        "delta_net_sharpe_lo",
        "delta_net_sharpe_hi",
        "base_turnover",
        "other_turnover",
        "other_net_exposure",
        "corr_gross",
    ]
    with pd.option_context("display.width", 250, "display.max_columns", None):
        for base, group in frame.groupby("base", sort=False):
            print(f"\n{base} (seeds {group.seeds.iloc[0]}) vs:")
            print(group[columns].round(2).to_string(index=False))
    print(f"\nWrote {RESULTS_DIR / 'full_comparison.csv'}")


if __name__ == "__main__":
    main()
