"""Derived runs from finished ones: seed ensembles and causal weight smoothing.

For a group of seeds ``<suite>__<variant>__s*`` this averages their daily weights, then
optionally smooths them over the last B days (paper III.M: arbitrage signals persist for
days, so overlapping holding cuts turnover more than it cuts return), re-neutralizes when
the members were dollar-neutral, and L1-normalizes. Only past weights are used, so there is
no look-ahead. Output: ``results/ensemble__<suite>-<variant>_b<B>__ens/`` in the standard
run format.

    uv run python -m experiments.postprocess final__cnn_neutral_cw0.25
    uv run python -m experiments.postprocess --suite final
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from experiments.data import run_targets
from experiments.run import RESULTS_DIR, evaluate, write_run
from experiments.trainer import RunConfig

SMOOTHING_DAYS = (1, 2, 3, 5, 10, 20)


def smooth(weights: np.ndarray, days: int) -> np.ndarray:
    """Causal overlapping-holding weights: mean of the last ``days`` daily allocations."""
    if days <= 1:
        return weights.copy()
    cumulative = np.cumsum(np.vstack([np.zeros_like(weights[:1]), weights]), axis=0)
    counts = np.minimum(np.arange(1, len(weights) + 1), days)[:, None]
    starts = np.maximum(np.arange(len(weights)) + 1 - days, 0)
    return (cumulative[1:] - cumulative[starts]) / counts


def combine(members: list[np.ndarray], days: int, neutralize: bool) -> np.ndarray:
    weights = smooth(np.mean(members, axis=0), days)
    if neutralize:
        weights = weights - weights.mean(axis=1, keepdims=True)
    gross = np.abs(weights).sum(axis=1, keepdims=True)
    return np.divide(weights, gross, out=np.zeros_like(weights), where=gross > 1e-12)


def _member_dirs(group: str, results_dir: Path) -> list[Path]:
    return sorted(
        path.parent
        for path in results_dir.glob(f"{group}__s*/metrics.json")
        if path.parent.name.rsplit("__s", 1)[-1].isdigit()
    )


def derive(group: str, days: int, results_dir: Path = RESULTS_DIR) -> dict:
    members = _member_dirs(group, results_dir)
    if len(members) < 2:
        raise ValueError(
            f"{group}: need at least two finished seeds, found {len(members)}"
        )
    configs = [json.loads((path / "config.json").read_text()) for path in members]
    weights, dates, tickers = [], None, None
    for path in members:
        with np.load(path / "predictions.npz") as data:
            weights.append(data["weights"].astype(np.float64))
            member_dates = data["dates"].astype("datetime64[D]")
            tickers = data["tickers"]
        if dates is not None and not np.array_equal(dates, member_dates):
            raise ValueError(f"{group}: members traded on different dates")
        dates = member_dates

    fields = {field.name for field in dataclasses.fields(RunConfig)}
    base = RunConfig(**{k: v for k, v in configs[0].items() if k in fields})
    combined = combine(weights, days, base.neutralize)
    output = {
        "weights": combined,
        "targets": run_targets(base.to_dict(), dates),
        "dates": dates,
        "block": np.zeros(len(dates), dtype=int),
        "history": [],
        "runtime_seconds": 0.0,
    }
    frame, metrics = evaluate(base, output)
    suite, variant = group.split("__", 1)
    name = f"ensemble__{suite}-{variant}_b{days}__ens"
    config = {
        **base.to_dict(),
        "name": name,
        "seed": -1,
        "kind": "ensemble",
        "members": [path.name for path in members],
        "smoothing_days": days,
    }
    write_run(results_dir / name, config, frame, metrics, output, tickers)
    return metrics


def groups_in_suite(suite: str, results_dir: Path = RESULTS_DIR) -> list[str]:
    groups = {
        path.parent.name.rsplit("__s", 1)[0]
        for path in results_dir.glob(f"{suite}__*__s*/metrics.json")
    }
    return sorted(
        group for group in groups if len(_member_dirs(group, results_dir)) >= 2
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="*", help="<suite>__<variant> groups of seeds")
    parser.add_argument(
        "--suite", action="append", default=[], help="all multi-seed groups of a suite"
    )
    parser.add_argument("--days", nargs="+", type=int, default=list(SMOOTHING_DAYS))
    args = parser.parse_args()
    groups = [
        *args.groups,
        *(g for suite in args.suite for g in groups_in_suite(suite)),
    ]
    for group in groups:
        for days in args.days:
            metrics = derive(group, days)
            print(
                f"{group} B={days:2d}: sharpe={metrics['sharpe']:+.2f} "
                f"net={metrics['net_sharpe']:+.2f} turnover={metrics['turnover']:.2f}"
            )


if __name__ == "__main__":
    main()
