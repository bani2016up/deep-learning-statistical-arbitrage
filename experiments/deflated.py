"""Deflated Sharpe ratio of the full-dataset runs on the test period (2006-01 → 2016-12).

Every full-data configuration we evaluated is a trial: trained runs of ``full_paper``,
``full_paper_costs``, ``full_recipe`` and ``full_bench`` and all ``ensemble__full_*`` derived
runs (``full_smoke`` / ``full_probe`` do not cover the test period). For each run the DSR is
the probability that its true Sharpe exceeds the maximum Sharpe expected from that many trials
under no skill (Bailey & López de Prado 2014), using ``dlsa_baseline``'s implementation.
Computed separately for gross returns and for returns net of 5 bp × turnover + 1 bp × short.

The deflation is conservative: trials are highly correlated (seeds, smoothing variants of one
model), and the spread of their Sharpe ratios, which sets the expected maximum, mostly reflects
real differences between models and turnover rather than luck. That matters most for net
returns, where turnover dominates. ``*_psr`` (probability that the Sharpe exceeds 0, no
multiple-testing correction) is reported alongside as the upper bound.

    uv run python -m experiments.deflated                     # → deflated_sharpe_full.csv
    uv run python -m experiments.deflated --family official   # authors' residuals, own trials
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from dlsa_baseline.analysis.statistics import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from experiments.compare import TEST_START, sharpe
from experiments.run import RESULTS_DIR

# Trial families: our WIKI panel and the authors' CRSP residuals (experiments.official) are
# separate studies on different data, each deflated for its own number of trials.
FAMILIES = {
    "full": (
        "full_paper__",
        "full_paper_costs__",
        "full_recipe__",
        "full_bench__",
        "ensemble__full_",
    ),
    "official": ("official_", "ensemble__official_"),
}


def trial_returns(
    results_dir: Path = RESULTS_DIR, prefixes: tuple[str, ...] = FAMILIES["full"]
) -> dict[str, pd.DataFrame]:
    trials = {}
    for path in sorted(results_dir.glob("*/returns.csv")):
        name = path.parent.name
        if not name.startswith(prefixes) or not (path.parent / "metrics.json").exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        trials[name] = frame[frame.date >= TEST_START]
    return trials


def deflate(trials: dict[str, pd.DataFrame]) -> pd.DataFrame:
    columns = {"gross": "return", "net": "net_return"}
    sharpes = {
        kind: [sharpe(frame[column].to_numpy()) for frame in trials.values()]
        for kind, column in columns.items()
    }
    rows = []
    for name, frame in trials.items():
        row = {"run": name, "days": len(frame)}
        for kind, column in columns.items():
            returns = frame[column].to_numpy()
            result = deflated_sharpe_ratio(returns, sharpes[kind])
            row[f"{kind}_sharpe"] = sharpe(returns)
            row[f"{kind}_expected_max_sharpe"] = result["expected_maximum_sharpe_under_selection"]
            row[f"{kind}_dsr"] = result["deflated_sharpe_probability"]
            row[f"{kind}_psr"] = probabilistic_sharpe_ratio(returns)
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["trials"] = len(trials)
    return frame.sort_values("gross_sharpe", ascending=False, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=list(FAMILIES), default="full")
    family = parser.parse_args().family
    output = RESULTS_DIR / f"deflated_sharpe_{family}.csv"
    frame = deflate(trial_returns(prefixes=FAMILIES[family]))
    frame.to_csv(output, index=False)
    print(
        f"{len(frame)} trials, test period from {TEST_START.date()}. "
        f"Expected max SR under no skill: gross {frame.gross_expected_max_sharpe.iloc[0]:.2f}, "
        f"net {frame.net_expected_max_sharpe.iloc[0]:.2f}"
    )
    with pd.option_context("display.width", 200):
        for column in ("gross_sharpe", "net_sharpe"):
            print(f"\nTop by {column}:")
            print(frame.nlargest(8, column).round(3).to_string(index=False))
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
