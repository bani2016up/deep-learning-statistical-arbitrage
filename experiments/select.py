"""Honest model selection: choose on the first part of the OOS period, report the rest.

Every run (trained or derived) shares the rolling OOS window 2020-02 → 2025-12. Picking the
best of ~200 configurations on that same window overstates performance, so this splits it:

    SELECTION  : OOS start … 2022-12-31   (choose here)
    HOLDOUT    : 2023-01-01 … OOS end     (report here, never used for choosing)

Writes results/selection.csv (one row per run) and prints the top candidates by selection
net Sharpe with their holdout numbers, plus the rank correlation between the two periods.

    uv run python -m experiments.select
"""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd
from scipy import stats

from experiments.run import RESULTS_DIR

SPLIT_DATE = pd.Timestamp("2023-01-01")


def _sharpe(returns: pd.Series) -> float:
    std = returns.std(ddof=0)
    return float(returns.mean() / std * math.sqrt(252)) if std > 0 else float("nan")


def period_metrics(frame: pd.DataFrame, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_sharpe": _sharpe(frame["return"]),
        f"{prefix}_net_sharpe": _sharpe(frame["net_return"]),
        f"{prefix}_mean": float(frame["return"].mean() * 252),
        f"{prefix}_net_mean": float(frame["net_return"].mean() * 252),
        f"{prefix}_turnover": float(frame["turnover"].iloc[1:].mean()),
        f"{prefix}_days": len(frame),
    }


def collect(split_date: pd.Timestamp = SPLIT_DATE, prefix: str = "") -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.name.startswith("_") or not run_dir.name.startswith(prefix):
            continue
        config = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        frame = pd.read_csv(run_dir / "returns.csv", parse_dates=["date"])
        selection, holdout = (
            frame[frame.date < split_date],
            frame[frame.date >= split_date],
        )
        if len(selection) < 60 or len(holdout) < 60:
            continue  # e.g. the `fixed` protocol only trades inside the holdout
        parts = run_dir.name.split("__")
        rows.append(
            {
                "run": run_dir.name,
                "suite": parts[0],
                "variant": parts[1] if len(parts) > 1 else run_dir.name,
                "model": config.get("model"),
                "n_factors": config.get("n_factors"),
                "neutralize": config.get("neutralize", False),
                "objective": config.get("objective"),
                "smoothing_days": config.get("smoothing_days", 1),
                "net_exposure": metrics["net_exposure"],
                **period_metrics(selection, "sel"),
                **period_metrics(holdout, "hold"),
            }
        )
    frame = pd.DataFrame(rows)
    # Suites share default configs; copies have identical returns and would double-count.
    key = (
        frame[["sel_sharpe", "hold_sharpe", "sel_turnover"]]
        .round(10)
        .astype(str)
        .agg("|".join, axis=1)
    )
    return frame.assign(duplicate_of_earlier=key.duplicated())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top", type=int, default=15)
    parser.add_argument(
        "--split-date", default=str(SPLIT_DATE.date()), help="first holdout date"
    )
    parser.add_argument(
        "--prefix", default="", help="only runs whose name starts with this"
    )
    parser.add_argument("--output", default="selection.csv")
    parser.add_argument(
        "--all",
        action="store_true",
        help="also rank non-neutral runs (their profit can be residual drift, not stat-arb)",
    )
    args = parser.parse_args()
    frame = collect(pd.Timestamp(args.split_date), args.prefix)
    frame.to_csv(RESULTS_DIR / args.output, index=False)
    frame = frame[~frame.duplicate_of_earlier]
    if not args.all:
        # Admissible stat-arb candidates only: dollar-neutral books, so drift cannot pay.
        frame = frame[frame.net_exposure.abs() < 0.05]
    rho, p_value = stats.spearmanr(frame["sel_net_sharpe"], frame["hold_net_sharpe"])
    gross_rho, _ = stats.spearmanr(frame["sel_sharpe"], frame["hold_sharpe"])
    print(
        f"{len(frame)} runs. Spearman(selection, holdout): net {rho:+.2f} (p={p_value:.3f}), gross {gross_rho:+.2f}"
    )
    columns = [
        "run",
        "sel_sharpe",
        "sel_net_sharpe",
        "hold_sharpe",
        "hold_net_sharpe",
        "hold_turnover",
    ]
    ranked = frame.sort_values("sel_net_sharpe", ascending=False)
    print("\nTop by SELECTION net Sharpe (holdout shown for honesty):")
    print(ranked[columns].head(args.top).round(2).to_string(index=False))
    print("\nTop by SELECTION gross Sharpe:")
    print(
        frame.sort_values("sel_sharpe", ascending=False)[columns]
        .head(args.top)
        .round(2)
        .to_string(index=False)
    )
    best = ranked.iloc[0]
    print(
        f"\nSelected: {best['run']}  holdout net Sharpe {best['hold_net_sharpe']:+.2f}"
    )


if __name__ == "__main__":
    main()
