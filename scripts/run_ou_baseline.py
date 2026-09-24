from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dlsa_baseline.config import ROOT
from dlsa_baseline.data.official_residuals import load_official_residuals
from dlsa_baseline.data.pca_residuals import load_residual_dataset
from dlsa_baseline.training.ou_backtest import OURule, annualized, backtest_ou

PAPER_OOS_START = "2002-01-01"


def main() -> None:
    parser = argparse.ArgumentParser(description="OU+Threshold / Avellaneda-Lee residual backtest")
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--family",
        choices=["FF", "PCA", "IPCA"],
        default="IPCA",
        help="official residual family (references/dlsa-public)",
    )
    source.add_argument("--residuals", type=Path, help="ResidualDataset .npz instead of official")
    parser.add_argument("--factors", type=int, default=5)
    parser.add_argument("--rule", choices=["paper", "al"], default="paper")
    parser.add_argument("--c-thresh", type=float, default=1.25)
    parser.add_argument("--c-crit", type=float, default=0.25)
    parser.add_argument(
        "--kappa-filter", action="store_true", help="Avellaneda-Lee speed filter b < exp(-2/L)"
    )
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument(
        "--start", default=None, help=f"first decision date (official default {PAPER_OOS_START})"
    )
    parser.add_argument("--end", default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/ou")
    args = parser.parse_args()

    if args.residuals:
        dataset, tag, start = load_residual_dataset(args.residuals), args.residuals.stem, args.start
    else:
        dataset = load_official_residuals(args.family, args.factors)
        tag, start = f"{args.family.lower()}{args.factors}", args.start or PAPER_OOS_START
    rule = OURule(
        kind=args.rule,
        c_thresh=args.c_thresh,
        c_crit=args.c_crit,
        b_max=float(np.exp(-2 / args.lookback)) if args.kappa_filter else 1.0,
    )
    result = backtest_ou(dataset, rule, args.lookback, start, args.end)

    name = f"ou_{args.rule}{'_kappa' if args.kappa_filter else ''}_{tag}"
    metrics = {
        "period": [str(result.dates[0]), str(result.dates[-1])],
        "gross": annualized(result.returns),
        "net_5bp_trade_1bp_short": annualized(result.net_returns()),
        "average_turnover": float(result.turnover.mean()),
        "average_positions": float((result.n_long + result.n_short).mean()),
        "average_universe": float(result.n_universe.mean()),
        "rule": rule.__dict__,
    }
    result.save_predictions(args.output_dir / f"{name}_predictions.npz")
    pd.DataFrame(
        {
            "date": result.dates,
            "ret": result.returns,
            "turnover": result.turnover,
            "short": result.short,
            "n_long": result.n_long,
            "n_short": result.n_short,
            "n_universe": result.n_universe,
            "holding_days": result.holding_days,
        }
    ).to_csv(args.output_dir / f"{name}_daily.csv", index=False)
    (args.output_dir / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"Predictions for scripts/analyze_run.py: {args.output_dir / f'{name}_predictions.npz'}")


if __name__ == "__main__":
    main()
