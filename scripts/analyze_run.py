from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dlsa_baseline.analysis.reporting import create_plots, write_json_report, write_markdown_report
from dlsa_baseline.analysis.statistics import analyze_predictions, load_predictions
from dlsa_baseline.config import ROOT


def named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(path)


def load_factors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame:
        raise ValueError("factor CSV must contain a date column")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.set_index("date").sort_index()
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        raise ValueError("factor CSV must contain decimal-return numeric columns")
    return numeric


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze an out-of-sample strategy run")
    parser.add_argument("--predictions", type=Path, default=ROOT / "outputs/test_predictions.npz")
    parser.add_argument("--benchmark", action="append", type=named_path, default=[])
    parser.add_argument("--factors", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/analysis")
    parser.add_argument("--cost-bps", nargs="+", type=float, default=[0, 1, 5, 10, 25, 50])
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--block-size", type=int, default=20)
    parser.add_argument("--pbo-blocks", type=int, default=8)
    parser.add_argument("--trial-sharpes", nargs="*", type=float, default=[])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    primary = load_predictions(args.predictions)
    benchmarks = {name: load_predictions(path) for name, path in args.benchmark}
    factors = load_factors(args.factors) if args.factors else None
    report, plot_data = analyze_predictions(
        primary,
        benchmarks=benchmarks,
        factors=factors,
        costs_bps=args.cost_bps,
        bootstrap_samples=args.bootstrap_samples,
        block_size=args.block_size,
        seed=args.seed,
        pbo_blocks=args.pbo_blocks,
        extra_trial_sharpes=args.trial_sharpes,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json_report(report, args.output_dir / "statistical_report.json")
    write_markdown_report(report, args.output_dir / "statistical_report.md")
    create_plots(primary, report, plot_data, args.output_dir)
    print(json.dumps(report["success_assessment"], indent=2))
    print(f"Reports and plots written to {args.output_dir}")


if __name__ == "__main__":
    main()
