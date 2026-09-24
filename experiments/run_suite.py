"""Run one or more named suites (resumable) and refresh results/summary.csv.

uv run python -m experiments.run_suite epochs models
uv run python -m experiments.run_suite --list
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from experiments.run import RESULTS_DIR, run_experiment
from experiments.suites import SUITES


def write_summary() -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(RESULTS_DIR.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        if run_dir.name.startswith("_"):
            continue
        config = json.loads((run_dir / "config.json").read_text())
        metrics = json.loads(metrics_path.read_text())
        parts = run_dir.name.split("__")
        rows.append(
            {
                "run": run_dir.name,
                "suite": parts[0],
                "variant": parts[1] if len(parts) > 1 else run_dir.name,
                **{f"cfg_{k}": v for k, v in config.items() if k != "name"},
                **{k: v for k, v in metrics.items() if not isinstance(v, (dict, list))},
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame.to_csv(RESULTS_DIR / "summary.csv", index=False)
    return frame


def _out_of_time(longest_run: float) -> bool:
    """``DLSA_DEADLINE`` (unix seconds): don't start a run that would likely overrun it.

    Kaggle kills a session at its time limit without saving outputs, so the runner sets a
    deadline below the limit and stops launching new runs once the longest run so far no
    longer fits (with 20% headroom). Unfinished suites resume in the next session.
    """
    deadline = os.environ.get("DLSA_DEADLINE")
    return bool(deadline) and time.time() + 1.2 * longest_run > float(deadline)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", choices=list(SUITES), metavar="SUITE")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--shard",
        default="0/1",
        help="i/n: run only configs with index %% n == i (one process per GPU)",
    )
    args = parser.parse_args()
    if args.list:
        for name, fn in SUITES.items():
            print(
                f"{name:14s} {len(list(fn())):3d} runs  {fn.__doc__.strip().splitlines()[0]}"
            )
        return
    longest_run = 0.0
    for suite in args.suites:
        shard, shards = (int(part) for part in args.shard.split("/"))
        configs = [c for i, c in enumerate(SUITES[suite]()) if i % shards == shard]
        for index, config in enumerate(configs, 1):
            if _out_of_time(longest_run):
                print(
                    f"[{suite}] deadline: skipping {config.name} and later runs",
                    flush=True,
                )
                break
            started = time.perf_counter()
            metrics = run_experiment(config, force=args.force)
            print(
                f"[{suite} {index}/{len(configs)}] {config.name}: sharpe={metrics['sharpe']:.2f} "
                f"net_sharpe={metrics['net_sharpe']:.2f} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
            longest_run = max(longest_run, time.perf_counter() - started)
        write_summary()
    print(f"Summary: {RESULTS_DIR / 'summary.csv'}")


if __name__ == "__main__":
    main()
