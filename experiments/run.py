"""Run one experiment and write results/<name>/ (config, metrics, returns, predictions, history)."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments import tracking
from experiments.data import REPO, load_run_data
from experiments.metrics import compute_metrics
from experiments.objectives import trading_costs
from experiments.trainer import RunConfig, run_protocol

RESULTS_DIR = Path(os.environ.get("DLSA_RESULTS_DIR", REPO / "results"))


def evaluate(config: RunConfig, output: dict[str, object]) -> tuple[pd.DataFrame, dict]:
    weights = output["weights"]
    returns = (weights * output["targets"]).sum(axis=1)
    costs, turnover, short = (
        t.numpy()
        for t in trading_costs(
            torch.from_numpy(weights), config.turnover_bps, config.short_bps
        )
    )
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(output["dates"]),
            "return": returns,
            "net_return": returns - costs,
            "turnover": turnover,
            "short_fraction": short,
            "block": output["block"],
        }
    )
    metrics = compute_metrics(
        returns, frame["net_return"].to_numpy(), weights, output["dates"]
    )
    metrics["runtime_seconds"] = output["runtime_seconds"]
    return frame, metrics


def _find_identical_run(config: RunConfig, results_dir: Path) -> Path | None:
    """Suites share their default variant; reuse a finished run with the same settings."""
    defaults = RunConfig(name="").to_dict()
    wanted = {k: v for k, v in config.to_dict().items() if k != "name"}
    for path in results_dir.glob("*/config.json"):
        if not (path.parent / "metrics.json").exists():
            continue
        existing = {**defaults, **json.loads(path.read_text())}
        existing.pop("name")
        if existing == wanted:
            return path.parent
    return None


def run_experiment(
    config: RunConfig, results_dir: Path = RESULTS_DIR, force: bool = False
) -> dict:
    run_dir = results_dir / config.name
    if (run_dir / "metrics.json").exists() and not force:
        return json.loads((run_dir / "metrics.json").read_text())
    twin = None if force else _find_identical_run(config, results_dir)
    if twin is not None:
        shutil.copytree(twin, run_dir, dirs_exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2))
        return json.loads((run_dir / "metrics.json").read_text())
    data = load_run_data(config.to_dict())
    tracking.start_run(config.to_dict(), group=config.name.split("__")[0])
    def on_block(block: int, values: dict) -> None:
        tracking.log_block(block, values)
        print(
            f"[{config.name}] block {block + 1} from {values['block_start']}: "
            f"oos_sharpe={values['block_oos_sharpe']:+.2f} loss={values['train_loss']:.3f}",
            flush=True,
        )

    output = run_protocol(
        config, data, checkpoint_dir=run_dir / "_blocks", on_block=on_block
    )
    frame, metrics = evaluate(config, output)
    tracking.finish(
        {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    )
    write_run(run_dir, config.to_dict(), frame, metrics, output, output["tickers"])
    shutil.rmtree(run_dir / "_blocks", ignore_errors=True)
    return metrics


def write_run(
    run_dir: Path,
    config: dict,
    frame: pd.DataFrame,
    metrics: dict,
    output: dict[str, object],
    tickers: np.ndarray,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    frame.to_csv(run_dir / "returns.csv", index=False, date_format="%Y-%m-%d")
    pd.DataFrame(output["history"]).to_csv(run_dir / "history.csv", index=False)
    # Same keys as the baseline's test_predictions.npz, so scripts/analyze_run.py works.
    np.savez_compressed(
        run_dir / "predictions.npz",
        returns=frame["return"].to_numpy(),
        weights=output["weights"].astype(np.float32),
        dates=np.asarray(output["dates"]).astype("datetime64[D]").astype(str),
        tickers=tickers,
    )
    # Written last: its presence marks a complete run (resume, dedupe, decompose rely on it).
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))


def _parse_bool(value: str) -> bool:
    if value.lower() not in ("true", "false", "1", "0"):
        raise argparse.ArgumentTypeError(f"expected true/false, got {value!r}")
    return value.lower() in ("true", "1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    parser.add_argument("--name", required=True)
    for name, field in fields.items():
        if name != "name":
            parser.add_argument(
                f"--{name.replace('_', '-')}",
                type=_parse_bool
                if isinstance(field.default, bool)
                else type(field.default),
                default=field.default,
            )
    parser.add_argument("--force", action="store_true")
    args = vars(parser.parse_args())
    force = args.pop("force")
    metrics = run_experiment(RunConfig(**args), force=force)
    print(
        json.dumps(
            {k: v for k, v in metrics.items() if k != "yearly_returns"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
