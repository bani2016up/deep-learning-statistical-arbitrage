"""Load runs from ``results/``, build a comparison table, and save plots.

    uv run python -m analysis.compare_runs --results-dir results --output-dir results/comparison_plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None


def _load_config_file(config_file: Path) -> Dict[str, Any]:
    """Loads config from YAML or JSON using PyYAML if available or standard library fallback."""
    if not config_file.exists():
        return {}
    if yaml is not None:
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[Warning] Failed to load config via yaml from {config_file}: {e}")

    try:
        content = config_file.read_text(encoding="utf-8").strip()
        if content.startswith("{") and content.endswith("}"):
            return json.loads(content)
        # Parse simple key-value YAML lines
        res: Dict[str, Any] = {}
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if v.lower() == "true":
                    res[k] = True
                elif v.lower() == "false":
                    res[k] = False
                elif v.lower() in ("null", "none"):
                    res[k] = None
                else:
                    try:
                        if "." in v or "e" in v.lower():
                            res[k] = float(v)
                        else:
                            res[k] = int(v)
                    except ValueError:
                        res[k] = v
        return res
    except Exception as e:
        print(f"[Warning] Failed to load config from {config_file}: {e}")
        return {}


from analysis.metrics import (
    annualized_return,
    annualized_sharpe,
    annualized_volatility,
    breakeven_cost_bps,
    calmar_ratio,
    compute_run_metrics,
    conditional_value_at_risk,
    evaluate_cost_impact,
    one_way_turnover,
    maximum_drawdown,
    sortino_ratio,
    value_at_risk,
    win_rate,
)
from analysis.plots import (
    plot_correlation_matrix,
    plot_cumulative_returns,
    plot_drawdowns,
    plot_executive_dashboard,
    plot_return_distributions,
    plot_rolling_sharpe,
    plot_turnover_vs_sharpe,
)


def load_run(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Loads run configuration, metrics, and return series from a run directory."""
    if not run_dir.is_dir():
        return None

    run_name = run_dir.name
    config_file = run_dir / "config.yaml"
    if not config_file.exists():
        config_file = run_dir / "config.json"
    metrics_file = run_dir / "metrics.json"
    returns_file = run_dir / "returns.csv"

    config: Dict[str, Any] = _load_config_file(config_file)

    metrics: Dict[str, Any] = {}
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f) or {}
        except Exception as e:
            print(f"[Warning] Failed to load metrics from {metrics_file}: {e}")

    returns_series: Optional[pd.Series] = None
    if returns_file.exists():
        try:
            df = pd.read_csv(returns_file)
            date_col = next(
                (c for c in df.columns if c.lower() in ("date", "time", "timestamp")),
                None,
            )
            ret_col = next(
                (
                    c
                    for c in df.columns
                    if c.lower()
                    in (
                        "return",
                        "portfolio_return",
                        "strategy_return",
                        "returns",
                        "ret",
                    )
                ),
                None,
            )
            if ret_col is not None:
                if date_col is not None:
                    df[date_col] = pd.to_datetime(df[date_col])
                    returns_series = pd.Series(
                        df[ret_col].values, index=df[date_col], name=run_name
                    )
                else:
                    returns_series = pd.Series(df[ret_col].values, name=run_name)
        except Exception as e:
            print(f"[Warning] Failed to load returns from {returns_file}: {e}")

    # Fallback to test_predictions.npz if present in run_dir or outputs
    if returns_series is None:
        npz_file = run_dir / "test_predictions.npz"
        if not npz_file.exists():
            npz_file = run_dir / "predictions.npz"
        if npz_file.exists():
            try:
                npz = np.load(npz_file, allow_pickle=True)
                if "portfolio_returns" in npz:
                    rets = npz["portfolio_returns"]
                    dates = npz["dates"] if "dates" in npz else None
                    if dates is not None:
                        dates = pd.to_datetime(dates)
                        returns_series = pd.Series(rets, index=dates, name=run_name)
                    else:
                        returns_series = pd.Series(rets, name=run_name)
            except Exception as e:
                print(f"[Warning] Failed to load npz from {npz_file}: {e}")

    if returns_series is None and not metrics:
        return None

    daily_to = one_way_turnover(metrics)
    if returns_series is not None and len(returns_series) > 0:
        computed = compute_run_metrics(
            returns_series, daily_turnover=daily_to
        )
        # Values from metrics.json win; computed values only fill missing keys.
        for k, v in computed.items():
            if k not in metrics or metrics[k] is None:
                metrics[k] = v

    return {
        "run_name": run_name,
        "config": config,
        "metrics": metrics,
        "returns": returns_series,
        "path": run_dir,
    }


def scan_runs(base_dir: Path) -> List[Dict[str, Any]]:
    """Recursively or directly scans base_dir for run subdirectories."""
    runs = []
    if not base_dir.exists():
        return runs

    for d in sorted(base_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("."):
            loaded = load_run(d)
            if loaded:
                runs.append(loaded)

    if not runs:
        loaded = load_run(base_dir)
        if loaded:
            runs.append(loaded)

    return runs


def generate_comparison_table(runs: List[Dict[str, Any]]) -> pd.DataFrame:
    """Creates a consolidated comparison DataFrame across all runs."""
    rows = []
    for r in runs:
        m = r["metrics"].copy()
        c = r["config"]
        row: Dict[str, Any] = {"run_name": r["run_name"]}
        if "model" in c:
            row["model"] = c["model"]
        if "lookback" in c:
            row["lookback"] = c["lookback"]
        if "seed" in c:
            row["seed"] = c["seed"]

        row["annualized_sharpe"] = m.get("annualized_sharpe", m.get("sharpe", np.nan))
        row["annualized_return"] = m.get(
            "annualized_return", m.get("mean_return", np.nan)
        )
        row["annualized_volatility"] = m.get(
            "annualized_volatility", m.get("volatility", np.nan)
        )
        row["max_drawdown"] = m.get("max_drawdown", m.get("maximum_drawdown", np.nan))
        row["sortino_ratio"] = m.get("sortino_ratio", m.get("sortino", np.nan))
        row["calmar_ratio"] = m.get("calmar_ratio", m.get("calmar", np.nan))
        row["daily_turnover"] = m.get("daily_turnover", m.get("turnover", np.nan))
        row["net_sharpe_5bps"] = m.get("net_sharpe_5bps", m.get("net_sharpe", np.nan))
        row["net_sharpe_10bps"] = m.get("net_sharpe_10bps", np.nan)
        row["breakeven_cost_bps"] = m.get(
            "breakeven_cost_bps", m.get("break_even_cost_bps", np.nan)
        )
        row["win_rate"] = m.get(
            "win_rate", m.get("hit_rate", m.get("positive_day_fraction", np.nan))
        )
        row["cvar_95"] = m.get("cvar_95", m.get("historical_cvar_95", np.nan))
        row["hac_t_stat"] = m.get("hac_t_stat", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by="annualized_sharpe", ascending=False).reset_index(
            drop=True
        )
    return df


def format_markdown_table(df: pd.DataFrame) -> str:
    """Formats the comparison dataframe into a clean Markdown table with rounded numbers."""
    if df.empty:
        return "No experiment runs found."

    display_df = df.copy()
    format_map = {
        "annualized_sharpe": "{:.2f}",
        "annualized_return": "{:+.2%}",
        "annualized_volatility": "{:.2%}",
        "max_drawdown": "{:.2%}",
        "sortino_ratio": "{:.2f}",
        "calmar_ratio": "{:.2f}",
        "daily_turnover": "{:.2f}",
        "net_sharpe_5bps": "{:.2f}",
        "net_sharpe_10bps": "{:.2f}",
        "breakeven_cost_bps": "{:.1f}",
        "win_rate": "{:.1%}",
        "cvar_95": "{:.2%}",
    }

    for col, fmt in format_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].apply(
                lambda x: fmt.format(x) if pd.notnull(x) else "-"
            )

    headers = [str(c) for c in display_df.columns]
    rows = [[str(val) for val in row] for row in display_df.values]

    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(val))

    header_line = (
        "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    )
    sep_line = (
        "|-" + "-|-".join("-" * col_widths[i] for i in range(len(headers))) + "-|"
    )
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def compare_and_report(
    results_dir: Union[str, Path] = "results",
    output_dir: Union[str, Path] = "results/comparison_plots",
) -> Tuple[pd.DataFrame, str]:
    """Scans results_dir, outputs markdown summary table, and saves plots."""
    results_path = Path(results_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    runs = scan_runs(results_path)
    if not runs:
        print(f"[Notice] No runs found in {results_path.resolve()}.")
        return pd.DataFrame(), "No runs found."

    print(f"Found {len(runs)} experiment runs in {results_path}.")
    summary_df = generate_comparison_table(runs)

    csv_path = output_path / "summary_metrics.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Saved metrics summary table to {csv_path}")

    md_table = format_markdown_table(summary_df)

    returns_dict = {
        r["run_name"]: r["returns"] for r in runs if r["returns"] is not None
    }

    if returns_dict:
        print("Generating comparison plots...")
        plot_cumulative_returns(
            returns_dict, output_path=output_path / "cumulative_returns.png"
        )
        plot_rolling_sharpe(
            returns_dict, output_path=output_path / "rolling_sharpe.png"
        )
        plot_drawdowns(returns_dict, output_path=output_path / "drawdowns.png")
        plot_return_distributions(
            returns_dict, output_path=output_path / "return_distributions.png"
        )
        plot_correlation_matrix(
            returns_dict, output_path=output_path / "correlation_matrix.png"
        )
        plot_turnover_vs_sharpe(
            summary_df, output_path=output_path / "turnover_vs_sharpe.png"
        )
        plot_executive_dashboard(
            returns_dict,
            summary_df,
            output_path=output_path / "executive_dashboard.png",
        )
        print(f"All standard plots saved to {output_path}/")

    from analysis.paper_posthoc import (
        aggregate_seed_dispersion,
        compute_cost_sensitivity_curves,
        evaluate_holding_periods,
        evaluate_sparse_portfolio,
        generate_table1_pivot,
    )
    from analysis.plots import (
        plot_cost_sensitivity_curves,
        plot_holding_period_persistence,
        plot_seed_dispersion,
        plot_sparse_portfolios,
        plot_table1_heatmap,
    )

    summary_file = results_path / "summary.csv"
    active_summary = summary_df
    if summary_file.exists():
        try:
            active_summary = pd.read_csv(summary_file)
            print(f"Loaded master summary from {summary_file}")
        except Exception as e:
            print(f"[Warning] Failed to load {summary_file}: {e}")

    if not active_summary.empty:
        sens_df = compute_cost_sensitivity_curves(active_summary)
        if not sens_df.empty:
            plot_cost_sensitivity_curves(
                sens_df, output_path=output_path / "cost_sensitivity.png"
            )

        if "variant" in active_summary.columns:
            disp_df = aggregate_seed_dispersion(active_summary)
            if not disp_df.empty and "sharpe_mean" in disp_df.columns:
                plot_seed_dispersion(
                    disp_df,
                    metric="sharpe",
                    output_path=output_path / "seed_dispersion.png",
                )
                disp_df.to_csv(output_path / "seed_dispersion_summary.csv", index=False)

        piv = generate_table1_pivot(active_summary)
        if not piv.empty:
            plot_table1_heatmap(piv, output_path=output_path / "table1_pivot.png")
            piv.to_csv(output_path / "table1_pivot.csv")

    sparse_results = {}
    holding_results = {}
    for r in runs:
        run_p = r["path"]
        npz_p = run_p / "predictions.npz"
        if npz_p.exists():
            try:
                npz = np.load(npz_p, allow_pickle=True)
                if "weights" in npz:
                    weights = npz["weights"]
                    targets = None
                    if "targets" in npz:
                        targets = npz["targets"]
                    elif "returns" in npz and npz["returns"].ndim == 2:
                        targets = npz["returns"]
                    else:
                        try:
                            from experiments.data import load_windows

                            cfg = r["config"]
                            nf = cfg.get("n_factors", 5)
                            lb = cfg.get("lookback", 30)
                            w_data = load_windows(nf, lb)
                            if w_data.targets.shape[0] == weights.shape[0]:
                                targets = w_data.targets
                            elif "dates" in npz:
                                date_set = set(npz["dates"])
                                mask = np.array(
                                    [d in date_set for d in w_data.dates.astype(str)]
                                )
                                if mask.sum() == weights.shape[0]:
                                    targets = w_data.targets[mask]
                        except Exception:
                            pass

                    if targets is not None and targets.shape == weights.shape:
                        sparse_df = evaluate_sparse_portfolio(weights, targets)
                        sparse_results[r["run_name"]] = sparse_df
                        hold_df = evaluate_holding_periods(weights, targets)
                        holding_results[r["run_name"]] = hold_df
            except Exception as e:
                print(f"[Notice] Post-hoc analysis skipped for {r['run_name']}: {e}")

    if sparse_results:
        plot_sparse_portfolios(
            sparse_results, output_path=output_path / "sparse_portfolios.png"
        )
        print("Generated sparse portfolio evaluation plot.")
    if holding_results:
        plot_holding_period_persistence(
            holding_results, output_path=output_path / "holding_persistence.png"
        )
        print("Generated holding persistence evaluation plot.")

    return summary_df, md_table


def main():
    parser = argparse.ArgumentParser(
        description="Compare and plot statistical arbitrage experiment runs."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing run folders (default: results)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/comparison_plots",
        help="Directory where comparison plots and CSV will be written",
    )
    args = parser.parse_args()

    df, md = compare_and_report(args.results_dir, args.output_dir)
    print("\n" + "=" * 50 + " EXPERIMENT COMPARISON " + "=" * 50 + "\n")
    print(md)
    print("\n" + "=" * 123 + "\n")


if __name__ == "__main__":
    main()
