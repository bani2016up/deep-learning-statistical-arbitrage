from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dlsa_baseline.analysis.statistics import PredictionData, drawdown_series


def json_safe(value: Any) -> Any:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json_report(report: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    performance = report["performance"]
    inference = report["inference"]
    assessment = report["success_assessment"]
    lines = [
        "# Statistical Run Analysis",
        "",
        "> Engineering diagnostics only. This report is not evidence of a scientific",
        "> replication or a guarantee of future returns.",
        "",
        "## Assessment",
        "",
        f"**{assessment['label']}** ({assessment['passed_fraction']:.0%} of available checks passed)",
        "",
        "| Check | Passed |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {name.replace('_', ' ')} | {'yes' if passed else 'no'} |"
        for name, passed in assessment["checks"].items()
    )
    lines.extend(
        [
            "",
            "## Performance",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    selected = [
        "observations",
        "start_date",
        "end_date",
        "total_return",
        "cagr",
        "annualized_mean",
        "annualized_volatility",
        "annualized_sharpe",
        "annualized_sortino",
        "calmar_ratio",
        "maximum_drawdown",
        "maximum_drawdown_duration_days",
        "skewness",
        "excess_kurtosis",
        "historical_var_95",
        "historical_cvar_95",
        "positive_day_fraction",
        "positive_month_fraction",
        "positive_year_fraction",
        "profit_factor",
    ]
    lines.extend(f"| {name} | {_format(performance.get(name))} |" for name in selected)
    hac = inference["newey_west_mean"]
    bootstrap = inference["block_bootstrap"]
    lines.extend(
        [
            "",
            "## Statistical Inference",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Newey-West mean t-statistic | {_format(hac['t_statistic'])} |",
            f"| Newey-West one-sided p-value | {_format(hac['one_sided_positive_p_value'])} |",
            f"| Bootstrap Sharpe 95% CI | {_format(bootstrap['sharpe_confidence_interval_95'])} |",
            f"| Probabilistic Sharpe Ratio | {_format(inference['probabilistic_sharpe_ratio'])} |",
            f"| Deflated Sharpe probability | {_format(inference['deflated_sharpe']['deflated_sharpe_probability'])} |",
            f"| Minimum track record days | {_format(inference['minimum_track_record_length_days'])} |",
            f"| Ljung-Box p-value | {_format(inference['autocorrelation']['ljung_box_p_value'])} |",
        ]
    )
    if "transaction_cost_sensitivity" in report:
        lines.extend(
            [
                "",
                "## Transaction Costs",
                "",
                f"Break-even cost: {_format(report['transaction_cost_sensitivity']['break_even_cost_bps'])} bps",
                "",
                "| Cost (bps) | Annual mean | Sharpe | Total return |",
                "|---:|---:|---:|---:|",
            ]
        )
        for cost, result in report["transaction_cost_sensitivity"]["scenarios"].items():
            lines.append(
                f"| {cost} | {_format(result['annualized_mean'])} | "
                f"{_format(result['annualized_sharpe'])} | {_format(result['total_return'])} |"
            )
    if report.get("comparisons"):
        lines.extend(
            [
                "",
                "## Benchmark Comparisons",
                "",
                "| Benchmark | Sharpe difference | Annual mean difference | HAC p-value |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, result in report["comparisons"].items():
            difference = result["primary_sharpe"] - result["benchmark_sharpe"]
            lines.append(
                f"| {name} | {_format(difference)} | {_format(result['annualized_mean_difference'])} | "
                f"{_format(result['hac_mean_difference']['one_sided_positive_p_value'])} |"
            )
    if report.get("factor_regression"):
        factor = report["factor_regression"]
        lines.extend(
            [
                "",
                "## Factor Regression",
                "",
                f"Annualized alpha: {_format(factor['annualized_alpha'])}",
                "",
                f"HAC alpha t-statistic: {_format(factor['alpha_hac_t_statistic'])}",
                "",
                f"Alpha p-value: {_format(factor['alpha_two_sided_p_value'])}",
            ]
        )
    if report.get("multiple_testing"):
        pbo = report["multiple_testing"]["pbo"]
        lines.extend(
            [
                "",
                "## Multiple Testing and PBO",
                "",
                f"PBO: {_format(pbo['probability_of_backtest_overfitting'])}",
                "",
                pbo["warning"],
            ]
        )
    lines.extend(["", "## Interpretation Warning", "", assessment["warning"], ""])
    path.write_text("\n".join(lines))


def create_plots(
    data: PredictionData,
    report: dict[str, Any],
    plot_data: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(data.dates)
    wealth, drawdowns = drawdown_series(data.returns)

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(dates, wealth, color="#1f4e79")
    axes[0].set_ylabel("Cumulative wealth")
    axes[0].grid(alpha=0.2)
    axes[1].fill_between(dates, drawdowns, 0, color="#a61c3c", alpha=0.75)
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "equity_and_drawdown.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 4.5))
    for window, series in plot_data["rolling_sharpes"].items():
        axis.plot(series.index, series.values, label=f"{window} days")
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_ylabel("Annualized rolling Sharpe")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_dir / "rolling_sharpe.png", dpi=160)
    plt.close(figure)

    monthly = pd.Series(report["performance"]["monthly_returns"], dtype=float)
    monthly.index = pd.to_datetime(monthly.index)
    table = monthly.to_frame("return")
    table["year"] = table.index.year
    table["month"] = table.index.month
    heatmap = table.pivot(index="year", columns="month", values="return")
    figure, axis = plt.subplots(figsize=(11, max(2.5, len(heatmap) * 0.65)))
    image = axis.imshow(heatmap.to_numpy(), cmap="RdYlGn", aspect="auto", vmin=-0.1, vmax=0.1)
    axis.set_xticks(
        np.arange(12),
        labels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    axis.set_yticks(np.arange(len(heatmap.index)), labels=heatmap.index)
    axis.set_title("Monthly compounded returns")
    figure.colorbar(image, ax=axis, label="Return")
    figure.tight_layout()
    figure.savefig(output_dir / "monthly_returns.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(data.returns, bins=40, density=True, color="#486f99", alpha=0.8)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Daily portfolio return")
    axis.set_ylabel("Density")
    figure.tight_layout()
    figure.savefig(output_dir / "return_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(plot_data["bootstrap_sharpes"], bins=50, color="#5d8c56", alpha=0.85)
    axis.axvline(0, color="black", linewidth=0.8)
    axis.set_xlabel("Circular block-bootstrap annualized Sharpe")
    axis.set_ylabel("Samples")
    figure.tight_layout()
    figure.savefig(output_dir / "bootstrap_sharpe.png", dpi=160)
    plt.close(figure)

    if "transaction_cost_sensitivity" in report:
        scenarios = report["transaction_cost_sensitivity"]["scenarios"]
        costs = np.array([float(value) for value in scenarios])
        sharpes = np.array([scenarios[key]["annualized_sharpe"] for key in scenarios])
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(costs, sharpes, marker="o", color="#8a4f19")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("Transaction cost (bps per unit turnover)")
        axis.set_ylabel("Net annualized Sharpe")
        axis.grid(alpha=0.2)
        figure.tight_layout()
        figure.savefig(output_dir / "turnover_cost_sensitivity.png", dpi=160)
        plt.close(figure)
