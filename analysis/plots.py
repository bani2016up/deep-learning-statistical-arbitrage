"""Matplotlib figures for comparing runs (returns, drawdowns, rolling Sharpe, costs, seeds).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.metrics import (
    annualized_return,
    annualized_sharpe,
    annualized_volatility,
    compute_cumulative_returns,
    compute_drawdown_series,
    maximum_drawdown,
)

PALETTE = [
    "#1f77b4",  # Muted Blue
    "#ff7f0e",  # Safety Orange
    "#2ca02c",  # Cooked Asparagus Green
    "#d62728",  # Brick Red
    "#9467bd",  # Muted Purple
    "#8c564b",  # Chestnut Brown
    "#e377c2",  # Middle Orchid
    "#7f7f7f",  # Middle Gray
    "#bcbd22",  # Curry Yellow-Green
    "#17becf",  # Blue-Teal
]


def _setup_style():
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Helvetica", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8


def plot_cumulative_returns(
    runs_dict: Dict[str, pd.Series],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Cumulative Out-of-Sample Performance",
    log_scale: bool = False,
    show_drawdown: bool = True,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plots cumulative return curves for multiple runs, with optional underwater drawdowns."""
    _setup_style()
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            continue
        cum_ret = compute_cumulative_returns(s)
        color = PALETTE[i % len(PALETTE)]
        ann_sr = annualized_sharpe(s)
        tot_ret = cum_ret.iloc[-1] * 100 if len(cum_ret) > 0 else 0.0
        label = f"{name} (Tot: {tot_ret:+.1f}%, SR: {ann_sr:.2f})"

        ax1.plot(
            cum_ret.index, cum_ret.values * 100, label=label, color=color, linewidth=1.8
        )

        if ax2 is not None:
            dd = compute_drawdown_series(s)
            ax2.plot(dd.index, dd.values * 100, label=name, color=color, linewidth=1.2)
            ax2.fill_between(dd.index, dd.values * 100, 0, color=color, alpha=0.15)

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11, fontweight="bold")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    if log_scale:
        ax1.set_yscale("symlog")
    ax1.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax1.grid(True, linestyle=":", alpha=0.6)

    if ax2 is not None:
        ax2.set_title("Underwater Drawdown", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
        ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_rolling_sharpe(
    runs_dict: Dict[str, pd.Series],
    window: int = 63,
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    freq: int = 252,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plots rolling annualized Sharpe ratio across runs."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if title is None:
        title = f"Rolling {window}-Day Annualized Sharpe Ratio"

    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) < window:
            continue
        rolling_mean = s.rolling(window).mean()
        rolling_std = s.rolling(window).std(ddof=1)
        rolling_sr = (rolling_mean / (rolling_std + 1e-8)) * np.sqrt(freq)
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            rolling_sr.index, rolling_sr.values, label=name, color=color, linewidth=1.6
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(2.0, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.set_ylabel("Sharpe Ratio", fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_drawdowns(
    runs_dict: Dict[str, pd.Series],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Comparative Drawdown Trajectories",
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """Plots underwater drawdowns for all runs."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            continue
        dd = compute_drawdown_series(s) * 100.0
        mdd = maximum_drawdown(s) * 100.0
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            dd.index,
            dd.values,
            label=f"{name} (Max DD: -{mdd:.1f}%)",
            color=color,
            linewidth=1.5,
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Drawdown (%)", fontsize=10, fontweight="bold")
    ax.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax.legend(loc="lower left", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_return_distributions(
    runs_dict: Dict[str, pd.Series],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Daily Return Distributions",
    figsize: tuple = (11, 5),
) -> plt.Figure:
    """Plots density/histogram of daily returns across models."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna() * 100.0
        if len(s) == 0:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.hist(
            s.values,
            bins=40,
            density=True,
            alpha=0.35,
            color=color,
            label=f"{name} (std={s.std():.2f}%)",
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Daily Return (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Density", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_turnover_vs_sharpe(
    summary_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Turnover vs. Sharpe Ratio Trade-Off",
    figsize: tuple = (9, 6),
) -> plt.Figure:
    """Scatter plot of Daily Turnover vs. Annualized Sharpe Ratio."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if (
        "daily_turnover" not in summary_df.columns
        or "annualized_sharpe" not in summary_df.columns
    ):
        ax.text(0.5, 0.5, "Turnover or Sharpe data missing", ha="center", va="center")
        return fig

    for i, (idx, row) in enumerate(summary_df.iterrows()):
        to = row["daily_turnover"]
        sr = row["annualized_sharpe"]
        name = row.get("run_name", str(idx))
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(to, sr, color=color, s=120, zorder=5, label=name)
        ax.annotate(
            name,
            (to, sr),
            textcoords="offset points",
            xytext=(6, 6),
            ha="left",
            fontsize=9,
            fontweight="bold",
        )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Daily 1-Way Turnover (L1 fraction)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_correlation_matrix(
    runs_dict: Dict[str, pd.Series],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Strategy Return Cross-Correlations",
    figsize: tuple = (7, 6),
) -> plt.Figure:
    """Heatmap showing pairwise correlations between strategy daily returns."""
    _setup_style()
    df = pd.DataFrame(runs_dict).dropna()
    fig, ax = plt.subplots(figsize=figsize)

    if df.empty or df.shape[1] < 2:
        ax.text(
            0.5,
            0.5,
            "Insufficient strategies for correlation matrix",
            ha="center",
            va="center",
        )
        return fig

    corr = df.corr()
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation", fontsize=9)

    ticks = range(len(corr.columns))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(corr.columns, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.5 else "black"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_executive_dashboard(
    runs_dict: Dict[str, pd.Series],
    summary_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Deep Learning Statistical Arbitrage — Executive Performance Dashboard",
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """4-panel overview of several runs."""
    _setup_style()
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            continue
        cum = compute_cumulative_returns(s) * 100.0
        color = PALETTE[i % len(PALETTE)]
        ax1.plot(cum.index, cum.values, label=name, color=color, linewidth=1.7)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_title(
        "1. Cumulative Out-of-Sample Return (%)", fontsize=11, fontweight="bold"
    )
    ax1.legend(loc="upper left", fontsize=8, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = fig.add_subplot(gs[0, 1])
    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) == 0:
            continue
        dd = compute_drawdown_series(s) * 100.0
        color = PALETTE[i % len(PALETTE)]
        ax2.plot(dd.index, dd.values, label=name, color=color, linewidth=1.4)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_title("2. Underwater Drawdown (%)", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3 = fig.add_subplot(gs[1, 0])
    for i, (name, s) in enumerate(runs_dict.items()):
        s = pd.Series(s).dropna()
        if len(s) < 40:
            continue
        rm = s.rolling(63).mean()
        rs = s.rolling(63).std(ddof=1)
        sr = (rm / (rs + 1e-8)) * np.sqrt(252)
        color = PALETTE[i % len(PALETTE)]
        ax3.plot(sr.index, sr.values, label=name, color=color, linewidth=1.5)
    ax3.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax3.set_title("3. Rolling 63-Day Sharpe Ratio", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    ax4 = fig.add_subplot(gs[1, 1])
    if not summary_df.empty and "annualized_sharpe" in summary_df.columns:
        names = summary_df.get("run_name", summary_df.index)
        sharpes = summary_df["annualized_sharpe"]
        x = np.arange(len(names))
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
        bars = ax4.bar(
            x, sharpes, color=colors, width=0.5, edgecolor="black", alpha=0.85
        )
        ax4.set_xticks(x)
        ax4.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax4.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            y_pos = height + 0.05 if height >= 0 else height - 0.15
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                y_pos,
                f"{height:.2f}",
                ha="center",
                fontsize=8,
                fontweight="bold",
            )
        ax4.set_title("4. Annualized Sharpe Comparison", fontsize=11, fontweight="bold")
    ax4.grid(True, linestyle=":", alpha=0.6)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(
        top=0.92, bottom=0.08, left=0.08, right=0.95, hspace=0.35, wspace=0.25
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_monthly_heatmap(
    returns_series: pd.Series,
    title: str = "Monthly Returns (%)",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plots a heatmap of monthly returns by year and month."""
    _setup_style()
    s = pd.Series(returns_series).dropna()
    fig, ax = plt.subplots(figsize=figsize)

    if not isinstance(s.index, pd.DatetimeIndex):
        try:
            s.index = pd.to_datetime(s.index)
        except Exception:
            ax.text(
                0.5,
                0.5,
                "DatetimeIndex required for monthly heatmap",
                ha="center",
                va="center",
            )
            return fig

    monthly = (
        s.groupby([s.index.year, s.index.month]).apply(lambda x: (1.0 + x).prod() - 1.0)
        * 100.0
    )
    monthly_df = monthly.unstack(level=1)

    all_months = list(range(1, 13))
    monthly_df = monthly_df.reindex(columns=all_months)
    month_names = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    monthly_df.columns = month_names

    yearly = s.groupby(s.index.year).apply(lambda x: (1.0 + x).prod() - 1.0) * 100.0
    monthly_df["Year"] = yearly

    data = monthly_df.values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 5.0)
    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Return (%)", fontsize=9)

    ax.set_xticks(range(len(monthly_df.columns)))
    ax.set_xticklabels(monthly_df.columns, fontsize=9, fontweight="bold")
    ax.set_yticks(range(len(monthly_df.index)))
    ax.set_yticklabels(monthly_df.index, fontsize=9, fontweight="bold")

    for i in range(len(monthly_df.index)):
        for j in range(len(monthly_df.columns)):
            val = monthly_df.iloc[i, j]
            if pd.notnull(val):
                text_color = "black" if abs(val) < vmax * 0.7 else "white"
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_sparse_portfolios(
    sparse_results_dict: Dict[str, pd.DataFrame],
    metric: str = "annualized_sharpe",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Sparse Portfolio Performance vs. Proportion of Retained Assets",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plots metric curves as a function of retained top fraction (Paper Section III.K, Figure 9)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, df) in enumerate(sparse_results_dict.items()):
        if df.empty or metric not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        x = df["top_fraction"] * 100.0
        y = df[metric]
        ax.plot(x, y, marker="o", linewidth=2.0, color=color, label=name)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Top Weight Assets Retained (%)", fontsize=10, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_holding_period_persistence(
    holding_results_dict: Dict[str, pd.DataFrame],
    metric: str = "annualized_sharpe",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Holding Period Persistence (Overlapping Multi-Day Weights)",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plots performance degradation across holding horizons B=1..30 (Paper Section III.M, Figure 12)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    for i, (name, df) in enumerate(holding_results_dict.items()):
        if df.empty or metric not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            df["holding_days"],
            df[metric],
            marker="s",
            linewidth=2.0,
            color=color,
            label=name,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Holding Period B (Trading Days)", fontsize=10, fontweight="bold")
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_cost_sensitivity_curves(
    sensitivity_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Net Sharpe Ratio vs. Transaction Cost (bps)",
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """Plots Net Sharpe decay as transaction cost increases from 0 to 30 bps."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if sensitivity_df.empty or "run" not in sensitivity_df.columns:
        ax.text(0.5, 0.5, "No sensitivity data available", ha="center", va="center")
        return fig

    for i, (name, group) in enumerate(sensitivity_df.groupby("run")):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(
            group["cost_bps"],
            group["net_sharpe"],
            linewidth=1.8,
            color=color,
            label=name,
        )

    ax.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.8)
    ax.axvline(
        5.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.6, label="5 bps (Base)"
    )
    ax.axvline(
        10.0, color="red", linestyle=":", linewidth=0.8, alpha=0.6, label="10 bps"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(
        "Transaction Cost (bps per unit turnover)", fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Annualized Net Sharpe Ratio", fontsize=10, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_seed_dispersion(
    dispersion_df: pd.DataFrame,
    metric: str = "sharpe",
    output_path: Optional[Union[str, Path]] = None,
    title: Optional[str] = None,
    figsize: tuple = (11, 5),
) -> plt.Figure:
    """Plots bar chart of mean +/- std across seeds per variant."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if dispersion_df.empty or mean_col not in dispersion_df.columns:
        ax.text(
            0.5,
            0.5,
            f"Metric {metric} not found in dispersion summary",
            ha="center",
            va="center",
        )
        return fig

    x_labels = dispersion_df.get("variant", dispersion_df.index.astype(str))
    means = dispersion_df[mean_col]
    stds = dispersion_df[std_col].fillna(0.0)

    x = np.arange(len(x_labels))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(x_labels))]

    bars = ax.bar(
        x, means, yerr=stds, capsize=4, color=colors, alpha=0.85, edgecolor="black"
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=25, ha="right", fontsize=9, fontweight="bold")

    if title is None:
        title = (
            f"Multi-Seed Robustness: {metric.replace('_', ' ').title()} (Mean ± Std)"
        )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10, fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_table1_heatmap(
    pivot_df: pd.DataFrame,
    metric_name: str = "Annualized Sharpe Ratio",
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Table I: Model Performance Across Factor Counts (K)",
    figsize: tuple = (8, 5),
) -> plt.Figure:
    """Heatmap visualization of Table-I (Model x K factors)."""
    _setup_style()
    fig, ax = plt.subplots(figsize=figsize)

    if pivot_df.empty:
        ax.text(0.5, 0.5, "Table I pivot data empty", ha="center", va="center")
        return fig

    data = pivot_df.values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 1.0)
    im = ax.imshow(data, cmap="coolwarm", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label(metric_name, fontsize=9)

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(
        [f"K={c}" for c in pivot_df.columns], fontsize=9, fontweight="bold"
    )
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=9, fontweight="bold")

    for i in range(len(pivot_df.index)):
        for j in range(len(pivot_df.columns)):
            val = pivot_df.iloc[i, j]
            if pd.notnull(val):
                text_color = "white" if abs(val) > vmax * 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_exposure_dynamics(
    runs_frames: Dict[str, pd.DataFrame],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Portfolio Exposure & Rebalancing Dynamics",
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """Plots Net Exposure, Short Fraction, and Daily Turnover over time across strategies."""
    _setup_style()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, sharex=True)

    for i, (name, df) in enumerate(runs_frames.items()):
        if df.empty or "date" not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        dates = pd.to_datetime(df["date"])

        if "net_exposure" in df.columns:
            net_exp = df["net_exposure"]
            ax1.plot(dates, net_exp, label=name, color=color, linewidth=1.4, alpha=0.85)

        if "short_fraction" in df.columns:
            ax2.plot(
                dates,
                df["short_fraction"] * 100.0,
                label=name,
                color=color,
                linewidth=1.4,
                alpha=0.85,
            )

        if "turnover" in df.columns:
            ax3.plot(
                dates, df["turnover"], label=name, color=color, linewidth=1.2, alpha=0.7
            )

    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Net Exposure", fontsize=10, fontweight="bold")
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.axhline(
        50.0, color="red", linestyle=":", linewidth=0.8, alpha=0.7, label="Balanced 50%"
    )
    ax2.set_ylabel("Short Fraction (%)", fontsize=10, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=8, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    ax3.set_ylabel("1-Way Turnover", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Date", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_drift_decomposition_bars(
    decomp_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Drift Decomposition: Residual Drift vs. Pure Neutral Arbitrage",
    figsize: tuple = (14, 7),
) -> plt.Figure:
    """Grouped bar chart per variant showing Sharpe from Drift vs. Sharpe from Neutral Arbitrage."""
    _setup_style()
    df = decomp_df.copy()
    if "suite" in df.columns:
        df.loc[df["suite"] == "neutral", "sharpe_drift"] = 0.0

    group_col = "variant" if "variant" in df.columns else "run"
    grouped = (
        df.groupby(group_col)[["sharpe_total", "sharpe_drift", "sharpe_neutral"]]
        .mean()
        .reset_index()
    )
    grouped = grouped.sort_values(by="sharpe_total", ascending=False).reset_index(
        drop=True
    )

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(grouped))
    width = 0.28

    ax.bar(
        x - width,
        grouped["sharpe_total"],
        width,
        label="Total Sharpe",
        color="#1f77b4",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x,
        grouped["sharpe_drift"],
        width,
        label="Drift Component (e_t × EW_res)",
        color="#ff7f0e",
        alpha=0.85,
        edgecolor="black",
    )
    ax.bar(
        x + width,
        grouped["sharpe_neutral"],
        width,
        label="Neutral Arbitrage Remainder",
        color="#2ca02c",
        alpha=0.85,
        edgecolor="black",
    )

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        grouped[group_col], rotation=35, ha="right", fontsize=9, fontweight="bold"
    )
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_selection_heatmaps(
    selection_df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Honest Model Selection: Hyperparameter Heatmaps (2020-2022 Selection vs. 2023-2025 Holdout)",
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """Plots 2x2 grid of heatmaps (Gross & Net Sharpe across Selection and Holdout) for (cost_weight x smoothing B)."""
    import re

    _setup_style()
    df = selection_df.copy()
    ens = df[df["run"].str.startswith("ensemble__final-cnn_neutral_cw")].copy()
    if ens.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No ensemble selection data found", ha="center", va="center")
        return fig

    ens["cw"] = ens["variant"].apply(
        lambda x: float(re.search(r"cw([0-9.]+)", x).group(1))
    )
    ens["B"] = ens["smoothing_days"].astype(int)

    metrics = [
        ("sel_sharpe", "Selection Gross Sharpe (2020-2022)", "coolwarm"),
        ("hold_sharpe", "Holdout Gross Sharpe (2023-2025)", "coolwarm"),
        ("sel_net_sharpe", "Selection Net Sharpe @ 5 bps", "RdYlGn"),
        ("hold_net_sharpe", "Holdout Net Sharpe @ 5 bps", "RdYlGn"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (col, panel_title, cmap) in enumerate(metrics):
        ax = axes[idx]
        piv = ens.pivot(index="cw", columns="B", values=col)
        data = piv.values
        vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.5)
        vmin = -vmax if cmap in ("coolwarm", "RdYlGn") else 0.0

        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Annualized Sharpe", fontsize=9)

        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels(
            [f"B={b}" for b in piv.columns], fontsize=9, fontweight="bold"
        )
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels(
            [f"cw={w:g}" for w in piv.index], fontsize=9, fontweight="bold"
        )

        for i in range(len(piv.index)):
            for j in range(len(piv.columns)):
                val = piv.iloc[i, j]
                if pd.notnull(val):
                    text_color = "white" if abs(val) > vmax * 0.6 else "black"
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=9,
                        fontweight="bold",
                    )

        ax.set_title(panel_title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Smoothing Horizon B (Days)", fontsize=9, fontweight="bold")
        ax.set_ylabel("Loss Cost Weight (cw)", fontsize=9, fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig
