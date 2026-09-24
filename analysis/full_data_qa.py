"""Checks for the full panel and the residual files under ``data/full/``.

Panel coverage over time, extreme returns, FF5 factor statistics, and eligible names per day.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = (
    REPO_ROOT / "data" / "processed" / "ff5_daily_panel.parquet"
)
OUTPUT_DIR = REPO_ROOT / "results" / "full_qa"


def audit_raw_panel(parquet_path: Path) -> pd.DataFrame:
    """Analyze the raw 148 MB FF5 daily parquet dataset."""
    print("=" * 60)
    print(f"AUDITING RAW PANEL: {parquet_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found at {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"])
    min_date, max_date = df["date"].min(), df["date"].max()
    unique_dates = df["date"].nunique()
    unique_tickers = df["ticker"].nunique()
    print(
        f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')} ({unique_dates} trading days)"
    )
    print(f"Total unique tickers: {unique_tickers:,}")

    daily_counts = df.groupby("date")["ticker"].count()
    print(
        f"Tickers per day: mean={daily_counts.mean():.1f}, min={daily_counts.min()}, max={daily_counts.max()}, median={daily_counts.median():.1f}"
    )

    ret = df["stock_return"]
    print("-" * 40)
    print("STOCK RETURN SANITY:")
    print(f"Min return: {ret.min():+.4%}")
    print(f"Max return: {ret.max():+.4%}")
    print(f"Mean return: {ret.mean():+.4%}, Std: {ret.std():.4%}")
    p1, p5, p50, p95, p99 = np.percentile(ret.dropna(), [1, 5, 50, 95, 99])
    print(
        f"Percentiles: 1%={p1:+.2%}, 5%={p5:+.2%}, 50%={p50:+.2%}, 95%={p95:+.2%}, 99%={p99:+.2%}"
    )

    outliers = df[df["stock_return"].abs() > 1.0]
    print(
        f"Outliers with |return| > 100%: {len(outliers):,} rows ({len(outliers) / len(df):.4%})"
    )
    if not outliers.empty:
        top_gainers = outliers.nlargest(5, "stock_return")[
            ["date", "ticker", "stock_return"]
        ]
        top_losers = outliers.nsmallest(5, "stock_return")[
            ["date", "ticker", "stock_return"]
        ]
        print("Top 5 Extreme Positive Spikes:\n", top_gainers.to_string(index=False))
        print("Top 5 Extreme Negative Drops:\n", top_losers.to_string(index=False))

    print("-" * 40)
    print("FF5 FACTORS SANITY (Daily annualized stats):")
    factor_cols = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    available_factors = [c for c in factor_cols if c in df.columns]

    factor_df = df.groupby("date")[available_factors].first()
    f_stats = pd.DataFrame(index=available_factors)
    f_stats["Annualized Mean"] = factor_df.mean() * 252
    f_stats["Annualized Vol"] = factor_df.std() * np.sqrt(252)
    f_stats["Sharpe"] = f_stats["Annualized Mean"] / f_stats["Annualized Vol"]
    print(f_stats.round(4).to_string())

    print("\nFactor Correlation Matrix:")
    print(factor_df[available_factors].corr().round(3).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(
        daily_counts.index, daily_counts.values, color="#1f77b4", linewidth=1.0
    )
    axes[0, 0].set_title(
        "Active Stocks Count per Trading Day (1998-2016)", fontweight="bold"
    )
    axes[0, 0].set_ylabel("Ticker Count")
    axes[0, 0].grid(True, linestyle=":", alpha=0.6)

    cum_factors = (1.0 + factor_df[["mkt_rf", "smb", "hml", "rmw", "cma"]]).cumprod()
    for col in cum_factors.columns:
        axes[0, 1].plot(cum_factors.index, cum_factors[col], label=col, linewidth=1.2)
    axes[0, 1].set_title("Fama-French 5 Cumulative Factor Growth", fontweight="bold")
    axes[0, 1].legend(loc="upper left")
    axes[0, 1].grid(True, linestyle=":", alpha=0.6)

    # Histogram truncated at +/-20% so the body of the distribution is visible.
    clipped_rets = np.clip(ret.dropna(), -0.20, 0.20) * 100.0
    axes[1, 0].hist(
        clipped_rets, bins=100, color="#2ca02c", edgecolor="black", alpha=0.7
    )
    axes[1, 0].set_title(
        "Daily Return Distribution (Clipped at ±20%)", fontweight="bold"
    )
    axes[1, 0].set_xlabel("Return (%)")
    axes[1, 0].grid(True, linestyle=":", alpha=0.6)

    outlier_years = outliers["date"].dt.year.value_counts().sort_index()
    axes[1, 1].bar(
        outlier_years.index,
        outlier_years.values,
        color="#d62728",
        edgecolor="black",
        alpha=0.8,
    )
    axes[1, 1].set_title(
        "Extreme Outliers Count (|r| > 100%) per Year", fontweight="bold"
    )
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plot_path = OUTPUT_DIR / "raw_panel_audit.png"
    plt.savefig(plot_path, dpi=200)
    print(f"\nSaved raw panel audit plot to {plot_path}")
    return df


def audit_processed_residuals(full_data_dir: Path) -> None:
    """Analyze processed residuals files under data/full/ if present."""
    npz_files = list(full_data_dir.glob("residuals_*.npz"))
    if not npz_files:
        print(
            f"\nNo residuals_*.npz found in {full_data_dir}. Run this again after data/full/*.npz residuals are generated."
        )
        return

    print("=" * 60)
    print(f"AUDITING PROCESSED RESIDUALS IN: {full_data_dir}")
    for npz_path in npz_files:
        print(f"\n--- Residuals file: {npz_path.name} ---")
        data = np.load(npz_path, allow_pickle=True)
        keys = list(data.keys())
        print(f"Keys: {keys}")

        residuals = data["residuals"]
        print(
            f"Residuals matrix shape: {residuals.shape} (T={residuals.shape[0]}, N={residuals.shape[1]})"
        )

        mask = data.get("eligible", data.get("mask"))
        if mask is not None:
            eligible_daily = mask.sum(axis=1)
            print(
                f"Eligible assets per day: mean={eligible_daily.mean():.1f}, min={eligible_daily.min()}, max={eligible_daily.max()}, median={np.median(eligible_daily):.1f}"
            )

        valid_res = residuals[~np.isnan(residuals)]
        print(
            f"Valid elements: {len(valid_res):,} / {residuals.size:,} ({len(valid_res) / residuals.size:.2%})"
        )
        print(f"Residual std: {valid_res.std():.4f}, mean: {valid_res.mean():.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Dataset QA and Sanity Audit")
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to raw parquet dataset",
    )
    parser.add_argument(
        "--residuals-dir",
        type=Path,
        default=REPO_ROOT / "data" / "full",
        help="Path to data/full/",
    )
    args = parser.parse_args()

    audit_raw_panel(args.parquet)
    audit_processed_residuals(args.residuals_dir)


if __name__ == "__main__":
    main()
