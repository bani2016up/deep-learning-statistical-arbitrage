from __future__ import annotations

import json
from pathlib import Path
from urllib.error import URLError

import pandas as pd
import yfinance as yf


def _yahoo_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=False, progress=False, threads=True
    )
    if raw.empty:
        return pd.DataFrame()
    field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
    prices = raw[field]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices.rename_axis(index="date", columns="ticker").sort_index()


def _stooq_price(ticker: str, start: str, end: str) -> pd.Series:
    symbol = ticker.replace("-", ".").lower() + ".us"
    url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start.replace('-', '')}&d2={end.replace('-', '')}&i=d"
    try:
        frame = pd.read_csv(url, parse_dates=["Date"])
    except (URLError, OSError, ValueError):
        return pd.Series(dtype=float, name=ticker)
    if "Close" not in frame or frame.empty:
        return pd.Series(dtype=float, name=ticker)
    return frame.set_index("Date")["Close"].rename(ticker).sort_index()


def download_adjusted_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_path: Path,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Download adjusted closes, cache long-form Parquet, and return a wide frame."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not force:
        long = pd.read_parquet(cache_path)
        prices = long.pivot(index="date", columns="ticker", values="adjusted_price")
        source = "cache"
    else:
        prices = _yahoo_prices(tickers, start, end)
        missing = [
            ticker for ticker in tickers if ticker not in prices or prices[ticker].isna().all()
        ]
        fallback = [_stooq_price(ticker, start, end) for ticker in missing]
        if fallback:
            fallback_frame = pd.concat(fallback, axis=1)
            prices = prices.drop(columns=missing, errors="ignore").join(fallback_frame, how="outer")
        prices = prices.reindex(columns=tickers).dropna(axis=1, how="all").sort_index()
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        long = prices.stack(future_stack=True).rename("adjusted_price").reset_index()
        long.columns = ["date", "ticker", "adjusted_price"]
        long.to_parquet(cache_path, index=False)
        source = "yfinance with Stooq fallback"

    successful = [column for column in prices if prices[column].notna().any()]
    missingness = prices[successful].isna().mean().to_dict() if successful else {}
    report: dict[str, object] = {
        "source": source,
        "start": str(prices.index.min().date()) if not prices.empty else None,
        "end": str(prices.index.max().date()) if not prices.empty else None,
        "tickers_requested": len(tickers),
        "tickers_successful": successful,
        "tickers_failed": sorted(set(tickers) - set(successful)),
        "missing_fraction_by_ticker": missingness,
        "usable_stocks": len(successful),
        "trading_days": len(prices),
    }
    cache_path.with_suffix(".quality.json").write_text(json.dumps(report, indent=2))
    return prices[successful], report
