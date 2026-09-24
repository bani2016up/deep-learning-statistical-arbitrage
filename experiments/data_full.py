"""Full dataset (team WIKI/FF5 panel, 1998-2016): universe, eligibility, residuals.

Inputs (``$DLSA_DATA_DIR``, default ``data/full``):
    raw/ff5_daily_panel.parquet               team panel (excess returns + FF5 factors)
    raw/quandl-wiki-prices-us-equites.zip     raw WIKI prices, only for dollar volume

Everything used at date t is computed from information through t-1, except the date-t
factor realization that defines the date-t residual (as in the paper and the baseline):
    universe   monthly top-N by median dollar volume over the 63 days *before* the month
    ff5        60-day rolling OLS (no intercept) of excess returns on the FF5 factors
    pca{K}     252-day correlation PCA over eligible names, 60-day loadings
    eligible   in universe, residuals observed on all 30 window days, no |return| > 50%
               in the previous 252 days
Output: ``residuals_{model}.npz`` with residuals [T, N] (NaN = unavailable), eligible [T, N],
dates, tickers and metadata. Build once:

    uv run python -m experiments.data_full --models ff5 pca5 pca8
"""

from __future__ import annotations

import argparse
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.data import REPO

FULL_DIR = Path(os.environ.get("DLSA_DATA_DIR", REPO / "data/full"))
PANEL_PATH = FULL_DIR / "raw/ff5_daily_panel.parquet"
WIKI_ZIP = FULL_DIR / "raw/quandl-wiki-prices-us-equites.zip"
FACTORS = ("mkt_rf", "smb", "hml", "rmw", "cma")

TOP_N = 500
DOLLAR_VOLUME_LOOKBACK = 63
WINDOW = 30
EXTREME_RETURN = 0.5
EXTREME_LOOKBACK = 252
FF_WINDOW = 60
COV_WINDOW = 252
LOADING_WINDOW = 60


@dataclass
class Panel:
    returns: np.ndarray  # [T, N] excess returns, NaN = missing
    factors: np.ndarray  # [T, 5] FF5 factor returns
    dates: np.ndarray  # [T] datetime64[D]
    tickers: np.ndarray  # [N]


def load_panel(path: Path = PANEL_PATH) -> Panel:
    frame = pd.read_parquet(path, columns=["date", "ticker", "excess_return", *FACTORS])
    returns = frame.pivot(
        index="date", columns="ticker", values="excess_return"
    ).sort_index()
    factors = frame.groupby("date")[list(FACTORS)].first().loc[returns.index]
    return Panel(
        returns=returns.to_numpy(np.float64),
        factors=factors.to_numpy(np.float64),
        dates=returns.index.to_numpy("datetime64[D]"),
        tickers=returns.columns.to_numpy(str),
    )


def load_dollar_volume(panel: Panel, zip_path: Path = WIKI_ZIP) -> np.ndarray:
    """adj_close × adj_volume aligned to the panel's [dates, tickers]; cached as .npy."""
    cache = FULL_DIR / "dollar_volume.npy"
    if cache.exists():
        values = np.load(cache)
        if values.shape == panel.returns.shape:
            return values
    with (
        zipfile.ZipFile(zip_path) as archive,
        archive.open("WIKI_PRICES.csv") as handle,
    ):
        raw = pd.read_csv(
            handle,
            usecols=["ticker", "date", "adj_close", "adj_volume"],
            dtype={"ticker": str, "adj_close": np.float64, "adj_volume": np.float64},
            parse_dates=["date"],
        )
    raw = raw[
        raw.ticker.isin(set(panel.tickers)) & (raw.date >= pd.Timestamp(panel.dates[0]))
    ]
    raw["dollar_volume"] = raw.adj_close * raw.adj_volume
    wide = raw.pivot_table(
        index="date", columns="ticker", values="dollar_volume", aggfunc="last"
    )
    wide = wide.reindex(index=pd.DatetimeIndex(panel.dates), columns=panel.tickers)
    values = wide.to_numpy(np.float64)
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, values)
    return values


def monthly_universe(
    dollar_volume: np.ndarray,
    dates: np.ndarray,
    top: int = TOP_N,
    lookback: int = DOLLAR_VOLUME_LOOKBACK,
) -> np.ndarray:
    """Top-N names by median dollar volume over the ``lookback`` days before each month."""
    universe = np.zeros(dollar_volume.shape, dtype=bool)
    months = pd.DatetimeIndex(dates).to_period("M")
    starts = np.flatnonzero(np.r_[True, months[1:] != months[:-1]])
    ends = np.r_[starts[1:], len(dates)]
    for start, end in zip(starts, ends):
        if start < lookback:
            continue
        history = dollar_volume[start - lookback : start]
        enough = np.isfinite(history).sum(axis=0) >= 0.8 * lookback
        median = np.where(
            enough, np.nanmedian(np.where(enough, history, 0.0), axis=0), -np.inf
        )
        ranked = np.argsort(-median)[:top]
        ranked = ranked[np.isfinite(median[ranked])]
        universe[start:end, ranked] = True
    return universe


def _recent_extreme(returns: np.ndarray, threshold: float, lookback: int) -> np.ndarray:
    """[T, N]: True if |return| > threshold on any of the ``lookback`` days before t."""
    extreme = (np.abs(np.nan_to_num(returns)) > threshold).astype(np.int32)
    cumulative = np.vstack(
        [np.zeros((1, returns.shape[1]), np.int32), np.cumsum(extreme, axis=0)]
    )
    index = np.arange(returns.shape[0])
    counts = cumulative[index] - cumulative[np.maximum(index - lookback, 0)]
    return counts > 0


def _complete_before(values: np.ndarray, window: int) -> np.ndarray:
    """[T, N]: True if ``values`` is finite on all ``window`` days before t."""
    missing = (~np.isfinite(values)).astype(np.int32)
    cumulative = np.vstack(
        [np.zeros((1, values.shape[1]), np.int32), np.cumsum(missing, axis=0)]
    )
    index = np.arange(values.shape[0])
    counts = cumulative[index] - cumulative[np.maximum(index - window, 0)]
    return (counts == 0) & (index[:, None] >= window)


def ff5_residuals(
    returns: np.ndarray, factors: np.ndarray, window: int = FF_WINDOW
) -> np.ndarray:
    """eps_t = r_t - beta'_{t-1} F_t with beta from OLS (no intercept) on [t-window, t-1]."""
    residuals = np.full(returns.shape, np.nan)
    complete = _complete_before(returns, window)
    for t in range(window, len(returns)):
        names = np.flatnonzero(complete[t] & np.isfinite(returns[t]))
        if names.size == 0:
            continue
        x = factors[t - window : t]
        beta = np.linalg.lstsq(x, returns[t - window : t, names], rcond=None)[
            0
        ]  # [5, n]
        residuals[t, names] = returns[t, names] - factors[t] @ beta
    return residuals


def pca_residuals(
    returns: np.ndarray,
    universe: np.ndarray,
    n_factors: tuple[int, ...] = (5, 8),
    cov_window: int = COV_WINDOW,
    loading_window: int = LOADING_WINDOW,
) -> dict[int, np.ndarray]:
    """Baseline rolling PCA, restricted each day to universe names with a clean history.

    The estimation set at t is: in the universe at t (fixed from pre-month data), complete
    returns on [t-cov_window, t-1], no extreme return in that window, and observed at t.
    One eigendecomposition per day serves every K.
    """
    out = {k: np.full(returns.shape, np.nan) for k in n_factors}
    usable = (
        universe
        & _complete_before(returns, cov_window)
        & ~_recent_extreme(returns, EXTREME_RETURN, cov_window)
        & np.isfinite(returns)
    )
    for t in range(cov_window, len(returns)):
        names = np.flatnonzero(usable[t])
        if names.size <= max(n_factors):
            continue
        history = returns[t - cov_window : t, names]
        scale = history.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        standardized = (history - history.mean(axis=0)) / scale
        eigenvalues, eigenvectors = np.linalg.eigh(
            standardized.T @ standardized / cov_window
        )
        order = np.argsort(eigenvalues)[::-1]
        for k in n_factors:
            components = eigenvectors[:, order[:k]]
            factor_history = (history / scale) @ components
            loadings = np.linalg.lstsq(
                factor_history[-loading_window:], history[-loading_window:], rcond=None
            )[0]
            factor_today = (returns[t, names] / scale) @ components
            out[k][t, names] = returns[t, names] - factor_today @ loadings
    return out


def eligibility(
    residuals: np.ndarray,
    returns: np.ndarray,
    universe: np.ndarray,
    window: int = WINDOW,
) -> np.ndarray:
    """Tradeable at t: in the universe, full residual window before t, no recent extreme."""
    return (
        universe
        & _complete_before(residuals, window)
        & ~_recent_extreme(returns, EXTREME_RETURN, EXTREME_LOOKBACK)
    )


def residual_path(model: str) -> Path:
    return FULL_DIR / f"residuals_{model}.npz"


def save(
    model: str,
    residuals: np.ndarray,
    eligible: np.ndarray,
    panel: Panel,
    metadata: dict,
) -> None:
    columns = np.flatnonzero(
        eligible.any(axis=0)
    )  # drop names that are never tradeable
    np.savez_compressed(
        residual_path(model),
        residuals=residuals[:, columns].astype(np.float32),
        eligible=eligible[:, columns],
        dates=panel.dates.astype(str),
        tickers=panel.tickers[columns],
        metadata=json.dumps(metadata),
    )


def build(models: list[str]) -> None:
    panel = load_panel()
    universe = monthly_universe(load_dollar_volume(panel), panel.dates)
    common = {
        "top_n": TOP_N,
        "dollar_volume_lookback": DOLLAR_VOLUME_LOOKBACK,
        "window": WINDOW,
        "extreme_return": EXTREME_RETURN,
        "extreme_lookback": EXTREME_LOOKBACK,
    }
    built: dict[str, np.ndarray] = {}
    if "ff5" in models:
        built["ff5"] = ff5_residuals(panel.returns, panel.factors)
    pca_ks = tuple(int(m[3:]) for m in models if m.startswith("pca"))
    if pca_ks:
        for k, values in pca_residuals(panel.returns, universe, pca_ks).items():
            built[f"pca{k}"] = values
    # Same first tradeable date for every model, so their OOS windows coincide.
    first = max(
        int(np.argmax(eligibility(v, panel.returns, universe).any(axis=1)))
        for v in built.values()
    )
    for model, residuals in built.items():
        eligible = eligibility(residuals, panel.returns, universe)
        eligible[:first] = False
        save(
            model,
            residuals,
            eligible,
            panel,
            {"model": model, "first_tradeable": str(panel.dates[first]), **common},
        )
        counts = eligible.sum(axis=1)
        print(
            f"{model}: first tradeable {panel.dates[first]}, eligible/day "
            f"median {int(np.median(counts[first:]))} (min {counts[first:].min()}), "
            f"columns {int(eligible.any(axis=0).sum())}"
        )


@dataclass
class FullData:
    residuals: np.ndarray  # [T, N] float32, NaN = unavailable
    eligible: np.ndarray  # [T, N] bool
    dates: np.ndarray
    tickers: np.ndarray
    metadata: dict


def load_full(model: str) -> FullData:
    with np.load(residual_path(model), allow_pickle=False) as data:
        return FullData(
            residuals=data["residuals"],
            eligible=data["eligible"],
            dates=data["dates"].astype("datetime64[D]"),
            tickers=data["tickers"].astype(str),
            metadata=json.loads(str(data["metadata"])),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["ff5", "pca5", "pca8"])
    build(parser.parse_args().models)


if __name__ == "__main__":
    main()
