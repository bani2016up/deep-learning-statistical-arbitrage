"""The paper authors' own out-of-sample residuals (CRSP, 1998-2016) in the full-data format.

Guijarro-Ordonez, Pelger & Zanotti publish their residual panels in ``gregzanotti/dlsa-public``
(``residuals/{famafrench,pca,ipca_normalized}/*.npy.gz``): 4,781 NYSE days 1998-01-02 →
2016-12-30 × 9,483 CRSP assets, 0 = outside the cap-filtered universe that day. This script
downloads the requested panels (about 40 MB each) and writes them as
``residuals_official_{family}{K}.npz`` next to the WIKI panels (``experiments.data_full``), so
any suite runs on them with ``factor_model="official_{family}"``.

    eligible at t   observed on all 30 days before t: the rule of the official preprocess.py.
                    No dollar-volume universe or extreme-return filter: the authors' 0.01%
                    market-cap filter is already in the data.
    dates           the 4,781 days of our WIKI panel over the same span (identical calendar).

The authors do not publish the loadings Φ, so books trade residuals and costs are measured on
residual weights, as on our own data.

    uv run python -m experiments.official --models ipca5 pca5 ff5
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from experiments.data_full import FULL_DIR, WINDOW, load_full, residual_path

SOURCE = "https://raw.githubusercontent.com/gregzanotti/dlsa-public/main/residuals/"
FILES = {
    "ff": "famafrench/DailyFamaFrench_OOSresiduals_{k}_factors_1998_initialOOSYear_"
    "60_rollingWindow_0.01_Cap.npy.gz",
    "pca": "pca/AvPCA_OOSresiduals_{k}_factors_1998_initialOOSYear_60_rollingWindow_"
    "252_covWindow_0.01_Cap.npy.gz",
    "ipca": "ipca_normalized/IPCA_DailyOOSresiduals_{k}_factors_420_initialMonths_240_"
    "window_12_reestimationFreq_0.01_cap.npy.gz",
}
FACTORS = {
    "ff": (0, 1, 2, 3, 4, 5, 8),
    "pca": (0, 1, 3, 5, 8, 10, 15),
    "ipca": (0, 1, 3, 5, 8, 10, 15),
}
START, END, DAYS = np.datetime64("1998-01-02"), np.datetime64("2016-12-30"), 4781
RAW_DIR = FULL_DIR / "raw/official"


def parse_model(model: str) -> tuple[str, int]:
    """``ipca5`` → ("ipca", 5)."""
    family = model.rstrip("0123456789")
    if family not in FILES or family == model:
        raise ValueError(f"model must be <family><K> with family in {sorted(FILES)}: {model}")
    k = int(model[len(family) :])
    if k not in FACTORS[family]:
        raise ValueError(f"{family} residuals exist only for K in {FACTORS[family]}")
    return family, k


def download(url: str, target: Path, attempts: int = 5, timeout: float = 60.0) -> Path:
    """Fetch ``url`` to ``target`` once (cached), retrying with exponential backoff."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(attempts):
        try:
            with (
                urllib.request.urlopen(url, timeout=timeout) as response,
                open(partial, "wb") as out,
            ):
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            partial.rename(target)
            return target
        except OSError as exc:  # URLError, HTTPError and socket timeouts are all OSErrors
            if attempt == attempts - 1:
                raise
            delay = 5.0 * 2**attempt
            print(f"download failed ({exc}); retrying in {delay:.0f}s")
            time.sleep(delay)
    raise AssertionError("unreachable")


def trading_days() -> np.ndarray:
    """The official panels' 4,781 rows: NYSE days 1998-01-02 … 2016-12-30, from our panel."""
    dates = load_full("ff5").dates
    dates = dates[(dates >= START) & (dates <= END)]
    if len(dates) != DAYS:
        raise ValueError(f"expected {DAYS} trading days in {START}…{END}, found {len(dates)}")
    return dates


def convert(raw: np.ndarray, lookback: int = WINDOW) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0-coded panel → (residuals with NaN = missing, eligible, kept columns)."""
    observed = raw != 0
    cumulative = np.vstack(
        [np.zeros((1, raw.shape[1]), np.int32), np.cumsum(observed, 0, np.int32)]
    )
    t = np.arange(lookback, len(raw))
    eligible = np.zeros(raw.shape, dtype=bool)
    eligible[lookback:] = cumulative[t] - cumulative[t - lookback] == lookback
    columns = np.flatnonzero(eligible.any(axis=0))
    residuals = np.where(observed, raw, np.nan).astype(np.float32)
    return residuals[:, columns], eligible[:, columns], columns


def build(model: str, dates: np.ndarray) -> None:
    family, k = parse_model(model)
    name = FILES[family].format(k=k)
    path = download(SOURCE + name, RAW_DIR / Path(name).name)
    with gzip.open(path) as handle:
        raw = np.load(handle).astype(np.float32)
    if raw.shape[0] != len(dates):
        raise ValueError(f"{name}: {raw.shape[0]} rows, expected {len(dates)}")
    residuals, eligible, columns = convert(raw)
    out = f"official_{model}"
    np.savez_compressed(
        residual_path(out),
        residuals=residuals,
        eligible=eligible,
        dates=dates.astype(str),
        tickers=np.array([f"asset_{c}" for c in columns]),
        metadata=json.dumps(
            {"model": out, "source": f"gregzanotti/dlsa-public residuals/{name}", "window": WINDOW}
        ),
    )
    counts = eligible.sum(axis=1)
    first = int(np.argmax(counts > 0))
    print(
        f"{out}: first tradeable {dates[first]}, eligible/day median "
        f"{int(np.median(counts[first:]))} (min {counts[first:].min()}), columns {len(columns)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["ipca5", "pca5", "ff5"])
    args = parser.parse_args()
    dates = trading_days()
    for model in args.models:
        build(model, dates)


if __name__ == "__main__":
    main()
