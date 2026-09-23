from __future__ import annotations

import argparse
import json

from dlsa_baseline.config import ROOT, SAMPLE_TICKERS, DataConfig
from dlsa_baseline.data.download import download_adjusted_prices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=len(SAMPLE_TICKERS))
    args = parser.parse_args()
    config = DataConfig()
    _, report = download_adjusted_prices(
        list(SAMPLE_TICKERS[: args.limit]),
        config.start,
        config.end,
        ROOT / "data/raw/adjusted_prices.parquet",
        force=args.force,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
