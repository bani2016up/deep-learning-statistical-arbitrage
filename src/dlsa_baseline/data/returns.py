from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(prices: pd.DataFrame, min_history_fraction: float = 0.95) -> pd.DataFrame:
    """Compute simple close-to-close returns without filling missing prices."""
    if not 0 < min_history_fraction <= 1:
        raise ValueError("min_history_fraction must be in (0, 1]")
    returns = prices.sort_index().pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    keep = returns.notna().mean() >= min_history_fraction
    returns = returns.loc[:, keep]
    # PCA needs a common cross-section. Dropping dates is explicit; prices are never imputed.
    return returns.dropna(axis=0, how="any").astype("float64")
