from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dlsa_baseline.data.pca_residuals import ResidualDataset
from dlsa_baseline.models.ou import hysteresis_positions, ou_signal, threshold_positions


@dataclass(frozen=True)
class OURule:
    """Trading rule on the OU s-score. ``kind`` is ``"paper"`` (memoryless) or ``"al"``."""

    kind: str = "paper"
    c_thresh: float = 1.25  # entry threshold (s_open for "al")
    c_crit: float = 0.25
    b_max: float = 1.0
    s_close_long: float = 0.50
    s_close_short: float = 0.75

    def __post_init__(self) -> None:
        if self.kind not in {"paper", "al"}:
            raise ValueError("kind must be 'paper' or 'al'")


@dataclass
class OUBacktest:
    returns: np.ndarray  # [dates] residual-space portfolio return on each decision date
    dates: np.ndarray  # [dates]
    turnover: np.ndarray  # sum |w_t - w_{t-1}|
    short: np.ndarray  # sum of |negative weights|
    n_long: np.ndarray
    n_short: np.ndarray
    n_universe: np.ndarray  # assets with a complete lookback window
    holding_days: np.ndarray  # mean age of open positions
    weights: np.ndarray  # [dates, traded assets], only assets that were ever held
    tickers: np.ndarray  # [traded assets]
    metadata: dict[str, object] = field(default_factory=dict)

    def net_returns(self, trade_cost: float = 0.0005, hold_cost: float = 0.0001) -> np.ndarray:
        """Official-code cost model: ``trade_cost * turnover + hold_cost * short exposure``."""
        return self.returns - trade_cost * self.turnover - hold_cost * self.short

    def save_predictions(self, path: Path) -> None:
        """Write the ``returns/dates/weights/tickers`` NPZ read by ``scripts/analyze_run.py``."""
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            returns=self.returns,
            dates=self.dates.astype("datetime64[D]").astype(str),
            weights=self.weights.astype(np.float32),
            tickers=self.tickers.astype(str),
        )


def backtest_ou(
    dataset: ResidualDataset,
    rule: OURule | None = None,
    lookback: int = 30,
    start: str | np.datetime64 | None = None,
    end: str | np.datetime64 | None = None,
) -> OUBacktest:
    """Daily OU backtest: position from residuals ``[t-lookback, t-1]``, return at ``t``.

    Only assets with no zero (= missing) residual inside the window are considered, as in the
    official preprocessing. Weights are normalized to ``sum |w| = 1`` in residual space; with
    no open position the day's return is zero. The official stock-space mapping through
    ``Phi`` is unavailable, see ``docs/OU_BASELINE.md``.
    """
    dataset.validate()
    rule = rule or OURule()
    residuals, dates = dataset.residuals, dataset.dates
    first = (
        lookback
        if start is None
        else max(lookback, int(np.searchsorted(dates, np.datetime64(start, "D"))))
    )
    last = (
        len(dates) if end is None else int(np.searchsorted(dates, np.datetime64(end, "D"), "right"))
    )
    if first >= last:
        raise ValueError("empty backtest period")
    n_dates, n_assets = last - first, residuals.shape[1]
    stats = np.zeros((n_dates, 7))
    weights = np.zeros((n_dates, n_assets), dtype=np.float32)
    positions = np.zeros(n_assets)
    age = np.zeros(n_assets)
    previous_weights = np.zeros(n_assets)
    for row, t in enumerate(range(first, last)):
        window = residuals[t - lookback : t]
        universe = np.flatnonzero(~np.any(window == 0, axis=0))
        signal = ou_signal(window[:, universe])
        if rule.kind == "paper":
            chosen = threshold_positions(signal, rule.c_thresh, rule.c_crit, rule.b_max)
        else:
            chosen = hysteresis_positions(
                signal,
                positions[universe],
                rule.c_thresh,
                rule.s_close_long,
                rule.s_close_short,
                rule.c_crit,
                rule.b_max,
            )
        new_positions = np.zeros(n_assets)
        new_positions[universe] = chosen
        held = new_positions != 0
        age = np.where(held & (new_positions == positions), age + 1, held.astype(float))
        positions = new_positions
        gross = np.abs(positions).sum()
        current = positions / gross if gross > 0 else np.zeros(n_assets)
        stats[row] = (
            current @ residuals[t],
            np.abs(current - previous_weights).sum(),
            -current[current < 0].sum(),
            (positions > 0).sum(),
            (positions < 0).sum(),
            len(universe),
            age[held].mean() if held.any() else 0.0,
        )
        weights[row] = current
        previous_weights = current
    traded = np.flatnonzero(np.any(weights != 0, axis=0))
    return OUBacktest(
        returns=stats[:, 0],
        dates=dates[first:last],
        turnover=stats[:, 1],
        short=stats[:, 2],
        n_long=stats[:, 3],
        n_short=stats[:, 4],
        n_universe=stats[:, 5],
        holding_days=stats[:, 6],
        weights=weights[:, traded],
        tickers=dataset.tickers[traded],
        metadata={"rule": rule.__dict__, "lookback": lookback, **dataset.metadata},
    )


def annualized(returns: np.ndarray) -> dict[str, float]:
    """Annualized mean, volatility (population) and Sharpe, as in paper footnote 12."""
    mean, vol = 252 * float(np.mean(returns)), float(np.sqrt(252) * np.std(returns))
    return {"SR": mean / vol if vol > 0 else float("nan"), "mu": mean, "sigma": vol}
