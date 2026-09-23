from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPSILON = 1e-12


@dataclass(frozen=True)
class PredictionData:
    returns: np.ndarray
    dates: np.ndarray
    weights: np.ndarray | None = None
    tickers: np.ndarray | None = None

    def validate(self) -> None:
        if self.returns.ndim != 1 or len(self.returns) < 3:
            raise ValueError("returns must be one-dimensional with at least three observations")
        if len(self.dates) != len(self.returns):
            raise ValueError("dates and returns must have equal length")
        if not np.isfinite(self.returns).all():
            raise ValueError("returns contain NaN or infinity")
        if len(np.unique(self.dates)) != len(self.dates):
            raise ValueError("dates must be unique")
        if np.any(self.dates[1:] <= self.dates[:-1]):
            raise ValueError("dates must be strictly increasing")
        if self.weights is not None:
            if self.weights.ndim != 2 or self.weights.shape[0] != len(self.returns):
                raise ValueError("weights must have shape [dates, assets]")
            if not np.isfinite(self.weights).all():
                raise ValueError("weights contain NaN or infinity")
            if self.tickers is not None and self.weights.shape[1] != len(self.tickers):
                raise ValueError("ticker count does not match weight columns")


def load_predictions(path: Path) -> PredictionData:
    with np.load(path, allow_pickle=False) as data:
        result = PredictionData(
            returns=data["returns"].astype(np.float64),
            dates=data["dates"].astype("datetime64[D]"),
            weights=data["weights"].astype(np.float64) if "weights" in data else None,
            tickers=data["tickers"].astype(str) if "tickers" in data else None,
        )
    result.validate()
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if abs(denominator) > EPSILON else float("nan")


def annualized_sharpe(returns: np.ndarray, annualization: int = 252) -> float:
    return _safe_ratio(
        float(np.mean(returns) * annualization), float(np.std(returns) * math.sqrt(annualization))
    )


def drawdown_series(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wealth = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[:-1]
    peaks = np.maximum(peaks, wealth)
    return wealth, wealth / peaks - 1.0


def _max_drawdown_duration(drawdowns: np.ndarray) -> int:
    longest = current = 0
    for value in drawdowns:
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def aggregate_period_returns(returns: np.ndarray, dates: np.ndarray, frequency: str) -> pd.Series:
    series = pd.Series(returns, index=pd.to_datetime(dates))
    return series.resample(frequency).apply(lambda values: (1.0 + values).prod() - 1.0)


def performance_metrics(
    returns: np.ndarray, dates: np.ndarray, annualization: int = 252
) -> dict[str, Any]:
    n = len(returns)
    annual_mean = float(np.mean(returns) * annualization)
    annual_volatility = float(np.std(returns, ddof=0) * math.sqrt(annualization))
    wealth, drawdowns = drawdown_series(returns)
    total_return = float(wealth[-1] - 1.0)
    cagr = float(wealth[-1] ** (annualization / n) - 1.0) if wealth[-1] > 0 else -1.0
    max_drawdown = float(drawdowns.min())
    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * math.sqrt(annualization))
    losses = returns[returns < 0]
    gains = returns[returns > 0]
    monthly = aggregate_period_returns(returns, dates, "ME")
    yearly = aggregate_period_returns(returns, dates, "YE")
    rolling_year = (
        pd.Series(1.0 + returns).rolling(min(annualization, n)).apply(np.prod, raw=True) - 1.0
    )
    var_cutoff = float(np.quantile(returns, 0.05))
    return {
        "observations": n,
        "start_date": str(dates[0]),
        "end_date": str(dates[-1]),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_mean": annual_mean,
        "annualized_volatility": annual_volatility,
        "annualized_sharpe": _safe_ratio(annual_mean, annual_volatility),
        "annualized_sortino": _safe_ratio(annual_mean, downside_deviation),
        "calmar_ratio": _safe_ratio(cagr, abs(max_drawdown)),
        "maximum_drawdown": max_drawdown,
        "maximum_drawdown_duration_days": _max_drawdown_duration(drawdowns),
        "downside_deviation": downside_deviation,
        "skewness": float(stats.skew(returns, bias=False)),
        "excess_kurtosis": float(stats.kurtosis(returns, fisher=True, bias=False)),
        "historical_var_95": -var_cutoff,
        "historical_cvar_95": float(-returns[returns <= var_cutoff].mean()),
        "positive_day_fraction": float(np.mean(returns > 0)),
        "positive_month_fraction": float(np.mean(monthly > 0)),
        "positive_year_fraction": float(np.mean(yearly > 0)),
        "best_day": float(returns.max()),
        "worst_day": float(returns.min()),
        "profit_factor": _safe_ratio(float(gains.sum()), float(-losses.sum())),
        "worst_rolling_252_day_return": (
            float(rolling_year.min()) if rolling_year.notna().any() else float("nan")
        ),
        "monthly_returns": {str(index.date()): float(value) for index, value in monthly.items()},
        "yearly_returns": {str(index.year): float(value) for index, value in yearly.items()},
    }


def default_hac_lags(n_observations: int) -> int:
    return max(1, math.floor(4 * (n_observations / 100) ** (2 / 9)))


def newey_west_mean_test(returns: np.ndarray, lags: int | None = None) -> dict[str, float]:
    n = len(returns)
    lags = min(default_hac_lags(n) if lags is None else lags, n - 1)
    demeaned = returns - returns.mean()
    long_run_variance = float(demeaned @ demeaned / n)
    for lag in range(1, lags + 1):
        covariance = float(demeaned[lag:] @ demeaned[:-lag] / n)
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1)) * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / n)
    t_statistic = _safe_ratio(float(returns.mean()), standard_error)
    return {
        "lags": lags,
        "daily_mean": float(returns.mean()),
        "standard_error": standard_error,
        "t_statistic": t_statistic,
        "two_sided_p_value": float(2 * stats.norm.sf(abs(t_statistic))),
        "one_sided_positive_p_value": float(stats.norm.sf(t_statistic)),
        "confidence_interval_95": [
            float(returns.mean() - 1.96 * standard_error),
            float(returns.mean() + 1.96 * standard_error),
        ],
    }


def autocorrelation_diagnostics(
    returns: np.ndarray, annualization: int = 252, max_lag: int = 10
) -> dict[str, Any]:
    n = len(returns)
    centered = returns - returns.mean()
    variance = float(centered @ centered)
    correlations = []
    for lag in range(1, min(max_lag, n - 1) + 1):
        correlations.append(float(centered[lag:] @ centered[:-lag] / variance))
    q_statistic = (
        n * (n + 2) * sum(rho**2 / (n - lag) for lag, rho in enumerate(correlations, start=1))
    )
    adjustment = max(1.0 + 2.0 * sum(correlations), EPSILON)
    return {
        "autocorrelations": {str(lag): value for lag, value in enumerate(correlations, start=1)},
        "ljung_box_lags": len(correlations),
        "ljung_box_q_statistic": float(q_statistic),
        "ljung_box_p_value": float(stats.chi2.sf(q_statistic, len(correlations))),
        "autocorrelation_adjusted_sharpe": annualized_sharpe(returns, annualization)
        / math.sqrt(adjustment),
    }


def circular_block_bootstrap(
    returns: np.ndarray,
    samples: int = 5000,
    block_size: int = 20,
    seed: int = 42,
    annualization: int = 252,
) -> dict[str, Any]:
    if samples < 100:
        raise ValueError("bootstrap samples must be at least 100")
    n = len(returns)
    block_size = min(max(1, block_size), n)
    rng = np.random.default_rng(seed)
    block_count = math.ceil(n / block_size)
    starts = rng.integers(0, n, size=(samples, block_count))
    offsets = np.arange(block_size)
    indices = ((starts[..., None] + offsets) % n).reshape(samples, -1)[:, :n]
    draws = returns[indices]
    means = draws.mean(axis=1)
    volatilities = draws.std(axis=1)
    sharpes = means / np.maximum(volatilities, EPSILON) * math.sqrt(annualization)
    return {
        "samples": samples,
        "block_size": block_size,
        "mean_confidence_interval_95": np.quantile(means, [0.025, 0.975]).tolist(),
        "sharpe_confidence_interval_95": np.quantile(sharpes, [0.025, 0.975]).tolist(),
        "probability_mean_nonpositive": float(np.mean(means <= 0)),
        "probability_sharpe_nonpositive": float(np.mean(sharpes <= 0)),
        "sharpe_samples": sharpes,
    }


def probabilistic_sharpe_ratio(
    returns: np.ndarray, benchmark_annual_sharpe: float = 0.0, annualization: int = 252
) -> float:
    n = len(returns)
    observed = float(returns.mean() / max(returns.std(ddof=0), EPSILON))
    benchmark = benchmark_annual_sharpe / math.sqrt(annualization)
    skewness = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))
    denominator = math.sqrt(
        max(1.0 - skewness * observed + (kurtosis - 1.0) * observed**2 / 4.0, EPSILON)
    )
    z_score = (observed - benchmark) * math.sqrt(n - 1) / denominator
    return float(stats.norm.cdf(z_score))


def deflated_sharpe_ratio(
    returns: np.ndarray, trial_annual_sharpes: list[float], annualization: int = 252
) -> dict[str, float]:
    trials = np.asarray(trial_annual_sharpes, dtype=float)
    if len(trials) <= 1 or np.std(trials, ddof=1) <= EPSILON:
        expected_maximum = 0.0
    else:
        euler_gamma = 0.5772156649015329
        trial_std = float(np.std(trials, ddof=1))
        expected_maximum = trial_std * (
            (1 - euler_gamma) * stats.norm.ppf(1 - 1 / len(trials))
            + euler_gamma * stats.norm.ppf(1 - 1 / (len(trials) * math.e))
        )
    return {
        "number_of_trials": len(trials),
        "expected_maximum_sharpe_under_selection": float(expected_maximum),
        "deflated_sharpe_probability": probabilistic_sharpe_ratio(
            returns, expected_maximum, annualization
        ),
    }


def minimum_track_record_length(
    returns: np.ndarray,
    benchmark_annual_sharpe: float = 0.0,
    confidence: float = 0.95,
    annualization: int = 252,
) -> float:
    observed = float(returns.mean() / max(returns.std(ddof=0), EPSILON))
    benchmark = benchmark_annual_sharpe / math.sqrt(annualization)
    difference = observed - benchmark
    if difference <= 0:
        return float("inf")
    skewness = float(stats.skew(returns, bias=False))
    kurtosis = float(stats.kurtosis(returns, fisher=False, bias=False))
    correction = max(1.0 - skewness * observed + (kurtosis - 1.0) * observed**2 / 4.0, EPSILON)
    return float(1.0 + correction * (stats.norm.ppf(confidence) / difference) ** 2)


def weight_diagnostics(weights: np.ndarray, tickers: np.ndarray | None = None) -> dict[str, Any]:
    turnover = np.concatenate(
        ([np.abs(weights[0]).sum()], np.abs(np.diff(weights, axis=0)).sum(axis=1))
    )
    gross = np.abs(weights).sum(axis=1)
    hhi = (weights**2).sum(axis=1) / np.maximum(gross**2, EPSILON)
    average_abs = np.abs(weights).mean(axis=0)
    top_indices = np.argsort(average_abs)[::-1][: min(10, weights.shape[1])]
    labels = tickers if tickers is not None else np.array([str(i) for i in range(weights.shape[1])])
    return {
        "average_turnover_including_entry": float(turnover.mean()),
        "median_turnover_including_entry": float(np.median(turnover)),
        "average_rebalance_turnover_excluding_entry": float(turnover[1:].mean()),
        "average_gross_exposure": float(gross.mean()),
        "maximum_gross_exposure": float(gross.max()),
        "average_net_exposure": float(weights.sum(axis=1).mean()),
        "average_long_exposure": float(np.maximum(weights, 0).sum(axis=1).mean()),
        "average_short_exposure": float(-np.minimum(weights, 0).sum(axis=1).mean()),
        "maximum_absolute_weight": float(np.abs(weights).max()),
        "average_hhi": float(hhi.mean()),
        "average_effective_positions": float(np.mean(1.0 / np.maximum(hhi, EPSILON))),
        "weight_changes": int(np.count_nonzero(np.abs(np.diff(weights, axis=0)) > 1e-8)),
        "top_average_absolute_weights": {
            str(labels[index]): float(average_abs[index]) for index in top_indices
        },
        "turnover": turnover,
    }


def transaction_cost_sensitivity(
    returns: np.ndarray,
    weights: np.ndarray,
    costs_bps: list[float],
    annualization: int = 252,
) -> dict[str, Any]:
    turnover = np.concatenate(
        ([np.abs(weights[0]).sum()], np.abs(np.diff(weights, axis=0)).sum(axis=1))
    )
    results: dict[str, Any] = {}
    for cost in costs_bps:
        net_returns = returns - turnover * cost / 10_000.0
        results[f"{float(cost):g}"] = {
            "annualized_mean": float(net_returns.mean() * annualization),
            "annualized_volatility": float(net_returns.std() * math.sqrt(annualization)),
            "annualized_sharpe": annualized_sharpe(net_returns, annualization),
            "total_return": float(np.prod(1.0 + net_returns) - 1.0),
        }
    break_even_bps = _safe_ratio(float(returns.mean() * 10_000), float(turnover.mean()))
    return {"break_even_cost_bps": break_even_bps, "scenarios": results}


def rolling_metrics(
    returns: np.ndarray, dates: np.ndarray, windows: tuple[int, ...] = (63, 126, 252)
) -> dict[str, pd.Series]:
    series = pd.Series(returns, index=pd.to_datetime(dates))
    output: dict[str, pd.Series] = {}
    for window in windows:
        if len(series) >= window:
            output[str(window)] = (
                series.rolling(window).mean() / series.rolling(window).std(ddof=0) * math.sqrt(252)
            ).dropna()
    return output


def cusum_stability(returns: np.ndarray) -> dict[str, Any]:
    centered = returns - returns.mean()
    scale = max(returns.std(ddof=0) * math.sqrt(len(returns)), EPSILON)
    path = np.cumsum(centered) / scale
    index = int(np.argmax(np.abs(path)))
    return {
        "maximum_absolute_cusum": float(np.abs(path[index])),
        "maximum_location_index": index,
        "maximum_location_fraction": float(index / len(returns)),
    }


def _newey_west_regression(
    y: np.ndarray, x: np.ndarray, lags: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n = len(y)
    lags = min(default_hac_lags(n) if lags is None else lags, n - 1)
    design = np.column_stack([np.ones(n), x])
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ y
    residuals = y - design @ coefficients
    scores = design * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1.0 - lag / (lags + 1)) * (cross + cross.T)
    covariance = inverse @ meat @ inverse
    return coefficients, np.sqrt(np.maximum(np.diag(covariance), 0.0))


def factor_regression(
    data: PredictionData, factors: pd.DataFrame, annualization: int = 252
) -> dict[str, Any]:
    strategy = pd.Series(data.returns, index=pd.to_datetime(data.dates), name="strategy")
    aligned = factors.join(strategy, how="inner").dropna()
    if len(aligned) < max(30, factors.shape[1] + 5):
        raise ValueError("Too few aligned factor observations")
    names = list(factors.columns)
    coefficients, errors = _newey_west_regression(
        aligned["strategy"].to_numpy(), aligned[names].to_numpy()
    )
    t_statistics = coefficients / np.maximum(errors, EPSILON)
    result: dict[str, Any] = {
        "observations": len(aligned),
        "annualized_alpha": float(coefficients[0] * annualization),
        "alpha_hac_t_statistic": float(t_statistics[0]),
        "alpha_two_sided_p_value": float(2 * stats.norm.sf(abs(t_statistics[0]))),
        "factor_loadings": {},
    }
    for index, name in enumerate(names, start=1):
        result["factor_loadings"][name] = {
            "coefficient": float(coefficients[index]),
            "hac_t_statistic": float(t_statistics[index]),
            "two_sided_p_value": float(2 * stats.norm.sf(abs(t_statistics[index]))),
        }
    return result


def align_prediction_returns(
    datasets: dict[str, PredictionData],
) -> tuple[np.ndarray, list[str], np.ndarray]:
    frames = [
        pd.Series(item.returns, index=pd.to_datetime(item.dates), name=name)
        for name, item in datasets.items()
    ]
    aligned = pd.concat(frames, axis=1, join="inner").dropna()
    if len(aligned) < 3:
        raise ValueError("Strategies have fewer than three common dates")
    return aligned.to_numpy(), list(aligned.columns), aligned.index.to_numpy(dtype="datetime64[D]")


def compare_strategies(
    primary: PredictionData,
    benchmarks: dict[str, PredictionData],
    bootstrap_samples: int,
    block_size: int,
    seed: int,
    annualization: int = 252,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for offset, (name, benchmark) in enumerate(benchmarks.items()):
        matrix, names, _ = align_prediction_returns({"primary": primary, name: benchmark})
        difference = matrix[:, 0] - matrix[:, 1]
        inference = newey_west_mean_test(difference)
        bootstrap = circular_block_bootstrap(
            difference, bootstrap_samples, block_size, seed + offset, annualization
        )
        output[name] = {
            "observations": len(difference),
            "primary_sharpe": annualized_sharpe(matrix[:, 0], annualization),
            "benchmark_sharpe": annualized_sharpe(matrix[:, 1], annualization),
            "annualized_mean_difference": float(difference.mean() * annualization),
            "information_ratio": annualized_sharpe(difference, annualization),
            "hac_mean_difference": inference,
            "bootstrap_difference": {
                key: value for key, value in bootstrap.items() if key != "sharpe_samples"
            },
            "aligned_names": names,
        }
    return output


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    names = list(p_values)
    values = np.array([p_values[name] for name in names])
    order = np.argsort(values)
    adjusted_sorted = np.minimum.accumulate(
        (values[order] * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return {name: float(adjusted[index]) for index, name in enumerate(names)}


def probability_of_backtest_overfitting(
    strategy_returns: np.ndarray,
    strategy_names: list[str],
    blocks: int = 8,
    annualization: int = 252,
) -> dict[str, Any]:
    observations, strategies = strategy_returns.shape
    if strategies < 2:
        raise ValueError("PBO requires at least two strategies")
    blocks = min(blocks, observations // 5)
    if blocks < 4:
        raise ValueError("PBO requires enough observations for at least four blocks")
    if blocks % 2:
        blocks -= 1
    partitions = np.array_split(np.arange(observations), blocks)
    logits: list[float] = []
    degradations: list[float] = []
    selected: dict[str, int] = {name: 0 for name in strategy_names}
    for train_blocks in itertools.combinations(range(blocks), blocks // 2):
        train_set = set(train_blocks)
        train_indices = np.concatenate([partitions[i] for i in train_blocks])
        test_indices = np.concatenate([partitions[i] for i in range(blocks) if i not in train_set])
        train_sharpes = np.array(
            [
                annualized_sharpe(strategy_returns[train_indices, column], annualization)
                for column in range(strategies)
            ]
        )
        test_sharpes = np.array(
            [
                annualized_sharpe(strategy_returns[test_indices, column], annualization)
                for column in range(strategies)
            ]
        )
        winner = int(np.nanargmax(train_sharpes))
        selected[strategy_names[winner]] += 1
        rank = stats.rankdata(test_sharpes, method="average")[winner]
        relative_rank = float((rank - 0.5) / strategies)
        logits.append(math.log(relative_rank / (1.0 - relative_rank)))
        degradations.append(float(train_sharpes[winner] - test_sharpes[winner]))
    return {
        "blocks": blocks,
        "combinations": len(logits),
        "probability_of_backtest_overfitting": float(np.mean(np.asarray(logits) <= 0)),
        "median_rank_logit": float(np.median(logits)),
        "average_sharpe_degradation": float(np.mean(degradations)),
        "in_sample_winner_counts": selected,
        "warning": "CSCV/PBO is informative only when all tried strategies are included.",
    }


def success_assessment(report: dict[str, Any]) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "positive_test_sharpe": report["performance"]["annualized_sharpe"] > 0,
        "positive_mean_hac_5pct": report["inference"]["newey_west_mean"][
            "one_sided_positive_p_value"
        ]
        < 0.05,
        "bootstrap_sharpe_ci_above_zero": report["inference"]["block_bootstrap"][
            "sharpe_confidence_interval_95"
        ][0]
        > 0,
        "probabilistic_sharpe_above_95pct": report["inference"]["probabilistic_sharpe_ratio"]
        > 0.95,
        "majority_positive_years": report["performance"]["positive_year_fraction"] >= 0.6,
    }
    cost_sensitivity = report.get("transaction_cost_sensitivity")
    if cost_sensitivity and "10" in cost_sensitivity["scenarios"]:
        checks["positive_sharpe_after_10bps"] = (
            cost_sensitivity["scenarios"]["10"]["annualized_sharpe"] > 0
        )
    factor = report.get("factor_regression")
    if factor:
        checks["positive_significant_factor_alpha"] = (
            factor["annualized_alpha"] > 0 and factor["alpha_two_sided_p_value"] < 0.05
        )
    comparisons = report.get("comparisons", {})
    if comparisons:
        checks["beats_all_benchmarks_hac_5pct"] = all(
            item["annualized_mean_difference"] > 0
            and item["hac_mean_difference"]["one_sided_positive_p_value"] < 0.05
            for item in comparisons.values()
        )
    score = float(np.mean(list(checks.values())))
    if score >= 0.8:
        label = "strong engineering evidence"
    elif score >= 0.6:
        label = "promising but not conclusive"
    else:
        label = "insufficient statistical evidence"
    return {
        "label": label,
        "passed_fraction": score,
        "checks": checks,
        "warning": "This heuristic is not a scientific acceptance test and must use untouched OOS data.",
    }


def analyze_predictions(
    primary: PredictionData,
    benchmarks: dict[str, PredictionData] | None = None,
    factors: pd.DataFrame | None = None,
    costs_bps: list[float] | None = None,
    bootstrap_samples: int = 5000,
    block_size: int = 20,
    seed: int = 42,
    annualization: int = 252,
    pbo_blocks: int = 8,
    extra_trial_sharpes: list[float] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    primary.validate()
    benchmarks = benchmarks or {}
    costs_bps = costs_bps or [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]
    bootstrap = circular_block_bootstrap(
        primary.returns, bootstrap_samples, block_size, seed, annualization
    )
    report: dict[str, Any] = {
        "performance": performance_metrics(primary.returns, primary.dates, annualization),
        "inference": {
            "newey_west_mean": newey_west_mean_test(primary.returns),
            "autocorrelation": autocorrelation_diagnostics(primary.returns, annualization),
            "block_bootstrap": {
                key: value for key, value in bootstrap.items() if key != "sharpe_samples"
            },
            "probabilistic_sharpe_ratio": probabilistic_sharpe_ratio(
                primary.returns, 0.0, annualization
            ),
            "minimum_track_record_length_days": minimum_track_record_length(
                primary.returns, 0.0, 0.95, annualization
            ),
        },
        "stability": {
            "cusum": cusum_stability(primary.returns),
            "rolling_sharpe_summary": {},
        },
    }
    rolling = rolling_metrics(primary.returns, primary.dates)
    report["stability"]["rolling_sharpe_summary"] = {
        window: {
            "minimum": float(series.min()),
            "median": float(series.median()),
            "maximum": float(series.max()),
            "last": float(series.iloc[-1]),
        }
        for window, series in rolling.items()
    }
    if primary.weights is not None:
        weights = weight_diagnostics(primary.weights, primary.tickers)
        report["portfolio"] = {key: value for key, value in weights.items() if key != "turnover"}
        report["transaction_cost_sensitivity"] = transaction_cost_sensitivity(
            primary.returns, primary.weights, costs_bps, annualization
        )
    if factors is not None:
        report["factor_regression"] = factor_regression(primary, factors, annualization)
    if benchmarks:
        report["comparisons"] = compare_strategies(
            primary,
            benchmarks,
            bootstrap_samples,
            block_size,
            seed,
            annualization,
        )
        all_datasets = {"primary": primary, **benchmarks}
        matrix, names, _ = align_prediction_returns(all_datasets)
        strategy_sharpes = [
            annualized_sharpe(matrix[:, i], annualization) for i in range(len(names))
        ]
        p_values = {
            name: newey_west_mean_test(matrix[:, index])["one_sided_positive_p_value"]
            for index, name in enumerate(names)
        }
        report["multiple_testing"] = {
            "strategy_sharpes": dict(zip(names, strategy_sharpes, strict=True)),
            "benjamini_hochberg_adjusted_p_values": benjamini_hochberg(p_values),
            "pbo": probability_of_backtest_overfitting(matrix, names, pbo_blocks, annualization),
        }
    trial_sharpes = [report["performance"]["annualized_sharpe"]]
    if benchmarks:
        trial_sharpes.extend(
            annualized_sharpe(item.returns, annualization) for item in benchmarks.values()
        )
    trial_sharpes.extend(extra_trial_sharpes or [])
    report["inference"]["deflated_sharpe"] = deflated_sharpe_ratio(
        primary.returns, trial_sharpes, annualization
    )
    report["success_assessment"] = success_assessment(report)
    plot_data = {
        "bootstrap_sharpes": bootstrap["sharpe_samples"],
        "rolling_sharpes": rolling,
    }
    return report, plot_data
