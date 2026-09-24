# Statistical Run Analysis

> Engineering diagnostics only. This report is not evidence of a scientific
> replication or a guarantee of future returns.

## Assessment

**strong engineering evidence** (86% of available checks passed)

| Check                            | Passed |
| -------------------------------- | -----: |
| positive test sharpe             |    yes |
| positive mean hac 5pct           |    yes |
| bootstrap sharpe ci above zero   |    yes |
| probabilistic sharpe above 95pct |    yes |
| majority positive years          |    yes |
| positive sharpe after 10bps      |     no |
| beats all benchmarks hac 5pct    |    yes |

## Performance

| Metric                         |      Value |
| ------------------------------ | ---------: |
| observations                   |       1483 |
| start_date                     | 2020-02-07 |
| end_date                       | 2025-12-31 |
| total_return                   |     0.3042 |
| cagr                           |     0.0462 |
| annualized_mean                |     0.0459 |
| annualized_volatility          |     0.0384 |
| annualized_sharpe              |     1.1939 |
| annualized_sortino             |     1.8433 |
| calmar_ratio                   |     1.1598 |
| maximum_drawdown               |    -0.0398 |
| maximum_drawdown_duration_days |        146 |
| skewness                       |     0.2857 |
| excess_kurtosis                |     2.9290 |
| historical_var_95              |     0.0035 |
| historical_cvar_95             |     0.0049 |
| positive_day_fraction          |     0.5367 |
| positive_month_fraction        |     0.5915 |
| positive_year_fraction         |     1.0000 |
| profit_factor                  |     1.2243 |

## Statistical Inference

| Metric                       |                                    Value |
| ---------------------------- | ---------------------------------------: |
| Newey-West mean t-statistic  |                                   2.8552 |
| Newey-West one-sided p-value |                                   0.0022 |
| Bootstrap Sharpe 95% CI      | [0.3834139793562837, 2.0123272985867007] |
| Probabilistic Sharpe Ratio   |                                   0.9982 |
| Deflated Sharpe probability  |                                   0.6943 |
| Minimum track record days    |                                 472.3505 |
| Ljung-Box p-value            |                                   0.2425 |

## Transaction Costs

Break-even cost: 7.6558 bps

| Cost (bps) | Annual mean |  Sharpe | Total return |
| ---------: | ----------: | ------: | -----------: |
|          0 |      0.0459 |  1.1939 |       0.3042 |
|          1 |      0.0399 |  1.0381 |       0.2590 |
|          5 |      0.0159 |  0.4145 |       0.0934 |
|         10 |     -0.0140 | -0.3660 |      -0.0833 |
|         25 |     -0.1039 | -2.7058 |      -0.4599 |
|         50 |     -0.2537 | -6.5593 |      -0.7765 |

## Benchmark Comparisons

| Benchmark        | Sharpe difference | Annual mean difference | HAC p-value |
| ---------------- | ----------------: | ---------------------: | ----------: |
| reversal_neutral |            1.1456 |                 0.0427 |      0.0442 |

## Multiple Testing and PBO

PBO: 0.0286

CSCV/PBO is informative only when all tried strategies are included.

## Interpretation Warning

This heuristic is not a scientific acceptance test and must use untouched OOS data.
