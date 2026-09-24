# Statistical Run Analysis

> Engineering diagnostics only. This report is not evidence of a scientific
> replication or a guarantee of future returns.

## Assessment

**insufficient statistical evidence** (29% of available checks passed)

| Check                            | Passed |
| -------------------------------- | -----: |
| positive test sharpe             |    yes |
| positive mean hac 5pct           |     no |
| bootstrap sharpe ci above zero   |     no |
| probabilistic sharpe above 95pct |     no |
| majority positive years          |    yes |
| positive sharpe after 10bps      |     no |
| beats all benchmarks hac 5pct    |     no |

## Performance

| Metric                         |      Value |
| ------------------------------ | ---------: |
| observations                   |       1483 |
| start_date                     | 2020-02-07 |
| end_date                       | 2025-12-31 |
| total_return                   |     0.1418 |
| cagr                           |     0.0228 |
| annualized_mean                |     0.0236 |
| annualized_volatility          |     0.0463 |
| annualized_sharpe              |     0.5099 |
| annualized_sortino             |     0.7262 |
| calmar_ratio                   |     0.3229 |
| maximum_drawdown               |    -0.0706 |
| maximum_drawdown_duration_days |        326 |
| skewness                       |    -0.2946 |
| excess_kurtosis                |     3.6523 |
| historical_var_95              |     0.0045 |
| historical_cvar_95             |     0.0066 |
| positive_day_fraction          |     0.5152 |
| positive_month_fraction        |     0.5211 |
| positive_year_fraction         |     1.0000 |
| profit_factor                  |     1.0919 |

## Statistical Inference

| Metric                       |                                      Value |
| ---------------------------- | -----------------------------------------: |
| Newey-West mean t-statistic  |                                     1.2570 |
| Newey-West one-sided p-value |                                     0.1044 |
| Bootstrap Sharpe 95% CI      | [-0.20968272843295022, 1.2465983451406681] |
| Probabilistic Sharpe Ratio   |                                     0.8906 |
| Deflated Sharpe probability  |                                     0.2228 |
| Minimum track record days    |                                  2651.5299 |
| Ljung-Box p-value            |                                     0.4205 |

## Transaction Costs

Break-even cost: 3.4302 bps

| Cost (bps) | Annual mean |  Sharpe | Total return |
| ---------: | ----------: | ------: | -----------: |
|          0 |      0.0236 |  0.5099 |       0.1418 |
|          1 |      0.0167 |  0.3613 |       0.0965 |
|          5 |     -0.0108 | -0.2334 |      -0.0675 |
|         10 |     -0.0452 | -0.9767 |      -0.2384 |
|         25 |     -0.1484 | -3.1989 |      -0.5852 |
|         50 |     -0.3204 | -6.8367 |      -0.8494 |

## Benchmark Comparisons

| Benchmark        | Sharpe difference | Annual mean difference | HAC p-value |
| ---------------- | ----------------: | ---------------------: | ----------: |
| reversal_neutral |            0.4616 |                 0.0205 |      0.2219 |

## Multiple Testing and PBO

PBO: 0.2857

CSCV/PBO is informative only when all tried strategies are included.

## Interpretation Warning

This heuristic is not a scientific acceptance test and must use untouched OOS data.
