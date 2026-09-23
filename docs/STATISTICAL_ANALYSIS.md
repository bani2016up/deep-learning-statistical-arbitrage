# Statistical Run Analysis

`scripts/analyze_run.py` evaluates an untouched out-of-sample daily return series. It does
not turn a backtest into scientific evidence by itself; conclusions remain conditional on
data quality, absence of leakage, declared model-selection history, and realistic costs.

## Standard Command

Training writes both the model and aligned reversal prediction artifacts. Analyze them with:

```bash
uv run python scripts/analyze_run.py \
  --predictions outputs/test_predictions.npz \
  --benchmark reversal=outputs/reversal_predictions.npz \
  --cost-bps 0 1 5 10 25 50 \
  --bootstrap-samples 5000 \
  --block-size 20
```

Artifacts are written under `outputs/analysis/`:

- `statistical_report.json`: machine-readable complete results.
- `statistical_report.md`: concise human-readable assessment.
- `equity_and_drawdown.png`.
- `rolling_sharpe.png` for 63, 126, and 252-day windows.
- `monthly_returns.png`.
- `return_distribution.png`.
- `bootstrap_sharpe.png`.
- `turnover_cost_sensitivity.png`.

## Metrics

Performance includes total return, CAGR, annualized arithmetic mean, volatility, Sharpe,
Sortino, Calmar, maximum drawdown and duration, downside deviation, skewness, excess
kurtosis, historical 95% VaR/CVaR, hit rates, profit factor, monthly/yearly returns, and the
worst rolling 252-day return.

Inference includes:

- Newey-West/HAC standard error and t-test for the mean.
- Return autocorrelations, Ljung-Box test, and autocorrelation-adjusted Sharpe.
- Circular block-bootstrap confidence intervals for mean and Sharpe.
- Probabilistic Sharpe Ratio against zero.
- Deflated Sharpe Ratio using all supplied strategy/trial Sharpes.
- Minimum Track Record Length at 95% confidence.
- CUSUM stability diagnostic and rolling Sharpe summaries.

Portfolio diagnostics include turnover, gross/net/long/short exposure, maximum absolute
weight, HHI concentration, effective number of positions, weight changes, and the most
concentrated average positions. Cost scenarios subtract
`turnover * bps / 10,000` daily and report the break-even cost.

With benchmarks, the analyzer aligns common dates and reports paired annual mean difference,
information ratio, HAC inference, paired block-bootstrap inference, Benjamini-Hochberg
adjusted p-values, and CSCV Probability of Backtest Overfitting (PBO). PBO is meaningful only
if every tried strategy/configuration is supplied, not only selected winners. Additional
historical trial Sharpes can be supplied using `--trial-sharpes` for a less optimistic
Deflated Sharpe calculation.

## Factor Alpha

Pass a CSV containing a `date` column and one or more numeric daily factor-return columns in
decimal units, not percentages:

```csv
date,market,smb,hml
2024-01-08,0.0012,-0.0003,0.0004
```

```bash
uv run python scripts/analyze_run.py \
  --factors data/reference/factor_returns.csv
```

The analyzer performs an OLS factor regression and reports annualized alpha, factor
loadings, and Newey-West/HAC t-statistics. It never downloads or silently substitutes a
factor proxy.

## Assessment Label

The summary label is deliberately conservative and based only on available checks:
positive Sharpe, HAC significance, bootstrap Sharpe interval above zero, PSR above 95%,
majority positive years, positive Sharpe after 10 bps, significant factor alpha when
provided, and statistically significant benchmark outperformance when provided.

This label is a triage heuristic, not a formal acceptance rule. It must never be used to tune
the model on the test period. Missing factor data is marked unavailable rather than treated
as a passed check.
