# Baseline Design

## Data Boundary

Data generation and model training communicate only through `ResidualDataset`: a finite
`[date, asset]` float matrix plus explicit dates, tickers, and metadata. Replacing
`pca_residuals.npz` with team residuals therefore does not require changing model code.

Adjusted daily prices come from Yahoo Finance through `yfinance`, with Stooq as a fallback,
and are cached in long-form Parquet. Missing prices are not forward-filled. For today's
rectangular PCA panel, stocks below 95% history coverage are removed and remaining dates with
missing cross-sectional observations are dropped.

## Provisional PCA

For each output date `t`:

1. Fit the correlation eigenspace on returns `[t-252, t-1]` only.
2. Form historical PCA factor returns using the fixed eigenspace.
3. Regress each stock on factors using `[t-60, t-1]` only.
4. Project the contemporaneous date-`t` cross-section into that already-estimated eigenspace.
5. Store date-`t` return minus its factor component.

The date-`t` return is used only to realize date-`t` factor and residual returns, never to fit
the eigenspace or loadings. This is an OOS decomposition, not a prediction of the factor.

## No-Lookahead Contract

For a decision/return date `t`, input is exactly residual returns
`[t-30, ..., t-1]`. Cumulative summation restarts at zero within each window. Tensor shapes:

| Object | Shape |
|---|---|
| Residual panel | `[dates, assets]` |
| Model windows | `[decision_dates, assets, 30]` |
| Flattened model input | `[decision_dates * assets, 30]` |
| CNN output | `[decision_dates * assets, 30, 8]` |
| Transformer output | `[decision_dates * assets, 30, 8]` |
| Scores / normalized weights | `[decision_dates, assets]` |
| Realized portfolio returns | `[decision_dates]` |

The score made from the window ending `t-1` is multiplied by residual return `t`. Tests pin
this convention explicitly.

## Model and Objective

The causal CNN detects local patterns. A small, single Transformer encoder combines those
patterns over the full known window. A linear FFN allocation head maps the final token to one
asset score. There is deliberately no positional encoding, matching the official code.
Scores are normalized each date so `sum(abs(weight)) = 1`.

Training minimizes negative daily Sharpe, `-mean(r)/(std(r)+epsilon)`, over complete date
batches. Evaluation annualizes mean and volatility by 252. Dates are never randomly shuffled:
the first 60% train, next 20% validate model selection, and final 20% test. Raw FFN and
contrarian reversal baselines are included.

## Current Paper Deviations

The paper uses excess returns, a much larger CRSP universe, PCA/FF/IPCA factor models,
stock-space `Phi` normalization, transaction costs in some experiments, and rolling
1,000-day retraining every 125 days. This baseline uses unadjusted-for-risk-free simple
returns, provisional rolling PCA, direct residual trading, no costs, and one chronological
split. It preserves the paper's small CNN-transformer intent but is not a reproduction.
