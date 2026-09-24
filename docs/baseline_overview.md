# Baseline overview

Source: <https://github.com/bani2016up/deep-learning-statistical-arbitrage-baseline>, the base of
this repository. It is a leak-free engineering baseline, **not** a reproduction.
Its own `docs/LIMITATIONS.md` and `RUN_REPORT.md` are worth reading first.

## Pipeline

```text
Yahoo adjusted closes (50 large caps, 2015-01 → 2025-12; PM failed → 49 usable)
  → simple daily returns (rectangular panel, dates with any NaN dropped)
  → rolling PCA residuals: 252-day standardized covariance, K=5, 60-day OLS loadings, all through t-1
  → windows: cumulative residuals over [t-30, t-1]  → shape [dates, assets, 30]
  → policy (shared across assets): CNNTransformer | RawFFN      → score per asset
  → L1-normalize scores across assets per date (residual space, no Phi mapping)
  → portfolio return_t = Σ w_{t,i} · eps_{t,i}
  → loss = −Sharpe of the batch's daily returns (batch = 125 consecutive dates)
```

## Components

| File                        | What it does                                                                           |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `data/pca_residuals.py`     | `rolling_pca_residuals`, `ResidualDataset` (npz: residuals, dates, tickers, metadata)  |
| `data/windows.py`           | `build_cumulative_windows`: sample t uses residuals t-L..t-1, target is residual t     |
| `models/cnn.py`             | 2 causal conv layers (k=2, 8 ch), instance norm, residual connection                   |
| `models/transformer.py`     | 1 `TransformerEncoderLayer` (4 heads, ff=16, dropout 0.25), no positional encoding     |
| `models/cnn_transformer.py` | CNN → Transformer → linear head on the last time step                                  |
| `models/baselines.py`       | `RawFFN` (30→16→8→4→1), `reversal_scores` (−last cumulative residual)                  |
| `training/trainer.py`       | 60/20/20 chronological split, Adam, best-validation-Sharpe checkpoint                  |
| `analysis/statistics.py`    | Rich OOS stats: HAC t-test, block bootstrap, PSR/DSR, MinTRL, costs, factor alpha, PBO |

## Reported baseline run (their `RUN_REPORT.md`)

Test 2024-01 → 2025-12 (498 days), 10 epochs, seed 42: **SR 0.68**, μ 2.3%, σ 3.3%,
turnover 0.82/day, HAC t = 1.05 (p = 0.15), best **validation** SR = −0.33, break-even cost
1.1 bp. Their own verdict: insufficient statistical evidence.

## Gaps vs the paper, and how `experiments/` closes them

| Gap                                  | Status in `experiments/`                                                                                                              |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Single 60/20/20 split, 498 test days | `protocol=rolling`: 1,000-day window, retrain every 125 days, **1,483 OOS days** (2020-02 → 2025-12). `fixed` is kept for comparison. |
| Only CNN+Trans and RawFFN            | Added **Fourier+FFN, OU+FFN, OU+Threshold**, plus reversal as a static model                                                          |
| Only K=5                             | `n_factors` ∈ {0,1,3,5,8,10,15}; K=0 = raw returns on identical dates                                                                 |
| Sharpe objective only                | `mean_variance` (γ=1) and `sharpe_costs` (paper friction model in the loss)                                                           |
| Costs only ex post                   | Every run also reports `net_*` metrics under the paper cost model                                                                     |
| One seed                             | Every trainable variant runs 3 seeds                                                                                                  |
| 10 epochs, early stopping            | Epoch budget calibrated in the `epochs` suite; rolling uses a fixed budget, as in the paper                                           |

Not addressed (data-limited): CRSP universe, IPCA/characteristics, `Phi` stock-space mapping,
risk-free rate, Fama-French residuals (feasible next step).
