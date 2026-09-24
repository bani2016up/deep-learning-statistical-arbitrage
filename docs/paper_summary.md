# Paper summary: Deep Learning Statistical Arbitrage

Guijarro-Ordonez, Pelger, Zanotti (arXiv 2106.04028v2, Sept 2022). Local copy:
`12. Deep Learning Statistical Arbitrage.pdf`.

## 1. Framework: stat-arb as three separable problems

| Step                    | Question                                         | Paper's choice                                                                                       | Conventional alternatives                 |
| ----------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| 1. Arbitrage portfolios | Which long-short portfolios of "similar" assets? | Out-of-sample residuals of a factor model: `eps_t = (I - beta_{t-1} w^F_{t-1}') R_t = Phi_{t-1} R_t` | pairs (distance, cointegration, copulas)  |
| 2. Signal               | What in the residual's recent path predicts it?  | CNN + Transformer on the last L=30 **cumulative** residuals                                          | OU fit (Avellaneda-Lee), FFT coefficients |
| 3. Allocation           | How to trade the signal?                         | FFN/linear head → weights, normalized so `                                                           |                                           | w^R |     | _1 = 1` in **stock** space | thresholds on OU s-score |

Steps 2 and 3 are trained **jointly** end to end on a trading objective (Sharpe or
mean-variance, optionally net of costs), **not** on a prediction loss. This is the key
methodological point: the network never forecasts returns explicitly.

Residuals are tradeable because the factors are traded portfolios, so the residual
weights `w^eps` map to stock weights `w^R = w^eps' Phi_{t-1} / ||w^eps' Phi_{t-1}||_1`.

## 2. Data and protocol

- CRSP daily, ~550 largest US stocks (mcap > 0.01% of total), 1998–2016 residuals; OOS trading
  **2002–2016**; excess returns over the 1M T-bill.
- Factor models, all rolling and out of sample:
  - Fama-French K ∈ {1,3,5,8}: 60-day rolling loadings.
  - PCA K ∈ {1,3,5,8,10,15}: 252-day correlation matrix, 60-day loadings.
  - IPCA K ∈ {1,...,15}: loadings = linear function of 46 characteristics, re-estimated yearly on 240 months.
  - K = 0: raw returns.
- Rolling training: **1,000-day window, retrain every 125 days**, 100 epochs, Adam lr 1e-3,
  batches = consecutive 125-day blocks (all stocks of a day together). No early stopping.
- Architecture: 2-layer causal CNN (D=8 filters, size 2, instance norm, residual connection),
  1 Transformer encoder layer (4 heads, FFN hidden 16, dropout 0.25), linear head on the last
  time step. Hyperparameters chosen on 1998–2001 only; the paper reports they are insensitive
  (validation SR 3.5–4.2 across 16 configs).

## 3. Headline results (OOS 2002–2016, Sharpe objective, no costs)

Table I, annualized Sharpe ratio:

| Model \ residuals |  FF5 | PCA5 |    IPCA5 | K=0 (raw returns) |
| ----------------- | ---: | ---: | -------: | ----------------: |
| CNN+Transformer   | 3.21 | 3.36 | **4.16** |              1.64 |
| Fourier+FFN       | 1.66 | 1.98 |     1.90 |              0.36 |
| OU+Threshold      | 0.38 | 0.73 |     0.97 |             −0.18 |

- PCA5 CNN+Trans: μ = 14.3%, σ = 4.2%. IPCA5: μ = 8.7%, σ = 2.1%.
- Alphas vs FF5 + momentum + ST/LT reversal are ≈ the mean return, and R² is ≈ 0–4%:
  the strategy is **orthogonal to known factors**.
- Mean-variance objective (γ=1): SR drops slightly (2.75 PCA5) but μ rises to ~20%.
- **With costs** (5 bp × turnover + 1 bp × short leg, in training and evaluation): SR
  **~1.0–1.24** (IPCA). Costs remove roughly 70% of the Sharpe.
- Lookback L=60 ≈ L=30. Constant model trained once (4y): SR ~1.9–2.1 on PCA/IPCA5, so
  retraining matters, though a static model still beats the benchmarks.
- Ablations (Table A.IX): FFN on raw windows and OU+FFN are clearly worse than CNN+Trans.
  Their claim: **signal extraction (step 2) is the separating element**; factor model choice
  matters much less once K≥3–5.
- Persistence: Sharpe half-life ≈ 7 trading days; overlapping 1-month holding still SR > 1.
- Sparsity: top 10% of |weights| still gives SR > 2.
- Interpretability: CNN filters look like local trend/reversal kernels; attention heads react
  to recent days in downtrends and earlier days in uptrends ("escalator up, elevator down").
- Simple reversal (buy bottom 20% / sell top 20% of lagged residuals): SR ≤ 0.3.

## 4. What we can and cannot replicate

| Paper ingredient                    | Available to us?                       | Consequence                                                                      |
| ----------------------------------- | -------------------------------------- | -------------------------------------------------------------------------------- |
| CRSP ~550 dynamic universe          | No, 49 current large caps (Yahoo)      | Survivorship bias; far fewer independent residuals (paper: ~550/day).            |
| IPCA + 46 characteristics           | No (Compustat)                         | We test PCA only. The paper's best numbers are IPCA.                             |
| Fama-French factors                 | Possible (Ken French library)          | Not yet included. Good extension.                                                |
| `Phi` mapping to stock weights      | Partial (PCA loadings are computable)  | Baseline trades residuals directly: weights are L1-normalized in residual space. |
| Risk-free subtraction               | Not done                               | Small effect for a market-neutral book.                                          |
| Period 2002–2016                    | Our OOS is 2020-02 → 2025-12           | Different regime (post-2020, higher retail/HFT arbitrage competition).           |
| Rolling 1000/125 protocol           | **Yes, implemented in `experiments/`** | Directly comparable protocol.                                                    |
| OU+Thresh, Fourier+FFN, FFN, OU+FFN | **Yes, implemented**                   | The full signal ablation can be reproduced.                                      |
| Cost model (5bp/1bp)                | **Yes, in objective and metrics**      | Directly comparable.                                                             |

## 5. Hypotheses to test in our setting

1. **H1 (signal matters most):** CNN+Trans > Fourier+FFN > OU+Thresh on the same residuals.
2. **H2 (residuals > returns):** K=0 underperforms K≥3.
3. **H3 (K plateau):** performance is flat for K ≥ 5.
4. **H4 (L robustness):** L=30 ≈ L=60.
5. **H5 (retraining helps):** rolling > constant model.
6. **H6 (costs):** gross profits largely survive 5 bp costs _only if_ costs are in the objective.
7. **H7 (orthogonality):** strategy returns are uncorrelated with market / reversal.

The small universe (49 names vs ~550) is the dominant difference. The paper's gains come from
diversifying across hundreds of weakly correlated residuals, so we should expect **much
lower absolute Sharpe** and should judge the hypotheses by **relative ordering** of models,
with seed dispersion and HAC t-stats.
