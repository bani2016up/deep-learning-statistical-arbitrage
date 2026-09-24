# Analysis summary by suite

Sample: 49 US large caps, PCA residuals, rolling 1000/125 protocol, OOS 2020-02 → 2025-12,
3 seeds per trainable variant. "SR" is the annualized gross Sharpe ratio; "net" deducts
5 bp × turnover + 1 bp × short leg; "neutral part" is the Sharpe of the dollar-neutral
component from `experiments/decompose.py`. Plots are in `results/comparison_plots/`. Full
tables are in `results/report_tables.md`, and the interpretation is in
`docs/experiment_report.md`.

## 1. Signal models (`models`)

| model           |                              SR | neutral part | turnover |   net |
| --------------- | ------------------------------: | -----------: | -------: | ----: |
| CNN+Transformer | 0.49 (seeds 0.63 / 0.31 / 0.52) |         0.32 |     0.71 | −2.46 |
| OU+FFN          |                            0.41 |         0.39 |     0.18 | −0.62 |
| reversal        |                            0.04 |         0.05 |     0.24 | −0.64 |
| Fourier+FFN     |                           −0.06 |        −0.15 |     0.18 | −1.38 |
| raw FFN         |                           −0.11 |        −0.18 |     0.04 | −0.82 |
| OU+Threshold    |                           −0.26 |        −0.27 |     1.06 | −1.95 |

None of the results is statistically significant (|HAC t| < 1.3). Plots:
`cumulative_returns.png`, `turnover_vs_sharpe.png`.

## 2. Number of factors (`factors`)

- K=0 (raw returns): SR 0.78, but the neutral part is −0.20, so the return is market drift.
- K=5: SR 0.49, neutral part 0.32. K=8: SR 0.43, neutral part 0.45, net exposure 0.23.
- K=10: SR −0.15 (neutral −0.20). K=15: SR −0.02 (neutral −0.11).

Plot: `table1_pivot.png`.

## 3. Lookback (`lookback`)

| L            |    10 |    20 |   30 |   60 |
| ------------ | ----: | ----: | ---: | ---: |
| SR           |  0.11 | −0.10 | 0.49 | 0.33 |
| neutral part | −0.01 | −0.26 | 0.32 | 0.22 |

Plot: `seed_dispersion.png`.

## 4. Epoch budget (`epochs`)

| epochs |    10 |    30 |   100 |
| ------ | ----: | ----: | ----: |
| SR     | −0.04 |  0.49 |  0.37 |
| net    | −2.54 | −2.47 | −2.94 |

One 30-epoch run takes about 3.8 min on Apple MPS.

## 5. Objective (`objective`)

| objective           |   SR | turnover | net exposure |   net |
| ------------------- | ---: | -------: | -----------: | ----: |
| Sharpe              | 0.49 |     0.71 |         0.76 | −2.46 |
| mean-variance (γ=1) | 0.24 |     1.14 |         0.20 | −2.87 |
| Sharpe net of costs | 0.22 |     0.14 |         0.97 | −0.43 |

With costs in the loss and no neutrality constraint, the model moves to a near-static net
long book (net exposure 0.97). Its neutral part is 0.04. Plots: `turnover_vs_sharpe.png`,
`cost_sensitivity.png`.

## 6. Dollar-neutral book (`neutral`)

The equal-weight residual portfolio alone has SR 0.44 over the OOS window. With scores
demeaned per date:

| model                              |                                          SR |   net |
| ---------------------------------- | ------------------------------------------: | ----: |
| CNN+Transformer                    | 0.43 (seeds 0.30 / 0.39 / 0.61; HAC t 1.11) | −2.91 |
| Fourier+FFN                        |                                        0.11 | −1.94 |
| OU+FFN                             |                                        0.10 | −1.22 |
| reversal                           |                                        0.05 | −0.63 |
| raw FFN                            |                                       −0.10 | −0.90 |
| CNN+Transformer, full cost penalty |                                       −0.04 | −1.08 |

Plot: `results/figures/drift_decomposition.png`.

## 7. Final recipe and selection (`final`, `select`)

Selection span 2020-02 → 2022-12; holdout 2023-01 → 2025-12. Dollar-neutral CNN,
3-seed weight ensemble, causal B-day smoothing:

- cost weight 0.25 keeps the per-seed SR (0.43, seed sd 0.05) at lower turnover (0.86 vs 1.10).
- Smoothing B=3 gives the highest gross SR (0.72–0.73 for cw 0–0.25). B=5 cuts turnover
  2.5× (cw 0.25: 0.86 → 0.35).
- B=20 was best on the selection span but failed in the holdout (gross −0.09).
- cw 0.5, B=5: selection 0.54 / −0.56, holdout 0.48 / −0.46 (gross / net).
- K=8 with the same recipe: 1.19 / +0.09 over the full OOS, but K was chosen after
  seeing results, and the neighbours are lower (K=6: 0.53, K=7: 0.66, K=10: −0.10).

Plots: `selection_heatmaps.png`, `holding_persistence.png`, `results/figures/k_sweep.png`.

## 8. Architecture (`architecture`)

| variant                                   |   SR | neutral part |
| ----------------------------------------- | ---: | -----------: |
| D=8, 4 heads, ff 16, dropout 0.25 (paper) | 0.49 |         0.32 |
| D=8, 2 heads                              | 0.37 |         0.19 |
| D=8, dropout 0.5                          | 0.28 |         0.07 |
| D=16, ff 32, dropout 0.5                  | 0.19 |         0.04 |

## 9. Training protocol (`protocol`)

| protocol                          |    SR | neutral part |
| --------------------------------- | ----: | -----------: |
| rolling, 1,000-day window         |  0.49 |         0.32 |
| rolling, 750                      |  0.42 |         0.30 |
| rolling, 500                      |  0.13 |         0.06 |
| constant (trained once)           |  0.22 |         0.08 |
| fixed 60/20/20 (OOS 2024–25 only) | −0.53 |        −0.44 |

## 10. Resulting recipe

Dollar-neutral CNN+Transformer (paper architecture), L=30, rolling 1000/125, 30 epochs,
Sharpe loss with 0.5 × paper costs, 3-seed ensemble, 5-day smoothing, K tuned in {5, 8}
on the validation span of the full dataset. Details: `docs/final_model.md`.
