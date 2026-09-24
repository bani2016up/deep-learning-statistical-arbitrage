# Final model recipe (for training on the full dataset)

Status: **recipe fixed from the sample experiments and trained on the full dataset**
(`full_recipe`). Final full-data results against the paper replication and the paper with
costs: `docs/full_report.md` (on 2006–2016 no arm is profitable after 5 bp + 1 bp costs).
Evidence: `docs/experiment_report.md`, `results/report_tables.md`, `results/selection.csv`.
The sample is 49 Yahoo large caps, OOS 2020-02 → 2025-12, rolling 1000/125.

## TL;DR

> **Dollar-neutral CNN+Transformer on PCA residuals, trained on Sharpe net of half the
> paper's trading costs, 3-seed ensemble, weights smoothed over the last 5 days.**
> Tune on a validation span _before_ the test period: `n_factors ∈ {5, 8}` (primary),
> then `cost_weight ∈ {0.25, 0.5}` × `smoothing ∈ {3, 5}`.

Two sample results bracket the recipe (same everything except K; full OOS 2020-02 → 2025-12):

|                                 | K=5 (paper default, conservative) | K=8 (best on the sample) |
| ------------------------------- | --------------------------------: | -----------------------: |
| gross SR / HAC t                |                       0.51 / 1.26 |          **1.19 / 2.86** |
| net SR (5 bp + 1 bp)            |                             −0.51 |                **+0.09** |
| selection 2020–22 (gross / net) |                      0.54 / −0.56 |             1.14 / +0.10 |
| holdout 2023–25 (gross / net)   |                      0.48 / −0.46 |             1.25 / +0.07 |
| Deflated SR prob. (all trials)  |                              0.22 |                     0.69 |
| break-even cost (turnover)      |                            3.4 bp |                   7.7 bp |
| max drawdown                    |                             −7.1% |                    −4.0% |

**How much to trust K=8.** It holds in both halves and at every smoothing length (holdout
gross SR 0.53–1.25 for B = 1…20), and all 3 seeds are positive. But K was picked after
seeing full-period results, and it is a **peak rather than a plateau**: K=6 → 0.53, K=7 →
0.66, K=8 → 1.19, K=10 → −0.10 (ensemble, B=5). Treat ~0.5–0.7 as the robust sample level
and K=8's 1.19 as optimistic. That is why K is a tuned parameter, not a fixed one. The
paper, with ~550 names, found PCA K=5 ≥ K=8 (3.36 vs 3.02).

## Recipe

| Component            | Choice                                                                                                                 | Evidence (sample)                                                                                                                                                                                                                                                                                                        |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Arbitrage portfolios | Out-of-sample factor residuals, **K tuned in {5, 8}** (PCA on the sample; IPCA/PCA per available data on the full set) | Raw returns (K=0) look best in total SR (0.78), but that is **pure drift**: neutral part −0.20. Neutral part by K: 1 → 0.26, 3 → −0.35, 5 → 0.32, 8 → 0.45, 10 → −0.20, 15 → −0.11 (`factors`). The final-recipe K sweep (neutral, cw 0.5, ensemble, B=5): 5 → 0.51, 6 → 0.53, 7 → 0.66, 8 → 1.19, 10 → −0.10 (`final`). |
| Signal + allocation  | CNN+Transformer (2×conv k=2, D=8, 1 encoder layer, 4 heads, ff 16, dropout 0.25), linear head                          | Only model with a positive dollar-neutral signal: 0.43 vs ≤ 0.11 for Fourier/OU/FFN/reversal (`neutral`). Same ordering as the paper. The paper default beats 2 heads (0.37), dropout 0.5 (0.28) and D=16 (0.19) (`architecture`).                                                                                       |
| Book                 | **Dollar-neutral**: demean scores per date, then L1-normalize (`neutralize=True`)                                      | Unconstrained models are net long (+0.35 to +0.97) and harvest residual drift (EW residual SR 0.44 in 2020–25). That is not stat-arb and not stable.                                                                                                                                                                     |
| Lookback L           | 30                                                                                                                     | SR (neutral part): L=10 0.11 (−0.01), L=20 −0.10 (−0.26), **L=30 0.49 (0.32)**, L=60 0.33 (0.22) (`lookback`).                                                                                                                                                                                                           |
| Objective            | Sharpe of returns net of `cost_weight × (5 bp·turnover + 1 bp·short)`, **cost_weight 0.5**                             | cw=1 kills the neutral signal (−0.04); cw=0 keeps it but turns over 110%/day. Per seed: cw 0.25 → SR 0.43 at 0.86 turnover, cw 0.5 → 0.24 at 0.52. With the ensemble and B=5 smoothing, cw 0.5 gives SR 0.51 at 0.27 (¼ of cw=0) (`final`).                                                                              |
| Training protocol    | Rolling: 1,000-day window, retrain every 125 days, fresh init per block, 30 epochs, Adam 1e-3, 125-day date batches    | 10 ep −0.04, **30 ep 0.49**, 100 ep 0.37 (`epochs`). Rolling 1000 0.49 > rolling 750 0.42 > constant 0.22 > rolling 500 0.13; the baseline fixed split is −0.53 (`protocol`).                                                                                                                                            |
| Ensembling           | Average the daily weights of **3 seeds**, then re-neutralize and L1-normalize                                          | Removes seed luck (single-seed SR sd ≈ 0.16). The ensemble's gross SR (0.54) beats the seed mean (0.43).                                                                                                                                                                                                                 |
| Execution smoothing  | Hold the mean of the last **B=5** days' target weights (`experiments/postprocess.py:smooth`)                           | B=3–5 raises gross SR (0.54 → 0.58–0.73) _and_ cuts turnover 2–3× in both periods. B ≥ 10 overfits the 2020–22 regime.                                                                                                                                                                                                   |
| Evaluation           | Gross and net (5 bp + 1 bp) SR, HAC t, drift/neutral decomposition, selection vs holdout split                         | `experiments/{metrics,decompose,select}.py`                                                                                                                                                                                                                                                                              |

Exact config (the `final__cnn_neutral_cw0.5` suite plus `postprocess --days 5`; the K=8 variant only changes `n_factors`). Machine-readable: `experiments/final_model.json`.

```json
{
  "model": "cnn_transformer",
  "n_factors": 5,
  "lookback": 30,
  "objective": "sharpe_costs",
  "cost_weight": 0.5,
  "turnover_bps": 5.0,
  "short_bps": 1.0,
  "neutralize": true,
  "protocol": "rolling",
  "train_window": 1000,
  "retrain_every": 125,
  "epochs": 30,
  "batch_dates": 125,
  "learning_rate": 0.001,
  "features": 8,
  "attention_heads": 4,
  "transformer_ff": 16,
  "dropout": 0.25,
  "seeds": [0, 1, 2],
  "ensemble": "mean weights",
  "smoothing_days": 5
}
```

## How the choice was made (and its caveats)

1. Every ingredient was ablated on the full OOS window with 3 seeds (`report_tables.md`).
2. The cost_weight × smoothing grid was chosen on **2020-02 → 2022-12** and checked on
   **2023-01 → 2025-12** (`experiments/select.py`).
3. The mechanical argmax of selection-period net SR (cw 0.5, B=20) **failed in the holdout**
   (net −0.66, gross −0.09): long smoothing fit the 2020–22 regime. We therefore do not fix a
   single B/cw. We fix the region that held up in both periods (B 3–5, cw 0.25–0.5) and
   default to its most consistent point (cw 0.5, B 5).
4. **Caveat:** the holdout has now been seen, so it is no longer untouched. The full
   dataset is the real out-of-sample test. On it, tune the 2×2 grid on the first years only.

## What must change for the full dataset (server)

- **Dynamic universe mask.** The current harness assumes a rectangular panel. The full data
  (like `dlsa-public`: T×N with 0 = missing, ~890 active names per day) needs an
  `[dates, assets]` eligibility mask: a stock is eligible if it has no missing residual in
  its 30-day window. Apply it to the scores before neutralization and L1 normalization, and
  use it to skip ineligible (date, asset) pairs in the forward pass. The single plug-in point
  is `experiments/data.py:load_windows`. `objectives.allocate` / `policy_returns` need a
  `mask` argument.
- **Epochs.** 30 was optimal on 49 names × 1,000 days. With ~18× more sequences per epoch
  the optimum may differ. Re-run the `epochs` suite (10 / 30 / 100) on the first 2–3 retrain
  blocks only.
- **Validation.** Use the paper's scheme: tune on the first years (e.g. the first 1,000 +
  250 days), then freeze. Never select on the reported test span.
- **Compute.** About 30 rolling blocks × 30 epochs × ~890 names. MPS ≈ 1–1.5 h per seed;
  a T4 is likely faster. 3 seeds × 4 grid points ≈ 12 runs, plus 3 benchmark models
  (Fourier+FFN, OU+Thresh, reversal) for the Table I comparison.
- **Checkpointing.** Save per-block weights so long runs can resume after a session limit.
- **Costs.** Keep evaluation at the paper's 5 bp + 1 bp. With `Phi` available, measure
  turnover in stock space (`use_residual_weights`) rather than residual space.

## Reproduce on the sample

```bash
uv run python -m experiments.run_suite final
uv run python -m experiments.postprocess --suite final
uv run python -m experiments.select
# K=5 recipe: results/ensemble__final-cnn_neutral_cw0.5_b5__ens/
# K=8 recipe: results/ensemble__final-cnn_neutral_cw0.5_k8_b5__ens/
```
