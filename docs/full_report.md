# Full-dataset results: paper replication vs our recipe

Data and protocol: `docs/kaggle_plan.md`. Top 500 US stocks by dollar volume, OOS
2003-02-10 → 2016-12-30 (3,499 days). Rolling 1,000-day window, retrained every 125 days,
30 epochs. Trained on a Kaggle T4 (`full_paper`, `full_recipe`, `full_probe`) and locally
(`full_bench`). Numbers: `results/report_tables.md`, `results/paper_vs_recipe.csv`,
`results/selection_full.csv`.

## TL;DR

- **Replication works.** CNN+Transformer on PCA5 residuals gives gross SR 2.84 (2-seed
  ensemble) vs 3.36 in the paper, and FF5 gives 2.12 vs 3.21. The model ranking of Table I
  holds: CNN+Transformer > Fourier+FFN > OU+Threshold.
- **Our recipe vs the paper's variant, test period 2006–2016, same seeds, same smoothing:**
  - _Gross_ SR is the same (Δ −0.06 / −0.07, not significant).
  - _Net_ SR (5 bp × turnover + 1 bp × short) is **better**: +1.38 without smoothing, **+0.50
    with 5-day smoothing, 95% CI [+0.25, +0.80]**.
- **Limits of that claim:**
  - Net Sharpe is **negative for both** (−0.84 ours vs −1.34 paper at B=5).
  - On the validation years 2003–2005 the ranking is reversed.
  - The pre-registered selection rule (choose on 2003–2005) picks the paper variant with
    20-day smoothing.
  - The advantage comes from lower turnover, not a better signal.

## 1. Table I: replication

Gross annualized SR, CNN+Transformer = mean of 2 seeds (2-seed ensemble in brackets),
benchmarks = 1 seed. Unconstrained books, Sharpe loss, no costs in training.

| Model \ residuals |         FF5 |        PCA5 | Paper FF5 | Paper PCA5 |
| ----------------- | ----------: | ----------: | --------: | ---------: |
| CNN+Transformer   | 1.95 [2.12] | 2.66 [2.84] |      3.21 |       3.36 |
| Fourier+FFN       |        0.94 |        1.32 |      1.66 |       1.98 |
| OU+Threshold      |        0.15 |        0.62 |      0.38 |       0.73 |
| Reversal          |        0.68 |        0.83 |         – |          – |

Our absolute level is ~15–35% below the paper. Plausible causes: a different universe
(top 500 by volume from Quandl WIKI vs the paper's CRSP large caps), no IPCA, and turnover
measured on residual-space weights (no `Phi` mapping). The ordering of models and residual
families matches the paper.

Drift decomposition (`results/decomposition.csv`): the paper's unconstrained book is nearly
dollar-neutral on PCA5 (net exposure 0.06, SR of the neutral part 2.66 = total). On FF5 it
is net long 0.15, and 0.40 SR comes from residual drift. On the 49-stock sample it was
the reverse: unconstrained books went net long and harvested drift. That is why neutrality
mattered there and hardly matters on PCA5 here.

## 2. Paper variant vs our recipe

- **Paper** (`full_paper`): CNN+Transformer, Sharpe loss, unconstrained book.
- **Ours** (`full_recipe`, `docs/final_model.md`): same network, dollar-neutral book,
  Sharpe net of `cost_weight × (5 bp + 1 bp)` in the loss.
- Both ensembled over the **common seeds 0 and 1** and smoothed with the same causal B-day
  weight average (`experiments/postprocess.py`).
- ΔSR = ours − paper, with a 95% moving-block bootstrap (21-day blocks) on paired days.
- `uv run python -m experiments.compare --days 1 5 20`.

### PCA5, cost weight 0.25 (main comparison)

|   B | Period     | SR paper | SR ours |   ΔSR gross [95% CI] | Net SR paper | Net SR ours |         ΔSR net [95% CI] | Turnover paper / ours |
| --: | ---------- | -------: | ------: | -------------------: | -----------: | ----------: | -----------------------: | --------------------: |
|   1 | 2003–2016  |     2.84 |    2.74 | −0.10 [−0.29, +0.10] |        −3.63 |       −2.58 | **+1.05 [+0.80, +1.32]** |           1.07 / 0.94 |
|   1 | validation |     4.45 |    4.17 | −0.28 [−0.70, +0.13] |        −2.20 |       −2.65 |     −0.46 [−0.86, −0.02] |           0.93 / 0.99 |
|   1 | **test**   |     2.53 |    2.47 | −0.06 [−0.27, +0.16] |        −3.95 |       −2.58 | **+1.38 [+1.12, +1.69]** |           1.10 / 0.93 |
|   5 | 2003–2016  |     1.93 |    1.83 | −0.11 [−0.34, +0.14] |        −1.06 |       −0.83 |     +0.24 [−0.01, +0.52] |           0.41 / 0.40 |
|   5 | validation |     3.19 |    2.77 | −0.42 [−1.06, +0.23] |        +0.33 |       −0.79 |     −1.12 [−1.71, −0.54] |           0.35 / 0.47 |
|   5 | **test**   |     1.70 |    1.63 | −0.07 [−0.31, +0.20] |        −1.34 |       −0.84 | **+0.50 [+0.25, +0.80]** |           0.43 / 0.38 |
|  20 | test       |     0.92 |    0.80 | −0.12 [−0.48, +0.26] |        −0.88 |       −0.60 |     +0.28 [−0.07, +0.68] |           0.17 / 0.18 |

Validation = 2003-02 → 2005-12, test = 2006-01 → 2016-12. Daily gross returns of the two
variants correlate at 0.90–0.92.

### Other recipe variants (test period, ΔSR ours − paper)

| Recipe      |   B |   ΔSR gross [95% CI] |     ΔSR net [95% CI] |
| ----------- | --: | -------------------: | -------------------: |
| PCA5 cw 0.5 |   1 | −0.53 [−0.98, −0.09] | +1.72 [+1.23, +2.22] |
| PCA5 cw 0.5 |   5 | −0.48 [−0.96, −0.01] | +0.37 [−0.09, +0.88] |
| FF5 cw 0.25 |   1 | −0.25 [−0.63, +0.10] | +0.48 [+0.12, +0.86] |
| FF5 cw 0.25 |   5 | −0.17 [−0.58, +0.21] | −0.09 [−0.50, +0.28] |
| FF5 cw 0.5  |   5 | −0.43 [−1.13, +0.27] | +0.17 [−0.53, +0.85] |

### Reading

1. **The signal is the same.** Gross SR differs by less than 0.1 with cost weight 0.25, and
   the daily returns correlate at 0.9. By year (B=1), ours is ahead in 6 of 14 years.
2. **The gain is turnover.** Penalizing costs in the loss cuts turnover by 12–15% at no gross
   cost (cw 0.25). Cost weight 0.5 cuts it further but significantly lowers gross SR (−0.5).
   On the full data 0.25 beats the sample's choice of 0.5.
3. **Smoothing captures most of it.** With 5-day smoothing on both, the test-period net
   advantage falls from +1.38 to +0.50. It stays significant for PCA5 and disappears for FF5.
4. **Not robust across periods.** In 2003–2005 the paper variant is better net at every B.
   Since the pre-registered protocol chooses on those years, it would have picked the paper
   variant (section 3).
5. **Neither variant is profitable after the paper's costs.** Break-even cost is 2.4–2.9 bp
   per unit turnover without smoothing and ~4 bp with 5-day smoothing, vs 5 bp assumed.
6. **The alpha decays.** Gross SR is ~5 in 2003–2006 and 0.3–0.5 in 2015–2016 for both
   variants (`results/paper_vs_recipe_yearly.csv`). This matches the weak sample results for
   2020–2025.

## 3. Pre-registered selection (validation → test)

`uv run python -m experiments.select --split-date 2006-01-01 --prefix ensemble__full
--output selection_full.csv`: 32 neutral ensembles from `postprocess` (paper: 2 seeds,
recipe: 3 seeds) (|net exposure| < 0.05; the paper PCA5
book qualifies at 0.05). Rank correlation between validation and test: net +0.21 (p = 0.25),
gross +0.82.

| Choice                                | Validation net SR | Test gross SR |         Test net SR |
| ------------------------------------- | ----------------: | ------------: | ------------------: |
| Selected: paper PCA5, B=20            |             +0.97 |          0.92 |               −0.88 |
| Ours, PCA5 cw 0.25, B=10 (main recipe K) | −0.41 | 1.28 | −0.53 |
| Best on test in hindsight: ours, PCA8 cw 0.25, B=20 | −0.62 | 1.39 | +0.05 |

Validation net SR does not predict test net SR (ρ = 0.21). Validation gross SR does
(ρ = 0.82), so the gross signal is stable while the cost/turnover trade-off is not. The
PCA8 line is the best test result, found after seeing it. It is not an out-of-sample
claim, and its validation net SR (−0.62) would not have selected it.

## 4. Conclusion

With the same network, seeds, data and smoothing, the recipe matches the paper's gross
Sharpe and has a higher net Sharpe on 2006–2016 (+0.50 at B=5, significant). The
improvement is lower turnover, not a stronger signal. It does not hold on 2003–2005, and it
does not make the strategy profitable at 5 bp.

## 5. Not done

- **Paper variant with costs in training** (Sharpe net of full costs, unconstrained book).
  The paper reports net SR ~1.0–1.24 (IPCA) for it. It is the missing third arm: it would
  show whether our negative net SR is due to the data or to the implementation.
- **Deflated Sharpe ratio** over the full-data trials (planned in `docs/kaggle_plan.md`).
- `full_probe` (epochs 10 / 30 / 100, first 3 blocks only, 1 seed): SR 3.46 / 4.18 / 5.50.
  More epochs look better on validation, but the main suites used 30 and no 100-epoch full
  run exists.
