# Full-dataset results: paper replication, paper with costs, our recipe

Status: **final.** All planned full-data runs are done. This is the concluding report.

Data and protocol: `docs/kaggle_plan.md`. Top 500 US stocks by dollar volume, OOS
2003-02-10 → 2016-12-30 (3,499 days). Rolling 1,000-day window, retrained every 125 days,
30 epochs. Trained on Kaggle T4s (`full_paper`, `full_paper_costs`, `full_recipe`,
`full_probe`) and locally (`full_bench`).

Numbers: `results/report_tables.md`, `results/full_comparison.csv` (paired comparisons),
`results/full_comparison_yearly.csv`, `results/decomposition.csv`,
`results/selection_full.csv`, `results/deflated_sharpe_full.csv`.

Validation = 2003-02 → 2005-12 (pre-registered selection span), test = 2006-01 → 2016-12.
Net = after 5 bp × turnover + 1 bp × short leg, the paper's cost model.

## TL;DR

- **Replication works gross.** CNN+Transformer on PCA5 residuals gives SR 2.84 (2-seed
  ensemble) vs 3.36 in the paper; FF5 gives 2.12 vs 3.21. The model ranking of Table I
  holds: CNN+Transformer > Fourier+FFN > OU+Threshold.
- **Nothing is profitable after costs on the test period.**
  - The best test net SR of each arm is ≈ 0: paper −0.88, paper with costs +0.11, ours
    +0.05.
  - None of these is distinguishable from zero (PSR ≤ 0.64). After deflation for 84 trials,
    the net DSR is 0 for every run.
  - The paper reports net SR ~1.0–1.24 for its with-costs variant.
- **Three arms, test period, PCA5:**
  - _Paper_ (Sharpe loss, unconstrained): keeps the full gross signal but trades too much.
    Net SR −1.34 at B=5.
  - _Paper with costs_ (Sharpe net of full costs, unconstrained): cuts turnover 5×, but
    turns into a **net-long, low-turnover book** (net exposure 0.86–0.91) with **no gross
    signal left** (SR ≈ 0). Net SR −0.81 at B=5.
  - _Ours_ (dollar-neutral, cost weight 0.25): **keeps the gross signal** (1.63 vs 1.70 for
    the paper at B=5) with 12–15% less turnover. Net SR −0.84 at B=5.
- **Our recipe vs each arm:**
  - Vs the paper: better net (+0.50 at B=5, 95% CI [+0.25, +0.80]).
  - Vs paper with costs on PCA5: equal net (−0.03 [−0.98, +0.87]).
  - Vs paper with costs on FF5: worse net (−1.20 [−2.05, −0.39]).
  - It is the only arm that is both cost-aware and keeps the signal. Its gross test SR is
    1.7 higher than paper with costs (PCA5, B=5).
- **Why the paper gets a positive net and we do not.** In 2003–2005, while the gross
  signal is strong (SR 3–5), paper with costs does what the paper describes: validation net
  SR +0.85 … +1.23 at B = 5–20, the paper's level. From 2006 the gross signal of every arm
  falls; the paper's is 0.3–0.5 SR by 2015–2016. The full-cost loss then finds no cheap alpha and
  settles on a drift book. Our universe and residuals carry less and faster-decaying alpha
  than the paper's CRSP/IPCA setup (section 5).

## 1. Table I: replication

Gross annualized SR, CNN+Transformer = mean of 2 seeds (2-seed ensemble in brackets),
benchmarks = 1 seed. Unconstrained books, Sharpe loss, no costs in training.

| Model \ residuals |         FF5 |        PCA5 | Paper FF5 | Paper PCA5 |
| ----------------- | ----------: | ----------: | --------: | ---------: |
| CNN+Transformer   | 1.95 [2.12] | 2.66 [2.84] |      3.21 |       3.36 |
| Fourier+FFN       |        0.94 |        1.32 |      1.66 |       1.98 |
| OU+Threshold      |        0.15 |        0.62 |      0.38 |       0.73 |
| Reversal          |        0.68 |        0.83 |         – |          – |

Our absolute level is ~15–35% below the paper. Plausible causes:

- a different universe (top 500 by volume from Quandl WIKI vs the paper's CRSP large caps);
- no IPCA;
- turnover measured on residual-space weights (no `Phi` mapping).

The ordering of models and residual families matches the paper.

## 2. The three arms

| Arm              | Suite              | Loss                              | Book           | Seeds   |
| ---------------- | ------------------ | --------------------------------- | -------------- | ------- |
| Paper            | `full_paper`       | Sharpe                            | unconstrained  | 0, 1    |
| Paper with costs | `full_paper_costs` | Sharpe net of full 5 bp + 1 bp    | unconstrained  | 0, 1    |
| Ours (recipe)    | `full_recipe`      | Sharpe net of 0.25 or 0.5 × costs | dollar-neutral | 0, 1, 2 |

Same network (CNN+Transformer), data and protocol.

For every pair, both sides are ensembled over their **common seeds** (0 and 1) and smoothed
with the same causal B-day weight average (`experiments/postprocess.py`). ΔSR has a 95%
moving-block bootstrap interval (21-day blocks) on paired days.

Command: `uv run python -m experiments.compare --days 1 5 20`.

### Test period 2006–2016

The net exposure column is the mean sum of weights (0 = dollar-neutral).

| Residuals |   B | Arm              | Gross SR | Net SR | Turnover | Net exposure |
| --------- | --: | ---------------- | -------: | -----: | -------: | -----------: |
| PCA5      |   1 | paper            |     2.53 |  −3.95 |     1.10 |        −0.07 |
|           |     | paper with costs |     0.06 |  −1.36 |     0.18 |         0.86 |
|           |     | ours (cw 0.25)   |     2.47 |  −2.58 |     0.93 |         0.00 |
| PCA5      |   5 | paper            |     1.70 |  −1.34 |     0.43 |        −0.14 |
|           |     | paper with costs |    −0.05 |  −0.81 |     0.08 |         0.87 |
|           |     | ours (cw 0.25)   |     1.63 |  −0.84 |     0.38 |         0.00 |
| PCA5      |  20 | paper            |     0.92 |  −0.88 |     0.17 |        −0.25 |
|           |     | paper with costs |    −0.20 |  −0.59 |     0.04 |         0.91 |
|           |     | ours (cw 0.25)   |     0.80 |  −0.60 |     0.18 |         0.00 |
| FF5       |   1 | paper            |     1.99 |  −3.19 |     1.04 |         0.16 |
|           |     | paper with costs |     0.60 |  −0.27 |     0.21 |         0.62 |
|           |     | ours (cw 0.25)   |     1.74 |  −2.71 |     0.86 |         0.00 |
| FF5       |   5 | paper            |     1.14 |  −1.14 |     0.43 |         0.24 |
|           |     | paper with costs |     0.46 |  −0.03 |     0.10 |         0.63 |
|           |     | ours (cw 0.25)   |     0.97 |  −1.23 |     0.37 |         0.00 |
| FF5       |  20 | paper            |     0.22 |  −0.91 |     0.20 |         0.41 |
|           |     | paper with costs |     0.39 |  +0.11 |     0.05 |         0.65 |
|           |     | ours (cw 0.25)   |     0.04 |  −1.25 |     0.18 |         0.00 |

### Paired differences, test period

| Pair (other − base)      | Residuals |   B |       ΔSR gross [95% CI] |         ΔSR net [95% CI] |
| ------------------------ | --------- | --: | -----------------------: | -----------------------: |
| ours − paper             | PCA5      |   1 |     −0.06 [−0.27, +0.16] | **+1.38 [+1.12, +1.69]** |
|                          | PCA5      |   5 |     −0.07 [−0.31, +0.20] | **+0.50 [+0.25, +0.80]** |
|                          | PCA5      |  20 |     −0.12 [−0.48, +0.26] |     +0.28 [−0.07, +0.68] |
|                          | FF5       |   5 |     −0.17 [−0.58, +0.21] |     −0.09 [−0.50, +0.28] |
| paper with costs − paper | PCA5      |   1 | **−2.47 [−3.37, −1.50]** | **+2.59 [+1.53, +3.75]** |
|                          | PCA5      |   5 | **−1.75 [−2.69, −0.77]** |     +0.53 [−0.43, +1.60] |
|                          | FF5       |   1 | **−1.39 [−2.19, −0.62]** | **+2.92 [+1.99, +3.86]** |
|                          | FF5       |   5 |     −0.68 [−1.46, +0.09] | **+1.11 [+0.30, +1.93]** |
| ours − paper with costs  | PCA5      |   1 | **+2.41 [+1.50, +3.30]** | **−1.21 [−2.27, −0.23]** |
|                          | PCA5      |   5 | **+1.68 [+0.78, +2.58]** |     −0.03 [−0.98, +0.87] |
|                          | PCA5      |  20 | **+1.00 [+0.15, +1.85]** |     −0.01 [−0.87, +0.85] |
|                          | FF5       |   5 |     +0.51 [−0.31, +1.30] | **−1.20 [−2.05, −0.39]** |
|                          | FF5       |  20 |     −0.35 [−1.10, +0.35] | **−1.36 [−2.11, −0.66]** |

The daily gross returns of ours and the paper correlate at 0.90–0.92. Paper with costs
correlates with the paper at 0.18 (PCA5) and 0.34 (FF5) on the test period: it is a
different strategy. More recipe variants (cost weight 0.5, PCA8) are in
`results/full_comparison.csv`.

Cost weight 0.5 significantly lowers gross SR vs the paper (PCA5 test, B=1: −0.53
[−0.98, −0.09]). On the full data, 0.25 beats the sample's choice of 0.5.

### Validation 2003–2005: when the alpha is strong

| Residuals |   B | Paper gross / net | Paper with costs gross / net | Ours (cw 0.25) gross / net |
| --------- | --: | ----------------: | ---------------------------: | -------------------------: |
| PCA5      |   1 |      4.45 / −2.20 |                 3.68 / +0.05 |               4.17 / −2.65 |
| PCA5      |   5 |      3.19 / +0.33 |                 2.41 / +0.93 |               2.77 / −0.79 |
| PCA5      |  20 |      2.21 / +0.97 |                 1.82 / +1.23 |               1.35 / −0.85 |
| FF5       |   5 |      2.77 / +0.28 |                 1.87 / +0.85 |               2.59 / −0.11 |
| FF5       |  20 |      1.49 / +0.37 |                 1.49 / +1.11 |               1.15 / −0.42 |

In these years paper with costs is the best arm net, with net SR around +1. It keeps most of
the gross signal (3.68 vs 4.45 at B=1), and its turnover is less than half the paper's.

### Gross SR by year (B=1)

| Year             | 2003 | 2004 | 2005 | 2006 |  2007 | 2008 |  2009 | 2010 |  2011 |  2012 |  2013 | 2014 |  2015 | 2016 |
| ---------------- | ---: | ---: | ---: | ---: | ----: | ---: | ----: | ---: | ----: | ----: | ----: | ---: | ----: | ---: |
| Paper PCA5       | 5.57 | 4.39 | 3.34 | 5.39 |  1.53 | 5.43 |  2.24 | 3.10 |  3.67 |  2.66 |  1.20 | 0.84 |  0.31 | 0.45 |
| Paper+costs PCA5 | 5.16 | 3.35 | 2.29 | 0.63 | −1.07 | 1.02 | −0.59 | 2.06 | −1.24 | −1.55 | −0.71 | 0.94 |  1.55 | 0.42 |
| Ours PCA5        | 4.87 | 4.10 | 3.53 | 5.59 |  2.16 | 5.16 |  2.07 | 3.34 |  3.29 |  2.44 |  1.14 | 1.46 | −0.51 | 0.45 |

## 3. Reading

1. **The full-cost loss collapses to a drift book once the alpha weakens.** From 2006,
   paper with costs holds 0.86–0.91 net long on PCA5 and 0.62–0.65 on FF5. It trades 0.04–0.2
   of the book per day.
   - Over 2003–2016, its PCA5 gross SR comes almost all from the neutral part: 0.75 of 0.74 at B=1, residual
     drift only 0.20 (`results/decomposition.csv`).
   - On FF5, drift is the larger part: drift 0.50, neutral 0.30.
   - The same degeneration appeared on the 49-stock sample, where it held a static long
     book.
   - The paper's unconstrained Sharpe-loss book stays near dollar-neutral on PCA5 (net
     exposure 0.06).
2. **Costs in the loss help net only against the cost-blind variant.** Both cost-aware
   arms beat the paper net on the test period. Neither turns net positive.
3. **Our recipe keeps the signal; paper with costs does not.** The recipe matches the
   paper's gross SR (Δ −0.06, n.s.) with lower turnover. Paper with costs gives up 1.4–2.5
   SR of gross signal to cut turnover 5×. Net on PCA5 the two end up equal, because neither
   gross signal survives 5 bp at the turnover it needs.
   - Recipe break-even: 2.4–2.9 bp without smoothing, ~4 bp at B=5.
   - Paper with costs PCA5 break-even: ≈ 0 bp on the test period (0.2 bp at B=1).
4. **Smoothing captures most of the recipe's advantage over the paper.** From B=1 to B=5
   the recipe's test net advantage falls from +1.38 to +0.50. It stays significant for PCA5
   and disappears for FF5.
5. **The alpha decays in every arm.** Gross SR is ~5 in 2003–2006 and 0.3–0.5 in 2015–2016.
   This matches the weak sample results for 2020–2025.
6. **Paper with costs on FF5 is the best net arm on the test period** (+0.11 at B=20).
   It earns this from residual drift of a 0.65 net-long book, not from stat-arb. It is also
   not significant (PSR 0.64).

## 4. Pre-registered selection (validation → test)

Commands:

- `uv run python -m experiments.select --split-date 2006-01-01 --prefix ensemble__full --output selection_full.csv`;
- `uv run python -m experiments.deflated`.

The selection rule, fixed before the runs:

- **choose on 2003–2005 net SR among near-neutral books** (|net exposure| < 0.05);
- report the choice on 2006–2016.

Paper with costs is excluded by the neutrality filter.

| Choice                                                      | Validation net SR | Test gross SR | Test net SR |
| ----------------------------------------------------------- | ----------------: | ------------: | ----------: |
| Selected: paper PCA5, B=20                                  |             +0.97 |          0.92 |       −0.88 |
| Without the neutrality filter: paper with costs PCA5, B=20  |             +1.23 |         −0.20 |       −0.59 |
| Ours, PCA5 cw 0.25, B=10 (main recipe K)                    |             −0.41 |          1.28 |       −0.53 |
| Best on test in hindsight: paper with costs FF5, B=20       |             +1.11 |          0.39 |       +0.11 |
| Best neutral on test in hindsight: ours, PCA8 cw 0.25, B=20 |             −0.62 |          1.39 |       +0.05 |

Rank correlation between validation and test for the neutral ensembles:

- net: +0.21 (p = 0.25);
- gross: +0.82.

The gross signal ranks stably; the cost/turnover trade-off does not. Every rule-based choice
ends negative net on test.

### Deflated Sharpe ratio

Setup:

- 84 full-data trials on the test period: every trained run and every ensemble/smoothing
  variant of `full_paper`, `full_paper_costs`, `full_recipe` and `full_bench`.
- Bailey & López de Prado (2014), `dlsa_baseline.analysis.statistics`.
- Expected maximum SR under no skill: 2.00 gross, 2.62 net.

| Run (test period)           | Gross SR | Gross DSR | Net SR | Net PSR | Net DSR |
| --------------------------- | -------: | --------: | -----: | ------: | ------: |
| Ours, PCA8 cw 0.25, B=1     |     2.63 |      0.98 |  −2.29 |    0.00 |    0.00 |
| Ours, PCA5 cw 0.25, B=1     |     2.59 |      0.98 |  −2.32 |    0.00 |    0.00 |
| Paper, PCA5, B=1            |     2.53 |      0.96 |  −3.95 |    0.00 |    0.00 |
| Paper with costs, FF5, B=20 |     0.39 |      0.00 |  +0.11 |    0.64 |    0.00 |
| Ours, PCA8 cw 0.25, B=20    |     1.39 |      0.02 |  +0.05 |    0.57 |    0.00 |

- **Gross:** the unsmoothed CNN+Transformer ensembles of the paper and our recipe survive
  deflation (DSR 0.96–0.98). The gross signal is real.
- **Net:** no run clears even the undeflated PSR bar of 0.95.

The deflation is conservative: the trials are highly correlated, and the spread of their
Sharpe ratios mostly reflects real model and turnover differences, not luck. That does not
change the net conclusion, which already fails without deflation.

## 5. Conclusion

1. **Replication.** The paper's gross result replicates in ranking and at 65–85% of its
   level on our data. The CNN+Transformer signal is statistically real after deflating for
   every configuration we tried.
2. **Our recipe vs the paper.** With the same network, seeds, data and smoothing, the recipe
   matches the paper's gross Sharpe and has a higher net Sharpe on 2006–2016 (+0.50 at B=5,
   significant for PCA5). The gain comes from lower turnover, not a stronger signal. It does
   not hold on 2003–2005.
3. **Our recipe vs paper with costs.** On PCA5 the recipe keeps 1.7 SR more gross signal (B=5). Net it is
   equal on PCA5 and worse on FF5, where paper with costs harvests residual drift from a
   net-long book.
4. **Costs.** None of the three arms is profitable at 5 bp + 1 bp on 2006–2016. The paper's
   net SR ~1 is reproduced only in 2003–2005, while the alpha is strong.

The paper's with-costs result is IPCA-based, on CRSP data and stock-space turnover. We
cannot separate these differences (universe, residual model, turnover measurement) from
alpha decay in our universe: that would need CRSP and IPCA, outside this study. The
research is complete: no further runs are planned.

`full_probe` (epochs 10 / 30 / 100, first 3 blocks only, 1 seed) gave SR 3.46 / 4.18 / 5.50.
More epochs look better on validation, but all main suites used 30 epochs, and no 100-epoch
full run exists.
