# Full-dataset plan: Kaggle T4 training

Budget: 10–15 GPU-hours on Kaggle T4 (2×T4 if available).

## Data

- **Returns**: `data/processed/ff5_daily_panel.parquet` (Git LFS, 148 MB): 3,182 tickers,
  1998-01-02 → 2016-12-30, `excess_return` and the FF5 factors.
- **Universe**: the panel has no size or volume data. The raw source
  `marketneutral/quandl-wiki-prices-us-equites` (Kaggle) provides adjusted close and volume.
  At the start of each month, stocks are ranked by median dollar volume over the previous
  63 trading days (data strictly before the month), and the top 500 are kept.
- **Eligibility** on date t: in the universe, residuals observed on all 30 window days, and
  no |return| > 50% over the previous 252 days. A missing target counts as a 0 return.
- **Residuals** (out of sample): `ff5` (60-day rolling OLS on the FF5 factors, no
  intercept), `pca5` and `pca8` (252-day correlation PCA over eligible names, 60-day
  loadings).
- `uv run python -m experiments.data_full` writes `data/full/residuals_{ff5,pca5,pca8}.npz`.
  Result: ~470–495 eligible names per day; first tradeable date 1999-02-17; OOS
  2003-02-10 → 2016-12-30 (3,499 days).

## Training

`dataset="full"` runs use `experiments/panel.py:MaskedPanel`, which builds windows per batch
on the device and scores only eligible names. Each finished retraining block is saved to
`<run>/_blocks/`, so a restarted run resumes from the first missing block. Environment
variables: `DLSA_DATA_DIR`, `DLSA_RESULTS_DIR`, `DLSA_DEVICE`, and `DLSA_DEADLINE` (no new
run starts if it would likely end after this time).

Timing: 4.4 s per epoch on Apple MPS (1,000 days × ~480 names), about 61 min per full
30-epoch run. Expected on one T4: 30–45 min.

## Runs

| suite         | runs | what                                                                                      |
| ------------- | ---: | ----------------------------------------------------------------------------------------- |
| `full_smoke`  |    1 | 1 epoch, 1 block: pipeline check                                                          |
| `full_probe`  |    3 | epochs 10 / 30 / 100 on the first 3 blocks (validation period only)                       |
| `full_paper`  |    4 | paper replication: CNN+Transformer, Sharpe loss, unconstrained book, ff5 / pca5 × 2 seeds |
| `full_paper_costs` | 4 | paper with costs: Sharpe net of full 5 bp + 1 bp, unconstrained book, ff5 / pca5 × 2 seeds |
| `full_recipe` |   18 | dollar-neutral, cost-aware loss, cost weight 0.25 / 0.5 × ff5 / pca5 / pca8 × 3 seeds     |
| `full_bench`  |    8 | Fourier+FFN, OU+Threshold, reversal (run locally on MPS, done)                            |

Runs are ordered seed-major, and `run_suite --shard i/n` splits them across GPUs.
Estimated Kaggle time: about 8 h on 2×T4 for smoke, probe, paper and recipe.

Model choice uses **validation = 2003-02 → 2005-12** only:
`python -m experiments.select --split-date 2006-01-01 --prefix ensemble__full`.
**Test = 2006-01 → 2016-12** is reported, not tuned on.

## Kaggle

- `kaggle/kaggle_push.py` stages `experiments/` and `src/` as a private code dataset,
  `data/full/*.npz` as a private residuals dataset, and the W&B key (if `.env` has one) as a
  private secrets dataset. It then pushes the kernel (`kaggle/kernel-metadata.json`, T4).
- `kaggle/run_pipeline.py` checks the GPUs, starts one `run_suite` process per GPU, sets
  `DLSA_DEADLINE` to 11 h after start, copies a previous session's results if a
  `dlsa-results-prev` input is attached, and zips `results/` at the end.
- `python kaggle/kaggle_pull_output.py --wait` downloads the output into `kaggle_output/` and
  adds the new run folders to `results/`. It never overwrites local aggregate tables or
  existing runs; rebuild them with `postprocess`, `decompose` and `report_tables`.

## Outputs

`results/full_*` runs in the standard run format, and `docs/full_report.md`: a Table I
comparison with the paper, three arms (paper, paper with costs, recipe) compared pairwise,
the drift decomposition, validation → test and the Deflated Sharpe ratio.

Status: Kaggle session 1 of 2026-09-24 (8.1 h on 2×T4) ran `full_smoke`, `full_probe`,
`full_paper` and `full_recipe`. Session 2 (1.5 h) ran `full_paper_costs`. Session 3
(7.1 h) ran the `official_*` suites below. Results: `docs/full_report.md` (section 6 for
session 3). Logs: `results/kaggle_logs/`.

## Session 3 (last): the authors' residuals (issue #2)

The authors publish their out-of-sample residuals on CRSP (`gregzanotti/dlsa-public`,
`residuals/`), including IPCA. `uv run python -m experiments.official` downloads the K=5
panels and writes `data/full/residuals_official_{ipca5,pca5,ff5}.npz`:

- cap filter > 0.01% of the market (in the data); eligible at t = observed on all 30 days
  before t (the official `preprocess.py` rule);
- median 863 eligible names per day (ours: ~480), first tradeable 1998-02-17;
- rolling 1,000 / 125 as before, so OOS = 2002-02-08 → 2016-12-30, the paper's OOS span;
- no Φ is published: books trade residuals, costs are on residual weights, as on our data.

Check: our OU+Threshold on `official_ipca5` gives gross SR 0.62, the same as the OU of PR #1.

| suite                  | runs | what                                                             |
| ---------------------- | ---: | ---------------------------------------------------------------- |
| `official_paper_costs` |    2 | paper with costs on IPCA5 (Table IX is IPCA-based), 2 seeds      |
| `official_paper`       |    6 | paper Table I: CNN+Trans, Sharpe loss, unconstrained, IPCA5 / PCA5 / FF5 × 2 seeds |
| `official_recipe`      |    2 | our recipe on IPCA5: dollar-neutral, cost weight 0.25, 2 seeds   |

10 runs, about 75–80 min each on a T4 (1.8× the names): about 7 h on 2×T4, one session.
This is the last run of the study; the results go into a separate section, and the WIKI
results above stay as they are.

**Readout, fixed before the run.** 2-seed ensembles, same periods as before
(validation = 2002-02 → 2005-12, test = 2006-01 → 2016-12); nothing is selected on either.

1. _Replication, gross:_ `official_paper` B=1, full OOS, vs Table I (IPCA 4.16, PCA 3.36,
   FF 3.21) and vs our WIKI numbers. Reported, no threshold.
2. _Why is our net negative? (main question):_ `official_paper_costs` IPCA5, **test net SR
   at B=5**, with its PSR and net exposure:
   - net SR ≥ 0.5 and PSR ≥ 0.95 → **the data explain it**: on the authors' universe and
     residuals the with-costs strategy survives 2006–2016, so our negative net comes from
     the WIKI universe / residuals;
   - net SR ≤ 0.2 → **not the data**: even on the authors' data the net does not survive
     2006–2016, consistent with alpha decay after costs;
   - in between → inconclusive.
   A book with |net exposure| > 0.5 counts as residual drift, not stat-arb, as before.
   Full-OOS net (2002–2016) is reported next to Table IX (~1.1) for reference only, as the
   paper measures turnover in stock space.
3. _Our recipe vs the paper:_ `official_recipe` − `official_paper` on IPCA5, test net ΔSR at
   B=5 with the paired bootstrap CI. Replicates the WIKI result (+0.50) if the CI is above 0.

`experiments.compare` pairs the three suites; `experiments.deflated --family official`
deflates them for their own trials (separate from the 84 WIKI trials).

Status: done (session 3, 7.1 h on 2×T4). Readout: `docs/full_report.md`, section 6.
