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
comparison with the paper, the recipe against the replication, the drift decomposition,
and validation → test (the Deflated Sharpe ratio is still open).

Status: done. Kaggle session of 2026-09-24 (8.1 h on 2×T4) finished `full_smoke`,
`full_probe`, `full_paper` and `full_recipe`; results in `docs/full_report.md`.
