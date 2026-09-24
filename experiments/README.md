# experiments/

A config-driven experiment harness on top of the `dlsa_baseline` package. Run everything
from the repository root:

```bash
uv sync --frozen && uv run python scripts/download_sample_data.py \
  && uv run python scripts/build_pca_residuals.py   # one-time data prep

uv run python -m experiments.run_suite --list          # show suites
uv run python -m experiments.run_suite models factors  # run (resumable)
uv run python -m experiments.run --name my_run --model fourier_ffn --n-factors 3
uv run pytest experiments                               # tests
```

## Modules

| File                         | Purpose                                                                                                                  |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `data.py`                    | Rolling-PCA residual panels for any K (cached in `data/residuals/pca_k{K}.npz`); K=0 = raw returns on the same dates     |
| `models.py`                  | `fourier_ffn`, `ou_ffn`, `ou_threshold`, `reversal`, plus the baseline `cnn_transformer`, `raw_ffn`                      |
| `objectives.py`              | `sharpe`, `mean_variance` (γ), `sharpe_costs` (paper friction model: 5 bp·turnover + 1 bp·short)                         |
| `trainer.py`                 | `RunConfig`; protocols `rolling` (paper 1000/125), `constant` (train once), `fixed` (baseline 60/20/20 + early stopping) |
| `metrics.py`                 | Unified metric set (gross + net-of-cost, HAC t-stat, drawdown, turnover, concentration)                                  |
| `run.py`                     | One run → `results/<name>/`; reuses an identical finished run instead of retraining                                      |
| `suites.py` / `run_suite.py` | Named grids mirroring the paper's tables → `results/summary.csv`                                                         |
| `decompose.py`               | Drift vs dollar-neutral decomposition of every run → `results/decomposition.csv`                                         |
| `postprocess.py`             | Seed ensembles + causal B-day weight smoothing → `results/ensemble__*__ens/` (standard run format)                       |
| `select.py`                  | Selection (2020-02→2022-12) vs holdout (2023→2025), neutral-only, deduplicated → `results/selection.csv`                 |
| `report_tables.py`           | All suite tables (mean ± seed sd) → `results/report_tables.md`                                                           |
| `figures.py`                 | Curated report figures (drift vs signal, K sweep) → `results/figures/`                                                   |
| `compare.py`                 | Full data: paper replication vs recipe on common seeds, same smoothing, paired bootstrap ΔSR → `results/paper_vs_recipe.csv` |
| `final_model.json`           | Final recipe (RunConfig + ensemble/smoothing + tuning grid + sample evidence); see `docs/final_model.md`                 |

## Output contract (`results/<suite>__<variant>__s<seed>/`)

- `config.json`: the full `RunConfig`
- `metrics.json`: `sharpe, mean_return, volatility, sortino, calmar, max_drawdown, hit_rate,
hac_t_stat, hac_p_value, turnover, short_fraction, effective_positions, break_even_cost_bps,
net_sharpe, net_mean_return, ...` + `yearly_returns`
- `returns.csv`: `date, return, net_return, turnover, short_fraction, block`
- `predictions.npz`: `returns, weights, dates, tickers` (works with
  `scripts/analyze_run.py --predictions`)
- `history.csv`: per retraining block and epoch training loss

## Suites

`epochs`, `models`, `factors`, `neutral`, `objective`, `lookback`, `protocol`,
`architecture`, `final` (`--list` shows run counts). Results and interpretation:
`docs/experiment_report.md`. Full-dataset suites (`full_smoke`, `full_probe`, `full_paper`,
`full_recipe`, `full_bench`): `docs/kaggle_plan.md` and `docs/full_report.md`. Full pipeline after the suites:

```bash
uv run python -m experiments.postprocess --suite final
uv run python -m experiments.decompose
uv run python -m experiments.select
uv run python -m experiments.report_tables
uv run python -m experiments.figures
```

## Design decisions

- **Same OOS window for all rolling/constant runs**: `test_start=1000` (window index) →
  2020-02-07 … 2025-12-31, 1,483 days. Shorter training windows start training later but test
  on the same days. `fixed` tests only the last 20%.
- **No early stopping in rolling mode.** This matches the paper (fixed epoch budget) and
  avoids carving a validation set from each 1,000-day window.
- **Fresh initialization per retraining block**, seeded by `(seed, block)`.
- **Costs**: turnover is measured on residual-space weights (no `Phi` mapping available).
  Net metrics are always reported, whatever the training objective.
- **OU+Threshold**: c_thresh = 1.25 (Avellaneda-Lee), R² > 0.25, requires 0 < b < 1.
- **FFN inputs are scale-normalized per window** (Fourier, OU features), analogous to the
  CNN's input instance norm. Without it, residual-scale inputs (~1e-2) barely train.
- **`neutralize=True`** demeans scores per date before L1 normalization, giving a
  dollar-neutral residual book. Without it, models go net long and harvest residual drift
  (see `decompose.py`).
- **`cost_weight`** scales the cost penalty in the _loss only_. Evaluation always uses the
  full 5 bp + 1 bp.
- **Dedupe**: a run whose config equals a finished run (missing keys filled with defaults)
  is copied, not retrained. `metrics.json` is written last and marks completion.
