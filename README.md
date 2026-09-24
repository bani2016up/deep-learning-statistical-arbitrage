# Deep Learning Statistical Arbitrage Baseline

A small, modern PyTorch research workspace inspired by Guijarro-Ordonez, Pelger, and
Zanotti's *Deep Learning Statistical Arbitrage*. This is a leak-free engineering baseline,
not a scientific reproduction.

## Setup

The project requires Python 3.13 or newer and uses
[`uv`](https://docs.astral.sh/uv/) exclusively for Python installation, dependency locking,
environment management, and command execution. PyTorch uses CUDA, Apple MPS, or CPU in that
order.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
uv sync --frozen
```

`uv sync --frozen` creates and manages the local environment from the committed `uv.lock`.
Do not create or activate a virtual environment manually.

Optionally restore the official reference repository used by
`docs/OFFICIAL_CODE_MAP.md`:

```bash
git clone --depth 1 https://github.com/gregzanotti/dlsa-public.git references/dlsa-public
```

It is not required to run this baseline and is intentionally excluded from Git because it
is large, independently versioned, and covered by its own non-commercial license.

On a CUDA machine, verify the selected wheel with
`uv run python -c "import torch; print(torch.cuda.is_available())"`. If a platform needs a
custom PyTorch index, follow the uv-specific PyTorch instructions at
<https://docs.astral.sh/uv/guides/integration/pytorch/> and regenerate `uv.lock`.

## End-to-End Commands

Run from this directory:

```bash
uv run python scripts/download_sample_data.py
uv run python scripts/build_pca_residuals.py
uv run pytest
uv run python scripts/run_smoke_test.py
uv run python scripts/train_baseline.py
uv run python scripts/evaluate_baseline.py
uv run python scripts/analyze_run.py \
  --benchmark reversal=outputs/reversal_predictions.npz
```

Useful development variants:

```bash
uv run python scripts/download_sample_data.py --force --limit 20
uv run python scripts/build_pca_residuals.py --factors 5
uv run python scripts/train_baseline.py --model raw_ffn --epochs 10 --device cpu
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
```

Downloads are cached at `data/raw/adjusted_prices.parquet`. Repeated normal runs do not
redownload. The default training run writes a checkpoint, metrics, predictions, training-loss
plot, and cumulative-return plot under `outputs/`.

## Pipeline

```text
adjusted prices -> daily returns -> rolling OOS PCA residuals
-> 30-day cumulative residual windows ending t-1
-> causal CNN -> small Transformer -> allocation head
-> per-date L1 weights -> residual return at t -> differentiable Sharpe
```

Data generation is separate from modeling. The interchangeable residual file stores a
`[date, asset]` matrix and explicit dates/tickers; see `ResidualDataset` in
`src/dlsa_baseline/data/pca_residuals.py`.

## OU+Threshold Baseline

The parametric OU+Threshold benchmark of the paper (Table I), plus Avellaneda-Lee variants,
runs on the official precomputed residuals (clone `dlsa-public` into `references/` as above).
No GPU is needed; the full experiment set takes about three minutes on CPU.

```bash
uv run python scripts/run_ou_baseline.py --family IPCA --factors 5
uv run python scripts/analyze_run.py --predictions outputs/ou/ou_paper_ipca5_predictions.npz \
  --output-dir outputs/ou/analysis_ipca5
uv run python scripts/ou_experiments.py all
uv run python scripts/ou_synthetic.py
```

Results, the replication of Table I, and the theory are in `docs/OU_BASELINE.md`.

## Outputs

- `data/raw/adjusted_prices.parquet`: long-form date/ticker/adjusted-price cache
- `data/raw/adjusted_prices.quality.json`: source, coverage, and missingness
- `data/processed/daily_returns.parquet`: rectangular simple-return panel
- `data/processed/pca_residuals.npz`: residuals, dates, tickers, metadata
- `outputs/cnn_transformer.pt`: selected model checkpoint
- `outputs/test_predictions.npz`: held-out returns and weights
- `outputs/test_metrics.json`: held-out annualized metrics
- `outputs/reversal_predictions.npz`: aligned non-trainable reversal benchmark
- `outputs/training_loss.png`: optimization trace
- `outputs/cumulative_test_return.png`: held-out cumulative wealth
- `outputs/analysis/`: statistical JSON/Markdown reports and diagnostic plots
- `outputs/ou/`: OU predictions, daily series, experiment tables, and plots

## Reading Guide

Read `docs/BASELINE_DESIGN.md` for the exact no-lookahead and tensor conventions,
`docs/OFFICIAL_CODE_MAP.md` for the relationship to official code, `docs/OU_BASELINE.md`
for the parametric OU benchmark, `docs/LIMITATIONS.md`
before interpreting any metric, `docs/STATISTICAL_ANALYSIS.md` for inference methodology,
and `RUN_REPORT.md` for the validated run on this host.
