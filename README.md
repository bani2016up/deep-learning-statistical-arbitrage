# Deep Learning Statistical Arbitrage Baseline

A small, modern PyTorch research workspace inspired by Guijarro-Ordonez, Pelger, and
Zanotti's *Deep Learning Statistical Arbitrage*. This is a leak-free engineering baseline,
not a scientific reproduction.

## Setup

Python 3.11 is recommended; 3.10-3.12 are supported. PyTorch will use CUDA, Apple MPS, or CPU
in that order.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Optionally restore the official reference repository used by
`docs/OFFICIAL_CODE_MAP.md`:

```bash
git clone --depth 1 https://github.com/gregzanotti/dlsa-public.git references/dlsa-public
```

It is not required to run this baseline and is intentionally excluded from Git because it
is large, independently versioned, and covered by its own non-commercial license.

On a CUDA machine, install the appropriate PyTorch 2.x wheel from
<https://pytorch.org/get-started/locally/> before `pip install -e '.[dev]'` if the default
wheel is not CUDA-enabled.

## End-to-End Commands

Run from this directory:

```bash
python scripts/download_sample_data.py
python scripts/build_pca_residuals.py
pytest
python scripts/run_smoke_test.py
python scripts/train_baseline.py
python scripts/evaluate_baseline.py
```

Useful development variants:

```bash
python scripts/download_sample_data.py --force --limit 20
python scripts/build_pca_residuals.py --factors 5
python scripts/train_baseline.py --model raw_ffn --epochs 10 --device cpu
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

## Outputs

- `data/raw/adjusted_prices.parquet`: long-form date/ticker/adjusted-price cache
- `data/raw/adjusted_prices.quality.json`: source, coverage, and missingness
- `data/processed/daily_returns.parquet`: rectangular simple-return panel
- `data/processed/pca_residuals.npz`: residuals, dates, tickers, metadata
- `outputs/cnn_transformer.pt`: selected model checkpoint
- `outputs/test_predictions.npz`: held-out returns and weights
- `outputs/test_metrics.json`: held-out annualized metrics
- `outputs/training_loss.png`: optimization trace
- `outputs/cumulative_test_return.png`: held-out cumulative wealth

## Reading Guide

Read `docs/BASELINE_DESIGN.md` for the exact no-lookahead and tensor conventions,
`docs/OFFICIAL_CODE_MAP.md` for the relationship to official code, `docs/LIMITATIONS.md`
before interpreting any metric, and `RUN_REPORT.md` for the validated run on this host.
