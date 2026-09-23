# Run Report

## 1. Work Completed

Built and executed a clean data-to-portfolio baseline with cached public prices, simple daily
returns, past-only rolling PCA, interchangeable residual serialization, leak-free cumulative
windows, CNN-transformer and FFN policies, L1 portfolio construction, differentiable Sharpe,
chronological train/validation/test, metrics, plots, and automated tests.

## 2. References and Materials

- Official `gregzanotti/dlsa-public` shallow clone at
  `references/dlsa-public/`, commit `ea8cc2958943eb1fe914aa4fad6998994a678323`.
- Paper HTML/public arXiv version 2106.04028v2 was consulted.
- No third-party replication repository was cloned; search results were low-signal or
  unlicensed, and the official code was sufficient.
- The official clone contains compressed residual arrays, but licensed CRSP/Compustat source
  data and required complete `Phi` mappings are unavailable. No external "Data for Code"
  payload was downloaded.

## 3. Environment

macOS 26.5.1 on Apple arm64 with 16 GiB RAM. The validated project environment is managed
exclusively by uv 0.9.13 from the committed lock file and uses Python 3.13.15 and PyTorch
2.14.0. No NVIDIA GPU, `nvidia-smi`, or CUDA was present. Apple MPS was available and
selected; CPU fallback was tested. See `docs/ENVIRONMENT.md`.

## 4. Public Data

Yahoo Finance adjusted closes via `yfinance`, with implemented Stooq fallback, for 50 liquid
US-listed stocks from 2015-01-02 through 2025-12-31. All 50 succeeded on the final run, with
2,766 common trading days and no reported missing adjusted closes in the downloaded panel.
The fixed current universe has survivorship bias.

The resulting return panel had 2,765 dates. Rolling PCA produced 2,513 OOS residual dates by
using 252 historical dates, five factors, and 60 dates for loadings.

PCA diagnostics:

| Diagnostic | Value |
|---|---:|
| Residual mean | 0.00003297 |
| Residual standard deviation | 0.0129582 |
| Mean five-factor explained variance | 0.58437 |
| Raw mean absolute cross-sectional correlation | 0.37308 |
| Residual mean absolute cross-sectional correlation | 0.04211 |

## 5. Architecture

The primary model has two causal kernel-size-2 convolutions, eight channels, instance
normalization and a residual connection, one four-head Transformer encoder with a 16-unit
internal FFN, dropout 0.25, and a scalar linear allocation head. There is no positional
encoding, following the official implementation. A `[date, asset, 30]` cumulative residual
tensor is flattened per asset for the shared policy, restored to `[date, asset]`, and L1
normalized across assets.

The raw FFN and a non-trainable reversal score are also implemented. Model tests execute both
trainable architectures.

## 6. Deviations From the Paper

Free fixed universe rather than CRSP; no risk-free subtraction; no Compustat/IPCA or 46
characteristics; provisional PCA; direct residual trading rather than stock-space `Phi`
mapping; simple 60/20/20 chronological split rather than rolling 1,000/125 retraining; no
costs; one fixed seed; modern independent PyTorch implementation. See
`docs/LIMITATIONS.md`.

## 7. Commands Executed

```bash
uv python install 3.13
uv lock
uv sync --frozen
git clone --depth 1 https://github.com/gregzanotti/dlsa-public.git references/dlsa-public
uv run python -c "import platform, torch; ..."
uv run ruff check src scripts tests
uv run ruff format --check src scripts tests
uv run pytest -q
uv run python scripts/run_smoke_test.py
uv run python scripts/download_sample_data.py
uv run python scripts/build_pca_residuals.py
uv run python scripts/train_baseline.py --epochs 10
uv run python scripts/evaluate_baseline.py
uv run python scripts/analyze_run.py --benchmark reversal=outputs/reversal_predictions.npz
```

The first PCA build found and led to correction of an explicit-cache-schema bug. The same
command then completed. No manual environment activation, direct `pip`, or `PYTHONPATH` is
required; every command is executed through uv from the locked environment.

## 8. Tests

Python 3.13 lock-file validation: `14 passed`; Ruff reported all checks passed and all Python
files formatted. Coverage includes return handling, PCA shape and future-mutation
no-lookahead, residual round-trip, exact window cutoff, CNN/Transformer/full-model shapes,
finite forward/backward, Sharpe gradients, L1 normalization, return alignment, and a complete
one-epoch integration train. Statistical tests cover HAC inference, block-bootstrap
reproducibility, transaction costs, factor alpha, multiple-testing correction, CSCV/PBO, and
complete report/plot generation.

## 9. Smoke Results

Synthetic mean-reverting process, four epochs, held-out 66 observations:

| Metric | Value |
|---|---:|
| Annualized mean | 2.0391 |
| Annualized volatility | 0.0644 |
| Annualized Sharpe | 31.66 |

This deliberately easy process only verifies learnability.

Public two-epoch smoke test produced Sharpe 0.05. The final fixed-seed ten-epoch engineering
run used 1,489 train dates, 496 validation dates, and 498 test dates. Final average training
batch objective was negative Sharpe `-0.0993`; best validation annualized Sharpe was `-0.3288`.
The selected checkpoint's held-out period was 2024-01-08 through 2025-12-31:

| Metric | Value |
|---|---:|
| Annualized mean | 0.02258 |
| Annualized volatility | 0.03331 |
| Annualized Sharpe | 0.6778 |
| Average one-way L1 turnover | 0.8167 |
| Asset weight changes counted | 24,850 |

No seeds were searched or cherry-picked. Negative validation performance is an important
warning against interpreting the positive test value.

The statistical analyzer classifies this run as **insufficient statistical evidence**. The
HAC t-statistic for mean return is 1.05 (one-sided p-value 0.147), the 95% circular
block-bootstrap Sharpe interval is `[-0.49, 1.78]`, PSR is 0.83, and Deflated Sharpe
probability is 0.65 when the reversal benchmark is counted as a tried strategy. Estimated
minimum track record is 1,483 trading days versus 498 observed. Break-even transaction cost
is only 1.10 bps per unit turnover; at 10 bps the annualized net Sharpe is -5.47. The model's
mean return exceeds reversal, but the paired HAC one-sided p-value is 0.171. PBO is 0.143 for
the two supplied strategies, but this estimate is not reliable unless every tried strategy
and configuration is included.

## 10. Runtime

Final Python 3.13 ten-epoch public run: 9.79 seconds. PCA generation and download completed
comfortably within the smoke workflow. Exact wall-clock timing was not instrumented for those
two steps.

## 11. Accelerator and Memory

Apple MPS was used. CUDA and RTX VRAM measurements were unavailable on this host. CUDA device
selection, GPU-name printing, and peak allocated VRAM reporting are implemented for CUDA.

## 12. Generated Artifacts

Cached prices and quality JSON, daily returns, PCA residual NPZ, model checkpoint, model and
reversal prediction NPZs, test metrics JSON, `outputs/training_loss.png`, and
`outputs/cumulative_test_return.png` were generated and read back where applicable. The
analyzer generated strict JSON and Markdown reports plus equity/drawdown, rolling Sharpe,
monthly returns, return distribution, bootstrap Sharpe, and transaction-cost plots.

## 13. Known Problems

The public source can transiently fail individual Yahoo requests; Stooq fallback and forced
cache refresh are available. Validation performance was weak. The full official code cannot
reproduce paper results without licensed inputs and stock-space mappings. No transaction
costs or rolling retraining are implemented yet.

## 14. Recommended Next Steps

1. Replace `pca_residuals.npz` with the team's residual panel while preserving dates/tickers.
2. Add and validate time-varying `Phi` matrices, then normalize in stock space.
3. Implement rolling 1,000-day training and 125-day retraining behind the same trainer API.
4. Add risk-free returns, transaction/borrow costs, and dynamic-universe masks.
5. Run multiple predeclared seeds and report distributions, not a selected result.

## WHAT DANIIL SHOULD READ FIRST

1. `docs/BASELINE_DESIGN.md`
2. `src/dlsa_baseline/data/pca_residuals.py`
3. `src/dlsa_baseline/data/windows.py`
4. `src/dlsa_baseline/models/cnn_transformer.py`
5. `src/dlsa_baseline/models/cnn.py`
6. `src/dlsa_baseline/training/objectives.py`
7. `src/dlsa_baseline/training/trainer.py`
8. `src/dlsa_baseline/training/evaluation.py`
9. `scripts/train_baseline.py`
10. `tests/test_data.py`
