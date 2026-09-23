# Official Code Map

## Reference

The official repository was shallow-cloned at commit
`ea8cc2958943eb1fe914aa4fad6998994a678323` into `references/dlsa-public/`.
It is reference-only and was not modified. Its custom license permits redistribution but
prohibits unauthorized commercial use and derivatives. This baseline is an independent
implementation based on the paper's public description and observed interfaces, not copied
source.

The repository contains about 792 MB of compressed, precomputed PCA, IPCA, and Fama-French
residual arrays. It does not contain the licensed CRSP/Compustat return and characteristic
inputs or all residual composition (`Phi`) matrices needed to reproduce stock-space results.
The official README explicitly attributes this to provider licensing. No separate public
"Data for Code" download with the missing licensed inputs was found, so nothing was copied
to `data/reference/`.

## Map

| Paper concept | Official file/function | Our implementation file |
|---|---|---|
| Entrypoint/configuration | `run_train_test.py`, YAML under `configs/` | scripts and `src/dlsa_baseline/config.py` |
| PCA residual generation | `factor_models/pca.py:PCA.OOSRollingWindowPermnos` | `data/pca_residuals.py:rolling_pca_residuals` |
| Residual loading | `run_train_test.py:run` | `data/pca_residuals.py:load_residual_dataset` |
| Cumulative windows | `preprocess.py:preprocess_cumsum` | `data/windows.py:build_cumulative_windows` |
| Causal CNN | `models/CNNTransformer.py:CNN_Block` | `models/cnn.py:CausalCNN` |
| Transformer | `models/CNNTransformer.py:CNNTransformer.encoder` | `models/transformer.py:TemporalTransformer` |
| Allocation layer | `models/CNNTransformer.py:CNNTransformer.linear` | `models/allocation.py:AllocationHead` |
| Raw FFN | `models/RawFFN.py:RawFFN` | `models/baselines.py:RawFFN` |
| L1 stock/residual weights | `train_test.py:train`, `get_returns` | `models/allocation.py:normalize_l1` |
| Sharpe/mean-variance objective | `train_test.py:train` | `training/objectives.py` |
| Training loop | `train_test.py:train` | `training/trainer.py:train_model` |
| Rolling 1000/125 experiment | `train_test.py:test` | Not yet reproduced; chronological split is in `training/trainer.py` |
| Performance reporting | `train_test.py:get_returns` | `training/evaluation.py:metrics` |

## Architecture Findings

The official default uses 30 observations, a CNN block containing two causal left-padded
convolutions, eight output features, instance normalization, residual addition, one
four-head Transformer encoder layer with a 16-unit internal FFN, dropout 0.25, and a linear
map from the last sequence position to one score. It uses no positional encoding. Scores are
normalized by gross stock exposure, and Sharpe is optimized directly. The full experiment
uses a 1,000-day training window and retrains every 125 days.

Our differences are explicit: modern batch-first PyTorch, stable epsilon handling, finite
gradient checks, a fixed rectangular liquid-stock universe, chronological train/validation/
test rather than full rolling retraining, and direct residual-space leverage normalization
because the public-data `Phi` mapping is unavailable.

## Legacy Run Status

The official model class imports and instantiates under PyTorch 2.5.1. Full official runs do
not run from the clone alone: factor generation expects absent files such as
`DailyReturns-RFadjusted-old.npz`, `MonthlyData.npz`, and Compustat characteristics, while
paper reproduction also requires residual composition matrices. Its lockfile pins NumPy
1.21.2, pandas 1.3.2, scikit-learn 0.24.2, and PyTorch 1.9.0, which are obsolete for modern
Python. Rebuilding that legacy environment was intentionally avoided.
