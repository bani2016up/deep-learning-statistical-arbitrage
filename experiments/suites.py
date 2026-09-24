"""Named experiment grids. Run names are ``<suite>__<variant>__s<seed>``."""

from __future__ import annotations

import os
from collections.abc import Iterator

import torch

from experiments.models import STATIC
from experiments.trainer import RunConfig

SEEDS = (0, 1, 2)
EPOCHS = 30  # chosen from the `epochs` calibration suite; see docs/experiment_report.md


def _device(model: str) -> str:
    return "mps" if model == "cnn_transformer" else "cpu"


def _runs(suite: str, variants: dict[str, dict], seeds=SEEDS) -> Iterator[RunConfig]:
    for variant, overrides in variants.items():
        model = overrides.get("model", "cnn_transformer")
        for seed in (0,) if model in STATIC else seeds:
            params = {
                "epochs": EPOCHS,
                "device": _device(model),
                "seed": seed,
                **overrides,
            }
            yield RunConfig(name=f"{suite}__{variant}__s{seed}", **params)


def epochs() -> Iterator[RunConfig]:
    """Calibration: how much does the epoch budget matter under rolling retraining?"""
    yield from _runs("epochs", {f"e{n}": {"epochs": n} for n in (10, 30, 100)})


def models() -> Iterator[RunConfig]:
    """Paper Table I / A.IX analogue at K=5: signal-extraction ablation."""
    names = (
        "cnn_transformer",
        "fourier_ffn",
        "raw_ffn",
        "ou_ffn",
        "ou_threshold",
        "reversal",
    )
    yield from _runs("models", {name: {"model": name} for name in names})


def factors() -> Iterator[RunConfig]:
    """Paper Table I: number of PCA factors K (K=0 trades raw returns)."""
    variants = {}
    for k in (0, 1, 3, 5, 8, 10, 15):
        for model in ("cnn_transformer", "fourier_ffn", "ou_threshold"):
            variants[f"{model}_k{k}"] = {"model": model, "n_factors": k}
    yield from _runs("factors", variants)


def lookback() -> Iterator[RunConfig]:
    """Paper Table V: signal lookback window L."""
    yield from _runs("lookback", {f"l{n}": {"lookback": n} for n in (10, 20, 30, 60)})


def objective() -> Iterator[RunConfig]:
    """Paper Tables III and IX: Sharpe vs mean-variance vs Sharpe net of trading frictions."""
    yield from _runs(
        "objective",
        {
            name: {"objective": name}
            for name in ("sharpe", "mean_variance", "sharpe_costs")
        },
    )


def protocol() -> Iterator[RunConfig]:
    """Paper Table VII + baseline: rolling retraining vs a constant model vs one fixed split."""
    variants = {
        "rolling_w1000": {"protocol": "rolling", "train_window": 1000},
        "rolling_w500": {"protocol": "rolling", "train_window": 500},
        "rolling_w750": {"protocol": "rolling", "train_window": 750},
        "constant_w1000": {"protocol": "constant", "train_window": 1000},
        "fixed_602020": {"protocol": "fixed"},
    }
    yield from _runs("protocol", variants)


def architecture() -> Iterator[RunConfig]:
    """Paper Table A.II/A.IV: small perturbations of the CNN+Transformer hyperparameters."""
    variants = {
        "d8_h4_drop25": {},
        "d16_h4_ff32_drop50": {"features": 16, "transformer_ff": 32, "dropout": 0.5},
        "d8_h2": {"attention_heads": 2},
        "d8_h4_drop50": {"dropout": 0.5},
    }
    yield from _runs("architecture", variants)


def final() -> Iterator[RunConfig]:
    """Final-model candidates: dollar-neutral CNN+Trans with a graded cost penalty in the loss.

    cost_weight 0 (plain Sharpe) and 1 (full paper costs) are reused from the `neutral` suite.
    Seed ensembles and weight smoothing are applied afterwards by experiments.postprocess.
    """
    variants = {
        f"cnn_neutral_cw{weight:g}": (
            {"neutralize": True}
            if weight == 0
            else {
                "neutralize": True,
                "objective": "sharpe_costs",
                "cost_weight": weight,
            }
        )
        for weight in (0.0, 0.25, 0.5, 1.0)
    }
    # `factors`: K=8 had the largest neutral part (0.45) but 3x the seed sd of K=5.
    # K=8 was picked after seeing full-OOS results, so its neighbours test that it is a
    # plateau rather than a lucky point.
    for k in (6, 7, 8, 10):
        variants[f"cnn_neutral_cw0.5_k{k}"] = {
            "neutralize": True,
            "objective": "sharpe_costs",
            "cost_weight": 0.5,
            "n_factors": k,
        }
    yield from _runs("final", variants)


def neutral() -> Iterator[RunConfig]:
    """Dollar-neutral residual book: removes the equal-weight residual drift the models ride."""
    variants = {
        f"{model}_neutral": {"model": model, "neutralize": True}
        for model in ("cnn_transformer", "ou_ffn", "fourier_ffn", "raw_ffn", "reversal")
    }
    variants["cnn_transformer_neutral_costs"] = {
        "neutralize": True,
        "objective": "sharpe_costs",
    }
    yield from _runs("neutral", variants)


# ---------------------------------------------------------------------------------------
# Full dataset (team WIKI/FF5 panel, ~480 eligible names/day, OOS 2003-02 → 2016-12).
# Runs are ordered by priority (seed 0 of every variant first), and
# `run_suite --shard i/n` deals them round-robin to GPUs.
# ---------------------------------------------------------------------------------------
FULL_RESIDUALS = {
    "ff5": {"factor_model": "ff5", "n_factors": 5},
    "pca5": {"n_factors": 5},
    "pca8": {"n_factors": 8},
}
RECIPE = {"neutralize": True, "objective": "sharpe_costs"}


def full_device() -> str:
    """``DLSA_DEVICE`` or the best available accelerator (cuda → mps → cpu)."""
    if os.environ.get("DLSA_DEVICE"):
        return os.environ["DLSA_DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _full_runs(
    suite: str, variants: dict[str, dict], seeds=SEEDS
) -> Iterator[RunConfig]:
    """Seed-major order: every variant's seed 0 before any seed 1."""
    for seed in seeds:
        for variant, overrides in variants.items():
            model = overrides.get("model", "cnn_transformer")
            if model in STATIC and seed != seeds[0]:
                continue
            params = {
                "dataset": "full",
                "epochs": EPOCHS,
                "device": full_device(),
                "seed": seed,
                **overrides,
            }
            yield RunConfig(name=f"{suite}__{variant}__s{seed}", **params)


def full_smoke() -> Iterator[RunConfig]:
    """Full data: 1 run, 1 epoch, 1 block — runner dry-run / first sanity step on Kaggle."""
    yield from _full_runs(
        "full_smoke",
        {"pca5": {**RECIPE, "cost_weight": 0.5, "epochs": 1, "max_blocks": 1}},
        seeds=(0,),
    )


def full_probe() -> Iterator[RunConfig]:
    """Full data: epoch budget on the first 3 retrain blocks (validation period only) + timing."""
    variants = {
        f"pca5_e{n}": {**RECIPE, "cost_weight": 0.5, "epochs": n, "max_blocks": 3}
        for n in (10, 30, 100)
    }
    yield from _full_runs("full_probe", variants, seeds=(0,))


def full_paper() -> Iterator[RunConfig]:
    """Full data: paper replication (CNN+Trans, Sharpe, unconstrained book) on ff5 / pca5."""
    variants = {name: dict(FULL_RESIDUALS[name]) for name in ("ff5", "pca5")}
    yield from _full_runs("full_paper", variants, seeds=(0, 1))


def full_paper_costs() -> Iterator[RunConfig]:
    """Full data: paper with costs (Sharpe net of full 5bp+1bp costs, unconstrained) on ff5 / pca5."""
    variants = {
        name: {**FULL_RESIDUALS[name], "objective": "sharpe_costs", "cost_weight": 1.0}
        for name in ("ff5", "pca5")
    }
    yield from _full_runs("full_paper_costs", variants, seeds=(0, 1))


def full_recipe() -> Iterator[RunConfig]:
    """Full data: our recipe (dollar-neutral, cost-aware loss) cw ∈ {0.25, 0.5} × ff5/pca5/pca8."""
    variants = {
        f"{name}_cw{weight:g}": {**residual, **RECIPE, "cost_weight": weight}
        for weight in (0.5, 0.25)
        for name, residual in FULL_RESIDUALS.items()
    }
    yield from _full_runs("full_recipe", variants)


def full_bench() -> Iterator[RunConfig]:
    """Full data: paper benchmarks (Fourier+FFN, OU+Thresh, reversal), unconstrained and neutral."""
    variants = {}
    for name in ("ff5", "pca5"):
        for model in ("fourier_ffn", "ou_threshold", "reversal"):
            variants[f"{name}_{model}"] = {**FULL_RESIDUALS[name], "model": model}
        variants[f"{name}_fourier_ffn_neutral"] = {
            **FULL_RESIDUALS[name],
            "model": "fourier_ffn",
            "neutralize": True,
        }
    yield from _full_runs("full_bench", variants, seeds=(0,))


# ---------------------------------------------------------------------------------------
# The authors' own OOS residuals (CRSP, cap > 0.01% of market; experiments.official), K = 5.
# One last Kaggle session (issue #2): does the negative net of full_* come from our data?
# Same seeds (0, 1) for every arm, so experiments.compare pairs them on common seeds.
# ---------------------------------------------------------------------------------------
OFFICIAL_RESIDUALS = {
    name: {"factor_model": f"official_{name[:-1]}", "n_factors": 5}
    for name in ("ipca5", "pca5", "ff5")
}


def official_paper() -> Iterator[RunConfig]:
    """Authors' residuals: paper Table I (CNN+Trans, Sharpe loss, unconstrained) on IPCA/PCA/FF."""
    yield from _full_runs("official_paper", OFFICIAL_RESIDUALS, seeds=(0, 1))


def official_paper_costs() -> Iterator[RunConfig]:
    """Authors' residuals: paper with costs (Table IX is IPCA-based) on IPCA5."""
    variants = {
        "ipca5": {**OFFICIAL_RESIDUALS["ipca5"], "objective": "sharpe_costs", "cost_weight": 1.0}
    }
    yield from _full_runs("official_paper_costs", variants, seeds=(0, 1))


def official_recipe() -> Iterator[RunConfig]:
    """Authors' residuals: our recipe (dollar-neutral, cost weight 0.25) on IPCA5."""
    variants = {"ipca5_cw0.25": {**OFFICIAL_RESIDUALS["ipca5"], **RECIPE, "cost_weight": 0.25}}
    yield from _full_runs("official_recipe", variants, seeds=(0, 1))


SUITES = {
    fn.__name__: fn
    for fn in (
        epochs,
        models,
        factors,
        lookback,
        objective,
        protocol,
        architecture,
        neutral,
        final,
        full_smoke,
        full_probe,
        full_paper,
        full_paper_costs,
        full_recipe,
        full_bench,
        official_paper,
        official_paper_costs,
        official_recipe,
    )
}
