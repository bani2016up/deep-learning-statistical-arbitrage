"""Curated report figures → results/figures/ (static PNG for the markdown docs).

    uv run python -m experiments.figures

1. drift_decomposition.png: total vs drift vs dollar-neutral Sharpe for the key variants.
2. k_sweep.png: final recipe (neutral, cw 0.5, 3-seed ensemble, B=5) gross/net SR vs K.
Palette: validated categorical slots 1-3 (blue, orange, aqua) on a #fcfcfb surface.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run import RESULTS_DIR

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#5f5e58", "#e4e3dd"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
FIGURES = RESULTS_DIR / "figures"
NEUTRAL_EXPOSURE = 0.05  # below this |net exposure| the drift term is float noise

DRIFT_VARIANTS = [  # (suite, variant, label)
    ("factors", "cnn_transformer_k0", "CNN+Trans, raw returns (K=0)"),
    ("models", "cnn_transformer", "CNN+Trans, K=5"),
    ("models", "ou_ffn", "OU+FFN, K=5"),
    ("models", "fourier_ffn", "Fourier+FFN, K=5"),
    ("objective", "sharpe_costs", "CNN+Trans, cost-aware loss"),
    ("neutral", "cnn_transformer_neutral", "CNN+Trans, dollar-neutral"),
    ("ensemble", "final-cnn_neutral_cw0.5_b5", "Final recipe, K=5"),
    ("ensemble", "final-cnn_neutral_cw0.5_k8_b5", "Final recipe, K=8"),
]
K_SWEEP = {
    5: "final-cnn_neutral_cw0.5_b5",
    **{k: f"final-cnn_neutral_cw0.5_k{k}_b5" for k in (6, 7, 8, 10)},
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.axvline(0, color=MUTED, linewidth=1) if ax.name == "rectilinear" else None


def drift_figure(decomposition: pd.DataFrame) -> None:
    rows = []
    for suite, variant, label in DRIFT_VARIANTS:
        group = decomposition[
            (decomposition.suite == suite) & (decomposition.variant == variant)
        ]
        if group.empty:
            continue
        drift = (
            0.0
            if group.net_exposure.abs().max() < NEUTRAL_EXPOSURE
            else group.sharpe_drift.mean()
        )
        rows.append(
            (label, group.sharpe_total.mean(), drift, group.sharpe_neutral.mean())
        )
    frame = pd.DataFrame(rows, columns=["label", "total", "drift", "neutral"])[::-1]

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=SURFACE)
    _style(ax)
    y = np.arange(len(frame))
    height = 0.26
    for offset, column, color, name in (
        (height, "total", BLUE, "Total SR"),
        (0, "drift", ORANGE, "Drift part (net exposure × EW residual)"),
        (-height, "neutral", AQUA, "Dollar-neutral part (stat-arb signal)"),
    ):
        ax.barh(
            y + offset,
            frame[column],
            height=height - 0.04,
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=0,
        )
    for yi, value in zip(y, frame["neutral"]):  # direct labels on the key series only
        ax.text(
            value + (0.02 if value >= 0 else -0.02),
            yi - height,
            f"{value:+.2f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8,
            color=INK,
        )
    ax.set_yticks(y, frame["label"], color=INK, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel(
        "Annualized Sharpe ratio, OOS 2020-02 → 2025-12 (mean over seeds)",
        color=MUTED,
        fontsize=9,
    )
    ax.set_title(
        "Where the Sharpe comes from: residual drift vs stat-arb signal",
        pad=24,
        loc="left",
        color=INK,
        fontsize=12,
    )
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )
    ax.set_xlim(
        min(-0.3, frame[["total", "drift", "neutral"]].min().min() - 0.1),
        frame[["total", "drift", "neutral"]].max().max() + 0.15,
    )
    fig.text(
        0.01,
        0.01,
        "Drift shown as 0 for dollar-neutral books (|net exposure| < 0.05). Source: results/decomposition.csv",
        color=MUTED,
        fontsize=7,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIGURES / "drift_decomposition.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def k_sweep_figure() -> None:
    points = []
    for k, variant in K_SWEEP.items():
        path = RESULTS_DIR / f"ensemble__{variant}__ens" / "metrics.json"
        if path.exists():
            metrics = json.loads(path.read_text())
            points.append((k, metrics["sharpe"], metrics["net_sharpe"]))
    frame = pd.DataFrame(points, columns=["k", "gross", "net"]).sort_values("k")

    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=MUTED, linewidth=1)
    for column, color, name in (
        ("gross", BLUE, "Gross SR"),
        ("net", ORANGE, "Net SR (5 bp + 1 bp)"),
    ):
        ax.plot(
            frame.k,
            frame[column],
            color=color,
            linewidth=2,
            marker="o",
            markersize=7,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
            label=name,
        )
        last = frame.iloc[-1]
        ax.annotate(
            name,
            (last.k, last[column]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=INK,
        )
    peak = frame.loc[frame.gross.idxmax()]
    ax.annotate(
        f"K={int(peak.k)}: {peak.gross:.2f} gross, {peak.net:+.2f} net\n(chosen post hoc: a peak, not a plateau)",
        (peak.k, peak.gross),
        xytext=(frame.k.min(), peak.gross - 0.1),
        textcoords="data",
        fontsize=8,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 0.8},
    )
    ax.set_xticks(frame.k, [str(k) for k in frame.k], color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlim(frame.k.min() - 0.5, frame.k.max() + 2.2)
    ax.set_ylim(min(frame.net.min(), 0) - 0.25, frame.gross.max() + 0.3)
    ax.set_xlabel("Number of PCA factors K", color=MUTED, fontsize=9)
    ax.set_ylabel("Annualized Sharpe ratio", color=MUTED, fontsize=9)
    ax.set_title(
        "Final recipe vs number of factors (dollar-neutral, cw 0.5, 3-seed ensemble, B=5)",
        loc="left",
        color=INK,
        fontsize=11,
    )
    ax.legend(loc="lower left", frameon=False, fontsize=8, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(FIGURES / "k_sweep.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    drift_figure(pd.read_csv(RESULTS_DIR / "decomposition.csv"))
    k_sweep_figure()
    print(f"Wrote {FIGURES}/drift_decomposition.png and k_sweep.png")


if __name__ == "__main__":
    main()
