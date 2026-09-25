"""Does the OU threshold rule trade with or against a deterministic trend?

The answer depends on the noise around the trend (docs/THEORY.md, section 8.5). With
mu_hat = mean(X_1..X_{L-1}) + d_bar / (1 - b_hat), mu_hat lands ahead of X_L only when
b_hat >~ 1 - 2/L; stationary noise keeps b_hat low (rule shorts the up-trend), persistent
(random-walk) noise keeps b_hat near 1 (rule follows the trend).

    uv run python scripts/ou_trend_sim.py
"""

from __future__ import annotations

import numpy as np

from dlsa_baseline.models.ou import ou_signal, threshold_positions

LOOKBACK, ASSETS, DAYS = 30, 1500, 330
rng = np.random.default_rng(0)


def ou_level(half_life: float, sd: float) -> np.ndarray:
    b = 0.5 ** (1 / half_life)
    shocks = rng.normal(0, sd * np.sqrt(1 - b**2), (DAYS, ASSETS))
    level = np.zeros((DAYS, ASSETS))
    level[0] = rng.normal(0, sd, ASSETS)
    for i in range(1, DAYS):
        level[i] = b * level[i - 1] + shocks[i]
    return level


def random_walk(sd: float) -> np.ndarray:
    return np.cumsum(rng.normal(0, sd, (DAYS, ASSETS)), axis=0)


def evaluate(level: np.ndarray, drift: float) -> dict[str, float]:
    """Paper rule on every 30-day window; the position earns the next day's change."""
    returns = np.diff(level, axis=0)
    returns[returns == 0] = 1e-12  # zero means "missing" to the OU code
    ahead, with_trend, against, pnl = [], [], [], []
    for t in range(LOOKBACK, len(returns)):
        signal = ou_signal(returns[t - LOOKBACK : t])
        weights = threshold_positions(signal)
        direction = np.sign(drift) if drift else 1.0
        if drift:
            ahead.append((direction * (signal.mu - signal.last) > 0)[signal.valid].mean())
        with_trend.append((weights * direction > 0).mean())
        against.append((weights * direction < 0).mean())
        pnl.append(weights * returns[t])
    pnl = np.array(pnl)
    return {
        "mu_ahead": float(np.mean(ahead)) if drift else float("nan"),
        "with": float(np.mean(with_trend)),
        "against": float(np.mean(against)),
        "pnl_bp": float(pnl.mean() * 1e4),
        "trend_part_bp": float((np.mean(with_trend) - np.mean(against)) * abs(drift) * 1e4),
    }


def main() -> None:
    time = np.arange(DAYS)[:, None]
    setups = {
        "trend + white noise 2%": lambda: rng.normal(0, 0.02, (DAYS, ASSETS)),
        "trend + white noise 1%": lambda: rng.normal(0, 0.01, (DAYS, ASSETS)),
        "trend + random walk 2%": lambda: random_walk(0.02),
        "trend + OU(5d, 2%) + RW 2%": lambda: ou_level(5, 0.02) + random_walk(0.02),
    }
    print(
        f"{'level':28s} {'d/day':>6s} {'mu ahead':>8s} {'with':>5s} {'against':>7s} "
        f"{'PnL bp':>7s} {'trend bp':>8s}"
    )
    for name, noise in setups.items():
        for drift in (0.0025, 0.005, 0.01):
            r = evaluate(drift * time + noise(), drift)
            print(
                f"{name:28s} {drift:6.2%} {r['mu_ahead']:8.0%} {r['with']:5.0%} "
                f"{r['against']:7.0%} {r['pnl_bp']:7.1f} {r['trend_part_bp']:8.1f}"
            )


if __name__ == "__main__":
    main()
