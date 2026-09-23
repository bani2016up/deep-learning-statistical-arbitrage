from __future__ import annotations

import json

import numpy as np

from dlsa_baseline.config import ROOT
from dlsa_baseline.training.evaluation import metrics


def main() -> None:
    path = ROOT / "outputs/test_predictions.npz"
    with np.load(path, allow_pickle=False) as data:
        result = metrics(data["returns"], data["weights"])
        start, end = data["dates"][0], data["dates"][-1]
    print(f"Evaluation period: {start} through {end}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
