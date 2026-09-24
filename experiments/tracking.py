"""Optional Weights & Biases logging for runs.

Every call is a no-op when ``wandb`` is not installed, ``WANDB_API_KEY`` is unset, or
``WANDB_DISABLED`` is set.
"""

from __future__ import annotations

import os
from typing import Any, Mapping

_ACTIVE_RUN: Any = None
_ENABLED: bool = False


def is_available() -> bool:
    """Check if tracking can be enabled."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        return False
    if not (os.environ.get("WANDB_API_KEY") or os.environ.get("WANDB_KEY")):
        return False
    try:
        import wandb  # noqa: F401

        return True
    except ImportError:
        return False


def start_run(
    config: Mapping[str, Any] | Any,
    project: str = "dlsa-full",
    group: str | None = None,
    tags: list[str] | None = None,
) -> None:
    """Initialize a tracking run for an experiment.

    Args:
        config: Dict or RunConfig dataclass/pydantic model.
        project: Target W&B project name.
        group: Optional run group (e.g. suite name).
        tags: Optional run tags.
    """
    global _ACTIVE_RUN, _ENABLED
    if not is_available():
        _ENABLED = False
        return

    try:
        import wandb

        if hasattr(config, "__dict__"):
            cfg_dict = {
                k: v for k, v in config.__dict__.items() if not k.startswith("_")
            }
        elif isinstance(config, Mapping):
            cfg_dict = dict(config)
        else:
            cfg_dict = {"config": str(config)}

        name = cfg_dict.get("name")
        suite = cfg_dict.get("suite") or group

        _ACTIVE_RUN = wandb.init(
            project=os.environ.get("WANDB_PROJECT", project),
            name=name,
            group=suite,
            tags=tags,
            config=cfg_dict,
            reinit=True,
        )
        _ENABLED = True
    except Exception as exc:
        print(
            f"[tracking] Warning: wandb.init failed ({exc}), continuing in no-op mode."
        )
        _ACTIVE_RUN = None
        _ENABLED = False


def log_block(block: int, metrics: Mapping[str, Any]) -> None:
    """Log block-level metrics during walk-forward retraining.

    Args:
        block: Current rolling retraining block index.
        metrics: Dictionary of metric name -> value.
    """
    if not _ENABLED or _ACTIVE_RUN is None:
        return

    try:
        import wandb

        payload = {"block": block, **metrics}
        wandb.log(payload)
    except Exception as exc:
        print(f"[tracking] Warning: log_block failed ({exc}).")


def finish(metrics: Mapping[str, Any] | None = None) -> None:
    """Finalize the active run, logging summary metrics.

    Args:
        metrics: Optional final summary metrics dict.
    """
    global _ACTIVE_RUN, _ENABLED
    if not _ENABLED or _ACTIVE_RUN is None:
        return

    try:
        import wandb

        if metrics:
            for k, v in metrics.items():
                if isinstance(v, (int, float, str, bool)):
                    wandb.run.summary[k] = v
        wandb.finish()
    except Exception as exc:
        print(f"[tracking] Warning: wandb.finish failed ({exc}).")
    finally:
        _ACTIVE_RUN = None
        _ENABLED = False
