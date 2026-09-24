"""Push the code, residuals and secrets as private Kaggle datasets, then push the kernel.

Code: ``experiments/`` and ``src/``. Residuals: ``data/full/*.npz``. Secrets: the W&B key
from the environment or ``.env``, if present.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = Path(__file__).resolve().parent

CODE_DIRS = ["experiments", "src"]
CODE_FILES = ["README.md"]


def _load_api():
    """Load and authenticate the Kaggle API client."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _dataset_id(metadata_file: Path) -> str:
    with open(metadata_file) as f:
        return json.load(f)["id"]


def _wait_dataset_ready(api, dataset_id: str, timeout: int = 180) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            status = api.dataset_status(dataset_id)
        except Exception:
            status = None
        if status == "ready":
            return True
        time.sleep(5)
    return False


def _push_dataset(
    api, staging_dir: Path, dataset_id: str, title: str = "dataset"
) -> None:
    print(f"Pushing {title} ({dataset_id}) from {staging_dir} ...")
    try:
        api.dataset_status(dataset_id)
        exists = True
    except Exception:
        exists = False

    if exists:
        resp = api.dataset_create_version(
            str(staging_dir),
            version_notes=f"Automated push at {time.strftime('%Y-%m-%d %H:%M:%S')}",
            quiet=False,
            dir_mode="zip",
        )
    else:
        resp = api.dataset_create_new(
            str(staging_dir),
            public=False,
            quiet=False,
            dir_mode="zip",
        )
    print(resp)

    if not _wait_dataset_ready(api, dataset_id):
        raise RuntimeError(
            f"Dataset {dataset_id} never reported 'ready' after push. "
            "Please check the Kaggle web dashboard."
        )


def _write_git_snapshot(staging_dir: Path) -> None:
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        diff = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "diff", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        (staging_dir / "GIT_COMMIT_AT_PUSH.txt").write_text(commit)
        (staging_dir / "GIT_DIFF_AT_PUSH.patch").write_text(diff)
    except Exception as e:
        print(f"Warning: could not capture git state ({e}); skipping snapshot.")


def stage_code_dataset() -> Path:
    staging_dir = Path(tempfile.mkdtemp(prefix="dlsa_code_"))

    for d in CODE_DIRS:
        src = REPO_ROOT / d
        if src.is_dir():
            dst = staging_dir / d
            shutil.copytree(
                src,
                dst,
                ignore=shutil.ignore_patterns(
                    "__pycache__", "*.pyc", ".ruff_cache", "*.npz"
                ),
            )

    for f in CODE_FILES:
        src = REPO_ROOT / f
        if src.is_file():
            shutil.copy2(src, staging_dir / f)

    _write_git_snapshot(staging_dir)
    shutil.copy2(
        KAGGLE_DIR / "dataset-metadata.json",
        staging_dir / "dataset-metadata.json",
    )
    return staging_dir


def stage_residuals_dataset() -> Path | None:
    full_data_dir = REPO_ROOT / "data" / "full"
    if not full_data_dir.is_dir() or not any(full_data_dir.glob("*.npz")):
        print(
            f"No .npz files found under {full_data_dir}. Skipping residuals dataset push."
        )
        return None

    staging_dir = Path(tempfile.mkdtemp(prefix="dlsa_residuals_"))
    for npz in full_data_dir.glob("*.npz"):
        shutil.copy2(npz, staging_dir / npz.name)

    shutil.copy2(
        KAGGLE_DIR / "data-metadata.json",
        staging_dir / "dataset-metadata.json",
    )
    return staging_dir


def stage_secrets_dataset() -> Path | None:
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        env_file = REPO_ROOT / ".env"
        if env_file.is_file():
            for line in env_file.read_text().splitlines():
                if line.startswith("WANDB_API_KEY="):
                    key = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break

    if not key:
        return None

    staging_dir = Path(tempfile.mkdtemp(prefix="dlsa_secrets_"))
    (staging_dir / "secrets.json").write_text(json.dumps({"WANDB_API_KEY": key}))
    shutil.copy2(
        KAGGLE_DIR / "secrets-dataset-metadata.json",
        staging_dir / "dataset-metadata.json",
    )
    return staging_dir


def main() -> int:
    api = _load_api()

    code_dataset_id = _dataset_id(KAGGLE_DIR / "dataset-metadata.json")
    kernel_metadata_path = KAGGLE_DIR / "kernel-metadata.json"
    with open(kernel_metadata_path) as f:
        kernel_meta = json.load(f)

    code_dir = stage_code_dataset()
    try:
        _push_dataset(api, code_dir, code_dataset_id, title="Code dataset")
    finally:
        shutil.rmtree(code_dir, ignore_errors=True)

    residuals_dir = stage_residuals_dataset()
    if residuals_dir:
        residuals_id = _dataset_id(KAGGLE_DIR / "data-metadata.json")
        try:
            _push_dataset(api, residuals_dir, residuals_id, title="Residuals dataset")
        finally:
            shutil.rmtree(residuals_dir, ignore_errors=True)

    secrets_dir = stage_secrets_dataset()
    if secrets_dir:
        secrets_id = _dataset_id(KAGGLE_DIR / "secrets-dataset-metadata.json")
        try:
            _push_dataset(api, secrets_dir, secrets_id, title="Secrets dataset")
            if secrets_id not in kernel_meta.get("dataset_sources", []):
                kernel_meta.setdefault("dataset_sources", []).append(secrets_id)
                with open(kernel_metadata_path, "w") as f:
                    json.dump(kernel_meta, f, indent=2)
        finally:
            shutil.rmtree(secrets_dir, ignore_errors=True)

    print(f"Pushing kernel from {KAGGLE_DIR} ...")
    api.kernels_push(str(KAGGLE_DIR))

    kernel_id = kernel_meta["id"]
    print("=" * 60)
    print(f"KERNEL PUSHED SUCCESSFULLY: {kernel_id}")
    print(f"Track live execution: https://www.kaggle.com/code/{kernel_id}")
    print(f"To wait and pull outputs when finished:")
    print(f"    python3 kaggle/kaggle_pull_output.py --wait")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
