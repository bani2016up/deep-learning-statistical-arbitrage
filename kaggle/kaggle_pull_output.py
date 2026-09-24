"""Pull outputs from finished Kaggle run into local results/ directory."""

from __future__ import annotations

import json
import sys
import time
import zipfile
from pathlib import Path

KAGGLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = KAGGLE_DIR.parent
OUTPUT_DIR = REPO_ROOT / "kaggle_output"
RESULTS_DIR = REPO_ROOT / "results"

TERMINAL_STATES = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}


def _status_string(raw: object) -> str:
    status = (
        raw.get("status", raw) if isinstance(raw, dict) else getattr(raw, "status", raw)
    )
    return getattr(status, "name", str(status))


def merge_runs(results_zip: Path) -> None:
    """Add the kernel's run folders to results/ without touching local aggregate tables.

    The kernel's summary.csv / decomposition.csv / report_tables.md cover only its own runs;
    extracting them would overwrite the local tables for every other suite. Top-level logs
    go to results/kaggle_logs/. Existing run folders are left as they are.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    added, skipped = set(), set()
    with zipfile.ZipFile(results_zip, "r") as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if len(parts) == 1:
                if member.filename.endswith(".log"):
                    zf.extract(member, RESULTS_DIR / "kaggle_logs")
                continue
            run = parts[0]
            if run in skipped or (run not in added and (RESULTS_DIR / run).exists()):
                skipped.add(run)
                continue
            added.add(run)
            zf.extract(member, RESULTS_DIR)
    print(f"Added {len(added)} run folders to {RESULTS_DIR}; kept {len(skipped)} existing.")
    print(
        "Rebuild the tables: uv run python -m experiments.postprocess --suite <suite> && "
        "uv run python -m experiments.decompose && uv run python -m experiments.report_tables"
    )


def main() -> int:
    from kaggle.api.kaggle_api_extended import KaggleApi

    metadata_path = KAGGLE_DIR / "kernel-metadata.json"
    with open(metadata_path) as f:
        kernel_id = json.load(f)["id"]

    api = KaggleApi()
    api.authenticate()

    wait = "--wait" in sys.argv

    print(f"Checking status for kernel: {kernel_id}")
    while True:
        status_obj = api.kernels_status(kernel_id)
        state = _status_string(status_obj)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Kernel status: {state}")

        if state in TERMINAL_STATES:
            break
        if not wait:
            print(
                "Kernel is still active. Run with --wait to poll until completion, "
                "or re-run this script later."
            )
            return 0
        time.sleep(30)

    if state != "COMPLETE":
        print(
            f"Warning: Kernel ended with non-complete status '{state}'. Downloading available output anyway."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading kernel output to {OUTPUT_DIR} ...")
    api.kernels_output(kernel_id, path=str(OUTPUT_DIR), force=True)
    print("Download complete.")

    results_zip = OUTPUT_DIR / "results.zip"
    if results_zip.is_file():
        merge_runs(results_zip)

    return 0


if __name__ == "__main__":
    sys.exit(main())
