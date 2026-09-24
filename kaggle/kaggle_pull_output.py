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
    status = raw.get("status", raw) if isinstance(raw, dict) else getattr(raw, "status", raw)
    return getattr(status, "name", str(status))


def _save_log(content: bytes, target: Path) -> None:
    """Write a log without overwriting a different one from an earlier run (worker_0_2.log, ...)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    candidate, n = target, 1
    while candidate.exists():
        if candidate.read_bytes() == content:
            return
        n += 1
        candidate = target.with_name(f"{target.stem}_{n}{target.suffix}")
    candidate.write_bytes(content)


def merge_runs(results_zip: Path) -> None:
    """Add the kernel's run folders to results/ without touching local aggregate tables.

    The kernel's summary.csv / decomposition.csv / report_tables.md cover only its own runs;
    extracting them would overwrite the local tables for every other suite. Top-level logs
    go to results/kaggle_logs/ (renamed if a different log of that name exists). Existing
    run folders are left as they are.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    added, skipped = set(), set()
    with zipfile.ZipFile(results_zip, "r") as zf:
        for member in zf.infolist():
            parts = Path(member.filename).parts
            if len(parts) == 1:
                if member.filename.endswith(".log"):
                    _save_log(zf.read(member), RESULTS_DIR / "kaggle_logs" / member.filename)
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


def _with_retries(call, attempts: int = 6, first_delay: float = 10.0):
    """Retry transient API/network failures with exponential backoff (10 s … ~5 min)."""
    for attempt in range(attempts):
        try:
            return call()
        except Exception as exc:  # the Kaggle SDK raises requests/HTTP errors of several types
            if attempt == attempts - 1:
                raise
            delay = first_delay * 2**attempt
            print(f"Kaggle API call failed ({type(exc).__name__}: {exc}); retrying in {delay:.0f}s")
            time.sleep(delay)


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
        status_obj = _with_retries(lambda: api.kernels_status(kernel_id))
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
    _with_retries(lambda: api.kernels_output(kernel_id, path=str(OUTPUT_DIR), force=True))
    print("Download complete.")

    results_zip = OUTPUT_DIR / "results.zip"
    if results_zip.is_file():
        merge_runs(results_zip)

    return 0


if __name__ == "__main__":
    sys.exit(main())
