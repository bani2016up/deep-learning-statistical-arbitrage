"""Kaggle kernel entry point: stage the code, check the GPUs, run one ``run_suite`` process
per GPU, and zip ``results/`` for the kernel output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Overridable so the runner can be exercised locally against a copy of the inputs.
KAGGLE_INPUT = Path(os.environ.get("KAGGLE_INPUT_DIR", "/kaggle/input"))
KAGGLE_WORKING = Path(os.environ.get("KAGGLE_WORKING_DIR", "/kaggle/working"))
REPO_DIR = KAGGLE_WORKING / "repo"
RESULTS_DIR = KAGGLE_WORKING / "results"

CODE_SLUG = "dlsa-code"
DATA_SLUG = "dlsa-residuals"
SECRETS_SLUG = "dlsa-secrets"
PREV_RESULTS_SLUG = "dlsa-results-prev"

DEFAULT_SUITES = ["full_smoke", "full_probe", "full_paper", "full_recipe"]


def _find_input_dir(slug: str) -> Path:
    """Find an attached dataset by folder name.

    Kaggle has mounted datasets both as /kaggle/input/<slug> and as
    /kaggle/input/datasets/<owner>/<slug>, so search a few levels down, preferring an
    exact name match.
    """
    candidates = []
    for root, dirs, _ in os.walk(KAGGLE_INPUT):
        depth = len(Path(root).relative_to(KAGGLE_INPUT).parts)
        for name in dirs:
            if slug in name:
                candidates.append((name != slug, depth, Path(root) / name))
        if depth >= 3:
            dirs[:] = []
    if candidates:
        return min(candidates)[2]
    tree = sorted(str(p.relative_to(KAGGLE_INPUT)) for p in KAGGLE_INPUT.glob("*/*/*"))
    raise FileNotFoundError(f"No dataset folder matching {slug!r} under {KAGGLE_INPUT}: {tree[:20]}")


def verify_environment() -> int:
    """Verify PyTorch and CUDA availability, returning the number of GPUs."""
    print("=" * 60)
    print("ENVIRONMENT & HARDWARE VERIFICATION")
    print(f"Python version: {sys.version}")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        cuda_ok = torch.cuda.is_available()
        print(f"CUDA available: {cuda_ok}")

        if not cuda_ok:
            print("WARNING: CUDA is NOT available. Running on CPU.")
            return 0

        gpu_count = torch.cuda.device_count()
        print(f"GPU device count: {gpu_count}")
        for i in range(gpu_count):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(
                f"  [GPU {i}]: {name} | Compute Capability: {cap} | Total VRAM: {mem:.1f} GB"
            )

        x = torch.randn(64, 64, device="cuda:0")
        _ = (x @ x).sum().item()
        print("CUDA smoke test passed successfully.")
        return gpu_count
    except Exception as exc:
        print(f"Error during environment verification: {exc}")
        return 0


def stage_code() -> None:
    """Stage code from /kaggle/input into a writable working directory."""
    print("=" * 60)
    print("STAGING CODEBASE")
    code_src = _find_input_dir(CODE_SLUG)
    print(f"Found code source at: {code_src}")

    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    shutil.copytree(code_src, REPO_DIR)
    for archive in REPO_DIR.glob("*.zip"):  # directories are uploaded zipped
        target = REPO_DIR / archive.stem
        if target.is_dir():
            continue
        unpacked = REPO_DIR / f".unpack_{archive.stem}"
        shutil.unpack_archive(archive, unpacked)
        entries = list(unpacked.iterdir())
        # Archives may or may not include the top-level folder itself.
        if len(entries) == 1 and entries[0].is_dir() and entries[0].name == archive.stem:
            entries[0].rename(target)
            unpacked.rmdir()
        else:
            unpacked.rename(target)
    print(f"Code staged to {REPO_DIR}: {sorted(p.name for p in REPO_DIR.iterdir())}")

    os.chdir(REPO_DIR)
    sys.path.insert(0, str(REPO_DIR))
    package_src = REPO_DIR / "src"
    sys.path.insert(0, str(package_src))
    current_pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{REPO_DIR}:{package_src}:{current_pp}"


def discover_secrets() -> None:
    """Set WANDB_API_KEY if available in environment, secrets dataset, or Kaggle Secrets."""
    if os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY found in environment.")
        return

    try:
        sec_dir = _find_input_dir(SECRETS_SLUG)
        sec_file = sec_dir / "secrets.json"
        if sec_file.is_file():
            payload = json.loads(sec_file.read_text())
            key = payload.get("WANDB_API_KEY")
            if key:
                os.environ["WANDB_API_KEY"] = key
                print("WANDB_API_KEY loaded from secrets dataset.")
                return
    except Exception:
        pass

    try:
        from kaggle_secrets import UserSecretsClient

        key = UserSecretsClient().get_secret("WANDB_API_KEY")
        if key:
            os.environ["WANDB_API_KEY"] = key
            print("WANDB_API_KEY loaded from Kaggle Secrets client.")
            return
    except Exception:
        pass

    print("WANDB_API_KEY not found. W&B tracking will run in no-op mode.")


def run_pipeline() -> None:
    """Orchestrate training runs across available GPUs."""
    gpu_count = verify_environment()
    stage_code()
    discover_secrets()

    try:
        data_dir = _find_input_dir(DATA_SLUG)
        print(f"Discovered residuals dataset at: {data_dir}")
        os.environ["DLSA_DATA_DIR"] = str(data_dir)
    except Exception as exc:
        print(f"Note on data discovery: {exc}")
        print("Falling back to local data/ directory if present.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["DLSA_RESULTS_DIR"] = str(RESULTS_DIR)

    # Cross-session resume: restore previous results if attached
    try:
        prev_dir = _find_input_dir(PREV_RESULTS_SLUG)
        print(f"Found previous results dataset at {prev_dir}, restoring...")
        for item in prev_dir.iterdir():
            dst = RESULTS_DIR / item.name
            if item.is_dir() and not dst.exists():
                shutil.copytree(item, dst)
            elif item.is_file() and not dst.exists():
                shutil.copy2(item, dst)
        print("Previous results restored into RESULTS_DIR.")
    except Exception:
        pass

    suites = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_SUITES
    print("=" * 60)
    print(f"LAUNCHING SUITES: {suites}")

    # Kaggle kills sessions at 12 h without saving outputs; stop launching runs at 11 h.
    deadline_ts = time.time() + 11 * 3600
    print(f"Set 11h session deadline guard: {time.ctime(deadline_ts)}")

    num_shards = max(gpu_count, 1)
    processes: list[subprocess.Popen] = []
    log_files: list[Path] = []

    for shard_idx in range(num_shards):
        env = os.environ.copy()
        env["DLSA_DEADLINE"] = str(deadline_ts)
        env["WANDB_SILENT"] = "true"
        if gpu_count > 0:
            env["CUDA_VISIBLE_DEVICES"] = str(shard_idx % gpu_count)
            env["DLSA_DEVICE"] = "cuda"
        else:
            env["DLSA_DEVICE"] = "cpu"

        log_path = RESULTS_DIR / f"worker_{shard_idx}.log"
        log_files.append(log_path)
        log_handle = open(log_path, "w", buffering=1)

        cmd = [
            sys.executable,
            "-u",
            "-m",
            "experiments.run_suite",
            *suites,
            "--shard",
            f"{shard_idx}/{num_shards}",
        ]

        print(
            f"Spawning Worker {shard_idx} (GPU {env.get('CUDA_VISIBLE_DEVICES', 'CPU')}): {' '.join(cmd)}"
        )
        p = subprocess.Popen(
            cmd,
            cwd=str(REPO_DIR),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append(p)

    print("=" * 60)
    print("MONITORING WORKERS...")
    start_time = time.time()

    while True:
        alive = [p.poll() is None for p in processes]
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)

        status_line = (
            f"[{mins:02d}:{secs:02d}] Active workers: {sum(alive)}/{num_shards}"
        )
        print(status_line)

        for idx, lp in enumerate(log_files):
            if lp.is_file() and lp.stat().st_size > 0:
                try:
                    lines = lp.read_text().strip().splitlines()
                    progress = [line for line in lines if line.startswith("[")]
                    if lines:
                        print(f"   Worker {idx}: {(progress or lines)[-1][:110]}")
                except Exception:
                    pass

        if not any(alive):
            break

        time.sleep(60)

    print("=" * 60)
    print("ALL WORKERS FINISHED. CHECKING STATUS...")
    all_success = True
    for idx, p in enumerate(processes):
        code = p.returncode
        print(f"Worker {idx} exited with code: {code}")
        if code != 0:
            all_success = False

    print("=" * 60)
    print("RUNNING POST-AGGREGATIONS...")
    try:
        subprocess.run(
            [sys.executable, "-m", "experiments.decompose"],
            cwd=str(REPO_DIR),
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "experiments.report_tables"],
            cwd=str(REPO_DIR),
            check=False,
        )
    except Exception as exc:
        print(f"Aggregation warning: {exc}")

    print("=" * 60)
    print("PACKAGING RESULTS ARTIFACTS...")
    zip_path = KAGGLE_WORKING / "results"
    shutil.make_archive(str(zip_path), "zip", RESULTS_DIR)
    print(
        f"Created {zip_path}.zip ({os.path.getsize(str(zip_path) + '.zip') / (1024 * 1024):.2f} MB)"
    )

    for fname in ["summary.csv", "decomposition.csv", "report_tables.md"]:
        src = RESULTS_DIR / fname
        if src.is_file():
            shutil.copy2(src, KAGGLE_WORKING / fname)
            print(f"Exposed {fname} in /kaggle/working/")

    print("=" * 60)
    print(f"PIPELINE COMPLETE (success={all_success}).")
    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    run_pipeline()
