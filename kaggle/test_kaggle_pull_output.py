import importlib.util
import zipfile
from pathlib import Path

# Loaded by path: this folder's name would shadow the installed ``kaggle`` package.
_spec = importlib.util.spec_from_file_location(
    "kaggle_pull_output", Path(__file__).with_name("kaggle_pull_output.py")
)
pull = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pull)


def _zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return path


def test_merge_keeps_existing_runs_tables_and_logs(tmp_path, monkeypatch):
    results = tmp_path / "results"
    monkeypatch.setattr(pull, "RESULTS_DIR", results)
    (results / "old__run__s0").mkdir(parents=True)
    (results / "old__run__s0" / "metrics.json").write_text("local")
    (results / "summary.csv").write_text("local")
    (results / "kaggle_logs").mkdir()
    (results / "kaggle_logs" / "worker_0.log").write_text("first run")

    archive = _zip(
        tmp_path / "results.zip",
        {
            "old__run__s0/metrics.json": "kernel",
            "new__run__s0/metrics.json": "kernel",
            "summary.csv": "kernel",
            "worker_0.log": "second run",
            "worker_1.log": "second run",
        },
    )
    pull.merge_runs(archive)
    pull.merge_runs(archive)  # pulling twice adds nothing

    assert (results / "old__run__s0" / "metrics.json").read_text() == "local"
    assert (results / "new__run__s0" / "metrics.json").read_text() == "kernel"
    assert (results / "summary.csv").read_text() == "local"
    logs = {p.name: p.read_text() for p in (results / "kaggle_logs").iterdir()}
    assert logs == {
        "worker_0.log": "first run",
        "worker_0_2.log": "second run",
        "worker_1.log": "second run",
    }
