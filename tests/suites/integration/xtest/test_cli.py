"""Canonical test-suite command-line behavior."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import psutil
import pytest
from pydantic import JsonValue

import xtest.cli
import xtest.harness.report
import xtest.harness.runner.execution
import xtest.harness.runner.plan
import xtest.harness.runner.selection
from xkit.results import RunStore
from xtest.harness.report import TestResultWriter, TestRunReport
from xtest.harness.runner.ctest import current_build_directory

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture
def source_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\naddopts = ["--import-mode=importlib"]\ntimeout = 15\n', encoding="utf-8"
    )
    (tmp_path / "uv.lock").touch()
    (tmp_path / "CMakeLists.txt").touch()
    (tmp_path / "tests").mkdir()
    shutil.copyfile(REPOSITORY_ROOT / "tests/tests.toml", tmp_path / "tests/tests.toml")
    shutil.copytree(REPOSITORY_ROOT / "configs/deployments", tmp_path / "configs/deployments")
    (tmp_path / "tests/conftest.py").write_text(
        'pytest_plugins = ["xtest.harness.runner.pytest_plugin"]\n', encoding="utf-8"
    )
    suite = tmp_path / "tests/suites/unit"
    suite.mkdir(parents=True)
    (suite / "test_example.py").write_text(
        "from pathlib import Path\nimport pytest\n"
        "@pytest.fixture(autouse=True)\n"
        "def observed_fixture():\n    Path('fixture-ran').touch()\n"
        "@pytest.mark.parametrize('outcome', ['pass', 'fail', 'skip'])\n"
        "def test_result(outcome):\n"
        "    if outcome == 'skip':\n        pytest.skip('declared skip')\n"
        "    assert outcome != 'fail', 'intentional fixture failure'\n",
        encoding="utf-8",
    )
    return tmp_path


def command(cwd: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_MPS_PIPE_DIRECTORY=str(cwd / "missing-mps"))
    return subprocess.run(
        ["uv", "run", "--project", str(REPOSITORY_ROOT), "--no-sync", "xtest", *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("success", [True, False])
def test_cli_lists_then_executes_source_cases_and_reports_original_outcomes(
    source_checkout: Path, success: bool, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    selector = "tests/suites/unit/test_example.py"
    if success:
        assert xtest.cli.main(["list", "--suite", "unit", selector]) == 0
        assert tuple(line.split("\t")[1] for line in capsys.readouterr().out.splitlines()) == tuple(
            f"{selector}::test_result[{outcome}]" for outcome in ("pass", "fail", "skip")
        )
        assert not (source_checkout / "fixture-ran").exists()
        assert not (source_checkout / ".xpool-cache/test-runs").exists()

    selected = f"{selector}::test_result[pass]" if success else selector
    root = source_checkout / "selected-results"
    arguments = ["run", "--suite", "unit", "--strict-requirements", "--result-root", str(root), selected]
    if success:
        completed = command(source_checkout, *arguments)
        assert completed.returncode == 0, completed.stderr
    else:
        assert xtest.cli.main(arguments) == 1
    assert (source_checkout / "fixture-ran").is_file()
    run = next(path for path in root.iterdir() if path.is_dir())
    summary = TestRunReport.load(run).summary()
    assert summary["original_result_code"] == int(not success)
    assert summary["passed"] == 1
    assert summary["failed"] == summary["skipped"] == int(not success)
    assert summary["cleanup_verified"] is True and summary["evidence_complete"] is True
    assert summary["strict_requirements"] is True
    if not success:
        return
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}

    outside = source_checkout / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    output = outside / "report"
    assert xtest.cli.main(["report", str(run), "--output", str(output), "--label", "original"]) == 0
    retained = json.loads((output / "summary.json").read_bytes())["runs"][0]
    assert retained["label"] == "original" and retained["summary"] == summary
    assert before == {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}


def test_list_declares_unavailable_requirements_without_resolving_them(
    source_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = source_checkout / "tests/suites/integration"
    suite.mkdir()
    (suite / "test_resource.py").write_text(
        "import xtest\n"
        "from xpool.model import ModelId\n"
        "@xtest.requirements(device_count=2, requires_config=True, "
        "model_ids=(ModelId('missing/model'),))\n"
        "def test_resource():\n    raise AssertionError('inventory executed a test')\n",
        encoding="utf-8",
    )
    assert xtest.cli.main(["list", "--suite", "integration"]) == 0
    assert "test_resource.py::test_resource\tdevices=2 config=True models=missing/model" in capsys.readouterr().out
    assert not (source_checkout / ".xpool-cache/test-runs").exists()


def test_list_reads_ctest_inventory_and_reports_missing_manifest_without_execution(
    source_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert xtest.cli.main(["list", "--suite", "cext"]) == 2
    assert "no CTest manifest" in capsys.readouterr().err
    build = current_build_directory(source_checkout)
    build.mkdir(parents=True)
    (build / "CTestTestfile.cmake").write_text(
        'add_test("cext.never-run" "cmake" "-E" "touch" "' + str(source_checkout / "native-ran") + '")\n',
        encoding="utf-8",
    )
    assert xtest.cli.main(["list", "--suite", "cext"]) == 0
    assert capsys.readouterr().out.splitlines() == ["cext\tcext.never-run"]
    assert not (source_checkout / "native-ran").exists()
    assert not (source_checkout / ".xpool-cache/test-runs").exists()


def test_list_reports_collection_failure_without_creating_durable_run(
    source_checkout: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (source_checkout / "tests/suites/unit/test_bad.py").write_text(
        "raise RuntimeError('collection unavailable')\n", encoding="utf-8"
    )
    assert xtest.cli.main(["list", "--suite", "unit"]) == 2
    assert "collection unavailable" in capsys.readouterr().err
    assert not (source_checkout / ".xpool-cache/test-runs").exists()
    assert not (source_checkout / "fixture-ran").exists()


@pytest.mark.parametrize("worker_loss", [False, True])
def test_interrupted_cli_retains_original_outcome_and_cleanup_proof(source_checkout: Path, worker_loss: bool) -> None:
    suite = source_checkout / "tests/suites/integration"
    suite.mkdir()
    (suite / "test_interrupt.py").write_text(
        "from pathlib import Path\nimport time\nimport pytest\n"
        "@pytest.mark.timeout(120)\ndef test_interrupt():\n"
        "    Path('body-started').touch()\n    time.sleep(120)\n",
        encoding="utf-8",
    )
    with (source_checkout / "cli.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                "uv",
                "run",
                "--project",
                str(REPOSITORY_ROOT),
                "--no-sync",
                "xtest",
                "run",
                "--suite",
                "integration",
                "--strict-requirements",
            ],
            cwd=source_checkout,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_MPS_PIPE_DIRECTORY=str(source_checkout / "missing-mps")),
            stdout=log,
            stderr=log,
            text=True,
        )
        try:
            deadline = time.monotonic() + 60
            while not (source_checkout / "body-started").is_file() and process.poll() is None:
                assert time.monotonic() < deadline, (source_checkout / "cli.log").read_text()
                time.sleep(0.05)
            assert process.poll() is None, (source_checkout / "cli.log").read_text()
            descendants = psutil.Process(process.pid).children(recursive=True)
            if worker_loss:
                worker = next(child for child in descendants if "xtest.harness.runner.worker" in child.cmdline())
                worker.kill()
            else:
                process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == (2 if worker_loss else 143), (source_checkout / "cli.log").read_text()
            assert all(not child.is_running() for child in descendants)
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
    root = source_checkout / ".xpool-cache/test-runs"
    run = next(path for path in root.iterdir() if path.is_dir())
    summary = TestRunReport.load(run).summary()
    assert summary["original_result_code"] == (2 if worker_loss else 143) and summary["cleanup_verified"] is True
    assert summary["strict_requirements"] is True and summary["passed"] == 0
    assert summary["evidence_complete"] is False
    if not worker_loss:
        outside = source_checkout / "outside"
        outside.mkdir()
        reported = command(outside, "report", str(run), "--output", str(outside / "report"))
        assert reported.returncode == 0, reported.stderr
        assert json.loads((outside / "report/summary.json").read_bytes())["runs"][0]["summary"] == summary


def test_result_checkpoint_failure_returns_infrastructure_error_after_cleanup(
    source_checkout: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write = xtest.harness.report.write_json
    failed = False

    def checkpoint(path: Path, value: JsonValue) -> None:
        nonlocal failed
        # Inject one failed terminal-task write; recovery checkpoints still work.
        if path.name == "results.json" and isinstance(value, dict) and value.get("tasks") and not failed:
            failed = True
            raise OSError("terminal checkpoint unavailable")
        write(path, value)

    monkeypatch.setattr(xtest.harness.report, "write_json", checkpoint)
    code = xtest.cli.main(
        ["run", "--suite", "unit", "--strict-requirements", "tests/suites/unit/test_example.py::test_result[pass]"]
    )
    assert code == 2 and failed
    root = source_checkout / ".xpool-cache/test-runs"
    run = next(path for path in root.iterdir() if path.is_dir())
    report = TestRunReport.load(run)
    assert report.results.finished and report.results.cleanup_verified
    assert report.results.overall_result_code == 2
    assert report.results.infrastructure_error is not None
    assert "terminal checkpoint unavailable" in report.results.infrastructure_error
    assert (run / ".completed").is_file()
    assert xtest.cli.main(["report", str(run), "--output", str(source_checkout / "report")]) == 0
    assert "terminal checkpoint unavailable" in (source_checkout / "report/report.md").read_text()


def test_completion_marker_failure_returns_infrastructure_error_with_readable_original_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_touch = Path.touch

    def touch(path: Path, mode: int = 0o666, exist_ok: bool = True) -> None:
        if path.name == ".completed":
            raise OSError("completion marker unavailable")
        original_touch(path, mode=mode, exist_ok=exist_ok)

    def execute(*args: object, result_writer: TestResultWriter, **kwargs: object) -> int:
        result_writer.cleanup(True)
        return 0

    monkeypatch.setattr(Path, "touch", touch)
    monkeypatch.setattr(xtest.harness.runner.execution, "execute_test_run", execute)
    root = tmp_path / "runs"
    assert xtest.cli.main(["run", "--suite", "unit", "--result-root", str(root)]) == 2
    directory = next(path for path in root.iterdir() if path.is_dir())
    with RunStore(root).read(directory.name) as protected:
        summary = TestRunReport.load(protected).summary()
    assert summary["original_result_code"] == 0 and summary["execution_finished"] is True
    assert summary["evidence_complete"] is False and summary["missing_artifacts"] == [".completed"]


def test_clean_defaults_to_twenty_retained_runs(source_checkout: Path) -> None:
    """Apply the documented retention default only for explicit cleanup."""

    result_root = source_checkout / ".xpool-cache" / "test-runs"
    for index in range(21):
        entry = result_root / f"unrecognized-{index:02d}"
        entry.mkdir(parents=True)
        os.utime(entry, ns=(index + 1, index + 1))

    assert xtest.cli.main(["clean"]) == 0
    assert len(tuple(path for path in result_root.iterdir() if path.name != ".cleanup.lock")) == 20


def test_clean_applies_explicit_dry_run_and_all(source_checkout: Path) -> None:
    result_root = source_checkout / ".xpool-cache" / "test-runs"
    entries = tuple(result_root / f"unrecognized-{index}" for index in range(3))
    for entry in entries:
        entry.mkdir(parents=True)

    assert xtest.cli.main(["clean", "--keep", "1", "--dry-run"]) == 0
    assert all(entry.is_dir() for entry in entries)
    assert xtest.cli.main(["clean", "--all"]) == 0
    assert tuple(path for path in result_root.iterdir() if path.name != ".cleanup.lock") == ()
