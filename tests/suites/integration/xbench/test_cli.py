from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from collections.abc import Mapping
from pathlib import Path

import httpx
import psutil
import pytest
import tomli_w
from matplotlib.figure import Figure
from tests.suites.integration.xbench.support import FakeServingServer, client_case

import xbench.cli
import xkit.config
from xbench.harness.serving.case import BenchCase, JsonlPrompts, TraceArrivals
from xbench.harness.serving.measure import BenchCaseManifest, BenchRunManifest, RepetitionManifest, RequestRecord
from xbench.harness.serving.report import list_bench_artifacts, load_series, report_bench_runs
from xbench.harness.serving.workload import read_jsonl
from xkit.case import CaseId
from xkit.results import write_json
from xkit.supervisor import SupervisedTaskScope, TaskStartFailure
from xtest.harness.support.config import TEST_CASE_ID, TEST_MODEL_ID, reset_development_config

pytestmark = pytest.mark.usefixtures(reset_development_config.__name__)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def command(outside: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(outside / "missing-mps-controller")
    return subprocess.run(
        ["uv", "run", "--project", str(REPOSITORY_ROOT), "--no-sync", "--no-env-file", "xbench", *arguments],
        cwd=outside,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )


def catalog_file(tmp_path: Path, cases: tuple[BenchCase, ...]) -> Path:
    catalog = tmp_path / "benches.toml"
    source = tmp_path / "suites/serving/multi_model.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REPOSITORY_ROOT / "benches/suites/serving/multi_model.py", source)
    catalog.write_text(
        tomli_w.dumps(
            {
                "serving_cases": {
                    str(case.id): case.model_dump(mode="json", exclude_none=True, exclude={"id"}) for case in cases
                }
            }
        ),
        encoding="utf-8",
    )
    return catalog


def test_continue_reuses_successful_replay_and_preserves_reportable_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    with FakeServingServer() as healthy, FakeServingServer(omit_done=True) as failing:
        first = client_case(tmp_path, healthy.url)
        second = first.model_copy(
            update={
                "id": CaseId.generate(),
                "targets": (
                    first.targets[0].model_copy(
                        update={"base_url": failing.url.replace("http://", "http://bench:password@")}
                    ),
                ),
            }
        )
        catalog = catalog_file(tmp_path, (first, second))
        assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "runs")]) == 1
        run = Path(capsys.readouterr().out.strip())
        first_attempt = run / f"cases/{first.id}/repetition-0001.attempt-0001"
        old_attempt = run / f"cases/{second.id}/repetition-0001.attempt-0001"
        assert first_attempt.is_dir() and old_attempt.is_dir()
        legacy = first_attempt.parent / "repetition-0001"
        legacy.unlink()
        first_attempt.rename(legacy)
        first_attempt = legacy
        first_manifest = BenchCaseManifest.model_validate_json((legacy.parent / "case.json").read_bytes())
        write_json(
            legacy.parent / "case.json",
            first_manifest.model_copy(update={"repetitions": (legacy.name,)}).model_dump(mode="json"),
        )
        original = {
            path: path.read_bytes()
            for attempt in (first_attempt, old_attempt)
            for path in attempt.rglob("*")
            if path.is_file()
        }
        saved = BenchCaseManifest.model_validate_json((old_attempt.parent / "case.json").read_bytes())
        assert saved.case == second
        assert isinstance(first.targets[0].prompts, JsonlPrompts) and isinstance(first.arrivals, TraceArrivals)
        first.targets[0].prompts.path.unlink()
        first.arrivals.path.unlink()
        manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
        write_json(
            run / "run.json",
            manifest.model_copy(update={"finished": False, "result_code": None}).model_dump(mode="json"),
        )
        (run / ".completed").unlink()
        failing.omit_done = False
        monkeypatch.setenv("XKIT_CONFIG", str(tmp_path / "missing-config.toml"))
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run)]) == 1, capsys.readouterr().err
        capsys.readouterr()
        assert failing.received == ["first", "second", "third"]
        assert not (old_attempt.parent / "repetition-0001.attempt-0002").exists()
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run), "--rerun-failure"]) == 0
        capsys.readouterr()
        assert healthy.received == ["first", "second", "third"]
        assert first_attempt.is_dir() and not first_attempt.is_symlink()
        assert failing.received == ["first", "second", "third"] * 2
        latest = old_attempt.parent / "repetition-0001.attempt-0002"
        assert (latest.parent / "repetition-0001").resolve() == latest
        assert original == {path: path.read_bytes() for path in original}
        assert RepetitionManifest.from_directory(old_attempt).result_code == 1
        assert RepetitionManifest.from_directory(latest).result_code == 0
        assert BenchRunManifest.model_validate_json((run / "run.json").read_bytes()).result_code == 0
        history = f"{run.name}/cases/{second.id}/{old_attempt.name}"
        assert list_bench_artifacts(run.parent) == (run.name, history)
        assert report_bench_runs((run, latest.parent / "repetition-0001", old_attempt, latest)) == (
            first_attempt / "report",
            latest / "report",
            old_attempt / "report",
        )
        before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in run.rglob("*") if path.is_file()}
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run)]) == 0
        assert before == {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before}
        assert len(failing.received) == 6


def test_continue_runs_unattempted_repetitions_without_retrying_interrupted_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeServingServer(omit_done=True) as server:
        case = client_case(tmp_path, server.url)
        catalog = catalog_file(tmp_path, (case,))
        assert (
            xbench.cli.main(
                [
                    "run",
                    "--all",
                    "--catalog",
                    str(catalog),
                    "--cache-root",
                    str(tmp_path / "runs"),
                    "--repetitions",
                    "2",
                    "--fast-fail",
                ]
            )
            == 1
        )
        run = Path(capsys.readouterr().out.strip())
        case_directory = run / "cases" / str(case.id)
        interrupted = case_directory / "repetition-0001.attempt-0001"
        checkpoint = RepetitionManifest.from_directory(interrupted)
        write_json(
            interrupted / "repetition.json",
            checkpoint.model_copy(update={"finished": False, "result_code": None, "cleanup_verified": None}).model_dump(
                mode="json"
            ),
        )
        before = {path: path.read_bytes() for path in interrupted.rglob("*") if path.is_file()}
        server.omit_done = False
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run), "--fast-fail"]) == 2
        capsys.readouterr()
        second = case_directory / "repetition-0002.attempt-0001"
        assert RepetitionManifest.from_directory(second).result_code == 0
        assert not (case_directory / "repetition-0001.attempt-0002").exists()
        assert before == {path: path.read_bytes() for path in before}
        second_before = {path: path.read_bytes() for path in second.rglob("*") if path.is_file()}
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run), "--rerun-failure"]) == 0
        capsys.readouterr()
        latest = case_directory / "repetition-0001.attempt-0002"
        assert RepetitionManifest.from_directory(latest).result_code == 0
        assert (case_directory / "repetition-0001").resolve() == latest
        assert before == {path: path.read_bytes() for path in before}
        assert second_before == {path: path.read_bytes() for path in second_before}
        assert server.received == ["first", "second", "third"] * 3


def test_continue_rejects_incomplete_preparation_and_changed_conditions_without_reopening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url)
        catalog = catalog_file(tmp_path, (case,))
        assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "runs")]) == 0
        run = Path(capsys.readouterr().out.strip())
        replay = run / f"cases/{case.id}/trace.jsonl"
        contents = replay.read_bytes()
        replay.unlink()
        before = (run / "run.json").read_bytes()
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run)]) == 2
        assert "trace.jsonl" in capsys.readouterr().err
        assert (run / "run.json").read_bytes() == before and (run / ".completed").exists()
        replay.write_bytes(contents)
        catalog_file(tmp_path, (case.model_copy(update={"max_inflight": case.max_inflight + 1}),))
        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run)]) == 2
        assert "experiment conditions changed" in capsys.readouterr().err
        assert (run / "run.json").read_bytes() == before and (run / ".completed").exists()
        assert len(server.received) == 3


@pytest.mark.parametrize("override", ["--config", "--catalog", "--repetitions", "--cache-root"])
def test_continue_rejects_configuration_overrides(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], override: str
) -> None:
    assert xbench.cli.main(["run", "--all", "--continue", str(tmp_path / "missing"), override, "2"]) == 2
    assert "exclusive" in capsys.readouterr().err


def test_rerun_failure_requires_continuation(capsys: pytest.CaptureFixture[str]) -> None:
    assert xbench.cli.main(["run", "--all", "--rerun-failure"]) == 2
    assert "requires run --all --continue" in capsys.readouterr().err


@pytest.mark.parametrize("fast_fail", [False, True])
def test_verified_startup_failure_keeps_later_repetitions_unless_fast_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fast_fail: bool
) -> None:
    def start(
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
        log_path: Path,
        timeout_seconds: float | None,
    ) -> SupervisedTaskScope:
        raise TaskStartFailure("startup rollback confirmed empty")

    monkeypatch.setattr(SupervisedTaskScope, "start", start)
    catalog = catalog_file(tmp_path, (client_case(tmp_path, "http://127.0.0.1:1"),))
    arguments = [
        "run",
        "--all",
        "--catalog",
        str(catalog),
        "--cache-root",
        str(tmp_path / "runs"),
        "--repetitions",
        "2",
    ]
    if fast_fail:
        arguments.append("--fast-fail")
    assert xbench.cli.main(arguments) == 2
    run = Path(capsys.readouterr().out.strip())
    first = run / f"cases/{TEST_CASE_ID}/repetition-0001"
    checkpoint = RepetitionManifest.from_directory(first.resolve())
    assert checkpoint.result_code == 2 and checkpoint.cleanup_verified
    following = first.parent / "repetition-0002"
    if fast_fail:
        assert not following.exists()
    else:
        checkpoint = RepetitionManifest.from_directory(following.resolve())
        assert checkpoint.result_code == 2 and checkpoint.cleanup_verified


def test_development_cli_runs_client_repetitions_outside_checkout(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with FakeServingServer() as server:
        metadata_path = tmp_path / "serving.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "devices": [{"uuid": "GPU-remote", "name": "remote device"}],
                    "target_device_uuids": {str(TEST_MODEL_ID): ["GPU-remote"]},
                    "packages": {"sglang": "declared"},
                }
            ),
            encoding="utf-8",
        )
        case = client_case(tmp_path, server.url).model_copy(
            update={"warmup_requests_per_target": 1, "serving_metadata_path": metadata_path}
        )
        catalog = catalog_file(tmp_path, (case,))
        completed = command(
            outside,
            "run",
            "--all",
            "--catalog",
            str(catalog),
            "--cache-root",
            str(tmp_path / "runs"),
            "--repetitions",
            "2",
        )
        assert completed.returncode == 0, completed.stderr
        run = Path(completed.stdout.strip())
        manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
        assert manifest.finished and manifest.result_code == 0
        for number in (1, 2):
            repetition = run / f"cases/{TEST_CASE_ID}" / f"repetition-{number:04d}"
            series = load_series(repetition, f"rep{number}")
            assert series.summary.cleanup_verified and series.summary.outcomes["success"] == 3
            assert not (repetition / "report").exists()
        assert series.case_manifest.serving_metadata == case.load_serving_metadata()
        environment = series.environment
        assert environment["serving_metadata_source"] == "declared"
        assert environment["environment_source"] == "local_client"
        assert environment["local_device_inventory"] is None
        software = environment["tool_software"]
        assert isinstance(software, dict) and software["source"] == "local_distribution_metadata"
        assert environment["cache_policy"] == {
            "controller": "external",
            "flush_performed": False,
            "serving_state_reset": False,
            "warmup_requests_per_target": 1,
        }
        assert len(server.received) == 8  # One warmup precedes each three-request replay.


@pytest.mark.parametrize("fast_fail", [False, True])
def test_request_and_warmup_failures_preserve_active_case_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fast_fail: bool
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with (
        FakeServingServer(omit_done=True) as failing,
        FakeServingServer(block_first=True, omit_done=True) as warmup_failure,
        FakeServingServer() as healthy,
    ):
        first = client_case(tmp_path, failing.url).model_copy(
            update={"id": CaseId.generate(), "warmup_requests_per_target": 0}
        )
        second = first.model_copy(
            update={
                "id": CaseId.generate(),
                "warmup_requests_per_target": 1,
                "targets": (first.targets[0].model_copy(update={"base_url": warmup_failure.url}),),
            }
        )
        third = first.model_copy(
            update={
                "id": CaseId.generate(),
                "targets": (first.targets[0].model_copy(update={"base_url": healthy.url}),),
            }
        )
        catalog = catalog_file(tmp_path, (first, second, third))
        warmup_failure.gate = healthy.first_started
        arguments = [
            "run",
            "--all",
            "--catalog",
            str(catalog),
            "--cache-root",
            str(tmp_path / "runs"),
            "--repetitions",
            "2",
        ]
        if fast_fail:
            arguments.append("--fast-fail")
        assert xbench.cli.main(arguments) == 2
        assert warmup_failure.gate_released
        run = Path(capsys.readouterr().out.strip())
        manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
        assert manifest.finished and manifest.result_code == 2
        measured = load_series(run / f"cases/{first.id}/repetition-0001", "first")
        assert measured.repetition_manifest.result_code == 1
        assert measured.summary.cleanup_verified and measured.summary.execution_complete
        assert measured.summary.outcomes["failed"] == 3
        repetition = run / f"cases/{second.id}/repetition-0001"
        warmup = load_series(repetition, "second")
        assert warmup.repetition_manifest.result_code == 2
        assert warmup.summary.cleanup_verified and warmup.summary.window_end_seconds is None
        assert warmup.summary.outcomes["not_sent"] == 3
        assert not (repetition / "measurement.json").exists()
        completed = load_series(run / f"cases/{third.id}/repetition-0001", "third")
        assert completed.repetition_manifest.result_code == 0
        assert completed.summary.cleanup_verified and completed.summary.execution_complete
        assert completed.summary.outcomes["success"] == 3
        assert healthy.received == ["first", "second", "third"] * (1 if fast_fail else 2)
        for case in (first, second, third):
            following = run / f"cases/{case.id}/repetition-0002"
            if fast_fail:
                assert not following.exists()
            else:
                assert RepetitionManifest.from_directory(following.resolve()).result_code == (
                    1 if case.id == first.id else 2 if case.id == second.id else 0
                )


def test_empty_schedule_does_not_start_requests_or_invent_t0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url).model_copy(update={"warmup_requests_per_target": 1})
        assert isinstance(case.arrivals, TraceArrivals)
        case.arrivals.path.write_text("", encoding="utf-8")
        catalog = catalog_file(tmp_path, (case,))
        assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "runs")]) == 1
        repetition = Path(capsys.readouterr().out.strip()) / f"cases/{TEST_CASE_ID}/repetition-0001"
        summary = load_series(repetition, "empty").summary
        assert summary.execution_complete and not summary.measurement_available
        assert summary.window_end_seconds is None and not (repetition / "measurement.json").exists()
        assert server.received == []


def test_source_collection_defers_body_and_worker_uses_external_roots_and_invocation_cwd(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    case = client_case(tmp_path, "http://127.0.0.1:1")
    catalog = catalog_file(tmp_path, (case,))
    (tmp_path / "local_resources.py").write_text(
        "from xkit import ResourceRequirements\ndef resources(case):\n    return ResourceRequirements(0, False, ())\n",
        encoding="utf-8",
    )
    (tmp_path / "suites/serving/multi_model.py").write_text(
        "import json\nfrom pathlib import Path\nimport xbench\nfrom local_resources import resources\n"
        "@xbench.parameterize('case')\n@xbench.requirements(resources)\n"
        "def bench_external(case, workdir):\n"
        "    (workdir / 'invocation.json').write_text(json.dumps({'cwd': str(Path.cwd()), 'case': str(case.id)}))\n"
        "    raise RuntimeError('intentional source failure')\n",
        encoding="utf-8",
    )
    assert isinstance(case.targets[0].prompts, JsonlPrompts)
    prompt_data = case.targets[0].prompts.path.read_bytes()
    case.targets[0].prompts.path.unlink()
    listed = command(outside, "list", "--catalog", str(catalog))
    assert listed.returncode == 0 and "devices=0" in listed.stdout, listed.stderr
    assert not tuple(tmp_path.rglob("invocation.json"))
    case.targets[0].prompts.path.write_bytes(prompt_data)
    root = tmp_path / "runs/bench-runs"
    completed = command(outside, "run", "--all", "--catalog", str(catalog), "--cache-root", str(root.parent))
    assert completed.returncode == 2, completed.stderr
    run = Path(completed.stdout.strip())
    repetition = run / f"cases/{TEST_CASE_ID}/repetition-0001"
    assert json.loads((repetition / "invocation.json").read_bytes()) == {"cwd": str(outside), "case": str(TEST_CASE_ID)}
    assert "intentional source failure" in (repetition / "logs/worker.log").read_text()
    assert load_series(repetition, "failed").summary.cleanup_verified


@pytest.mark.parametrize("checkpoint", [False, True])
def test_client_source_requirements_fail_before_worker_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], checkpoint: bool
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    catalog = catalog_file(tmp_path, (client_case(tmp_path, "http://127.0.0.1:1"),))
    (tmp_path / "suites/serving/multi_model.py").write_text(
        "import xbench\nfrom xpool.model import ModelId\n"
        "@xbench.parameterize('case')\n"
        + (
            f"@xbench.requirements(requires_config=True, model_ids=(ModelId({str(TEST_MODEL_ID)!r}),))\n"
            if checkpoint
            else "@xbench.requirements(requires_config=True)\n"
        )
        + "def bench_required(case, workdir):\n"
        "    (workdir / 'executed').touch()\n",
        encoding="utf-8",
    )
    if checkpoint:
        config = tmp_path / "runtime.toml"
        config.write_text(
            tomli_w.dumps(
                {
                    "vendor": {"model_base_uri": str(tmp_path / "missing-models")},
                    "atn": {"devices": [0]},
                    "ffn": {"devices": [1]},
                    "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                    "models": [{"id": str(TEST_MODEL_ID)}],
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("XPOOL_CONFIG", str(config))
    else:
        monkeypatch.delenv("XPOOL_CONFIG", raising=False)
    monkeypatch.delenv("UV_ENV_FILE", raising=False)
    assert (
        xbench.cli.main(
            ["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "runs"), "--repetitions", "2"]
        )
        == 2
    )
    repetition = Path(capsys.readouterr().out.strip()) / f"cases/{TEST_CASE_ID}/repetition-0001"
    assert not (repetition / "executed").exists()
    series = load_series(repetition, "missing requirement")
    assert series.summary.infrastructure_error is not None
    assert ("weight directory" if checkpoint else "set XPOOL_CONFIG") in series.summary.infrastructure_error
    following = repetition.parent / "repetition-0002"
    assert not (following / "executed").exists()
    assert RepetitionManifest.from_directory(following.resolve()).result_code == 2


@pytest.mark.parametrize(
    "operation,source",
    [
        ("list", "def helper(): pass\n"),
        (
            "run",
            "import xbench\n@xbench.parameterize('case')\ndef one(case, workdir): pass\n"
            "@xbench.parameterize('case')\ndef two(case, workdir): pass\n",
        ),
    ],
)
def test_invalid_source_entries_fail_inventory_and_run_before_repetition_allocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], operation: str, source: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    monkeypatch.chdir(outside)
    catalog = catalog_file(tmp_path, (client_case(tmp_path, "http://127.0.0.1:1"),))
    (tmp_path / "suites/serving/multi_model.py").write_text(source, encoding="utf-8")
    root = tmp_path / "runs/bench-runs"
    arguments = [operation, "--catalog", str(catalog)]
    if operation == "run":
        arguments.extend(("--all", "--cache-root", str(root.parent)))
    assert xbench.cli.main(arguments) == 2
    assert "expected one catalogue-bound entry" in capsys.readouterr().err
    assert not root.exists()


def test_unsealed_invocation_preserves_measurements_but_is_not_reportable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_touch = Path.touch

    def touch(path: Path, mode: int = 0o666, exist_ok: bool = True) -> None:
        if path.name == ".completed":
            raise OSError("completion marker unavailable")
        original_touch(path, mode=mode, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "touch", touch)
    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    root = tmp_path / "runs/bench-runs"
    with FakeServingServer() as server:
        catalog = catalog_file(tmp_path, (client_case(tmp_path, server.url),))
        assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(root.parent)]) == 2
    run = next(path for path in root.iterdir() if path.is_dir())
    assert BenchRunManifest.model_validate_json((run / "run.json").read_bytes()).result_code == 0
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    output = run / f"cases/{TEST_CASE_ID}/repetition-0001/report"
    monkeypatch.setattr(xkit.config, "global_config", None)
    assert xbench.cli.main(["report", run.name, "--cache-root", str(root.parent)]) == 2
    assert not output.exists()
    series = load_series(output.parent, run.name)
    assert series.repetition_manifest.result_code == 0 and series.summary.outcomes["success"] == 3
    assert series.summary.measurement_available and not series.summary.evidence_complete
    assert before == {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file() and not path.is_relative_to(output)
    }


@pytest.mark.parametrize("worker_loss", [False, True])
def test_interrupted_cli_retains_outcomes_and_leaves_external_server_alive(tmp_path: Path, worker_loss: bool) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "runs/bench-runs"
    with FakeServingServer(block_first=True) as server:
        case = client_case(tmp_path, server.url, future=True)
        catalog = catalog_file(tmp_path, (case,))
        with (tmp_path / "cli.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    "uv",
                    "run",
                    "--project",
                    str(REPOSITORY_ROOT),
                    "--no-sync",
                    "--no-env-file",
                    "xbench",
                    "run",
                    "--all",
                    "--catalog",
                    str(catalog),
                    "--cache-root",
                    str(root.parent),
                ],
                cwd=outside,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_MPS_PIPE_DIRECTORY=str(outside / "missing-mps")),
                stdout=log,
                stderr=log,
                text=True,
            )
            try:
                # The whole-test timeout bounds cold startup before cancellation.
                server.first_started.wait()
                descendants = psutil.Process(process.pid).children(recursive=True)
                if worker_loss:
                    worker = next(child for child in descendants if "xbench.harness.serving.worker" in child.cmdline())
                    worker.kill()
                else:
                    process.send_signal(signal.SIGTERM)
                assert process.wait(timeout=30) == (2 if worker_loss else 143), (tmp_path / "cli.log").read_text()
                assert all(not child.is_running() for child in descendants)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
        run = next(path for path in root.iterdir() if path.is_dir())
        manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
        assert manifest.finished and manifest.result_code == (2 if worker_loss else 143)
        repetition = run / f"cases/{TEST_CASE_ID}/repetition-0001"
        summary = load_series(repetition, "interrupted").summary
        assert summary.cleanup_verified and not summary.execution_complete
        assert sum(summary.outcomes.values()) == 3
        records = {record.request_id: record for record in read_jsonl(repetition / "requests.jsonl", RequestRecord)}
        if worker_loss:
            assert all(record.error_kind == "evidence_missing" for record in records.values())
            assert not summary.evidence_complete
        else:
            assert records["first"].outcome == "cancelled"
            assert records["second"].outcome == records["third"].outcome == "not_sent"
            assert records["third"].enqueued_at_seconds is None
            assert summary.evidence_complete
            assert summary.window_kind == "interrupted"
            assert summary.window_end_seconds is not None and summary.window_end_seconds < 1000.0
        # A benchmark owns its clients, not this independently launched server.
        server.gate.set()
        response = httpx.post(server.url + "/generate", json={"rid": "after-benchmark"}, timeout=5)
        assert response.status_code == 200 and b"[DONE]" in response.content
        assert server.received == ["first", "after-benchmark"]
