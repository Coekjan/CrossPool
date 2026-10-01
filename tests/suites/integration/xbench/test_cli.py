from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
from pathlib import Path

import httpx
import psutil
import pytest
import tomli_w
from tests.suites.integration.xbench.harness.serving.test_client import FakeServingServer, client_case

import xbench.cli
from xbench.harness.serving.case import BenchCase, JsonlPrompts, TraceArrivals
from xbench.harness.serving.measure import BenchRunManifest, RequestRecord
from xbench.harness.serving.report import load_series
from xbench.harness.serving.workload import read_jsonl
from xtest.harness.support.config import TEST_MODEL_ID

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def command(outside: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["CUDA_MPS_PIPE_DIRECTORY"] = str(outside / "missing-mps-controller")
    return subprocess.run(
        ["uv", "run", "--project", str(REPOSITORY_ROOT), "--no-sync", "xbench", *arguments],
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
                    case.id: case.model_dump(mode="json", exclude_none=True, exclude={"id"}) for case in cases
                }
            }
        ),
        encoding="utf-8",
    )
    return catalog


def test_installed_cli_outside_checkout_replays_repetitions_and_reports_then_cleans(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with FakeServingServer() as server:
        metadata = tmp_path / "serving.json"
        declared = {
            "schema_version": 1,
            "gpus": [{"uuid": "GPU-remote", "name": "remote GPU", "total_memory_bytes": 85899345920}],
            "target_gpu_uuids": {str(TEST_MODEL_ID): ["GPU-remote"]},
            "packages": {"sglang": "externally-declared"},
        }
        metadata.write_text(json.dumps(declared), encoding="utf-8")
        case = client_case(tmp_path, server.url).model_copy(
            update={"repetitions": 2, "warmup_requests_per_target": 1, "serving_metadata_path": Path("serving.json")}
        )
        catalog = catalog_file(tmp_path, (case,))
        listed = command(outside, "list", "--catalog", str(catalog), "--case", case.id)
        assert listed.returncode == 0, listed.stderr
        assert f"case\tmode=client targets={TEST_MODEL_ID}" in listed.stdout
        root = tmp_path / "runs"
        completed = command(outside, "run", "--catalog", str(catalog), "--result-root", str(root))
        assert completed.returncode == 0, completed.stderr
        run = Path(completed.stdout.strip())
        manifest = BenchRunManifest.model_validate_json((run / "run.json").read_bytes())
        assert manifest.finished and manifest.result_code == 0
        case_directory = run / "cases/case"
        measured_replays = []
        measured_summaries = []
        for number in (1, 2):
            repetition = case_directory / f"repetition-{number:04d}"
            series = load_series(repetition, f"rep{number}")
            assert series.summary.cleanup_verified
            assert series.summary.outcomes["success"] == 3
            assert series.summary.targets["aggregate"].input_tokens == 15
            measured_summaries.append(series.summary.model_dump(mode="json"))
            assert series.environment["local_gpu_inventory"] is None
            assert series.environment["serving_metadata_source"] == "declared"
            assert series.environment["environment_source"] == "local_client"
            serving = series.case_manifest.serving_metadata
            assert serving is not None and serving.target_gpu_uuids == {TEST_MODEL_ID: ("GPU-remote",)}
            assert serving.packages == declared["packages"] and serving.cuda_build_version is None
            software = series.environment["tool_software"]
            assert isinstance(software, dict) and software["source"] == "local_distribution_metadata"
            assert isinstance(software["packages"], dict) and "matplotlib" not in software["packages"]
            assert {"xpool-dev", "xpool"} <= software["packages"].keys()
            assert series.environment["cache_policy"] == {
                "controller": "external",
                "flush_performed": False,
                "serving_state_reset": False,
                "warmup_requests_per_target": 1,
            }
            assert not (repetition / "report").exists()
            checkpoint = json.loads((repetition / "repetition.json").read_bytes())
            assert set(checkpoint["artifact_sha256"]) == {"requests.jsonl", "events.jsonl", "measurement.json"}
            measured_replays.append(
                [json.loads(line)["request_id"] for line in (repetition / "requests.jsonl").read_text().splitlines()]
            )
        assert measured_replays == [["first", "second", "third"]] * 2
        assert len(server.received) == 8  # Independent warmup once before each three-request replay.
        catalog.unlink()
        (tmp_path / "suites/serving/multi_model.py").unlink()
        reported = command(outside, "report", str(run), "--label", "client run")
        assert reported.returncode == 0, reported.stderr
        outputs = tuple(case_directory / f"repetition-{number:04d}/report" for number in (1, 2))
        assert reported.stdout.splitlines() == [str(output) for output in outputs]
        for output, measured_summary in zip(outputs, measured_summaries, strict=True):
            projection = json.loads((output / "summary.json").read_bytes())
            assert (output / "throughput.pdf").is_file()
            assert projection["summary"] == measured_summary
            assert projection["directory"] == str(output.parent)
        dry_run = command(outside, "clean", "--result-root", str(root), "--all", "--dry-run")
        assert dry_run.returncode == 0 and run.exists()
        cleaned = command(outside, "clean", "--result-root", str(root), "--all")
        assert cleaned.returncode == 0 and not run.exists()


@pytest.mark.parametrize("warmup_failure", [False, True])
def test_request_failures_continue_cases_but_warmup_failure_stops_admission(
    tmp_path: Path, warmup_failure: bool
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with FakeServingServer(omit_done=True) as failing, FakeServingServer() as healthy:
        first = client_case(tmp_path, failing.url).model_copy(
            update={"id": "first", "warmup_requests_per_target": int(warmup_failure)}
        )
        second = first.model_copy(
            update={"id": "second", "targets": (first.targets[0].model_copy(update={"base_url": healthy.url}),)}
        )
        catalog = catalog_file(tmp_path, (first, second))
        completed = command(outside, "run", "--catalog", str(catalog), "--result-root", str(tmp_path / "runs"))
        assert completed.returncode == (2 if warmup_failure else 1), completed.stderr
        run = Path(completed.stdout.strip())
        repetition = run / "cases/first/repetition-0001"
        summary = load_series(repetition, "first").summary
        assert summary.cleanup_verified
        if warmup_failure:
            assert summary.window_end_seconds is None and summary.outcomes["not_sent"] == 3
            assert not (repetition / "measurement.json").exists()
            assert healthy.received == [] and not (run / "cases/second").exists()
        else:
            assert summary.outcomes["failed"] == 3 and summary.execution_complete
            assert healthy.received == ["first", "second", "third"]


def test_empty_schedule_and_catalog_listing_do_not_start_requests_or_invent_t0(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    with FakeServingServer() as server:
        case = client_case(tmp_path, server.url).model_copy(update={"warmup_requests_per_target": 1})
        assert isinstance(case.arrivals, TraceArrivals)
        case.arrivals.path.write_text("", encoding="utf-8")
        catalog = catalog_file(tmp_path, (case,))
        completed = command(outside, "run", "--catalog", str(catalog), "--result-root", str(tmp_path / "runs"))
        assert completed.returncode == 1, completed.stderr
        repetition = Path(completed.stdout.strip()) / "cases/case/repetition-0001"
        summary = load_series(repetition, "empty").summary
        assert summary.execution_complete and not summary.measurement_available
        assert summary.window_end_seconds is None and not (repetition / "measurement.json").exists()
        assert server.received == []
    # Inventory resolves declarations even with deliberately unavailable datasets.
    assert isinstance(case.targets[0].prompts, JsonlPrompts)
    case.targets[0].prompts.path.unlink()
    listed = command(outside, "list", "--catalog", str(catalog))
    assert listed.returncode == 0, listed.stderr


def test_source_collection_defers_body_and_worker_uses_external_roots_and_invocation_cwd(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    case = client_case(tmp_path, "http://127.0.0.1:1")
    catalog = catalog_file(tmp_path, (case,))
    (tmp_path / "local_resources.py").write_text(
        "from xkit import ResourceRequirements\n"
        "def resources(case):\n    return ResourceRequirements(0, False, False, ())\n",
        encoding="utf-8",
    )
    (tmp_path / "suites/serving/multi_model.py").write_text(
        "import json\nfrom pathlib import Path\nimport xbench\nfrom local_resources import resources\n"
        "@xbench.parameterize('case')\n@xbench.requirements(resources)\n"
        "def bench_external(case, workdir):\n"
        "    (workdir / 'invocation.json').write_text(json.dumps({'cwd': str(Path.cwd()), 'case': case.id}))\n"
        "    raise RuntimeError('intentional source failure')\n",
        encoding="utf-8",
    )
    listed = command(outside, "list", "--catalog", str(catalog))
    assert listed.returncode == 0 and "gpus=0" in listed.stdout, listed.stderr
    assert not tuple(tmp_path.rglob("invocation.json"))
    root = tmp_path / "runs"
    completed = command(outside, "run", "--catalog", str(catalog), "--result-root", str(root))
    assert completed.returncode == 2, completed.stderr
    run = Path(completed.stdout.strip())
    repetition = run / "cases/case/repetition-0001"
    assert json.loads((repetition / "invocation.json").read_bytes()) == {"cwd": str(outside), "case": "case"}
    assert "intentional source failure" in (repetition / "logs/worker.log").read_text()
    assert load_series(repetition, "failed").summary.cleanup_verified


@pytest.mark.parametrize("checkpoint", [False, True])
def test_client_source_requirements_fail_before_worker_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: bool
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
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
    listed = command(outside, "list", "--catalog", str(catalog))
    assert listed.returncode == 0, listed.stderr
    completed = command(outside, "run", "--catalog", str(catalog), "--result-root", str(tmp_path / "runs"))
    assert completed.returncode == 2, completed.stderr
    repetition = Path(completed.stdout.strip()) / "cases/case/repetition-0001"
    assert not (repetition / "executed").exists()
    series = load_series(repetition, "missing requirement")
    assert series.summary.infrastructure_error is not None
    assert ("weight directory" if checkpoint else "set XPOOL_CONFIG") in series.summary.infrastructure_error


@pytest.mark.parametrize(
    "source",
    [
        "def helper(): pass\n",
        "import xbench\n@xbench.parameterize('case')\ndef one(case, workdir): pass\n"
        "@xbench.parameterize('case')\ndef two(case, workdir): pass\n",
    ],
)
def test_invalid_source_entries_fail_inventory_and_run_before_repetition_allocation(
    tmp_path: Path, source: str
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    catalog = catalog_file(tmp_path, (client_case(tmp_path, "http://127.0.0.1:1"),))
    (tmp_path / "suites/serving/multi_model.py").write_text(source, encoding="utf-8")
    root = tmp_path / "runs"
    for arguments in (("list",), ("run", "--result-root", str(root))):
        result = command(outside, *arguments, "--catalog", str(catalog))
        assert result.returncode == 2 and "expected one catalogue-bound entry" in result.stderr
    assert not root.exists()


def test_completion_marker_failure_preserves_final_measurement_and_allows_incomplete_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_touch = Path.touch

    def touch(path: Path, mode: int = 0o666, exist_ok: bool = True) -> None:
        if path.name == ".completed":
            raise OSError("completion marker unavailable")
        original_touch(path, mode=mode, exist_ok=exist_ok)

    monkeypatch.setattr(Path, "touch", touch)
    root = tmp_path / "runs"
    with FakeServingServer() as server:
        catalog = catalog_file(tmp_path, (client_case(tmp_path, server.url),))
        assert xbench.cli.main(["run", "--catalog", str(catalog), "--result-root", str(root)]) == 2
    run = next(path for path in root.iterdir() if path.is_dir())
    assert BenchRunManifest.model_validate_json((run / "run.json").read_bytes()).result_code == 0
    before = {path.relative_to(run): path.read_bytes() for path in run.rglob("*") if path.is_file()}
    output = run / "cases/case/repetition-0001/report"
    assert xbench.cli.main(["report", str(run)]) == 0
    series = json.loads((output / "summary.json").read_bytes())
    assert series["original_result_code"] == 0 and series["summary"]["outcomes"]["success"] == 3
    assert series["summary"]["measurement_available"] and not series["summary"]["evidence_complete"]
    assert before == {
        path.relative_to(run): path.read_bytes()
        for path in run.rglob("*")
        if path.is_file() and not path.is_relative_to(output)
    }


@pytest.mark.parametrize("worker_loss", [False, True])
def test_interrupted_cli_retains_outcomes_and_leaves_external_server_alive(tmp_path: Path, worker_loss: bool) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "runs"
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
                    "xbench",
                    "run",
                    "--catalog",
                    str(catalog),
                    "--result-root",
                    str(root),
                ],
                cwd=outside,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES="", CUDA_MPS_PIPE_DIRECTORY=str(outside / "missing-mps")),
                stdout=log,
                stderr=log,
                text=True,
            )
            try:
                assert server.first_started.wait(30), (tmp_path / "cli.log").read_text()
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
        repetition = run / "cases/case/repetition-0001"
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
        reported = command(outside, "report", str(run))
        assert reported.returncode == 0, reported.stderr
        projection = json.loads((repetition / "report/summary.json").read_bytes())["summary"]
        assert projection == summary.model_dump(mode="json")
