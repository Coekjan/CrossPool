from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest
import tomli_w
from tests.suites.integration.xbench.support import FakeServingServer, client_case, load_retained_series

import xbench.cli
import xbench.harness.serving.hosted
import xkit.config
import xkit.process
from xbench.harness.serving.case import BenchCatalog, BenchTarget, HostedBenchCase, HostedRequirements, JsonlPrompts
from xbench.harness.serving.hosted import HostedServingAdapter, collect_hosted_launch
from xbench.harness.serving.measure import BenchCaseManifest, RepetitionManifest
from xkit.case import CaseId
from xkit.network import TcpEndpointReservation, TcpPortSpace, reserve_namespace_lock
from xkit.process import OwnedProcessGroup, process_group_has_live_members
from xpool.model import ModelId
from xtest.harness.support.config import TEST_CASE_ID, TEST_MODEL_ID, reset_development_config

pytestmark = pytest.mark.usefixtures(reset_development_config.__name__)

LAUNCHER = r"""
import argparse
import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--settings", type=Path, required=True)
parser.add_argument("--model", action="append", default=[])
args = parser.parse_args()
settings = json.loads(args.settings.read_text())
if args.command == "requirements":
    print(json.dumps({"device_count": 0, "listeners": settings["listeners"]}))
    raise SystemExit(0)

Path(settings["started"]).touch()
if Path(settings["fail"]).exists():
    raise SystemExit("baseline startup failed")
if settings["behavior"] == "wait":
    signal.pause()
models = {}
for item in args.model:
    model, _, address = item.partition("@")
    models.setdefault(address, []).append(model)
if settings["behavior"] == "swapped":
    first, second = models
    models[first], models[second] = models[second], models[first]
if settings["behavior"] == "duplicate":
    first, second = models
    models[second] += models[first]
if settings["behavior"] == "invalid-record":
    Path("launch/commands.json").write_text("{")
elif settings["behavior"] == "commands-only":
    Path("launch/commands.json").write_text(json.dumps({"commands": [{"argv": sys.argv, "environment": {}}]}))
class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if settings["behavior"] == "stall-model-list":
            Path("model-list-started").touch()
            signal.pause()
        payload = json.dumps({"data": [
            {"id": model, "max_model_len": self.server.context_length} for model in self.server.models
        ]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["content-length"])))
        assert self.path == "/v1/completions" and body["model"] in self.server.models
        assert body["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
        if settings["behavior"] == "exit":
            os._exit(42)
        usage = {"completion_tokens": 2, "prompt_tokens": 2}
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for choices in ([{"text": "", "finish_reason": "length"}], []):
            self.wfile.write(("data: " + json.dumps({"choices": choices, "usage": usage}) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")

for address, names in models.items():
    if os.fork() == 0:
        url = urlsplit(address)
        server = ThreadingHTTPServer((url.hostname, url.port), Handler)
        server.models = names
        server.context_length = 3 if settings["behavior"] == "over-limit" else 16
        server.serve_forever()
        raise SystemExit(1)
raise SystemExit(os.waitstatus_to_exitcode(os.wait()[1]) or 1)
"""


def hosted_catalogue(tmp_path: Path, *, behavior: str) -> Path:
    model_ids = (TEST_MODEL_ID, ModelId("test/peer"))
    script = tmp_path / "launch.py"
    script.write_text(f"#!{sys.executable}\n" + LAUNCHER, encoding="utf-8")
    script.chmod(0o755)
    (tmp_path / "prompts.jsonl").write_text('{"prompt_id":"p","input_ids":[1,2]}\n', encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "request_id": str(model_id),
                    "model_id": str(model_id),
                    "arrival_seconds": 0.0,
                    "prompt_id": "p",
                    "max_new_tokens": 2,
                }
            )
            + "\n"
            for model_id in model_ids
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "benches.toml"
    catalog.write_text(
        tomli_w.dumps(
            {
                "serving_cases": {
                    str(TEST_CASE_ID): {
                        "description": "Serve two models through a local launcher.",
                        "mode": "hosted",
                        "script": "launch.py",
                        "launch_settings": {
                            "listeners": [[str(model_id)] for model_id in model_ids],
                            "behavior": behavior,
                            "started": str(tmp_path / "started"),
                            "fail": str(tmp_path / "fail"),
                        },
                        "warmup_requests_per_target": 0,
                        "startup_timeout_seconds": 3.0 if behavior == "stall-model-list" else 20.0,
                        "arrivals": {"kind": "jsonl", "path": "trace.jsonl", "duration_seconds": 0.1},
                        "targets": [
                            {"model_id": str(model_id), "prompts": {"kind": "jsonl", "path": "prompts.jsonl"}}
                            for model_id in model_ids
                        ],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return catalog


def test_list_keeps_cases_when_a_hosted_interpreter_is_unavailable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = hosted_catalogue(tmp_path, behavior="missing-record")
    launcher = tmp_path / "unavailable.py"
    launcher.write_text(f"#!{tmp_path / 'missing-python'}\n", encoding="utf-8")
    launcher.chmod(0o755)
    cases = BenchCatalog.from_file(catalog).cases
    healthy = cases[0]
    unavailable_id = CaseId.generate()
    client = client_case(tmp_path, "http://127.0.0.1:1").model_copy(update={"id": CaseId.generate()})
    catalog.write_text(
        tomli_w.dumps(
            {
                "serving_cases": {
                    str(unavailable_id): {
                        **healthy.model_dump(mode="json", exclude_none=True, exclude={"id"}),
                        "description": "Unavailable launcher.",
                        "script": str(launcher),
                    },
                    **{
                        str(case.id): case.model_dump(mode="json", exclude_none=True, exclude={"id"})
                        for case in (healthy, client)
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    assert xbench.cli.main(["list", "--catalog", str(catalog)]) == 0
    listing = capsys.readouterr()
    assert f"{unavailable_id}\tUnavailable launcher.\n  mode=hosted devices=-" in listing.out
    assert str(healthy.id) in listing.out and "mode=hosted devices=0" in listing.out
    assert str(client.id) in listing.out and "mode=client devices=0" in listing.out
    assert str(unavailable_id) in listing.err and "cannot start" in listing.err
    assert not (tmp_path / "started").exists()

    monkeypatch.setattr(xkit.config, "global_config", None)
    cache = tmp_path / "cache"
    assert (
        xbench.cli.main(["run", "--case", str(unavailable_id), "--catalog", str(catalog), "--cache-root", str(cache)])
        == 2
    )
    assert not (cache / "bench-runs").exists()


@pytest.mark.parametrize(
    ("behavior", "code", "diagnostic"),
    [
        ("missing-record", 0, "launch/commands.json"),
        ("invalid-record", 0, "launch/commands.json"),
        ("commands-only", 0, ""),
        ("swapped", 2, "must appear only at"),
        ("duplicate", 2, "must appear only at"),
        ("stall-model-list", 2, "timed out waiting for hosted listeners and model lists"),
        ("over-limit", 2, "exceeds hosted context limit"),
        ("exit", 2, "hosted serving exited"),
    ],
)
def test_hosted_runs_and_retires_two_listeners(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    behavior: str,
    code: int,
    diagnostic: str,
) -> None:
    catalog = hosted_catalogue(tmp_path, behavior=behavior)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:1")
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    if behavior == "missing-record":
        assert xbench.cli.main(["list", "--catalog", str(catalog)]) == 0
        assert "mode=hosted devices=0" in capsys.readouterr().out
        assert not (tmp_path / "started").exists()
        monkeypatch.setattr(xkit.config, "global_config", None)
    assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "cache")]) == code
    directory = Path(capsys.readouterr().out.strip()) / f"cases/{TEST_CASE_ID}/repetition-0001.attempt-0001"
    series = load_retained_series(directory, behavior)
    assert series.summary.cleanup_verified
    if code == 0:
        assert series.summary.outcomes["success"] == 2
        serving = series.environment["serving"]
        assert isinstance(serving, dict)
        if diagnostic:
            assert serving["commands"] is None
            assert diagnostic in str(series.environment["capture_errors"])
        else:
            commands = serving["commands"]
            assert isinstance(commands, list)
            (command,) = commands
            assert isinstance(command, dict) and command["software"] is None
            argv = command["argv"]
            assert isinstance(argv, list) and argv[0] == str(tmp_path / "launch.py")
            assert not series.environment["capture_errors"]
        endpoints = serving["endpoints"]
        assert isinstance(endpoints, dict)
        assert len({record["base_url"] for record in endpoints.values() if isinstance(record, dict)}) == 2
    else:
        assert diagnostic in (directory / "logs/worker.log").read_text()
        if behavior == "stall-model-list":
            assert (directory / "model-list-started").exists()
        if behavior == "over-limit":
            assert "serving" in series.environment
            assert not (directory / "measurement.json").exists()


def test_hosted_cancellation_during_startup_retires_the_launcher(tmp_path: Path) -> None:
    catalog = hosted_catalogue(tmp_path, behavior="wait")
    cache = tmp_path / "cache"
    with (
        (tmp_path / "xbench.log").open("w", encoding="utf-8") as log,
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                "from xbench.cli import main; raise SystemExit(main())",
                "run",
                "--all",
                "--catalog",
                str(catalog),
                "--cache-root",
                str(cache),
            ],
            cwd=tmp_path,
            stdout=log,
            stderr=log,
        ) as process,
    ):
        try:
            while not (tmp_path / "started").exists():
                assert process.poll() is None, (tmp_path / "xbench.log").read_text()
                time.sleep(0.05)
            descendants = psutil.Process(process.pid).children(recursive=True)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=30) == 143, (tmp_path / "xbench.log").read_text()
            assert all(not child.is_running() for child in descendants)
        finally:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
                process.wait(timeout=30)
    run = next(path for path in (cache / "bench-runs").iterdir() if path.is_dir())
    directory = run / f"cases/{TEST_CASE_ID}/repetition-0001.attempt-0001"
    assert RepetitionManifest.from_directory(directory).cleanup_verified
    assert not (directory / "measurement.json").exists()


def test_hosted_continuation_checks_the_script_and_replays_original_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = hosted_catalogue(tmp_path, behavior="missing-record")
    fail = tmp_path / "fail"
    fail.touch()
    assert xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "cache")]) == 2
    run = Path(capsys.readouterr().out.strip())
    case = run / "cases" / str(TEST_CASE_ID)
    checkpoint = case / "case.json"
    saved = checkpoint.read_bytes()
    assert BenchCaseManifest.model_validate_json(saved).hosted_launch is not None
    script = tmp_path / "launch.py"
    original = script.read_bytes()
    script.write_bytes(original + b"\n")
    fail.unlink()
    continuation = ["run", "--all", "--continue", str(run), "--rerun-failure"]
    monkeypatch.setattr(xkit.config, "global_config", None)
    assert xbench.cli.main(continuation) == 2
    assert "script content or requirements changed" in capsys.readouterr().err
    assert checkpoint.read_bytes() == saved
    script.write_bytes(original)
    (tmp_path / "trace.jsonl").unlink()
    monkeypatch.setattr(xkit.config, "global_config", None)
    assert xbench.cli.main(continuation) == 0
    first = RepetitionManifest.from_directory(case / "repetition-0001.attempt-0001")
    latest = RepetitionManifest.from_directory(case / "repetition-0001.attempt-0002")
    assert first.result_code == 2 and latest.result_code == 0 and latest.cleanup_verified


def test_client_continuation_reuses_completed_hosted_case_without_its_script(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = hosted_catalogue(tmp_path, behavior="missing-record")
    hosted = BenchCatalog.from_file(catalog).cases[0]
    assert isinstance(hosted, HostedBenchCase)
    with FakeServingServer(omit_done=True) as peer:
        client = client_case(tmp_path, peer.url).model_copy(update={"id": CaseId.generate()})
        catalog.write_text(
            tomli_w.dumps(
                {
                    "serving_cases": {
                        str(case.id): case.model_dump(mode="json", exclude_none=True, exclude={"id"})
                        for case in (hosted, client)
                    }
                }
            ),
            encoding="utf-8",
        )
        assert (
            xbench.cli.main(["run", "--all", "--catalog", str(catalog), "--cache-root", str(tmp_path / "cache")]) == 1
        )
        run = Path(capsys.readouterr().out.strip())
        hosted_case = run / "cases" / str(hosted.id)
        hosted_result = hosted_case / "repetition-0001.attempt-0001"
        retained = RepetitionManifest.from_directory(hosted_result)
        assert retained.result_code == 0 and retained.cleanup_verified
        case_manifest = (hosted_case / "case.json").read_bytes()
        hosted.script.unlink()
        peer.omit_done = False

        monkeypatch.setattr(xkit.config, "global_config", None)
        assert xbench.cli.main(["run", "--all", "--continue", str(run), "--rerun-failure"]) == 0
        client_result = run / "cases" / str(client.id) / "repetition-0001.attempt-0002"
        assert RepetitionManifest.from_directory(client_result).result_code == 0
        assert peer.received == ["first", "second", "third"] * 2
        assert (hosted_case / "case.json").read_bytes() == case_manifest


@pytest.mark.parametrize("cancelled", [False, True])
def test_requirements_timeout_retires_helper_after_launcher_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool
) -> None:
    case = BenchCatalog.from_file(hosted_catalogue(tmp_path, behavior="missing-record")).cases[0]
    assert isinstance(case, HostedBenchCase)
    marker = tmp_path / "helper.pid"
    case.script.write_text(
        f"#!{sys.executable}\n"
        "import os, signal\n"
        "from pathlib import Path\n"
        "reader, writer = os.pipe()\n"
        "if os.fork() == 0:\n"
        "    os.close(reader)\n"
        "    signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "    signal.alarm(15)\n"
        f"    Path({str(marker)!r}).write_text(f'{{os.getpid()}} {{os.getpgrp()}}')\n"
        "    os.write(writer, b'1')\n"
        "    os.close(writer)\n"
        "    signal.pause()\n"
        "else:\n"
        "    os.close(writer)\n"
        "    os.read(reader, 1)\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(xkit.process, "PROCESS_TERMINATE_TIMEOUT_SECONDS", 0.1)
    if cancelled:
        terminate = OwnedProcessGroup.terminate

        def signal_then_terminate(owner: OwnedProcessGroup) -> None:
            signal.raise_signal(signal.SIGTERM)
            terminate(owner)

        monkeypatch.setattr(OwnedProcessGroup, "terminate", signal_then_terminate)
    try:
        with pytest.raises(SystemExit if cancelled else TimeoutError) as caught:
            collect_hosted_launch(case, settings_path=tmp_path / "settings.json", deadline=time.monotonic() + 2)
        if isinstance(caught.value, SystemExit):
            assert caught.value.code == 143
        else:
            assert "requirements timed out" in str(caught.value)
        _, group = map(int, marker.read_text().split())
        assert not process_group_has_live_members(group)
    finally:
        if marker.is_file():
            child, group = map(int, marker.read_text().split())
            with suppress(ProcessLookupError):
                if os.getpgid(child) == group:
                    os.kill(child, signal.SIGKILL)


@pytest.mark.parametrize("retiring", [False, True])
def test_hosted_releases_listeners_when_startup_is_cancelled_or_the_script_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retiring: bool
) -> None:
    case = BenchCatalog.from_file(hosted_catalogue(tmp_path, behavior="missing-record")).cases[0]
    assert isinstance(case, HostedBenchCase)
    launch = collect_hosted_launch(case, settings_path=tmp_path / "settings.json", deadline=time.monotonic() + 10)
    attempt = tmp_path / "attempt"
    (attempt / "launch").mkdir(parents=True)
    (attempt / "logs").mkdir()
    adapter = HostedServingAdapter(case, launch, [])
    if retiring:
        root = SimpleNamespace(
            sealed=False, activate=lambda: None, register_scope=lambda: SimpleNamespace(complete=lambda: None)
        )
        monkeypatch.setattr(xbench.harness.serving.hosted, "get_task_root", lambda: root)
        reserve = adapter.reserve_listeners

        def reserve_then_seal(deadline: float) -> None:
            reserve(deadline)
            root.sealed = True

        monkeypatch.setattr(adapter, "reserve_listeners", reserve_then_seal)
    else:
        case.script.write_bytes(case.script.read_bytes() + b"\n")

    async def start() -> None:
        async with adapter.start(workdir=attempt, startup_timeout_seconds=10):
            pytest.fail("the attempt started its script")

    with pytest.raises(
        InterruptedError if retiring else ValueError,
        match="startup is retiring" if retiring else "changed since requirements collection",
    ):
        asyncio.run(start())
    assert not (tmp_path / "started").exists()
    for reservation in adapter.reservations:
        TcpEndpointReservation.reserve_exact(*reservation.address).close()


def test_hosted_skips_a_locked_but_tcp_free_port(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case = BenchCatalog.from_file(hosted_catalogue(tmp_path, behavior="missing-record")).cases[0]
    assert isinstance(case, HostedBenchCase)
    deadline = time.monotonic() + 10.0
    declaration = collect_hosted_launch(case, settings_path=tmp_path / "settings.json", deadline=deadline)
    adapter = HostedServingAdapter(case, declaration, [])
    space = TcpPortSpace.local()
    candidates = tuple(TcpEndpointReservation.reserve("127.0.0.1", port_space=space) for _ in range(3))
    ports = tuple(item.port for item in candidates)
    for item in candidates:
        item.close()
    monkeypatch.setattr(TcpPortSpace, "candidates", lambda self: iter(ports))
    with reserve_namespace_lock(ports[0]):
        try:
            adapter.reserve_listeners(deadline)
            assert tuple(item.port for item in adapter.reservations) == ports[1:]
        finally:
            for item in adapter.reservations:
                item.close()


def test_hosted_rejects_a_listener_outside_its_process_group(tmp_path: Path) -> None:
    case = BenchCatalog.from_file(hosted_catalogue(tmp_path, behavior="missing-record")).cases[0]
    assert isinstance(case, HostedBenchCase)
    declaration = collect_hosted_launch(case, settings_path=tmp_path / "settings.json", deadline=time.monotonic() + 10)
    adapter = HostedServingAdapter(case, declaration, [])
    with FakeServingServer() as peer, reserve_namespace_lock(peer.server.server_port) as lock:
        adapter.reservations.append(TcpEndpointReservation(("127.0.0.1", peer.server.server_port), None, lock))
        owner = OwnedProcessGroup.spawn_logged(
            "hosted-launcher",
            [sys.executable, "-c", "import signal; signal.pause()"],
            cwd=tmp_path,
            env={},
            log_path=tmp_path / "launcher.log",
        )
        try:
            with pytest.raises(RuntimeError, match="outside the supervised process group"):
                adapter.owns_listeners(owner)
            assert peer.thread.is_alive()
        finally:
            owner.terminate()
            owner.close()


@pytest.mark.parametrize("listeners", [[], [[]], [[str(TEST_MODEL_ID), str(TEST_MODEL_ID)]], [["test/missing"]]])
def test_hosted_requirements_partition_the_case_targets(listeners: list[list[str]]) -> None:
    target = BenchTarget(model_id=TEST_MODEL_ID, prompts=JsonlPrompts(kind="jsonl", path=Path("prompts.jsonl")))
    with pytest.raises(ValueError, match="listeners"):
        HostedRequirements.model_validate_json(
            json.dumps({"device_count": 0, "listeners": listeners})
        ).validate_targets((target,))


@pytest.mark.parametrize("fields", [{"device_count": -1}, {"device_count": True}, {"other_resource": 1}])
def test_hosted_requirements_reject_invalid_resource_fields(fields: dict[str, int]) -> None:
    with pytest.raises(ValueError, match=next(iter(fields))):
        HostedRequirements.model_validate_json(
            json.dumps({"device_count": 0, "listeners": [[str(TEST_MODEL_ID)]], **fields})
        )
