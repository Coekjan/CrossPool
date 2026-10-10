from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import cast

import httpx
import pytest

import xkit.serving.sglang.server
from xkit.process import OwnedProcessGroup
from xkit.serving.cluster import XpoolClusterLaunch
from xkit.serving.readiness import ReadinessEvidence
from xkit.serving.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import SglangLaunchModel
from xkit.serving.sglang.server import (
    SglangServerProcess,
    server_command,
)
from xpool.config import XpoolConfig
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.mark.parametrize(
    ("model", "enable_dp_attention", "decode_backend", "prefill_backend"),
    [
        (
            SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
            False,
            "disabled",
            "disabled",
        ),
        (
            SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            False,
            "full",
            "breakable",
        ),
        (
            SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE),
            True,
            "full",
            "breakable",
        ),
    ],
)
def test_server_command_projects_pinned_cli_policy(
    model: SglangLaunchModel,
    enable_dp_attention: bool,
    decode_backend: str,
    prefill_backend: str,
    tmp_path: Path,
) -> None:
    dp_size = 2 if enable_dp_attention else 1
    launch = server_launch(tmp_path, dp_size=dp_size)
    command = server_command(
        launch=launch,
        model=model,
        endpoint=SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, dp_size),
    )

    assert command[:5] == ["xpool", "exec", "--", "sglang", "serve"]
    assert command[command.index("--base-gpu-id") + 1] == "0"
    assert command[command.index("--gpu-id-step") + 1] == "1"
    assert "--max-total-tokens" not in command
    assert command[command.index("--nccl-port") + 1] == "19001"
    assert command[command.index("--tensor-parallel-size") + 1] == str(launch.config.atn_world_size)
    assert command[command.index("--data-parallel-size") + 1] == str(dp_size)
    assert command[command.index("--cuda-graph-backend-decode") + 1] == decode_backend
    assert command[command.index("--cuda-graph-backend-prefill") + 1] == prefill_backend
    assert ("--enable-dp-attention" in command) is enable_dp_attention


def test_server_start_prepares_process_environment_and_log_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    launch = replace(
        server_launch(tmp_path),
        environment=MappingProxyType({"SGLANG_GRPC_PORT": "discarded", "XPOOL_TEST_VALUE": "preserved"}),
    )
    model = SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER)
    family = SglangEndpointFamily("127.0.0.1", 19_000, 19_001, 19_002, 1)
    events: list[str] = []
    endpoint = cast(
        SglangEndpointFamilyLease,
        SimpleNamespace(family=family, release_tcp_for_spawn=lambda: events.append("released")),
    )
    captured: dict[str, object] = {}
    captured_command: tuple[str, ...] = ()

    def spawn_logged(
        name: str,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
    ) -> OwnedProcessGroup:
        nonlocal captured_command
        captured_command = tuple(command)
        assert log_path.parent.is_dir()
        events.append("spawned")
        captured.update(name=name, cwd=cwd, env=env, log_path=log_path)
        return cast(OwnedProcessGroup, SimpleNamespace())

    monkeypatch.setattr(OwnedProcessGroup, "spawn_logged", staticmethod(spawn_logged))

    server = SglangServerProcess.start(
        launch=launch,
        model=model,
        endpoint=endpoint,
        workdir=tmp_path / "artifacts",
    )

    assert events == ["released", "spawned"]
    assert captured["env"] == {
        "SGLANG_GRPC_PORT": "19002",
        "SGLANG_PLUGINS": "xpool",
        "XPOOL_TEST_VALUE": "preserved",
    }
    assert launch.environment == {"SGLANG_GRPC_PORT": "discarded", "XPOOL_TEST_VALUE": "preserved"}
    assert server.endpoint is family
    assert captured["cwd"] == launch.cwd
    assert captured["log_path"] == tmp_path / "artifacts/models" / TEST_MODEL_ID.uri_encode() / "server.log"
    assert server.command == captured_command


@pytest.mark.parametrize("status_code", [200, 503, None])
def test_server_health_records_live_http_observations(
    status_code: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: None),
        ),
    )
    server = SglangServerProcess(
        model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
        owner=owner,
        endpoint=family,
    )

    def health_response(*args: object, **kwargs: object) -> httpx.Response:
        if status_code is None:
            raise httpx.ConnectError("not ready")
        return httpx.Response(status_code)

    monkeypatch.setattr(xkit.serving.sglang.server.httpx, "get", health_response)
    evidence = ReadinessEvidence("SGLang health", "http://127.0.0.1:20000/health")

    assert server.healthy(evidence) is (status_code == 200)
    assert evidence.attempt_count == 1
    assert evidence.last_status_code == status_code
    assert evidence.last_error_type == ("ConnectError" if status_code is None else None)


def test_server_health_records_early_exit_before_raising() -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: 1),
            tail=lambda: "ValueError: metrics_port at 20237 is not available in 30 seconds.",
        ),
    )
    server = SglangServerProcess(
        model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
        owner=owner,
        endpoint=family,
    )
    evidence = ReadinessEvidence("SGLang health", "http://127.0.0.1:20000/health")

    with pytest.raises(RuntimeError, match="exited before readiness with code 1"):
        server.healthy(evidence)

    assert evidence.attempt_count == 1
    assert evidence.last_error_type == "RuntimeError"
    assert evidence.last_error_message is not None
    assert "exited before readiness with code 1" in evidence.last_error_message


def test_server_close_owns_only_process_resources(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=SimpleNamespace(poll=lambda: 0),
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
        owner=owner,
        endpoint=family,
    )
    monkeypatch.setattr(xkit.serving.sglang.server, "wait_for_process_group", lambda process, timeout: True)

    server.close()
    server.close()

    assert events == ["process-close"]


def test_server_close_signals_only_live_leader_on_orderly_path(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []

    def send_signal(signum: int) -> None:
        events.append(f"signal:{signum}")
        process.returncode = 0

    process = SimpleNamespace(
        pid=123,
        returncode=None,
        poll=lambda: process.returncode,
        send_signal=send_signal,
    )
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=process,
            terminate=lambda: events.append("fallback"),
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
        owner=owner,
        endpoint=family,
    )
    monkeypatch.setattr(
        xkit.serving.sglang.server, "wait_for_process_group", lambda process, timeout: process.poll() is not None
    )

    server.close()

    assert events == [f"signal:{xkit.serving.sglang.server.signal.SIGTERM}", "process-close"]


def test_server_close_retains_owner_past_deadline_until_confirmed_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    family = SglangEndpointFamily("127.0.0.1", 20_000, 21_000, 22_000, 1)
    events: list[str] = []
    clock = [0.0]
    process = SimpleNamespace(
        pid=123,
        poll=lambda: None,
        send_signal=lambda signum: events.append(f"signal:{signum}"),
    )
    owner = cast(
        OwnedProcessGroup,
        SimpleNamespace(
            name="sglang-test",
            process=process,
            close=lambda: events.append("process-close"),
        ),
    )
    server = SglangServerProcess(
        model=SglangLaunchModel(TEST_MODEL_ID, SglangGraphMode.EAGER),
        owner=owner,
        endpoint=family,
    )

    def wait(process: object, timeout: float) -> bool:
        assert timeout == 0.0
        assert not server.closed
        assert "process-close" not in events
        if clock[0] == 0.0:
            clock[0] = 2.0
            return False
        return True

    monkeypatch.setattr(xkit.serving.sglang.server, "monotonic", lambda: clock[0])
    monkeypatch.setattr(xkit.serving.sglang.server, "wait_for_process_group", wait)

    server.close(deadline=1.0)

    assert server.closed
    assert server.cleanup_deadline == 1.0
    assert events == [f"signal:{xkit.serving.sglang.server.signal.SIGTERM}", "process-close"]


def server_launch(tmp_path: Path, *, dp_size: int = 1) -> XpoolClusterLaunch:
    model_root = tmp_path / "models"
    model_id = str(TEST_MODEL_ID)
    model_path = model_root / model_id
    model_path.mkdir(parents=True)
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19000},
            "vendor": {"model_base_uri": str(model_root)},
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": list(range(dp_size))},
            "ffn": {"devices": [dp_size]},
            "models": [{"id": model_id, "atn_dp_size": dp_size}],
        },
        env={},
    )
    return XpoolClusterLaunch(
        cwd=Path.cwd(),
        config=config,
        config_path=tmp_path / "xpool.toml",
        environment=MappingProxyType({}),
    )
