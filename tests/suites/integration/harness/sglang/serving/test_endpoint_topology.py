from __future__ import annotations

import errno
import json
from pathlib import Path

import pytest
from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA, ZMQ_TCP_PORT_DELTA, PortArgs, ServerArgs

from tests.harness.runner.child import PythonChildProcess
from tests.harness.runner.network import TcpPortSpace
from tests.harness.runner.supervisor import prepare_task_supervision
from tests.harness.sglang.serving.endpoints import SglangEndpointFamilyLease, probe_namespace_lock


def test_pinned_sglang_resolves_explicit_grpc_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "model_type": "qwen3",
                "num_attention_heads": 8,
                "num_hidden_layers": 2,
                "num_key_value_heads": 8,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SGLANG_GRPC_PORT", "19002")

    server_args = ServerArgs(model_path=str(model_path), port=65_000, device="cpu")
    server_args.resolve_once()

    assert server_args.resolved_dict()["grpc_port"] == 19_002


def test_endpoint_family_lock_coordinates_independent_interpreters(tmp_path: Path) -> None:
    prepare_task_supervision()
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    endpoint = (lease.family.host, lease.family.http_port)
    try:
        occupied = PythonChildProcess.start(
            "occupied-endpoint-lock",
            probe_namespace_lock,
            endpoint,
            log_path=tmp_path / "occupied.log",
        )
        try:
            assert occupied.receive(int, timeout_seconds=5) == errno.EADDRINUSE
            occupied.wait(timeout_seconds=5)
        finally:
            if occupied.process.is_alive():
                PythonChildProcess.terminate_all((occupied,))
            occupied.close()
    finally:
        lease.close()

    available = PythonChildProcess.start(
        "available-endpoint-lock",
        probe_namespace_lock,
        endpoint,
        log_path=tmp_path / "available.log",
    )
    try:
        assert available.receive(type(None), timeout_seconds=5) is None
        available.wait(timeout_seconds=5)
    finally:
        if available.process.is_alive():
            PythonChildProcess.terminate_all((available,))
        available.close()


def test_endpoint_family_matches_pinned_dp_attention_port_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3ForCausalLM"],
                "hidden_size": 64,
                "intermediate_size": 128,
                "model_type": "qwen3",
                "num_attention_heads": 8,
                "num_hidden_layers": 2,
                "num_key_value_heads": 8,
                "vocab_size": 128,
            }
        ),
        encoding="utf-8",
    )
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=2,
        port_space=TcpPortSpace.local(),
    )
    family = lease.family
    lease.release_tcp_for_spawn()
    try:
        monkeypatch.setenv("SGLANG_GRPC_PORT", str(family.grpc_port))
        server_args = ServerArgs(
            model_path=str(model_path),
            host=family.host,
            port=family.http_port,
            nccl_port=family.nccl_port,
            device="cpu",
            enable_dp_attention=True,
            dp_size=2,
            tp_size=2,
        )
        server_args.resolve_once()
        ports = PortArgs.init_new(server_args)
    finally:
        lease.close()

    zmq_port = family.http_port + ZMQ_TCP_PORT_DELTA
    if zmq_port > 65_535:
        zmq_port = family.http_port - ZMQ_TCP_PORT_DELTA
    expected_addresses = tuple(f"tcp://127.0.0.1:{zmq_port + offset}" for offset in range(1, 7))
    assert server_args.resolved_dict()["grpc_port"] == family.grpc_port
    assert family.http_port + DP_ATTENTION_HANDSHAKE_PORT_DELTA in family.ports
    assert ports.nccl_port == family.nccl_port
    assert (
        ports.tokenizer_ipc_name,
        ports.detokenizer_ipc_name,
        ports.rpc_ipc_name,
        ports.metrics_ipc_name,
        ports.scheduler_input_ipc_name,
        ports.load_collector_ipc_name,
    ) == expected_addresses
