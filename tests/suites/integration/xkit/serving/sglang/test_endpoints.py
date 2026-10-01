from __future__ import annotations

import errno
import json
import os
import socket
from pathlib import Path
from unittest.mock import patch

import pytest
from sglang.srt.server_args import DP_ATTENTION_HANDSHAKE_PORT_DELTA, ZMQ_TCP_PORT_DELTA, PortArgs

from xkit.network import TcpEndpointReservation, TcpPortSpace
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease, reserve_namespace_lock
from xtest.harness.support.sglang.fakes import ServerArgs


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
        with patch.dict(os.environ):
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


def test_endpoint_family_lease_reserves_and_releases_every_tcp_port() -> None:
    port_space = TcpPortSpace.local()
    lease = SglangEndpointFamilyLease.acquire("127.0.0.1", dp_size=2, port_space=port_space)
    try:
        assert all(port_space.is_eligible(port) for port in lease.family.ports)
        for port in lease.family.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
                with pytest.raises(OSError) as error:
                    competitor.bind((lease.family.host, port))
                assert error.value.errno == errno.EADDRINUSE

        lease.release_tcp_for_spawn()
        for port in lease.family.ports:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as competitor:
                competitor.bind((lease.family.host, port))
    finally:
        lease.close()


def test_endpoint_namespace_lock_survives_tcp_release() -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    try:
        lease.release_tcp_for_spawn()
        with pytest.raises(OSError) as error:
            reserve_namespace_lock(lease.family.host, lease.family.http_port)
        assert error.value.errno == errno.EADDRINUSE
    finally:
        lease.close()


def test_endpoint_family_reacquires_all_released_ports_and_reports_conflicts() -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=2,
        port_space=TcpPortSpace.local(),
    )
    occupied_ports = (lease.family.ports[1], lease.family.ports[-1])
    competitors: list[socket.socket] = []
    try:
        lease.release_tcp_for_spawn()
        for port in occupied_ports:
            competitor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            competitor.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            competitor.bind((lease.family.host, port))
            competitor.listen()
            competitors.append(competitor)

        assert lease.reacquire_tcp() == occupied_ports
        for reservation in lease.tcp_reservations:
            if reservation.port in occupied_ports:
                assert reservation.listener is None
            else:
                assert reservation.listener is not None
    finally:
        for competitor in competitors:
            competitor.close()
        lease.close()


def test_endpoint_family_reacquire_propagates_non_conflict_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lease = SglangEndpointFamilyLease.acquire(
        "127.0.0.1",
        dp_size=1,
        port_space=TcpPortSpace.local(),
    )
    failed_port = lease.family.nccl_port
    original_reacquire = TcpEndpointReservation.reacquire

    def reacquire(reservation: TcpEndpointReservation) -> None:
        if reservation.port == failed_port:
            raise OSError(errno.EACCES, "permission denied")
        original_reacquire(reservation)

    monkeypatch.setattr(TcpEndpointReservation, "reacquire", reacquire)
    try:
        with pytest.raises(RuntimeError, match="were not released"):
            lease.reacquire_tcp()
        lease.release_tcp_for_spawn()
        with pytest.raises(OSError) as error:
            lease.reacquire_tcp()
        assert error.value.errno == errno.EACCES
    finally:
        lease.close()


def test_endpoint_family_acquisition_exhausts_each_http_candidate_once() -> None:
    occupied = TcpEndpointReservation.reserve("127.0.0.1", port_space=TcpPortSpace.local())
    port_space = TcpPortSpace(
        unprivileged_port_start=occupied.port,
        ephemeral=range(occupied.port + 1, 65_536),
        administratively_reserved=frozenset(),
    )
    try:
        with pytest.raises(RuntimeError, match="no eligible SGLang endpoint family"):
            SglangEndpointFamilyLease.acquire("127.0.0.1", dp_size=1, port_space=port_space)
    finally:
        occupied.close()
