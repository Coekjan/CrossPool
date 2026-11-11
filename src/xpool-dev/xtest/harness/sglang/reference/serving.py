"""Original SGLang HTTP serving over the same encoded native media requests."""

from __future__ import annotations

import os
import time
from pathlib import Path

import httpx

from xkit.network import TcpPortSpace
from xkit.process import OwnedProcessGroup
from xkit.results import write_json
from xkit.serving.sglang.endpoints import SglangEndpointFamilyLease
from xkit.serving.sglang.graph import SglangGraphMode
from xkit.serving.sglang.launch import SglangLaunchModel
from xkit.serving.sglang.server import SglangServerProcess
from xpool.model import ModelId
from xtest.harness.sglang.catalog import E2eServingInput
from xtest.harness.sglang.serving.server import SglangProbeServer


def run_reference(
    *,
    model_id: ModelId,
    model_path: Path,
    inputs: tuple[E2eServingInput, ...],
    server_arguments: tuple[str, ...],
    workdir: Path,
    timeout_seconds: float,
) -> tuple[tuple[int, ...], ...]:
    """Compare cold HTTP requests against a plugin-isolated original model.

    Media decoding, processor policy and request transport match the installed
    production endpoint. The caller retains its device lease until the server
    and every descendant exit; this server creates no CrossPool generation.
    """

    workdir.mkdir(parents=True, exist_ok=False)
    deadline = time.monotonic() + timeout_seconds
    endpoint = SglangEndpointFamilyLease.acquire(
        "127.0.0.1", dp_size=1, port_space=TcpPortSpace.local(), deadline=deadline
    )
    environment = dict(os.environ)
    environment["SGLANG_PLUGINS"] = "__xpool_native_serving_reference_no_plugins__"
    environment["SGLANG_GRPC_PORT"] = str(endpoint.family.grpc_port)
    environment["SGLANG_KILLPG_ON_SCHEDULER_EXCEPTION"] = "0"
    command = [
        "sglang",
        "serve",
        "--model-path",
        str(model_path),
        "--host",
        endpoint.family.host,
        "--port",
        str(endpoint.family.http_port),
        "--nccl-port",
        str(endpoint.family.nccl_port),
        "--trust-remote-code",
        "--random-seed",
        "0",
        "--mem-fraction-static",
        "0.7",
        "--max-total-tokens",
        "4096",
        "--cuda-graph-backend-decode",
        "disabled",
        "--cuda-graph-backend-prefill",
        "disabled",
        "--log-level",
        "error",
        *server_arguments,
    ]
    server = None
    try:
        endpoint.release_tcp_for_spawn()
        owner = OwnedProcessGroup.spawn_logged(
            "native-sglang-reference", command, cwd=Path.cwd(), env=environment, log_path=workdir / "server.log"
        )
        server = SglangServerProcess(SglangLaunchModel(model_id, SglangGraphMode.EAGER), owner, endpoint.family)
        while not server.healthy(timeout_seconds=min(1, max(0.01, deadline - time.monotonic()))):
            if time.monotonic() >= deadline:
                raise TimeoutError(f"original SGLang did not become ready: {server.diagnostics()}")
            time.sleep(0.1)
        results = []
        for index, native in enumerate(inputs):
            probe = SglangProbeServer(server, workdir / f"input-{index}.json", native_input=native)
            results.append(probe.result().output_ids)
            response = httpx.get(f"{server.url()}/flush_cache", timeout=30)
            response.raise_for_status()
        write_json(workdir / "reference.json", dict(output_ids=results, arguments=command))
        return tuple(results)
    finally:
        if server is not None:
            server.close()
        endpoint.close()
