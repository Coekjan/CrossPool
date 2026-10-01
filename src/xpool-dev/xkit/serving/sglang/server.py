"""One installed SGLang process; test requests and evidence are tool-owned."""

from __future__ import annotations

import signal
from dataclasses import dataclass
from pathlib import Path

import httpx

from xkit.process import OwnedProcessGroup, wait_for_process_group
from xkit.serving.cluster import XpoolClusterLaunch, process_diagnostics
from xkit.serving.readiness import ReadinessEvidence
from xkit.serving.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from xkit.serving.sglang.launch import SglangLaunchModel
from xpool.integrations.sglang.placement import SglangCudaPlacement

HTTP_TIMEOUT_SECONDS = 30.0
PROCESS_EXIT_TIMEOUT_SECONDS = 30.0


@dataclass(slots=True)
class SglangServerProcess:
    """Own one installed ``sglang serve`` process and public HTTP endpoint."""

    model: SglangLaunchModel
    owner: OwnedProcessGroup
    endpoint: SglangEndpointFamily
    command: tuple[str, ...] = ()
    closed: bool = False

    @classmethod
    def start(
        cls,
        *,
        launch: XpoolClusterLaunch,
        model: SglangLaunchModel,
        endpoint: SglangEndpointFamilyLease,
        workdir: Path,
    ) -> SglangServerProcess:
        """Consume one reservation and launch the pinned installed CLI."""

        environment = dict(launch.environment)
        environment["SGLANG_GRPC_PORT"] = str(endpoint.family.grpc_port)
        command = server_command(
            launch=launch,
            model=model,
            endpoint=endpoint.family,
        )
        log_path = workdir / "models" / model.model_id.uri_encode() / "server.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        endpoint.release_tcp_for_spawn()
        owner = OwnedProcessGroup.spawn_logged(
            f"sglang-{model.model_id}",
            command,
            cwd=launch.cwd,
            env=environment,
            log_path=log_path,
        )
        return cls(
            model=model,
            owner=owner,
            endpoint=endpoint.family,
            command=tuple(command),
        )

    def healthy(
        self, evidence: ReadinessEvidence | None = None, *, timeout_seconds: float = HTTP_TIMEOUT_SECONDS
    ) -> bool:
        """Record one health observation or raise a recorded terminal error."""

        returncode = self.owner.process.poll()
        if returncode is not None:
            log = self.owner.tail()
            error = RuntimeError(f"{self.owner.name} exited before readiness with code {returncode}\n{log}")
            if evidence is not None:
                evidence.record_error(error)
            raise error
        try:
            response = httpx.get(f"{self.url()}/health", timeout=timeout_seconds)
        except httpx.HTTPError as error:
            if evidence is not None:
                evidence.record_error(error)
            return False
        if evidence is not None:
            evidence.record_response(response)
        return response.is_success

    def diagnostics(self) -> str:
        """Return bounded process state and log tail."""

        return process_diagnostics([self.owner])

    def close(self) -> None:
        """Request orderly server shutdown, then close owned process resources."""

        if self.closed:
            return
        self.closed = True
        try:
            if self.owner.process.poll() is None:
                self.owner.process.send_signal(signal.SIGTERM)
                if not wait_for_process_group(self.owner.process, PROCESS_EXIT_TIMEOUT_SECONDS):
                    self.owner.terminate()
            elif not wait_for_process_group(self.owner.process, PROCESS_EXIT_TIMEOUT_SECONDS):
                raise RuntimeError(f"{self.owner.name} left live descendants after exit")
        finally:
            self.owner.close()

    def url(self) -> str:
        """Return this server's loopback URL."""

        host = f"[{self.endpoint.host}]" if ":" in self.endpoint.host else self.endpoint.host
        return f"http://{host}:{self.endpoint.http_port}"


def server_command(
    *,
    launch: XpoolClusterLaunch,
    model: SglangLaunchModel,
    endpoint: SglangEndpointFamily,
) -> list[str]:
    """Project one E2E model and graph mode to the pinned SGLang CLI."""

    placement = SglangCudaPlacement.derive(launch.config.atn.devices)
    model_config = launch.config.model_by_id[model.model_id]
    graph_settings = model.graph_mode.settings()
    command = [
        "sglang",
        "serve",
        "--model-path",
        str(launch.config.model_path_of(model.model_id)),
        "--host",
        endpoint.host,
        "--port",
        str(endpoint.http_port),
        "--nccl-port",
        str(endpoint.nccl_port),
        "--trust-remote-code",
        "--tensor-parallel-size",
        str(launch.config.atn_world_size),
        "--data-parallel-size",
        str(model_config.atn_dp_size),
        "--attention-context-parallel-size",
        "1",
        "--base-gpu-id",
        str(placement.base_gpu_id),
        "--gpu-id-step",
        str(placement.gpu_id_step),
        "--random-seed",
        "0",
        "--log-level",
        "error",
        "--log-level-http",
        "error",
    ]
    if model_config.atn_dp_size > 1:
        command.append("--enable-dp-attention")
    command.extend(
        (
            "--cuda-graph-backend-decode",
            graph_settings.decode_backend,
            "--cuda-graph-backend-prefill",
            graph_settings.prefill_backend,
        )
    )
    return command
