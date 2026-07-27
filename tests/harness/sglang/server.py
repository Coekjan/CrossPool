"""One installed SGLang server process and its public HTTP boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import JsonValue, TypeAdapter

from tests.harness.process import OwnedProcessGroup, wait_for_process_group
from tests.harness.sglang.e2e import E2eLaunch, E2eLaunchModel, model_id_slug
from tests.harness.sglang.endpoints import SglangEndpointFamily, SglangEndpointFamilyLease
from tests.harness.sglang.graph import SglangGraphSettings

PROMPT = "The quick brown fox jumps over the lazy dog. " * 8
NEW_TOKENS = 8
HTTP_TIMEOUT_SECONDS = 30.0
PROCESS_EXIT_TIMEOUT_SECONDS = 30.0
UNAVAILABLE_ENDPOINT_PATTERN = re.compile(
    r"(?P<name>[A-Za-z_][A-Za-z0-9_]*) at (?P<port>[0-9]{1,5}) is not available\b"
)
ADDRESS_IN_USE_PATTERN = re.compile(
    r"Address already in use.{0,1000}?tcp://(?:\[[^\]]+\]|[^:\s'\"),]+):(?P<port>[0-9]{1,5})",
    re.DOTALL,
)


class SglangEndpointConflict(RuntimeError):
    """A pinned-SGLang startup diagnostic for one owned family endpoint.

    Attributes:
        port: Conflicting TCP port parsed from the SGLang log.
        endpoint_name: SGLang endpoint label, or ``tcp_bind`` for a raw bind
            diagnostic.
    """

    __slots__ = ("endpoint_name", "port")

    def __init__(self, port: int, endpoint_name: str) -> None:
        self.port = port
        self.endpoint_name = endpoint_name
        super().__init__(f"SGLang endpoint conflict: {endpoint_name} at {port}")

    @classmethod
    def from_log(cls, family: SglangEndpointFamily, log: str) -> SglangEndpointConflict | None:
        """Parse strict conflict evidence for a member of ``family``."""

        family_ports = frozenset(family.ports)
        for match in UNAVAILABLE_ENDPOINT_PATTERN.finditer(log):
            port = int(match.group("port"))
            if port in family_ports:
                return cls(port, match.group("name"))
        for match in ADDRESS_IN_USE_PATTERN.finditer(log):
            port = int(match.group("port"))
            if port in family_ports:
                return cls(port, "tcp_bind")
        return None


@dataclass(frozen=True, slots=True)
class SglangServerResult:
    """Resolved graph behavior and deterministic token result from one server."""

    model_id: str
    resolved_graph_settings: SglangGraphSettings
    output_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class SglangInferenceRecord:
    """Exact public HTTP request and response retained as E2E evidence."""

    model_id: str
    url: str
    request: dict[str, JsonValue]
    response_status_code: int
    response_content_type: str | None
    response: JsonValue

    def write(self, path: Path) -> None:
        """Exclusively write one versionless inference record."""

        with path.open("xb") as output:
            output.write(TypeAdapter(type(self)).dump_json(self, indent=2))


@dataclass(slots=True)
class SglangServerProcess:
    """Own one installed ``sglang serve`` process and public HTTP endpoint."""

    model: E2eLaunchModel
    owner: OwnedProcessGroup
    endpoint: SglangEndpointFamilyLease
    host: str
    port: int
    inference_path: Path
    closed: bool = False

    @classmethod
    def start(
        cls,
        *,
        launch: E2eLaunch,
        model: E2eLaunchModel,
        graph_settings: SglangGraphSettings,
        endpoint: SglangEndpointFamilyLease,
        workdir: Path,
    ) -> SglangServerProcess:
        """Consume one reservation and launch the pinned installed CLI."""

        slug = model_id_slug(model.model_id)
        environment = dict(launch.environment)
        environment["SGLANG_GRPC_PORT"] = str(endpoint.family.grpc_port)
        command = server_command(
            launch=launch,
            model=model,
            graph_settings=graph_settings,
            endpoint=endpoint.family,
        )
        endpoint.release_tcp_for_spawn()
        owner = OwnedProcessGroup.spawn_logged(
            f"sglang-{slug}",
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            log_path=workdir / f"{slug}.log",
        )
        return cls(
            model=model,
            owner=owner,
            endpoint=endpoint,
            host=endpoint.family.host,
            port=endpoint.family.http_port,
            inference_path=workdir / f"{slug}.inference.json",
        )

    def healthy(self) -> bool:
        """Return HTTP health or fail immediately after an early process exit."""

        returncode = self.owner.process.poll()
        if returncode is not None:
            log = self.owner.tail()
            conflict = SglangEndpointConflict.from_log(self.endpoint.family, log)
            if conflict is not None:
                raise conflict
            raise RuntimeError(f"{self.owner.name} exited before readiness with code {returncode}\n{log}")
        try:
            response = httpx.get(f"{self.url()}/health", timeout=1.0)
        except httpx.HTTPError:
            return False
        return response.is_success

    def result(self) -> SglangServerResult:
        """Read resolved graph settings and execute one deterministic request."""

        request: dict[str, JsonValue] = {
            "text": PROMPT,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": NEW_TOKENS,
                "min_new_tokens": NEW_TOKENS,
                "ignore_eos": True,
            },
            "stream": False,
        }
        with httpx.Client(base_url=self.url(), timeout=HTTP_TIMEOUT_SECONDS) as client:
            server_info = client.get("/server_info")
            server_info.raise_for_status()
            info = server_info.json()
            disable_cuda_graph = require_bool(info, "disable_cuda_graph")
            disable_piecewise_cuda_graph = require_bool(info, "disable_piecewise_cuda_graph")
            response = client.post("/generate", json=request)
            try:
                payload: JsonValue = response.json()
            except ValueError:
                payload = response.text
            SglangInferenceRecord(
                model_id=self.model.model_id,
                url=f"{self.url()}/generate",
                request=request,
                response_status_code=response.status_code,
                response_content_type=response.headers.get("content-type"),
                response=payload,
            ).write(self.inference_path)
            response.raise_for_status()
        output_ids = payload.get("output_ids") if isinstance(payload, dict) else None
        if (
            not isinstance(output_ids, list)
            or len(output_ids) != NEW_TOKENS
            or not all(isinstance(token_id, int) and not isinstance(token_id, bool) for token_id in output_ids)
        ):
            raise RuntimeError(f"{self.owner.name} returned invalid output_ids: {output_ids!r}")
        return SglangServerResult(
            model_id=self.model.model_id,
            resolved_graph_settings=SglangGraphSettings(
                cuda_graph=not disable_cuda_graph,
                piecewise_cuda_graph=not disable_piecewise_cuda_graph,
            ),
            output_ids=tuple(output_ids),
        )

    def diagnostics(self) -> str:
        """Return bounded process state and log tail."""

        returncode = self.owner.process.poll()
        status = "running" if returncode is None else f"exited({returncode})"
        return f"--- {self.owner.name}: {status}; log={self.owner.log_path} ---\n{self.owner.tail()}"

    def close(self) -> None:
        """Terminate the complete server process group and close owned logs."""

        if self.closed:
            return
        self.closed = True
        try:
            if self.owner.process.poll() is None:
                self.owner.terminate()
            elif not wait_for_process_group(self.owner.process, PROCESS_EXIT_TIMEOUT_SECONDS):
                raise RuntimeError(f"{self.owner.name} left live descendants after exit")
        finally:
            self.owner.close()
            self.endpoint.close()

    def url(self) -> str:
        """Return this server's loopback URL."""

        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


def server_command(
    *,
    launch: E2eLaunch,
    model: E2eLaunchModel,
    graph_settings: SglangGraphSettings,
    endpoint: SglangEndpointFamily,
) -> list[str]:
    """Project one E2E model and graph mode to the pinned SGLang CLI."""

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
        str(model.atn_tp_size * model.atn_dp_size),
        "--data-parallel-size",
        str(model.atn_dp_size),
        "--attention-context-parallel-size",
        "1",
        "--base-gpu-id",
        "0",
        "--gpu-id-step",
        "1",
        "--random-seed",
        "0",
        "--max-total-tokens",
        str(model.max_total_tokens),
        "--log-level",
        "error",
        "--log-level-http",
        "error",
    ]
    if model.atn_dp_size > 1:
        command.append("--enable-dp-attention")
    if not graph_settings.cuda_graph:
        command.append("--disable-cuda-graph")
    if model.atn_dp_size == 1 and not graph_settings.piecewise_cuda_graph:
        command.append("--disable-piecewise-cuda-graph")
    return command


def require_bool(payload: object, name: str) -> bool:
    """Read one strict boolean from a server-info object."""

    value = payload.get(name) if isinstance(payload, dict) else None
    if not isinstance(value, bool):
        raise RuntimeError(f"SGLang /server_info returned invalid {name}: {value!r}")
    return value
