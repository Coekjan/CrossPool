"""Read hosted launcher requirements and supervise its servers."""

from __future__ import annotations

import asyncio
import errno
import os
import signal
import subprocess
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FrameType

import httpx
import psutil
from pydantic import BaseModel, ConfigDict, Field

from xbench.harness.serving.case import HostedBenchCase, HostedLaunchDeclaration, HostedRequirements, ServingEndpoint
from xbench.harness.serving.environment import HostedLaunchRecord, ServingEnvironment, capture_serving_environment
from xbench.harness.serving.workload import PreparedWorkload
from xkit.case import CaseId
from xkit.network import TcpEndpointReservation, TcpEndpointUnreachable, TcpPortSpace
from xkit.process import PROCESS_POLL_INTERVAL_SECONDS, OwnedProcessGroup
from xkit.results import file_sha256, write_json
from xkit.task import get_task_root
from xpool.model import ModelId
from xpool.utils.sighandler import sighandle

REQUIREMENTS_TIMEOUT_SECONDS = 30.0


def collect_hosted_launch(case: HostedBenchCase, *, settings_path: Path, deadline: float) -> HostedLaunchDeclaration:
    """Read declared local resources before the scheduler acquires a lease."""
    try:
        digest = file_sha256(case.script)
    except OSError as error:
        raise ValueError(f"cannot read hosted script {case.script}: {error}") from error
    command = [str(case.script), "requirements"]
    if case.launch_settings is not None:
        write_json(settings_path, case.launch_settings)
        command.extend(("--settings", str(settings_path)))
    if time.monotonic() >= deadline:
        raise TimeoutError("hosted requirements collection exceeded its deadline")

    owner: OwnedProcessGroup | None = None
    receiving = False
    cancelled_signal: int | None = None

    def cancel(signum: int, frame: FrameType | None) -> None:
        nonlocal cancelled_signal
        if cancelled_signal is None:
            # A signal during spawn must wait until we can stop the process group.
            cancelled_signal = signum
            if receiving:
                raise SystemExit(128 + signum)

    with sighandle(signal.SIGINT, cancel), sighandle(signal.SIGTERM, cancel):
        try:
            try:
                process = subprocess.Popen(
                    command,
                    cwd=case.script.parent,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                    text=True,
                    encoding="utf-8",
                )
            except OSError as error:
                raise ValueError(f"cannot start {case.script} requirements: {error}") from error
            owner = OwnedProcessGroup("hosted-requirements", process)
            try:
                receiving = True
                if cancelled_signal is not None:
                    raise SystemExit(128 + cancelled_signal)
                stdout, stderr = process.communicate(timeout=deadline - time.monotonic())
            except subprocess.TimeoutExpired as error:
                raise TimeoutError(f"{case.script} requirements timed out") from error
            finally:
                receiving = False
        finally:
            if owner is not None:
                try:
                    owner.terminate()
                finally:
                    owner.close()
            if cancelled_signal is not None:
                raise SystemExit(128 + cancelled_signal)

    if process.returncode:
        raise ValueError(f"{case.script} requirements exited with status {process.returncode}: {stderr.strip()}")
    requirements = HostedRequirements.model_validate_json(stdout)
    requirements.validate_targets(case.targets)
    return HostedLaunchDeclaration(script_sha256=digest, requirements=requirements)


def collect_hosted_launches(cases: Iterable[HostedBenchCase]) -> dict[CaseId, HostedLaunchDeclaration]:
    """Collect launch declarations within one execution preparation deadline."""
    deadline = time.monotonic() + REQUIREMENTS_TIMEOUT_SECONDS
    with TemporaryDirectory(prefix="xpool-hosted-settings-") as temporary:
        workdir = Path(temporary)
        return {
            case.id: collect_hosted_launch(case, settings_path=workdir / f"{case.id}.json", deadline=deadline)
            for case in cases
        }


class ModelCard(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str
    max_model_len: int | None = Field(default=None, gt=0)


class ModelList(BaseModel):
    model_config = ConfigDict(strict=True)

    data: tuple[ModelCard, ...]


class HostedServingAdapter:
    """Own one local launcher group and every assigned HTTP listener."""

    def __init__(self, case: HostedBenchCase, declaration: HostedLaunchDeclaration, errors: list[str]) -> None:
        self.case = case
        self.script_sha256 = declaration.script_sha256
        self.requirements = declaration.requirements
        self.errors = errors
        self.reservations: list[TcpEndpointReservation] = []
        self.owner: OwnedProcessGroup | None = None
        self.context_limits: dict[ModelId, int] = {}

    @asynccontextmanager
    async def start(self, *, workdir: Path, startup_timeout_seconds: float) -> AsyncIterator[ServingEnvironment]:
        deadline = time.monotonic() + startup_timeout_seconds
        root = get_task_root()
        scope = root.register_scope() if root is not None else None
        try:
            if root is not None:
                root.activate()
            self.reserve_listeners(deadline)
            endpoints = {
                model_id: ServingEndpoint(
                    base_url=f"http://{reservation.host}:{reservation.port}", served_model_name=str(model_id)
                )
                for group, reservation in zip(self.requirements.listeners, self.reservations, strict=True)
                for model_id in group
            }
            command = [str(self.case.script), "start"]
            for model_id, endpoint in endpoints.items():
                command.extend(("--model", f"{model_id}@{endpoint.base_url}"))
            if self.case.launch_settings is not None:
                settings_path = (workdir / "launch" / "settings.json").resolve()
                write_json(settings_path, self.case.launch_settings)
                command.extend(("--settings", str(settings_path)))
            environment = dict(os.environ)
            if self.requirements.device_count == 0:
                environment["CUDA_VISIBLE_DEVICES"] = ""
            if file_sha256(self.case.script) != self.script_sha256:
                raise ValueError(f"{self.case.script}: hosted script changed since requirements collection")
            if root is not None and root.sealed:
                raise InterruptedError("hosted serving startup is retiring")
            for reservation in self.reservations:
                reservation.release_for_spawn()
            self.owner = OwnedProcessGroup.spawn_logged(
                "hosted-serving", command, cwd=workdir, env=environment, log_path=workdir / "logs" / "server.log"
            )
            await self.wait_for_readiness(endpoints, deadline=deadline)
            commands = None
            try:
                record = (workdir / "launch" / "commands.json").read_bytes()
                commands = HostedLaunchRecord.model_validate_json(record).commands
            except (OSError, ValueError) as error:
                self.errors.append(f"launch/commands.json: {error}")
            devices = tuple(environment["CUDA_VISIBLE_DEVICES"].split(",")) if self.requirements.device_count else ()
            yield capture_serving_environment(endpoints, devices, commands=commands, errors=self.errors)
        finally:
            if self.owner is not None:
                try:
                    self.owner.terminate()
                finally:
                    self.owner.close()
            for reservation in self.reservations:
                reservation.close()
            if scope is not None:
                scope.complete()

    def reserve_listeners(self, deadline: float) -> None:
        for port in TcpPortSpace.local().candidates():
            if time.monotonic() >= deadline:
                raise TimeoutError("hosted listener allocation exceeded the startup deadline")
            try:
                reservation = TcpEndpointReservation.reserve_exact("127.0.0.1", port, deadline=deadline)
            except TcpEndpointUnreachable:
                continue
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            self.reservations.append(reservation)
            if len(self.reservations) == len(self.requirements.listeners):
                return
        raise RuntimeError("no eligible hosted TCP listener addresses are available")

    def check_alive(self) -> None:
        if self.owner is not None and (status := self.owner.process.poll()) is not None:
            raise RuntimeError(f"hosted serving exited with status {status}\n{self.owner.tail()}")

    def owns_listeners(self, owner: OwnedProcessGroup) -> bool:
        expected = {reservation.address for reservation in self.reservations}
        owned: set[tuple[str, int]] = set()
        for connection in psutil.net_connections(kind="tcp4"):
            if connection.status != psutil.CONN_LISTEN or not connection.laddr:
                continue
            address = (connection.laddr.ip, connection.laddr.port)
            if address not in expected:
                continue
            try:
                group = os.getpgid(connection.pid) if connection.pid is not None else None
            except ProcessLookupError:
                continue
            if group != owner.process.pid:
                raise RuntimeError(f"hosted listener {address[0]}:{address[1]} is outside the supervised process group")
            owned.add(address)
        return owned == expected

    async def wait_for_readiness(self, endpoints: dict[ModelId, ServingEndpoint], *, deadline: float) -> None:
        owner = self.owner
        if owner is None:
            raise RuntimeError("hosted readiness requires a launched service")
        urls = tuple(dict.fromkeys(endpoint.base_url for endpoint in endpoints.values()))
        error: httpx.HTTPError | TimeoutError | None = None
        async with httpx.AsyncClient(trust_env=False, timeout=None) as client:
            while time.monotonic() < deadline:
                self.check_alive()
                if self.owns_listeners(owner):
                    try:
                        lists: dict[str, tuple[ModelCard, ...]] = {}
                        async with asyncio.timeout(deadline - time.monotonic()):
                            for url in urls:
                                response = await client.get(f"{url}/v1/models")
                                response.raise_for_status()
                                lists[url] = ModelList.model_validate_json(response.content).data
                    except (httpx.HTTPError, TimeoutError) as failure:
                        error = failure
                    else:
                        for model_id, endpoint in endpoints.items():
                            matches = [
                                (url, model)
                                for url, models in lists.items()
                                for model in models
                                if model.id == endpoint.served_model_name
                            ]
                            if len(matches) != 1 or matches[0][0] != endpoint.base_url:
                                raise ValueError(f"hosted model {model_id} must appear only at {endpoint.base_url}")
                            limit = matches[0][1].max_model_len
                            if limit is not None:
                                self.context_limits[model_id] = limit
                        self.check_alive()
                        return
                await asyncio.sleep(min(PROCESS_POLL_INTERVAL_SECONDS, max(0.0, deadline - time.monotonic())))
        raise TimeoutError("timed out waiting for hosted listeners and model lists") from error

    async def validate_workload(self, workload: PreparedWorkload) -> None:
        prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
        for request in (*workload.requests, *workload.warmup):
            limit = self.context_limits.get(request.model_id)
            if limit is None:
                continue
            input_tokens = prompts[request.model_id, request.prompt_id].input_tokens
            if (input_tokens or 0) + request.max_new_tokens > limit:
                raise ValueError(
                    f"{request.model_id}: request {request.request_id!r} exceeds hosted context limit {limit} "
                    f"(input={input_tokens}, output={request.max_new_tokens})"
                )
