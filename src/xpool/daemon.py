"""Host control-plane for xpool system."""

from __future__ import annotations

import os
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from xpool.config import XpoolConfig
from xpool.runtime.mps import MpsHealthMonitor

type ProcessLivenessChecker = Callable[[int], bool]


class InstanceRegistration(BaseModel):
    """SGLang instance registration payload accepted by the daemon."""

    model_config = ConfigDict(extra="forbid")

    instance_id: str = Field(description="SGLang instance id derived from the configured model id.")
    model_id: str = Field(description="Configured model id served by the registering SGLang instance.")
    attention_cuda_devices: list[int] = Field(
        min_length=1,
        description="Attention CUDA devices used by the registering SGLang instance.",
    )
    pid: int = Field(ge=1, description="Host process id of the registering SGLang server.")


class DeviceAgentRegistration(BaseModel):
    """Device-agent registration payload accepted by the daemon."""

    model_config = ConfigDict(extra="forbid")

    device_agent_id: str = Field(description="Derived xpool device-agent id, such as cuda0.")
    cuda_device: int = Field(ge=0, description="CUDA device index owned by the registering device agent.")
    nvshmem_rank: int = Field(ge=0, description="NVSHMEM rank assigned to the registering device agent.")
    pid: int = Field(ge=1, description="Host process id of the registering device agent.")


type RegistrationPayload = InstanceRegistration | DeviceAgentRegistration


def pid_is_alive(pid: int) -> bool:
    """Return whether a host process id appears alive.

    Args:
        pid: Positive host process id to probe.

    Returns:
        ``True`` when the process exists or cannot be signaled due to permissions.

    Raises:
        ValueError: If ``pid`` is not positive.

    Side Effects:
        Sends signal ``0`` to the operating system process table.
    """

    if pid <= 0:
        raise ValueError(f"pid must be positive, got {pid}")
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def registrations_conflict(
    existing: RegistrationPayload,
    candidate: RegistrationPayload,
    *,
    process_alive: ProcessLivenessChecker,
) -> bool:
    """Return whether an existing live registration conflicts with a candidate.

    Args:
        existing: Current registration stored by the daemon.
        candidate: New registration payload for the same registration id.
        process_alive: Probe used to decide whether ``existing.pid`` is stale.

    Returns:
        ``True`` when the current registration belongs to a live process and the
        candidate changes either pid or non-pid semantics.

    Side Effects:
        Calls ``process_alive`` for ``existing.pid``.
    """

    if not process_alive(existing.pid):
        return False
    return existing.pid != candidate.pid or existing.model_dump(exclude={"pid"}) != candidate.model_dump(
        exclude={"pid"}
    )


@dataclass(slots=True)
class DaemonState:
    """Mutable daemon state shared by FastAPI route handlers.

    Attributes:
        config: Validated xpool config that defines expected registrations.
        mps_monitor: Periodic CUDA MPS health monitor used by health and readiness checks.
        started_at: Unix timestamp recorded when the daemon state was created.
        instances: Registered SGLang instances keyed by instance id.
        device_agents: Registered device agents keyed by device-agent id.
        process_alive: Liveness probe used to ignore stale registrations.
        _lock: Thread lock protecting registration dictionaries.
    """

    config: XpoolConfig
    mps_monitor: MpsHealthMonitor
    process_alive: ProcessLivenessChecker = pid_is_alive
    started_at: float = field(default_factory=time.time)
    instances: dict[str, InstanceRegistration] = field(default_factory=dict)
    device_agents: dict[str, DeviceAgentRegistration] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def readiness_snapshot(self) -> dict[str, object]:
        """Return daemon readiness from one MPS sample and one registration pass.

        Returns:
            JSON-compatible readiness fields containing health, registered ids,
            and missing participant ids.

        Side Effects:
            Reads the cached MPS health snapshot once, then takes the
            daemon-state lock to prune stale process registrations. The
            registration portion is internally consistent; MPS health is a
            cached monitor sample rather than part of the same lock domain.
        """

        mps = self.mps_monitor.snapshot()
        configured_agents = {agent.id for agent in self.config.device_agents}
        configured_instances = {instance.id for instance in self.config.sglang_instances}
        with self._lock:
            registered_agents: set[str] = set()
            stale_agents: list[str] = []
            for agent_id, registration in list(self.device_agents.items()):
                if self.process_alive(registration.pid):
                    registered_agents.add(agent_id)
                else:
                    stale_agents.append(agent_id)
                    del self.device_agents[agent_id]

            registered_instances: set[str] = set()
            stale_instances: list[str] = []
            for instance_id, registration in list(self.instances.items()):
                if self.process_alive(registration.pid):
                    registered_instances.add(instance_id)
                else:
                    stale_instances.append(instance_id)
                    del self.instances[instance_id]
        missing_agents = sorted(configured_agents - registered_agents)
        missing_instances = sorted(configured_instances - registered_instances)
        return {
            "ready": mps.healthy and not missing_agents and not missing_instances,
            "mps_healthy": mps.healthy,
            "registered_device_agents": sorted(registered_agents),
            "registered_instances": sorted(registered_instances),
            "stale_device_agents": sorted(stale_agents),
            "stale_instances": sorted(stale_instances),
            "missing_device_agents": missing_agents,
            "missing_instances": missing_instances,
        }

    def config_view(self) -> dict[str, object]:
        """Return the daemon's lightweight runtime config view.

        Returns:
            Validated raw config plus derived device-agent and SGLang-instance
            launch views.

        Side Effects:
            Does not read model files or resolve SGLang model metadata.
        """

        return {
            "config": self.config.model_dump(mode="json"),
            "derived": {
                "device_agents": [agent.model_dump(mode="json") for agent in self.config.device_agents],
                "sglang_instances": [instance.model_dump(mode="json") for instance in self.config.sglang_instances],
            },
        }


def create_app(
    config: XpoolConfig,
    *,
    mps_monitor: MpsHealthMonitor | None = None,
    process_alive: ProcessLivenessChecker = pid_is_alive,
) -> FastAPI:
    """Create the FastAPI daemon app for one resolved xpool config.

    Args:
        config: Validated xpool config used to verify registrations and expose
            the runtime config endpoint.
        mps_monitor: Optional monitor for tests or preflight-seeded startup. If
            omitted, the app creates a monitor that performs its own initial MPS check.
        process_alive: Optional process-id liveness probe. Tests may inject a
            deterministic fake; production uses ``os.kill(pid, 0)``.

    Returns:
        FastAPI application with health, readiness, config, and registration routes.

    Side Effects:
        The app lifespan starts and stops the MPS health monitor thread.
    """

    state = DaemonState(config=config, mps_monitor=mps_monitor or MpsHealthMonitor(), process_alive=process_alive)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        state.mps_monitor.start()
        try:
            yield
        finally:
            state.mps_monitor.stop()

    app = FastAPI(title="xpool daemon", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health(response: Response) -> dict[str, object]:
        mps = state.mps_monitor.snapshot()
        if not mps.healthy:
            response.status_code = 503
        return {
            "status": "ok" if mps.healthy else "unhealthy",
            "mps": mps.model_dump(mode="json"),
        }

    @app.get("/ready")
    def ready() -> dict[str, object]:
        return state.readiness_snapshot()

    @app.get("/config")
    def get_config() -> dict[str, object]:
        return state.config_view()

    @app.put("/device-agents/register")
    def register_device_agent(registration: DeviceAgentRegistration) -> dict[str, str]:
        configured = {agent.id: agent for agent in state.config.device_agents}
        agent = configured.get(registration.device_agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="unknown device agent")
        if agent.cuda_device != registration.cuda_device or agent.nvshmem_rank != registration.nvshmem_rank:
            raise HTTPException(status_code=409, detail="device agent registration does not match configuration")
        with state._lock:
            existing = state.device_agents.get(registration.device_agent_id)
            if existing is not None and registrations_conflict(
                existing,
                registration,
                process_alive=state.process_alive,
            ):
                raise HTTPException(status_code=409, detail="device agent is already registered with another payload")
            state.device_agents[registration.device_agent_id] = registration
        return {"status": "registered"}

    @app.put("/instances/register")
    def register_instance(registration: InstanceRegistration) -> dict[str, str]:
        configured = {instance.id: instance for instance in state.config.sglang_instances}
        instance = configured.get(registration.instance_id)
        if instance is None:
            raise HTTPException(status_code=404, detail="unknown SGLang instance")
        if (
            instance.model_id != registration.model_id
            or instance.attention_cuda_devices != registration.attention_cuda_devices
        ):
            raise HTTPException(status_code=409, detail="instance registration does not match configuration")
        with state._lock:
            existing = state.instances.get(registration.instance_id)
            if existing is not None and registrations_conflict(
                existing,
                registration,
                process_alive=state.process_alive,
            ):
                raise HTTPException(status_code=409, detail="instance is already registered with another payload")
            state.instances[registration.instance_id] = registration
        return {"status": "registered"}

    return app
