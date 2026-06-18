"""Host control-plane for xpool system."""

from __future__ import annotations

import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field

from xpool.config import XpoolConfig
from xpool.runtime.mps import MpsHealthMonitor


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


@dataclass(slots=True)
class DaemonState:
    """Mutable daemon state shared by FastAPI route handlers.

    Attributes:
        config: Validated xpool config that defines expected registrations.
        mps_monitor: Periodic CUDA MPS health monitor used by health and readiness checks.
        started_at: Unix timestamp recorded when the daemon state was created.
        instances: Registered SGLang instances keyed by instance id.
        device_agents: Registered device agents keyed by device-agent id.
        _lock: Thread lock protecting registration dictionaries.
    """

    config: XpoolConfig
    mps_monitor: MpsHealthMonitor
    started_at: float = field(default_factory=time.time)
    instances: dict[str, InstanceRegistration] = field(default_factory=dict)
    device_agents: dict[str, DeviceAgentRegistration] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def ready(self) -> bool:
        """Return whether all configured runtime participants are registered and healthy.

        Returns:
            ``True`` only when CUDA MPS is healthy and every configured SGLang
            instance and device agent has registered.

        Side Effects:
            Takes the daemon-state lock briefly and reads the cached MPS health snapshot.
        """

        configured_agents = {agent.id for agent in self.config.device_agents}
        configured_instances = {instance.id for instance in self.config.sglang_instances}
        with self._lock:
            registered_agents = set(self.device_agents)
            registered_instances = set(self.instances)
        return (
            self.mps_monitor.snapshot().healthy
            and configured_agents.issubset(registered_agents)
            and configured_instances.issubset(registered_instances)
        )


def create_app(config: XpoolConfig, *, mps_monitor: MpsHealthMonitor | None = None) -> FastAPI:
    """Create the FastAPI daemon app for one resolved xpool config.

    Args:
        config: Validated xpool config used to verify registrations and expose
            the runtime config endpoint.
        mps_monitor: Optional monitor for tests or preflight-seeded startup. If
            omitted, the app creates a monitor that performs its own initial MPS check.

    Returns:
        FastAPI application with health, readiness, config, and registration routes.

    Side Effects:
        The app lifespan starts and stops the MPS health monitor thread.
    """

    state = DaemonState(config=config, mps_monitor=mps_monitor or MpsHealthMonitor())

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
        mps = state.mps_monitor.snapshot()
        with state._lock:
            registered_device_agents = sorted(state.device_agents)
            registered_instances = sorted(state.instances)
        return {
            "ready": state.ready(),
            "mps_healthy": mps.healthy,
            "registered_device_agents": registered_device_agents,
            "registered_instances": registered_instances,
        }

    @app.get("/config")
    def get_config() -> dict[str, object]:
        return state.config.model_dump(mode="json")

    @app.put("/device-agents/register")
    def register_device_agent(registration: DeviceAgentRegistration) -> dict[str, str]:
        configured = {agent.id: agent for agent in state.config.device_agents}
        agent = configured.get(registration.device_agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail="unknown device agent")
        if agent.cuda_device != registration.cuda_device or agent.nvshmem_rank != registration.nvshmem_rank:
            raise HTTPException(status_code=409, detail="device agent registration does not match configuration")
        with state._lock:
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
            state.instances[registration.instance_id] = registration
        return {"status": "registered"}

    return app
