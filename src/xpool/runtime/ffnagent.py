"""FfnAgent process lifecycle."""

from __future__ import annotations

import logging

import xpool.native
from xpool.abi import ABI_VERSION
from xpool.runtime import RuntimeRole
from xpool.runtime.agent import Agent, AgentError, AgentHeartbeat
from xpool.service.client import XpoolClientError
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import FfnAgentRegistration, HeartbeatResponse

logger = logging.getLogger(__name__)


class FfnAgent(Agent):
    """FfnAgent owning generation-scoped Fabric progress."""

    def __init__(self, *, cuda_device: int) -> None:
        """Initialize native FFN role state and daemon ownership."""

        super().__init__(cuda_device=cuda_device, runtime_role=RuntimeRole.FFNAGENT)
        self.registration = FfnAgentRegistration(
            cuda_device=cuda_device,
            abi_version=ABI_VERSION,
            pid=self.proc_id.pid,
        )
        self.heartbeat_worker = AgentHeartbeat(agent=self)

    def activate_fabric_plan(self) -> None:
        """Start persistent Fabric progress on this FfnAgent PE."""

        xpool.native.fabric.activate()

    def quiesce_fabric(self) -> None:
        """Keep coordinator progress alive while AtnAgents quiesce producers."""

    def register(self) -> None:
        """Register this FfnAgent with the daemon."""

        try:
            self.client.register_ffnagent(self.registration)
        except (XpoolDaemonError, XpoolClientError) as exc:
            self.registered = False
            if not exc.is_recoverable:
                raise AgentError(f"FfnAgent registration received unrecoverable daemon error: {exc}") from exc
            logger.warning("FfnAgent registration failed: %s", exc)
            return
        self.registered = True

    def send_heartbeat(self) -> HeartbeatResponse:
        """Publish this FfnAgent's heartbeat to its role-specific endpoint."""

        return self.client.heartbeat_ffnagent(self.cuda_device, self.process_ref)

    def prepare_fabric(self) -> bool:
        """Report that FFN-side role state is ready for collective join."""

        return True

    def close_role(self) -> None:
        """Release FFN-side resources after Fabric shutdown."""
