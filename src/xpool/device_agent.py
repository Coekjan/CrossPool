"""Device-agent launch-plan skeleton."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from xpool.config import DeviceRole, XpoolConfig
from xpool.runtime.mps import MpsPreflight


class DeviceAgentLaunch(BaseModel):
    """Host launch metadata for one xpool device-agent process."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(description="Derived device-agent id, such as cuda0.")
    cuda_device: int = Field(description="CUDA device index owned by the device agent.")
    nvshmem_rank: int = Field(description="NVSHMEM rank assigned to the device agent.")
    roles: list[DeviceRole] = Field(description="Exclusive role list exposed for launch-plan consumers.")
    runs_attention_agent: bool = Field(description="Whether this agent hosts attention-side shim scheduling.")
    runs_ffn_executor: bool = Field(description="Whether this agent hosts FFN execution.")


class DeviceAgentLaunchPlan(BaseModel):
    """Derived launch plan for all device agents in one xpool config."""

    model_config = ConfigDict(extra="forbid")

    mps: MpsPreflight = Field(description="CUDA MPS preflight result captured while deriving the launch plan.")
    device_agents: list[DeviceAgentLaunch] = Field(description="Device agents ordered by derived NVSHMEM rank.")

    @classmethod
    def from_config(cls, config: XpoolConfig) -> DeviceAgentLaunchPlan:
        """Derive a launch plan from validated xpool config.

        Args:
            config: Validated xpool config with derived device-agent placement.

        Returns:
            Launch plan containing current MPS preflight state and device agents.

        Side Effects:
            Runs CUDA MPS preflight detection.
        """

        return cls(
            mps=MpsPreflight.detect(),
            device_agents=[
                DeviceAgentLaunch(
                    id=agent.id,
                    cuda_device=agent.cuda_device,
                    nvshmem_rank=agent.nvshmem_rank,
                    roles=[agent.role],
                    runs_attention_agent=agent.role is DeviceRole.ATTENTION,
                    runs_ffn_executor=agent.role is DeviceRole.FFN,
                )
                for agent in config.device_agents
            ],
        )
