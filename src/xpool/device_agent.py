"""Device-agent launch-plan skeleton."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from xpool.config import DeviceRole, XpoolConfig
from xpool.runtime.mps import MpsPreflight


class DeviceAgentLaunch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    cuda_device: int
    nvshmem_rank: int
    roles: list[DeviceRole]
    runs_attention_agent: bool
    runs_ffn_executor: bool


class DeviceAgentLaunchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mps: MpsPreflight
    device_agents: list[DeviceAgentLaunch]

    @classmethod
    def from_config(cls, config: XpoolConfig) -> "DeviceAgentLaunchPlan":
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
