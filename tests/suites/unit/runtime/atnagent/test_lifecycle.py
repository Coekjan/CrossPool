from __future__ import annotations

from typing import cast

import pytest

import xpool.native
import xpool.runtime.agent
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from tests.harness.support.runtime.atnagent import (
    create_atnagent,
    reset_agent_runtime,
    reset_atnagent_runtime,
)
from xpool.abi import TensorDType
from xpool.config import XpoolConfig
from xpool.fabric import (
    FabricGeneration,
    FabricModelPlan,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnLayerKind,
    FfnLayerSpec,
    FfnWorkload,
    FifoSchedulerPlan,
)
from xpool.runtime import RuntimeRole
from xpool.runtime.agent import AgentError
from xpool.runtime.atnagent import AtnAgent
from xpool.runtime.ffnagent import FfnAgent
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.wire import FabricParticipantReport

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__,
    reset_agent_runtime.__name__,
    reset_atnagent_runtime.__name__,
)


@pytest.mark.parametrize(
    ("agent_type", "cuda_device", "role"),
    [
        (AtnAgent, 0, RuntimeRole.ATNAGENT),
        (FfnAgent, 1, RuntimeRole.FFNAGENT),
    ],
)
def test_agent_construction_initializes_role_and_devkit(
    monkeypatch: pytest.MonkeyPatch,
    agent_type: type[AtnAgent] | type[FfnAgent],
    cuda_device: int,
    role: RuntimeRole,
) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    install_test_config(config=config)
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        xpool.runtime.agent.bootstrap,
        "init",
        lambda cuda_device, role: events.append(("init", cuda_device, role)),
    )
    monkeypatch.setattr(xpool.runtime.agent.devkit, "install", lambda: events.append(("devkit",)))

    agent_type(cuda_device=cuda_device)

    assert events == [("init", cuda_device, role), ("devkit",)]


def test_cuda_device_selection_rejects_unknown_device() -> None:
    config = synthetic_config()

    with pytest.raises(AgentError, match="CUDA device 9 has no local instance-rank arenas"):
        create_atnagent(config, cuda_device=9)


def test_participant_report_commits_only_after_daemon_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = create_atnagent(config, cuda_device=0)
    workload = FfnWorkload(
        model_config_digest="a" * 64,
        dtype=TensorDType.BF16,
        hidden_size=4,
        layers=(FfnLayerSpec(layer_id=0, kind=FfnLayerKind.DENSE),),
        max_decode_rows=1,
        max_prefill_rows=1,
    )
    agent.fabric_plan = FabricPlan(
        generation=FabricGeneration(high=1, low=2),
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(pe=0, role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(pe=1, role=FabricRole.FFNAGENT, cuda_device=1),
        ),
        executor_count=1,
        scheduler=FifoSchedulerPlan(),
        models=(FabricModelPlan(workload=workload, atn_tp_size=1, atn_dp_size=1),),
    )
    reports: list[FabricParticipantReport] = []

    class RetryingClient:
        def report_fabric_participant(self, report: FabricParticipantReport) -> None:
            reports.append(report)
            if len(reports) < xpool.runtime.agent.FABRIC_REPORT_RETRY_ATTEMPTS:
                raise XpoolClientError("transport", "test daemon unavailable")

    sleeps: list[float] = []
    agent.client = cast(XpoolClient, RetryingClient())
    monkeypatch.setattr(xpool.runtime.agent.time, "sleep", sleeps.append)

    agent.report_fabric_phase(FabricParticipantPhase.JOINING)

    assert len(reports) == xpool.runtime.agent.FABRIC_REPORT_RETRY_ATTEMPTS
    assert all(report == reports[0] for report in reports)
    assert sleeps == [
        xpool.runtime.agent.FABRIC_REPORT_RETRY_DELAY_S,
        xpool.runtime.agent.FABRIC_REPORT_RETRY_DELAY_S,
    ]
    assert agent.participant_report == reports[-1]


def test_atnagent_loopback_joins_fabric_before_activating_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """AtnAgent loopback shares the production Fabric and Transport lifecycle."""

    config = XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_LOOPBACK_ENABLE": "1",
            "XPOOL_DEBUG_LOOPBACK_SITE": "atnagent",
        },
    )
    agent = create_atnagent(config, cuda_device=0)
    workload = FfnWorkload(
        model_config_digest="a" * 64,
        dtype=TensorDType.BF16,
        hidden_size=4,
        layers=(FfnLayerSpec(layer_id=0, kind=FfnLayerKind.DENSE),),
        max_decode_rows=1,
        max_prefill_rows=1,
    )
    plan = FabricPlan(
        generation=FabricGeneration(high=1, low=2),
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(pe=0, role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(pe=1, role=FabricRole.FFNAGENT, cuda_device=1),
        ),
        executor_count=1,
        scheduler=FifoSchedulerPlan(),
        models=(FabricModelPlan(workload=workload, atn_tp_size=1, atn_dp_size=1),),
    )
    events: list[object] = []

    class FabricClient:
        def fabric_plan(self) -> FabricPlan:
            events.append("plan")
            return plan

        def report_fabric_participant(self, report: FabricParticipantReport) -> None:
            events.append(report.phase)

    agent.client = cast(XpoolClient, FabricClient())
    monkeypatch.setattr(xpool.native.fabric, "join", lambda metadata: events.append("join"))
    monkeypatch.setattr(xpool.native.fabric, "check_health", lambda: events.append("health"))
    monkeypatch.setattr(agent.catalog, "activate", lambda: events.append("transport"))

    agent.join_fabric()

    assert events == [
        "plan",
        FabricParticipantPhase.JOINING,
        "join",
        FabricParticipantPhase.JOINED,
        "transport",
        "health",
        FabricParticipantPhase.ACTIVE,
    ]
