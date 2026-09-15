from __future__ import annotations

import logging
from typing import cast

import pytest
import torch

import xpool.native
import xpool.runtime.agent
import xpool.runtime.ffnagent.agent
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from tests.harness.support.runtime.atnagent import (
    create_atnagent,
    reset_agent_runtime,
    reset_atnagent_runtime,
)
from tests.harness.support.service.daemon import ffn_model_spec
from xpool.config import XpoolConfig
from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricGenerationId,
    FabricGenerationPhase,
    FabricInstancePlan,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnModelPlan,
    FifoSchedulerPolicy,
    InstanceFfnLayerProfile,
    InstanceFfnProfile,
    InstanceRankTopology,
)
from xpool.ffn import FfnModelSpec
from xpool.native import RuntimeRole
from xpool.native.ffn import LayerKind
from xpool.runtime.agent import AgentError
from xpool.runtime.atnagent import AtnAgent
from xpool.runtime.ffnagent import FfnAgent
from xpool.service.client import XpoolClient, XpoolClientError
from xpool.service.wire import FabricParticipantReport, KvCapacityChannelRef

pytestmark = pytest.mark.usefixtures(
    reset_global_config.__name__,
    reset_agent_runtime.__name__,
    reset_atnagent_runtime.__name__,
)


def fabric_plan(profile: InstanceFfnProfile) -> FabricPlan:
    """Build the final one-Instance Plan used by Agent lifecycle tests."""

    return FabricPlan(
        generation=FabricGenerationId(high=1, low=2),
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=1),
        ),
        executor_lane_count=1,
        scheduler=FifoSchedulerPolicy(),
        model_plans=(
            FfnModelPlan(
                model_spec_digest="b" * 64,
                layers=(DenseFfnLayerPlan(ffnagent_indices=(0,), local_intermediate_size=8),),
            ),
        ),
        instance_plans=(
            FabricInstancePlan(
                instance_id="m",
                ffn_profile=profile,
                instance_rank_topology=InstanceRankTopology(
                    atn_tp_size=1,
                    atn_dp_size=1,
                    atnagent_indices=(0,),
                ),
            ),
        ),
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
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
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
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "load",
        lambda **kwargs: FfnModelSpec.model_validate(ffn_model_spec()),
    )
    if agent_type is FfnAgent:
        monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
        monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 2))

    agent_type(cuda_device=cuda_device)

    assert events == [("init", cuda_device, role), ("devkit",)]


@pytest.mark.parametrize(
    "agent_type",
    [AtnAgent, FfnAgent],
    ids=["atnagent", "ffnagent"],
)
def test_agent_startup_registration_failure_is_debug(
    caplog: pytest.LogCaptureFixture,
    agent_type: type[AtnAgent] | type[FfnAgent],
) -> None:
    class UnavailableClient:
        def register_atnagent(self, registration: object) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

        def register_ffnagent(self, registration: object) -> None:
            raise XpoolClientError("transport", "daemon unavailable")

    agent = object.__new__(agent_type)
    agent.client = cast(XpoolClient, UnavailableClient())
    agent.cuda_device = 0
    agent.registration = object()
    agent.registered = True
    if isinstance(agent, AtnAgent):
        agent.registration_epoch = 0

    with caplog.at_level(logging.DEBUG, logger=agent_type.__module__):
        agent.register()

    assert not agent.registered
    assert [record.levelno for record in caplog.records] == [logging.DEBUG]


def test_cuda_device_selection_rejects_unknown_device() -> None:
    config = synthetic_config()

    with pytest.raises(AgentError, match="CUDA device 9 has no local instance-rank arenas"):
        create_atnagent(config, cuda_device=9)


def test_participant_report_commits_only_after_daemon_acknowledgement(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = create_atnagent(config, cuda_device=0)
    profile = InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.bfloat16,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=1,
        group_sum_complete_admitted=False,
    )
    agent.fabric_plan = fabric_plan(profile)
    reports: list[FabricParticipantReport] = []

    class RetryingClient:
        def report_fabric_participant(self, report: FabricParticipantReport) -> None:
            reports.append(report)
            if len(reports) < xpool.runtime.agent.FABRIC_REPORT_RETRY_ATTEMPTS:
                raise XpoolClientError("transport", "test daemon unavailable")

    sleeps: list[float] = []
    agent.client = cast(XpoolClient, RetryingClient())
    monkeypatch.setattr(xpool.runtime.agent.time, "sleep", sleeps.append)

    agent.report_fabric_phase(FabricParticipantPhase.JOIN_READY)

    assert len(reports) == xpool.runtime.agent.FABRIC_REPORT_RETRY_ATTEMPTS
    assert all(report == reports[0] for report in reports)
    assert sleeps == [
        xpool.runtime.agent.FABRIC_REPORT_RETRY_DELAY_S,
        xpool.runtime.agent.FABRIC_REPORT_RETRY_DELAY_S,
    ]
    assert agent.participant_report == reports[-1]


def test_post_join_value_error_is_reported_as_control_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """An ordinary post-join validation error reaches the daemon trust boundary."""

    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    agent = create_atnagent(config, cuda_device=0)
    profile = InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.bfloat16,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=1,
        group_sum_complete_admitted=False,
    )
    plan = fabric_plan(profile)
    agent.fabric_plan = plan
    agent.fabric_phase = FabricGenerationPhase.PREPARING_EXECUTION
    agent.participant_report = FabricParticipantReport(
        owner=agent.process_ref,
        generation=plan.generation,
        pe=0,
        phase=FabricParticipantPhase.JOINED,
    )
    failures: list[str] = []

    def fail_execution_preparation() -> None:
        raise ValueError("bad projection")

    monkeypatch.setattr(agent, "prepare_fabric_execution", fail_execution_preparation)
    monkeypatch.setattr(agent, "report_local_control_failure", failures.append)

    with pytest.raises(AgentError, match="bad projection"):
        agent.advance_fabric_lifecycle()

    assert failures == ["Fabric lifecycle failed: bad projection"]


def test_atnagent_joins_fabric_before_activating_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """AtnAgent activates Transport only after joining Fabric."""

    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
    )
    agent = create_atnagent(config, cuda_device=0)
    profile = InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=torch.bfloat16,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=1,
        group_sum_complete_admitted=False,
    )
    plan = fabric_plan(profile)
    events: list[object] = []

    class FabricClient:
        def fabric_plan(self) -> FabricPlan:
            events.append("plan")
            return plan

        def report_fabric_participant(self, report: FabricParticipantReport) -> None:
            events.append(report.phase)

        def kv_capacity_channel(self, generation: FabricGenerationId) -> KvCapacityChannelRef:
            events.append("capacity_channel")
            return KvCapacityChannelRef(generation=generation, name="/xpool-kv-test")

    agent.client = cast(XpoolClient, FabricClient())
    monkeypatch.setattr(agent, "prepare_fabric_join", lambda: events.append("prepare") or True)
    monkeypatch.setattr(xpool.native.fabric, "join", lambda projection, pe: events.append("join"))
    monkeypatch.setattr(
        xpool.native.kv.AtnAgentCapacityChannel,
        "attach",
        lambda name, pool_index, partition_indices: (
            events.append(("capacity_attach", name, pool_index, partition_indices)) or object()
        ),
    )
    monkeypatch.setattr(agent.transport, "activate", lambda: events.append("transport"))
    monkeypatch.setattr(agent.transport, "check_health", lambda: events.append("health"))

    agent.advance_fabric_lifecycle()
    agent.fabric_phase = FabricGenerationPhase.JOINING
    agent.advance_fabric_lifecycle()
    agent.fabric_phase = FabricGenerationPhase.PREPARING_EXECUTION
    agent.advance_fabric_lifecycle()
    agent.fabric_phase = FabricGenerationPhase.ACTIVATING
    agent.advance_fabric_lifecycle()

    assert events == [
        "plan",
        "prepare",
        FabricParticipantPhase.JOIN_READY,
        FabricParticipantPhase.JOINING,
        "join",
        FabricParticipantPhase.JOINED,
        "capacity_channel",
        ("capacity_attach", "/xpool-kv-test", 0, [0]),
        FabricParticipantPhase.EXECUTION_READY,
        "transport",
        "health",
        FabricParticipantPhase.ACTIVE,
    ]


def test_atnagent_publishes_device_memory_once_after_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    publications: list[tuple[int, int]] = []

    class CapacityChannel:
        def captures_complete(self) -> bool:
            return True

        def publish_device_memory(self, total_bytes: int, free_bytes: int) -> None:
            publications.append((total_bytes, free_bytes))

    agent = object.__new__(AtnAgent)
    agent.capacity_channel = CapacityChannel()
    agent.capacity_memory_published = False
    agent.participant_report = None
    agent.fabric_phase = None
    agent.cuda_device = 3
    monkeypatch.setattr(xpool.runtime.agent.Agent, "poll_fabric_health", lambda self: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (40, 100))

    agent.poll_fabric_health()
    agent.poll_fabric_health()

    assert publications == [(100, 40)]
