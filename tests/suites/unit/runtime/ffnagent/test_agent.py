from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest
import torch

import xpool.runtime.ffnagent.agent
from xpool.fabric import FabricDeploymentPlan, FabricPePlacement, FabricRole
from xpool.memory import DeviceMemoryEstimate
from xpool.native import RuntimeRole
from xpool.runtime.agent import Agent, AgentError
from xpool.runtime.ffnagent.agent import FfnAgent
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import (
    TEST_CALIBRATION_ENVIRONMENT,
    install_test_config,
    reset_global_config,
    synthetic_config,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def execution_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.memory_calibration,
        "observe_environment",
        lambda device: TEST_CALIBRATION_ENVIRONMENT,
    )


def test_ffnagent_rejects_cuda_initialized_before_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(AgentError, match="CUDA initialized before"):
        FfnAgent(device=1)


def test_ffnagent_records_baseline_after_loading_and_cache_trim(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("trim"))

    def baseline(device: int) -> tuple[int, int]:
        events.append("baseline")
        return 1, TEST_CALIBRATION_ENVIRONMENT.total_memory_bytes

    def bootstrap_agent(self: Agent, **kwargs: object) -> None:
        events.append("bootstrap")
        self.proc_id = ProcUniqId.current()
        self.local_rank = 0
        self.device = 1

    monkeypatch.setattr(torch.cuda, "mem_get_info", baseline)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.memory_calibration,
        "observe_environment",
        lambda device: pytest.fail("unprofiled execution must omit compatibility-only observation"),
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent.Agent, "__init__", bootstrap_agent)
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "load", lambda **kwargs: SimpleNamespace(layers=()))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent, "FfnAgentRegistration", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "AgentHeartbeat", lambda **kwargs: object())

    agent = FfnAgent(device=1)

    assert agent.registration.calibration_environment is None
    assert events == ["bootstrap", "trim", "baseline"]


def test_ffnagent_rejects_plan_lanes_before_loading() -> None:
    install_test_config(synthetic_config())
    agent = object.__new__(FfnAgent)
    agent.device = 1
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.local_rank = 0
    agent.fabric_plan = cast(
        FabricDeploymentPlan,
        SimpleNamespace(
            pe_placements=(
                FabricPePlacement(role=FabricRole.ATNAGENT, device=0),
                FabricPePlacement(role=FabricRole.FFNAGENT, device=1),
            ),
            executor_lane_count=2,
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace()
    with pytest.raises(AgentError, match="Lane count"):
        agent.prepare_fabric_join()


@pytest.mark.parametrize("fails", (False, True))
def test_ffnagent_retains_registry_through_materialization(monkeypatch: pytest.MonkeyPatch, fails: bool) -> None:
    config = synthetic_config()
    install_test_config(config)
    agent = object.__new__(FfnAgent)
    agent.fabric_plan = cast(
        FabricDeploymentPlan,
        SimpleNamespace(
            pe_placements=(
                FabricPePlacement(role=FabricRole.ATNAGENT, device=0),
                FabricPePlacement(role=FabricRole.FFNAGENT, device=1),
            ),
            instance_plans=(),
            executor_lane_count=1,
            model_plans=(),
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace(
        model_specs=("spec",),
        calibration_environment=TEST_CALIBRATION_ENVIRONMENT,
    )
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.device = 1
    agent.local_rank = 0
    agent.layer_weights = None
    agent.execution_registry = None
    layer_weights = (("weights",),)
    observed: list[tuple[str, dict[str, object]]] = []

    def materialize(registry: FfnExecutionRegistry, **arguments: object) -> None:
        assert agent.execution_registry is registry
        assert agent.layer_weights is None
        assert registry.layer_weights is layer_weights
        observed.append(("registry", arguments))
        if fails:
            raise RuntimeError("materialization failed")

    monkeypatch.setattr(agent, "fabric_pe", lambda: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        torch.cuda, "synchronize", lambda device: pytest.fail("Agent repeated Registry-owned completion")
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "materialize_layer_weights",
        lambda **arguments: observed.append(("weights", arguments)) or layer_weights,
    )
    monkeypatch.setattr(
        FfnExecutionRegistry,
        "materialize",
        materialize,
    )

    assert agent.prepare_fabric_join()
    if fails:
        with pytest.raises(RuntimeError, match="materialization failed"):
            agent.prepare_fabric_execution()
    else:
        agent.prepare_fabric_execution()
    assert agent.layer_weights is None

    assert agent.execution_registry is not None
    assert agent.execution_registry.layer_weights is layer_weights
    assert [name for name, _ in observed] == ["weights", "registry"]

    retained_registry = agent.execution_registry
    with pytest.raises(AgentError, match="cannot be retried"):
        agent.prepare_fabric_execution()
    assert agent.execution_registry is retained_registry
    assert [name for name, _ in observed] == ["weights", "registry"]

    def release(registry: FfnExecutionRegistry) -> None:
        raise RuntimeError("capture retirement failed")

    monkeypatch.setattr(FfnExecutionRegistry, "close", release)
    with pytest.raises(RuntimeError, match="capture retirement failed"):
        agent.close_role()
    assert agent.execution_registry is not None
    assert agent.execution_registry.layer_weights is layer_weights
    monkeypatch.setattr(FfnExecutionRegistry, "close", lambda self: None)
    agent.close_role()
    assert agent.execution_registry is None
    assert agent.layer_weights is None


def test_ffnagent_rejects_prejoin_memory_shortfall(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    agent = object.__new__(FfnAgent)
    agent.fabric_plan = cast(
        FabricDeploymentPlan,
        SimpleNamespace(
            pe_placements=(
                FabricPePlacement(role=FabricRole.ATNAGENT, device=0),
                FabricPePlacement(role=FabricRole.FFNAGENT, device=1),
            ),
            instance_plans=(),
            executor_lane_count=1,
            model_plans=(),
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace(
        model_specs=("spec",),
        calibration_environment=TEST_CALIBRATION_ENVIRONMENT,
    )
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.device = 1
    agent.local_rank = 0
    agent.layer_weights = None
    agent.execution_registry = None
    monkeypatch.setattr(agent, "fabric_pe", lambda: 1)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (9, 10))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "materialize_layer_weights",
        lambda **arguments: pytest.fail("weights must not load after failed admission"),
    )

    with pytest.raises(AgentError, match="analytic memory admission requires 10 bytes"):
        agent.prepare_fabric_join()
