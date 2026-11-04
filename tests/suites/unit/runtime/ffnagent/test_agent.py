from __future__ import annotations

import os
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
    TEST_EXECUTION_CONTRACT,
    install_test_config,
    lane_resource_declaration,
    reset_global_config,
    synthetic_config,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@pytest.fixture(autouse=True)
def blas_workspace_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.memory_calibration,
        "observe_environment",
        lambda device: TEST_CALIBRATION_ENVIRONMENT,
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.capture_workspace,
        "prepare_execution_sources",
        lambda: (TEST_EXECUTION_CONTRACT, ()),
    )


def test_ffnagent_rejects_cuda_initialized_before_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(AgentError, match="CUDA initialized before"):
        FfnAgent(device=1)


def test_ffnagent_preserves_policy_and_registers_discovery_before_baseline(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    events: list[str] = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("trim"))

    def baseline(device: int) -> tuple[int, int]:
        events.append("baseline")
        return 1, TEST_CALIBRATION_ENVIRONMENT.total_memory_bytes

    def bootstrap_agent(self: Agent, **kwargs: object) -> None:
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
        events.append("bootstrap")
        self.proc_id = ProcUniqId.current()
        self.local_rank = 0
        self.device = 1

    def discover():
        events.append("discover")
        return TEST_EXECUTION_CONTRACT, ()

    monkeypatch.setattr(torch.cuda, "mem_get_info", baseline)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.memory_calibration,
        "observe_environment",
        lambda device: pytest.fail("unprofiled execution must omit compatibility-only observation"),
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent.Agent, "__init__", bootstrap_agent)
    monkeypatch.setattr(xpool.runtime.ffnagent.agent.capture_workspace, "prepare_execution_sources", discover)
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "load", lambda **kwargs: SimpleNamespace(layers=()))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent, "FfnAgentRegistration", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "AgentHeartbeat", lambda **kwargs: object())

    agent = FfnAgent(device=1)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":16:8"
    assert agent.registration.execution_contract == TEST_EXECUTION_CONTRACT
    assert agent.registration.calibration_environment is None
    assert events == ["bootstrap", "discover", "trim", "baseline"]


@pytest.mark.parametrize("mismatch", ("facts", "lanes"))
def test_ffnagent_rejects_plan_facts_or_lanes_before_loading(mismatch: str) -> None:
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
            executor_lane_count=2 if mismatch == "lanes" else 1,
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,),
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace(
        execution_contract=TEST_EXECUTION_CONTRACT.model_copy(
            update={
                "lane_resources": (
                    lane_resource_declaration(key="scratch", max_bytes=1024, alignment_bytes=256, aliases=("scratch",)),
                ),
            }
        ),
    )
    with pytest.raises(AgentError, match="Lane count" if mismatch == "lanes" else "Plan differs from registered"):
        agent.prepare_fabric_join()


def test_ffnagent_rejects_source_failure_before_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_source() -> None:
        raise RuntimeError("source preparation failed")

    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.Agent,
        "__init__",
        lambda self, **kwargs: setattr(self, "proc_id", SimpleNamespace(pid=1)),
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.capture_workspace,
        "prepare_execution_sources",
        reject_source,
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "load",
        lambda **kwargs: pytest.fail("model loading must not run with an source preparation failed"),
    )

    with pytest.raises(RuntimeError, match="source preparation failed"):
        FfnAgent(device=1)


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
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,),
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace(
        model_specs=("spec",),
        execution_contract=TEST_EXECUTION_CONTRACT,
        calibration_environment=TEST_CALIBRATION_ENVIRONMENT,
    )
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.device = 1
    agent.local_rank = 0
    agent.layer_weights = None
    agent.execution_registry = None
    agent.external_resources = ()
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
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,),
            ffn_memory_estimates=(DeviceMemoryEstimate(retained_bytes=1, peak_bytes=10),),
        ),
    )
    agent.registration = SimpleNamespace(
        model_specs=("spec",),
        execution_contract=TEST_EXECUTION_CONTRACT,
        calibration_environment=TEST_CALIBRATION_ENVIRONMENT,
    )
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.device = 1
    agent.local_rank = 0
    agent.layer_weights = None
    agent.execution_registry = None
    agent.external_resources = ()
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
