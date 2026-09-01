from __future__ import annotations

import os
from types import SimpleNamespace
from typing import cast

import pytest
import torch

import xpool.runtime.ffnagent.agent
from tests.harness.support.config import install_test_config, reset_global_config, synthetic_config
from xpool.fabric import FabricPlan
from xpool.memory import DeviceMemoryEstimate
from xpool.native import RuntimeRole
from xpool.runtime.agent import AgentError
from xpool.runtime.ffnagent.agent import FfnAgent
from xpool.runtime.ffnagent.registry import FfnExecutionRegistry
from xpool.runtime.ffnagent.weights import FfnLayerWeights

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_ffnagent_rejects_cuda_initialized_before_workspace_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(AgentError, match="CUDA initialized before"):
        FfnAgent(cuda_device=1)


def test_ffnagent_installs_zero_workspace_policy_before_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1, 2))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.Agent,
        "__init__",
        lambda self, **kwargs: setattr(self, "proc_id", SimpleNamespace(pid=1)),
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "ensure_supported_cuda_allocator", lambda: None)
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "load", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "FfnAgentRegistration", lambda **kwargs: object())
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "AgentHeartbeat", lambda **kwargs: object())

    FfnAgent(cuda_device=1)

    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":0:0"


def test_ffnagent_rejects_incompatible_calibration_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.Agent,
        "__init__",
        lambda self, **kwargs: setattr(self, "proc_id", SimpleNamespace(pid=1)),
    )
    monkeypatch.setattr(xpool.runtime.ffnagent.agent, "ensure_supported_cuda_allocator", lambda: None)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "load_memory_calibration_profile",
        lambda: SimpleNamespace(
            environment=SimpleNamespace(
                ffnagent_gpus=(
                    SimpleNamespace(
                        name="expected",
                        compute_capability=(8, 0),
                        total_memory_bytes=80,
                    ),
                )
            )
        ),
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(name="actual", major=8, minor=0, total_memory=80),
    )

    with pytest.raises(AgentError, match="name: expected 'expected', found 'actual'"):
        FfnAgent(cuda_device=1)


def test_ffnagent_rejects_allocator_before_model_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def reject_allocator() -> None:
        raise RuntimeError("unsupported allocator")

    install_test_config(synthetic_config())
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent.Agent,
        "__init__",
        lambda self, **kwargs: setattr(self, "proc_id", SimpleNamespace(pid=1)),
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "ensure_supported_cuda_allocator",
        reject_allocator,
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "load",
        lambda **kwargs: pytest.fail("model loading must not run with an unsupported allocator"),
    )

    with pytest.raises(RuntimeError, match="unsupported allocator"):
        FfnAgent(cuda_device=1)


def test_ffnagent_prepares_weights_then_installs_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    config = synthetic_config()
    install_test_config(config)
    agent = object.__new__(FfnAgent)
    agent.fabric_plan = cast(FabricPlan, SimpleNamespace(pe_placements=(), instance_plans=()))
    agent.registration = SimpleNamespace(model_specs=("spec",))
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.cuda_device = 1
    agent.layer_weights = None
    agent.execution_registry = None
    layer_weights = cast(tuple[tuple[FfnLayerWeights | None, ...], ...], (("weights",),))
    registry = cast(FfnExecutionRegistry, object())
    observed: list[tuple[str, object]] = []
    monkeypatch.setattr(agent, "fabric_pe", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (1 << 30, 2 << 30))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "DeviceMemoryEstimator",
        lambda **arguments: SimpleNamespace(
            coefficients=None,
            estimate=lambda **estimate_arguments: DeviceMemoryEstimate(retained_bytes=1, peak_bytes=2),
        ),
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "materialize_layer_weights",
        lambda **arguments: observed.append(("weights", arguments)) or layer_weights,
    )
    monkeypatch.setattr(
        FfnExecutionRegistry,
        "materialize",
        lambda **arguments: observed.append(("registry", arguments)) or registry,
    )

    assert agent.prepare_fabric_join()
    agent.prepare_fabric_execution()

    assert agent.layer_weights is None
    assert agent.execution_registry is registry
    assert [name for name, _ in observed] == ["weights", "registry"]
    assert cast(dict[str, object], observed[1][1])["layer_weights"] is layer_weights


def test_ffnagent_rejects_prejoin_memory_shortfall(monkeypatch: pytest.MonkeyPatch) -> None:
    install_test_config(synthetic_config())
    agent = object.__new__(FfnAgent)
    agent.fabric_plan = cast(FabricPlan, SimpleNamespace(pe_placements=(), instance_plans=()))
    agent.registration = SimpleNamespace(model_specs=("spec",))
    agent.runtime_role = RuntimeRole.FFNAGENT
    agent.cuda_device = 1
    agent.layer_weights = None
    agent.execution_registry = None
    monkeypatch.setattr(agent, "fabric_pe", lambda: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (9, 10))
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "DeviceMemoryEstimator",
        lambda **arguments: SimpleNamespace(
            coefficients=None,
            estimate=lambda **estimate_arguments: DeviceMemoryEstimate(retained_bytes=8, peak_bytes=10),
        ),
    )
    monkeypatch.setattr(
        xpool.runtime.ffnagent.agent,
        "materialize_layer_weights",
        lambda **arguments: pytest.fail("weights must not load after failed admission"),
    )

    with pytest.raises(AgentError, match="analytic memory admission requires 10 bytes"):
        agent.prepare_fabric_join()
