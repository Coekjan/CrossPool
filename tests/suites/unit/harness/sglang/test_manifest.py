"""Declarative SGLang E2E manifest schema and catalog behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.harness.sglang.graph import SglangGraphMode
from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eManifest, E2eModelPlacement, E2eServingCase


def test_manifest_loads_complete_serving_catalog() -> None:
    manifest = E2eManifest.load(E2E_MANIFEST_PATH)

    assert manifest.models
    assert manifest.model_serving_cases
    assert manifest.loopback_serving_cases
    assert all(
        case.required_gpu_count == case.atnagent_count + case.ffnagent_count for case in manifest.model_serving_cases
    )


def test_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "e2e.toml"
    manifest_path.write_text(
        """
[models.synthetic]
model_id = "organization/model"
architecture = "SyntheticForCausalLM"
unknown = true

[[model_serving_cases]]
id = "synthetic"
models = [{ model = "synthetic", atn_tp_size = 1, atn_dp_size = 1 }]
ffnagent_count = 1
executor_count = 1
graph_modes = ["eager"]
estimated_duration_seconds = 1
timeout_seconds = 1
transport_trace_capacity = 1
fabric_trace_capacity = 1

[[loopback_serving_cases]]
id = "synthetic-sites"
models = [{ model = "synthetic", atn_tp_size = 1, atn_dp_size = 1 }]
ffnagent_count = 1
executor_count = 1
sites = ["instance"]
graph_modes = ["eager"]
estimated_duration_seconds = 1
timeout_seconds = 1
transport_trace_capacity = 1
fabric_trace_capacity = 1
""",
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="unknown"):
        E2eManifest.load(manifest_path)


def test_serving_case_allows_fewer_executors_than_ffnagents() -> None:
    case = E2eServingCase(
        id="shared-executor",
        models=(E2eModelPlacement(model="synthetic", atn_tp_size=1, atn_dp_size=1),),
        ffnagent_count=2,
        executor_count=1,
        graph_modes=(SglangGraphMode.EAGER,),
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_trace_capacity=1,
        fabric_trace_capacity=1,
    )

    assert case.ffnagent_count == 2
    assert case.executor_count == 1


def test_model_placement_rejects_combined_attention_tp_by_dp() -> None:
    with pytest.raises(ValidationError, match="combined attention TP-by-DP"):
        E2eModelPlacement(model="synthetic", atn_tp_size=2, atn_dp_size=2)


def test_serving_case_rejects_heterogeneous_attention_topology() -> None:
    with pytest.raises(ValidationError, match="same attention topology"):
        serving_case(
            models=(
                E2eModelPlacement(model="first", atn_tp_size=1, atn_dp_size=1),
                E2eModelPlacement(model="second", atn_tp_size=2, atn_dp_size=1),
            )
        )


def test_serving_case_rejects_piecewise_graph_under_attention_dp() -> None:
    with pytest.raises(ValidationError, match="attention DP does not support piecewise"):
        serving_case(
            models=(E2eModelPlacement(model="synthetic", atn_tp_size=1, atn_dp_size=2),),
            graph_modes=(SglangGraphMode.PIECEWISE,),
        )


def serving_case(
    *,
    models: tuple[E2eModelPlacement, ...],
    graph_modes: tuple[SglangGraphMode, ...] = (SglangGraphMode.EAGER,),
) -> E2eServingCase:
    return E2eServingCase(
        id="synthetic-case",
        models=models,
        ffnagent_count=1,
        executor_count=1,
        graph_modes=graph_modes,
        estimated_duration_seconds=1,
        timeout_seconds=1,
        transport_trace_capacity=1,
        fabric_trace_capacity=1,
    )
