"""Explicit SGLang numerical and graph qualification for zai-org/GLM-4.7-Flash."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.sglang import numerical
from tests.harness.sglang.manifest import (
    E2E_MANIFEST_PATH,
    E2eFfnInputMatrix,
    E2eFfnNumericalCase,
    E2eManifest,
    E2eModel,
    E2eModelPlacement,
    E2eServingCase,
)
from tests.harness.sglang.serving import qualification
from tests.harness.sglang.serving.graph import SglangGraphMode
from xpool.config import XpoolConfig

pytest_plugins = ("tests.harness.support.config",)

MODEL = E2eModel(model_id="zai-org/GLM-4.7-Flash", architecture="Glm4MoeLiteForCausalLM")
SERVING_SLO = E2eManifest.load(E2E_MANIFEST_PATH).serving_slo
NUMERICAL_CASES: tuple[E2eFfnNumericalCase, ...] = (
    E2eFfnNumericalCase(
        id="glm4-7-flash-ffn-numerical",
        model_placement=E2eModelPlacement(model_id=MODEL.model_id, atn_tp_size=1, atn_dp_size=1),
        layer_ids=(0, 1, 46),
        input_matrix=E2eFfnInputMatrix(seed=17, row_counts=(1, 32, 4096)),
        ffnagent_count=2,
        executor_lane_count=1,
        ffn_tp_size=2,
        estimated_duration_seconds=900,
        timeout_seconds=1800,
    ),
)
SERVING_CASES: tuple[E2eServingCase, ...] = (
    E2eServingCase(
        id="glm4-7-flash-tp1-dp1-f2-e1",
        models=(E2eModelPlacement(model_id=MODEL.model_id, atn_tp_size=1, atn_dp_size=1),),
        ffnagent_count=2,
        executor_lane_count=1,
        graph_modes=(SglangGraphMode.EAGER, SglangGraphMode.DECODE_FULL, SglangGraphMode.PREFILL_BREAKABLE),
        estimated_duration_seconds=700,
        timeout_seconds=3600,
        transport_record_capacity=32768,
        fabric_record_capacity=32768,
    ),
    E2eServingCase(
        id="glm4-7-flash-tp1-dp2-f2-e1",
        models=(E2eModelPlacement(model_id=MODEL.model_id, atn_tp_size=1, atn_dp_size=2),),
        ffnagent_count=2,
        executor_lane_count=1,
        graph_modes=(SglangGraphMode.DECODE_FULL,),
        estimated_duration_seconds=800,
        timeout_seconds=3600,
        transport_record_capacity=32768,
        fabric_record_capacity=32768,
    ),
)


@pytest.mark.parametrize(
    "case",
    tuple(numerical.case_parameter(case, model=MODEL) for case in NUMERICAL_CASES),
)
def test_ffn_numerical(
    case: E2eFfnNumericalCase,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Compare installed FFN with an independent SGLang reference."""

    numerical.run_numerical_case(case, e2e_base_config, tmp_path, task_artifact_dir, model=MODEL)


@pytest.mark.parametrize(
    ("case", "graph_mode"),
    tuple(
        qualification.case_parameter(case, graph_mode, models=(MODEL,), compare_modes=len(case.graph_modes) > 1)
        for case in SERVING_CASES
        for graph_mode in case.graph_modes
    ),
)
def test_serving_graph(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed graph modes and their cross-mode alignment."""

    qualification.run_serving_case(
        case,
        graph_mode,
        e2e_base_config,
        tmp_path,
        task_artifact_dir,
        models=(MODEL,),
        serving_slo=SERVING_SLO,
        compare_modes=len(case.graph_modes) > 1,
    )
