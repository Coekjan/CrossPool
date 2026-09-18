"""Routine installed-serving regression on the two small Qwen checkpoints."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eManifest, E2eServingCase
from tests.harness.sglang.serving.graph import SglangGraphMode
from tests.harness.sglang.serving.qualification import case_parameter, run_serving_case
from xpool.config import XpoolConfig

pytest_plugins = ("tests.harness.support.config",)
MANIFEST = E2eManifest.load(E2E_MANIFEST_PATH)


@pytest.mark.parametrize(
    ("case", "graph_mode"),
    tuple(
        case_parameter(
            case,
            graph_mode,
            models=tuple(MANIFEST.model(placement.model_id) for placement in case.models),
            compare_modes=False,
        )
        for case in MANIFEST.model_serving_cases
        if case.elastic_kv is None
        for graph_mode in case.graph_modes
    ),
)
def test_e2e_model_serving(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: XpoolConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed serving for the selected graph mode."""

    run_serving_case(
        case,
        graph_mode,
        e2e_base_config,
        tmp_path,
        task_artifact_dir,
        models=tuple(MANIFEST.model(placement.model_id) for placement in case.models),
        serving_slo=MANIFEST.serving_slo,
        compare_modes=False,
    )
