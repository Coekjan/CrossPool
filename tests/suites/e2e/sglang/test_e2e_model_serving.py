"""Routine installed-serving regression on the two small Qwen checkpoints."""

from __future__ import annotations

from pathlib import Path

import xtest
from xkit.serving.sglang.graph import SglangGraphMode
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import E2eServingCase
from xtest.harness.sglang.serving import qualification

pytest_plugins = ("xtest.harness.support.config",)


@xtest.parameterize(("case", "graph_mode"), rows=qualification.graph_rows)
@xtest.requirements(qualification.requirements_of)
def test_e2e_model_serving(
    case: E2eServingCase,
    graph_mode: SglangGraphMode,
    e2e_base_config: ResolvedConfig,
    tmp_path: Path,
    task_artifact_dir: Path | None,
) -> None:
    """Prove installed serving for the selected graph mode."""

    qualification.run_serving_case(
        case,
        graph_mode,
        e2e_base_config,
        tmp_path,
        task_artifact_dir,
        compare_modes=False,
    )
