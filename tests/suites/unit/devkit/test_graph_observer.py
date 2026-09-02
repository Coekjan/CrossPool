from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import cuda.bindings.runtime  # ty: ignore[unresolved-import]
import pytest

import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.devkit import graph_observer
from xpool.fabric import FabricGenerationId
from xpool.runtime.ffnagent.agent import FfnAgent

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_graph_snapshot_serializes_native_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "graph-observer"
    output.mkdir()
    install_test_config(observer_enabled_config("graph_observer", output))
    generation = FabricGenerationId(high=1, low=2)
    agent = cast(
        FfnAgent,
        SimpleNamespace(
            fabric_plan=SimpleNamespace(generation=generation),
            cuda_device=2,
            fabric_pe=lambda: 5,
        ),
    )
    snapshot = cast(
        xpool.native.devkit.graph_observer.Snapshot,
        SimpleNamespace(
            primary_graphs=(
                SimpleNamespace(
                    node_counts={int(cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeKernel): 3},
                    binding_site_count=7,
                ),
            ),
            lane_graphs=(
                SimpleNamespace(
                    node_counts={int(cuda.bindings.runtime.cudaGraphNodeType.cudaGraphNodeTypeConditional): 2},
                    compute_branch_count=1,
                    delivery_branch_count=2,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        graph_observer.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(uuid="GPU-test"),
    )

    path = graph_observer.write_graph_snapshot(agent, snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == f"xpool.graph-observer.{generation.format()}.5.json"
    assert payload["gpu_uuid"] == "GPU-test"
    assert payload["primary_graphs"][0]["node_counts"] == {"cudaGraphNodeTypeKernel": 3}
    assert payload["primary_graphs"][0]["binding_site_count"] == 7
    assert payload["lane_graphs"][0]["node_counts"] == {"cudaGraphNodeTypeConditional": 2}
