from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import safetensors
import torch

import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.devkit.ffn_routing_observer import write_routing_snapshot
from xpool.fabric import FabricGenerationId

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_routing_snapshot_preserves_compact_tensor_evidence(tmp_path: Path) -> None:
    output = tmp_path / "routing-observer"
    output.mkdir()
    install_test_config(observer_enabled_config("ffn_routing_observer", output))
    generation = FabricGenerationId(high=1, low=2)
    ids = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]], dtype=torch.float32)
    snapshot = cast(
        "xpool.native.devkit.ffn_routing_observer.Snapshot",
        SimpleNamespace(
            sequence=2,
            dropped=1,
            records=(
                SimpleNamespace(
                    key=SimpleNamespace(instance_index=7, invocation_sequence=11),
                    layer_ordinal=13,
                    topk_ids=ids,
                    topk_weights=weights,
                ),
            ),
        ),
    )

    path = write_routing_snapshot(generation, 5, snapshot)

    assert path.name == f"xpool.ffn-routing-observer.{generation.format()}.5.safetensors"
    with safetensors.safe_open(path, framework="pt", device="cpu") as artifact:
        assert artifact.metadata() == {
            "generation": generation.format(),
            "pe": "5",
            "sequence": "2",
            "dropped": "1",
            "records": json.dumps(
                [{"instance_index": 7, "invocation_sequence": 11, "layer_ordinal": 13}],
                separators=(",", ":"),
            ),
        }
        assert set(artifact.keys()) == {"records.0.topk_ids", "records.0.topk_weights"}
        assert torch.equal(artifact.get_tensor("records.0.topk_ids"), ids)
        assert torch.equal(artifact.get_tensor("records.0.topk_weights"), weights)
