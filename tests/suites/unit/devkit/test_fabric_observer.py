from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.devkit import observer_enabled_config
from xpool.devkit.fabric_observer import record_payload, write_fabric_snapshot
from xpool.fabric import FabricGenerationId
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


@dataclass(frozen=True)
class FakeKey:
    """Minimal bound Invocation Key replacement."""

    instance_index: int = 2
    invocation_sequence: int = 3


@dataclass(frozen=True)
class FakeRecord:
    """Minimal target Fabric trace record replacement."""

    kind: object
    local_trace_id: int = 1
    key: FakeKey = FakeKey()
    layer_ordinal: int = 4
    payload_rows: int = 8
    output_requirement: OutputRequirement = OutputRequirement.GROUP_SUM_COMPLETE
    dp_row_layout: DpRowLayout = DpRowLayout.NONE
    dp_rank_payload_rows: int = 8
    forward_mode: ForwardMode = ForwardMode.DECODE
    executor_lane_index: int = 1
    executor_lease_sequence: int = 7
    ready_ticket: int = 9
    payload_row_capacity: int = 16
    delivery: xpool.native.fabric.DeliveryVariant = xpool.native.fabric.DeliveryVariant.SINGLE_COMPLETE

    def timestamp(self, event: object) -> int:
        """Return one deterministic nonzero event timestamp."""

        return int(cast(xpool.native.devkit.fabric_observer.AtnAgentEvent, event)) + 1


@pytest.mark.parametrize(
    ("kind", "expected_kind", "fact_name", "expected_fact"),
    [
        (xpool.native.devkit.fabric_observer.RecordKind.ATNAGENT, "atnagent", "forward_mode", "decode"),
        (
            xpool.native.devkit.fabric_observer.RecordKind.COORDINATOR,
            "coordinator",
            "scheduler",
            {"policy": "fifo", "ready_ticket": 9},
        ),
        (xpool.native.devkit.fabric_observer.RecordKind.FFNAGENT, "ffnagent", "delivery", "single_complete"),
    ],
)
def test_record_payload_uses_target_trace_domains(
    kind: object,
    expected_kind: str,
    fact_name: str,
    expected_fact: object,
) -> None:
    payload = record_payload(cast("xpool.native.devkit.fabric_observer.Record", FakeRecord(kind)))

    assert payload["kind"] == expected_kind
    assert payload["payload_rows"] == 8
    assert payload["output_requirement"] == "group_sum_complete"
    assert payload["dp_row_layout"] == "none"
    assert cast(dict[str, object], payload["facts"])[fact_name] == expected_fact


def test_write_fabric_snapshot_uses_generation_pe_identity(tmp_path: Path) -> None:
    output = tmp_path / "fabric-observer"
    output.mkdir()
    install_test_config(observer_enabled_config("fabric_observer", output))
    generation = FabricGenerationId(high=1, low=2)
    snapshot = cast(
        "xpool.native.devkit.fabric_observer.Snapshot",
        SimpleNamespace(
            pe=5,
            atnagent_count=1,
            ffnagent_count=2,
            model_topologies=(SimpleNamespace(atn_tp_size=1, atn_dp_size=1),),
            sequence=1,
            dropped=0,
            records=(FakeRecord(xpool.native.devkit.fabric_observer.RecordKind.FFNAGENT),),
        ),
    )

    path = write_fabric_snapshot(generation, snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert path.name == f"xpool.fabric-observer.{generation.format()}.5.json"
    assert payload["records"][0]["kind"] == "ffnagent"
    assert payload["model_topologies"] == [{"atn_tp_size": 1, "atn_dp_size": 1}]
