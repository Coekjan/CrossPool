from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import TypeAdapter, ValidationError

from xpool.integrations.sglang.devkit import SglangGraphEvent
from xtest.harness.sglang.serving.graph import read_graph_events
from xtest.harness.support.config import TEST_MODEL_ID


@pytest.mark.parametrize("pid", [7, "invalid"])
def test_graph_reader_validates_fields_and_preserves_order_and_extensions(tmp_path: Path, pid: int | str) -> None:
    """Read producer-shaped JSONL records without closing the event-name domain."""

    event: SglangGraphEvent = {
        "pid": 7,
        "model_id": TEST_MODEL_ID,
        "time_ns": 100,
        "forward_phase": "decode",
        "backend_class": "FullCudaGraphBackend",
        "event": "diagnostic",
    }
    external = {**event, "pid": pid, "extension": {"enabled": True}}
    (tmp_path / "xpool.graph-observer.2.jsonl").write_text(
        json.dumps({**external, "model_id": str(TEST_MODEL_ID)}) + "\n\n", encoding="utf-8"
    )
    (tmp_path / "xpool.graph-observer.1.jsonl").write_bytes(TypeAdapter(SglangGraphEvent).dump_json(event) + b"\n")

    if isinstance(pid, str):
        with pytest.raises(ValidationError, match="pid"):
            read_graph_events(tmp_path)
    else:
        assert read_graph_events(tmp_path) == [event, external]
