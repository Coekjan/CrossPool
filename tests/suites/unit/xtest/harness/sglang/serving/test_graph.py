from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from xpool.integrations.sglang.devkit import SglangGraphEvent
from xtest.harness.sglang.serving.graph import read_graph_events


@pytest.mark.parametrize("pid", [7, "invalid"])
def test_graph_reader_validates_fields_and_preserves_order_and_extensions(tmp_path: Path, pid: int | str) -> None:
    """Read producer-shaped JSONL records without closing the event-name domain."""

    event: SglangGraphEvent = {
        "pid": 7,
        "time_ns": 100,
        "forward_phase": "decode",
        "backend_class": "FullCudaGraphBackend",
        "event": "diagnostic",
    }
    external = {**event, "pid": pid, "extension": {"enabled": True}}
    (tmp_path / "xpool.graph-observer.2.jsonl").write_text(json.dumps(external) + "\n\n", encoding="utf-8")
    (tmp_path / "xpool.graph-observer.1.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")

    if isinstance(pid, str):
        with pytest.raises(ValidationError, match="pid"):
            read_graph_events(tmp_path)
    else:
        assert read_graph_events(tmp_path) == [event, external]
