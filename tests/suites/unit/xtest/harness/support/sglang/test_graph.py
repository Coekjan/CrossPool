from pathlib import Path

import pytest

from xkit.serving.sglang.graph import SglangGraphMode
from xpool.integrations.sglang.devkit import SglangGraphEvent
from xtest.harness.sglang.serving.attempt import ProbeRun
from xtest.harness.sglang.serving.server import SglangInferenceRecord, SglangServerResult
from xtest.harness.support.config import TEST_MODEL_ID
from xtest.harness.support.sglang.graph import assert_run_graph_evidence


@pytest.mark.parametrize("inside_request", [False, True])
def test_graph_proof_requires_execution_during_measured_request(tmp_path: Path, inside_request: bool) -> None:
    path = tmp_path / "inference.json"
    SglangInferenceRecord(TEST_MODEL_ID, "http://localhost/generate", {}, 200, "application/json", {}, 100, 200).write(
        path
    )
    settings = SglangGraphMode.DECODE_FULL.settings()
    result = SglangServerResult(TEST_MODEL_ID, settings, "generation", (1,), tuple(range(8)), 0, path, None)
    events = [
        SglangGraphEvent(
            pid=1, time_ns=timestamp, forward_phase="decode", backend_class="FullCudaGraphBackend", event=event
        )
        for timestamp, event in ((1, "capture_begin"), (2, "capture_end"))
    ]
    start = 110 if inside_request else 210
    events.extend(
        SglangGraphEvent(
            pid=1,
            time_ns=start + index * 2 + offset,
            forward_phase="decode",
            backend_class="FullCudaGraphBackend",
            event=event,
        )
        for index in range(8)
        for offset, event in enumerate(("execute_begin", "execute_end"))
    )
    run = ProbeRun((TEST_MODEL_ID,), tmp_path, settings, (result,), events, 0.0, 1.0)

    if inside_request:
        assert_run_graph_evidence(run)
    else:
        with pytest.raises(AssertionError):
            assert_run_graph_evidence(run)
