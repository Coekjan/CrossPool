from pathlib import Path

import pytest

from xkit.serving.sglang.graph import SglangGraphMode
from xpool.integrations.sglang.devkit import SglangGraphEvent
from xpool.model import ModelId
from xtest.harness.sglang.serving.attempt import ProbeRun
from xtest.harness.sglang.serving.server import SglangInferenceRecord, SglangServerResult
from xtest.harness.support.config import TEST_MODEL_ID
from xtest.harness.support.sglang.graph import assert_run_graph_evidence


@pytest.mark.parametrize(
    ("inside_request", "second_model_operations", "second_model_replays", "passed"),
    [
        (True, ("capture", "execute"), (8,), True),
        (False, ("capture", "execute"), (8,), False),
        (True, ("capture",), (), False),
        (True, ("execute",), (8,), False),
        (True, (), (), False),
        (True, ("capture", "execute"), (4, 4), False),
        (True, ("capture", "execute"), (8, 8), True),
    ],
)
def test_graph_proof_belongs_to_each_model_and_measured_request(
    tmp_path: Path,
    inside_request: bool,
    second_model_operations: tuple[str, ...],
    second_model_replays: tuple[int, ...],
    passed: bool,
) -> None:
    settings = SglangGraphMode.DECODE_FULL.settings()
    model_ids = (TEST_MODEL_ID, ModelId("test/second-model"))
    results = []
    events: list[SglangGraphEvent] = []
    for pid, model_id in enumerate(model_ids, start=1):
        path = tmp_path / f"inference-{pid}.json"
        SglangInferenceRecord(model_id, "http://localhost/generate", {}, 200, "application/json", {}, 100, 200).write(
            path
        )
        results.append(SglangServerResult(model_id, settings, "generation", (1,), tuple(range(8)), 0, path, None))
        operations = ("capture", "execute") if pid == 1 else second_model_operations
        if "capture" in operations:
            events.extend(
                SglangGraphEvent(
                    pid=pid,
                    model_id=model_id,
                    time_ns=timestamp,
                    forward_phase="decode",
                    backend_class="FullCudaGraphBackend",
                    event=event,
                )
                for timestamp, event in ((1, "capture_begin"), (2, "capture_end"))
            )
        if "execute" in operations:
            start = 110 if pid == 1 or inside_request else 210
            replays = (8,) if pid == 1 else second_model_replays
            events.extend(
                SglangGraphEvent(
                    pid=pid + rank * len(model_ids),
                    model_id=model_id,
                    time_ns=start + index * 2 + offset,
                    forward_phase="decode",
                    backend_class="FullCudaGraphBackend",
                    event=event,
                )
                for rank, count in enumerate(replays)
                for index in range(count)
                for offset, event in enumerate(("execute_begin", "execute_end"))
            )
    run = ProbeRun(model_ids, tmp_path, settings, tuple(results), events, 0.0, 1.0)

    if passed:
        assert_run_graph_evidence(run)
    else:
        with pytest.raises(AssertionError):
            assert_run_graph_evidence(run)
