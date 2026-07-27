from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.devkit.common.fabric_observer
import xpool.native
from tests.harness.config import install_test_config
from xpool.config import XpoolConfig
from xpool.fabric import FabricGeneration, FabricParticipantPhase

production_advance = xpool.devkit.common.fabric_observer.Agent.advance_fabric_lifecycle


@dataclass(slots=True)
class FakeFabricRecord:
    """Behavioral fake for the bound read-only trace projection."""

    kind: object
    events: dict[object, int]
    local_trace_id: int = 1
    key: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(model_index=2, invocation_sequence=3))
    layer_ordinal: int = 4
    layer_id: int = 6
    result_handoff: int = 1
    dp_padding_mode: int = 0
    submission_payload_rows: int | None = None
    invocation_payload_rows: int | None = None
    local_token_count: int | None = None
    forward_mode: int | None = None
    input_pe: int | None = None
    executor_index: int | None = None
    execution_mode: int | None = None
    result_contribution: int | None = None
    ready_ticket: int | None = None

    def timestamp(self, event: object) -> int:
        """Return one raw event timestamp, including zero for absence."""

        return self.events.get(event, 0)

    def recorded(self, event: object) -> bool:
        """Return whether one event is present."""

        return self.timestamp(event) != 0


@pytest.fixture
def reset_fabric_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the production Fabric lifecycle method around each test."""

    monkeypatch.setattr(xpool.devkit.common.fabric_observer, "installed", False)
    monkeypatch.setattr(xpool.devkit.common.fabric_observer.Agent, "advance_fabric_lifecycle", production_advance)
    yield


pytestmark = pytest.mark.usefixtures(reset_fabric_observer.__name__, "reset_global_config")


def test_write_fabric_snapshot_projects_atnagent_trace(tmp_path: Path) -> None:
    """Fabric output preserves topology, semantic facts, raw events, and valid durations."""

    event = xpool.native.AtnAgentTraceEvent
    record = FakeFabricRecord(
        kind=xpool.native.FabricTraceKind.ATNAGENT,
        events={
            event.SUBMISSION_PREPARED: 100,
            event.DECODE_INPUT_STAGED: 110,
            event.SUBMISSION_PUBLISHED: 120,
            event.ADMISSION_OBSERVED: 130,
            event.RESULT_OBSERVED: 190,
            event.OUTPUT_PREPARED: 200,
            event.TRANSPORT_EVALUATED_PUBLISHED: 205,
            event.ACKNOWLEDGEMENT_PUBLISHED: 210,
        },
        submission_payload_rows=7,
        local_token_count=7,
        forward_mode=2,
        executor_index=1,
        execution_mode=2,
        result_contribution=1,
    )
    snapshot = cast(
        xpool.native.FabricTraceSnapshot,
        SimpleNamespace(
            pe=0,
            atnagent_count=1,
            ffnagent_count=1,
            model_topologies=(SimpleNamespace(atn_tp_size=1, atn_dp_size=1),),
            sequence=1,
            dropped=0,
            records=(record,),
        ),
    )
    install_test_config(config=fabric_observer_config(tmp_path))

    path = xpool.devkit.common.fabric_observer.write_fabric_snapshot(FabricGeneration(high=1, low=2), snapshot)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "generation": "00000000000000010000000000000002",
        "pe": 0,
        "atnagent_count": 1,
        "ffnagent_count": 1,
        "model_topologies": [{"atn_tp_size": 1, "atn_dp_size": 1}],
        "sequence": 1,
        "dropped": 0,
        "records": [
            {
                "local_trace_id": 1,
                "kind": "atnagent",
                "model_index": 2,
                "invocation_sequence": 3,
                "layer_ordinal": 4,
                "layer_id": 6,
                "result_handoff": "replicated_full",
                "dp_padding_mode": "none",
                "facts": {
                    "submission_payload_rows": 7,
                    "local_token_count": 7,
                    "forward_mode": "decode",
                    "executor_index": 1,
                    "execution_mode": "decode",
                    "result_contribution": "full",
                },
                "events_ns": {
                    "submission_prepared": 100,
                    "decode_input_staged": 110,
                    "submission_published": 120,
                    "admission_observed": 130,
                    "prefill_input_staged": 0,
                    "prefill_input_published": 0,
                    "result_observed": 190,
                    "output_prepared": 200,
                    "transport_evaluated_published": 205,
                    "acknowledgement_published": 210,
                },
                "durations_ns": {
                    "submission": 20,
                    "decode_input_staging": 10,
                    "admission_wait": 10,
                    "result_wait": 60,
                    "output_preparation": 10,
                    "transport_evaluation_publication": 5,
                    "acknowledgement_publication": 5,
                    "total": 110,
                },
            }
        ],
    }


@pytest.mark.parametrize(
    ("ready_ticket", "expected_scheduler"),
    [
        (7, {"policy": "fifo", "ready_ticket": 7}),
        (None, {"policy": "random"}),
    ],
)
def test_coordinator_record_projects_scheduler_policy(
    ready_ticket: int | None,
    expected_scheduler: dict[str, object],
) -> None:
    """Coordinator traces expose the policy-specific admission fact."""

    event = xpool.native.CoordinatorTraceEvent
    record = FakeFabricRecord(
        kind=xpool.native.FabricTraceKind.COORDINATOR,
        events={
            event.ENQUEUED: 100,
            event.SCHEDULED: 110,
            event.ADMISSIONS_PUBLISHED: 120,
            event.INVOCATIONS_PUBLISHED: 130,
            event.COMPLETIONS_OBSERVED: 160,
            event.RESULTS_PUBLISHED: 170,
            event.ACKNOWLEDGEMENTS_OBSERVED: 180,
            event.SCHEDULER_RELEASED: 190,
        },
        invocation_payload_rows=7,
        input_pe=0,
        execution_mode=2,
        executor_index=1,
        ready_ticket=ready_ticket,
    )
    payload = xpool.devkit.common.fabric_observer.record_payload(cast(xpool.native.FabricTraceRecord, record))

    assert payload["kind"] == "coordinator"
    assert payload["facts"] == {
        "invocation_payload_rows": 7,
        "input_pe": 0,
        "execution_mode": "decode",
        "executor_index": 1,
        "scheduler": expected_scheduler,
    }
    assert cast(dict[str, int], payload["durations_ns"])["scheduling"] == 10
    assert cast(dict[str, int], payload["durations_ns"])["total"] == 90


@pytest.mark.parametrize(
    ("execution_mode", "mode_event", "expected_mode"),
    [
        (1, "prefill", "prefill"),
        (2, "decode", "decode"),
    ],
)
def test_execution_record_projects_mode_specific_events(
    execution_mode: int,
    mode_event: str,
    expected_mode: str,
) -> None:
    """Execution traces derive readiness from the selected data-transfer protocol."""

    event = xpool.native.ExecutionTraceEvent
    events: dict[object, int] = {
        event.INVOCATION_OBSERVED: 100,
        event.EXECUTION_STARTED: 130,
        event.EXECUTION_COMPLETED: 160,
        event.COMPLETION_PUBLISHED: 170,
    }
    if mode_event == "decode":
        events[event.DECODE_INPUT_PULL_STARTED] = 110
        events[event.DECODE_INPUT_PULL_COMPLETED] = 120
    else:
        events[event.PREFILL_INPUT_READY_OBSERVED] = 120
    record = FakeFabricRecord(
        kind=xpool.native.FabricTraceKind.EXECUTION,
        events=events,
        invocation_payload_rows=7,
        input_pe=0,
        execution_mode=execution_mode,
        executor_index=1,
    )
    payload = xpool.devkit.common.fabric_observer.record_payload(cast(xpool.native.FabricTraceRecord, record))

    assert payload["kind"] == "execution"
    assert cast(dict[str, object], payload["facts"])["execution_mode"] == expected_mode
    assert cast(dict[str, int], payload["durations_ns"])["input_readiness"] == 20
    assert cast(dict[str, int], payload["durations_ns"])["execution"] == 30


def test_install_reads_snapshot_when_local_drain_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed observer reads one local snapshot after terminal drain."""

    generation = FabricGeneration(high=1, low=2)
    snapshot = cast(
        xpool.native.FabricTraceSnapshot,
        SimpleNamespace(
            pe=1,
            atnagent_count=1,
            ffnagent_count=1,
            model_topologies=(),
            sequence=0,
            dropped=0,
            records=(),
        ),
    )
    calls: list[str] = []

    def complete_drain(agent: object) -> None:
        calls.append("advance")
        cast(SimpleNamespace, agent).participant_report.phase = FabricParticipantPhase.DRAINED

    monkeypatch.setattr(xpool.devkit.common.fabric_observer.Agent, "advance_fabric_lifecycle", complete_drain)
    monkeypatch.setattr(xpool.native.fabric, "read_trace", lambda: snapshot)
    install_test_config(config=fabric_observer_config(tmp_path))
    xpool.devkit.common.fabric_observer.install()

    agent = cast(
        xpool.devkit.common.fabric_observer.Agent,
        SimpleNamespace(
            fabric_plan=SimpleNamespace(generation=generation),
            participant_report=SimpleNamespace(phase=FabricParticipantPhase.DRAINING),
        ),
    )
    xpool.devkit.common.fabric_observer.Agent.advance_fabric_lifecycle(agent)

    assert calls == ["advance"]
    assert len(list(tmp_path.glob("xpool.fabric-observer.*.json"))) == 1


def test_native_lifecycle_failure_prevents_trace_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Native drain failures propagate before observation begins."""

    reads: list[str] = []

    def fail_advance(agent: object) -> None:
        raise RuntimeError("fabric drain failed")

    monkeypatch.setattr(xpool.devkit.common.fabric_observer.Agent, "advance_fabric_lifecycle", fail_advance)
    monkeypatch.setattr(xpool.native.fabric, "read_trace", lambda: reads.append("read"))
    install_test_config(config=fabric_observer_config(tmp_path))
    xpool.devkit.common.fabric_observer.install()

    agent = cast(
        xpool.devkit.common.fabric_observer.Agent,
        SimpleNamespace(
            fabric_plan=SimpleNamespace(generation=FabricGeneration(high=1, low=2)),
            participant_report=SimpleNamespace(phase=FabricParticipantPhase.DRAINING),
        ),
    )
    with pytest.raises(RuntimeError, match="fabric drain failed"):
        xpool.devkit.common.fabric_observer.Agent.advance_fabric_lifecycle(agent)

    assert reads == []


def test_snapshot_write_failure_warns_without_blocking_drain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observer I/O failure does not turn successful native drain into failure."""

    snapshot = cast(
        xpool.native.FabricTraceSnapshot,
        SimpleNamespace(
            pe=1,
            atnagent_count=1,
            ffnagent_count=1,
            model_topologies=(),
            sequence=0,
            dropped=0,
            records=(),
        ),
    )

    def complete_drain(agent: object) -> None:
        cast(SimpleNamespace, agent).participant_report.phase = FabricParticipantPhase.DRAINED

    def fail_write(generation: FabricGeneration, snapshot: xpool.native.FabricTraceSnapshot) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(xpool.devkit.common.fabric_observer.Agent, "advance_fabric_lifecycle", complete_drain)
    monkeypatch.setattr(xpool.native.fabric, "read_trace", lambda: snapshot)
    monkeypatch.setattr(xpool.devkit.common.fabric_observer, "write_fabric_snapshot", fail_write)
    install_test_config(config=fabric_observer_config(tmp_path))
    xpool.devkit.common.fabric_observer.install()
    agent = cast(
        xpool.devkit.common.fabric_observer.Agent,
        SimpleNamespace(
            fabric_plan=SimpleNamespace(generation=FabricGeneration(high=1, low=2)),
            participant_report=SimpleNamespace(phase=FabricParticipantPhase.DRAINING),
        ),
    )

    with caplog.at_level("WARNING", logger="xpool.devkit.common.fabric_observer"):
        assert xpool.devkit.common.fabric_observer.Agent.advance_fabric_lifecycle(agent) is None

    assert agent.participant_report is not None
    assert agent.participant_report.phase is FabricParticipantPhase.DRAINED
    assert "Failed to record xpool Fabric observer snapshot" in caplog.text


def fabric_observer_config(outdir: Path) -> XpoolConfig:
    """Return a config with native Fabric observation enabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_FABRIC_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_FABRIC_OBSERVER_OUTDIR": str(outdir),
        },
    )
