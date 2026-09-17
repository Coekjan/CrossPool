"""Transport observer behavior tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.devkit.transport_observer
import xpool.native
from tests.harness.support.config import install_test_config, reset_global_config
from tests.harness.support.runtime.atnagent import transport_entry
from xpool.config import XpoolConfig
from xpool.devkit.transport_observer import write_transport_snapshot
from xpool.native import ABI_VERSION
from xpool.native.ffn import DpRowLayout, ForwardMode, OutputRequirement, ResultCode
from xpool.runtime.atnagent import AtnAgentTransportArenaState, AtnAgentTransportRuntime
from xpool.service.client import XpoolClient
from xpool.service.wire import ProcessRef
from xpool.transport import TransportArenaHandle

production_quiesce = AtnAgentTransportRuntime.quiesce


@pytest.fixture
def reset_transport_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the production transport_runtime quiesce method around each test."""

    monkeypatch.setattr(xpool.devkit.transport_observer, "installed", False)
    monkeypatch.setattr(AtnAgentTransportRuntime, "quiesce", production_quiesce)
    yield


pytestmark = pytest.mark.usefixtures(reset_transport_observer.__name__, reset_global_config.__name__)


def trace_record(**overrides: object) -> SimpleNamespace:
    """Return one complete acknowledged mailbox trace record."""

    values: dict[str, object] = {
        "trace_id": 1,
        "payload_rows": 3,
        "layer_ordinal": 2,
        "forward_mode": ForwardMode.DECODE,
        "output_requirement": OutputRequirement.PER_RANK_COMPLETE,
        "dp_row_layout": DpRowLayout.NONE,
        "result_code": ResultCode.OK,
        "request_staging_started": 100,
        "request_staging_completed": 110,
        "request_published": 120,
        "request_observed": 130,
        "execution_started": 140,
        "execution_completed": 160,
        "result_published": 170,
        "result_observed": 180,
        "output_copied": 190,
        "result_acknowledged": 200,
        "closed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def trace_snapshot(*records: SimpleNamespace) -> xpool.native.devkit.transport_observer.EndpointSnapshot:
    """Return a typed test view of one native snapshot."""

    return cast(
        xpool.native.devkit.transport_observer.EndpointSnapshot,
        SimpleNamespace(instance_index=0, instance_rank=0, sequence=len(records), dropped=0, records=records),
    )


def atnagent_snapshot(*records: SimpleNamespace) -> SimpleNamespace:
    """Return one AtnAgent-local observer snapshot."""

    return SimpleNamespace(endpoints=[trace_snapshot(*records)])


def test_write_transport_snapshot_serializes_structured_records(tmp_path: Path) -> None:
    """Observer output derives semantic values and phases from one mailbox trace."""

    install_test_config(config=transport_observer_config(tmp_path))
    path = write_transport_snapshot(
        site="instance",
        instance_id="model/0",
        rank=0,
        handle=TransportArenaHandle("00" * 64),
        snapshot=trace_snapshot(trace_record()),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["sequence"] == 1
    assert payload["dropped"] == 0
    assert payload["record_counts"] == {
        "retained": 1,
        "completed": 1,
        "closed": 0,
        "incomplete": 0,
    }
    record = payload["records"][0]
    assert record["forward_mode"] == "decode"
    assert record["output_requirement"] == "per_rank_complete"
    assert record["dp_row_layout"] == "none"
    assert record["result_code"] == "ok"
    assert record["durations_ns"]["result_acknowledged_total"] == 100
    assert payload["phase_summary"]["execution"]["median_ns"] == 20


def test_write_transport_snapshot_classifies_closed_record(tmp_path: Path) -> None:
    """A Staging-to-Closed request is terminal rather than incomplete."""

    record = trace_record(
        result_code=ResultCode.SHUTDOWN,
        request_staging_completed=0,
        request_published=0,
        request_observed=0,
        execution_started=0,
        execution_completed=0,
        result_published=0,
        result_observed=0,
        output_copied=0,
        result_acknowledged=0,
        closed=110,
    )
    install_test_config(config=transport_observer_config(tmp_path))
    path = write_transport_snapshot(
        site="instance",
        instance_id="model/0",
        rank=0,
        handle=TransportArenaHandle("00" * 64),
        snapshot=trace_snapshot(record),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["record_counts"] == {
        "retained": 1,
        "completed": 0,
        "closed": 1,
        "incomplete": 0,
    }
    assert payload["records"][0]["durations_ns"]["closed_total"] == 10


def test_install_records_snapshot_after_production_quiesce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed observer reads a snapshot after Resident drain and before destroy."""

    calls: list[str] = []

    def fake_quiesce(owner: AtnAgentTransportRuntime) -> None:
        calls.extend(resource.instance_id for resource in owner.resources)

    monkeypatch.setattr(AtnAgentTransportRuntime, "quiesce", fake_quiesce)
    monkeypatch.setattr(
        xpool.devkit.transport_observer,
        "get_runtime_role",
        lambda: xpool.devkit.transport_observer.RuntimeRole.ATNAGENT,
    )
    monkeypatch.setattr(
        xpool.devkit.transport_observer.xpool.native.devkit.transport_observer,
        "read",
        lambda: atnagent_snapshot(),
    )
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.transport_observer.install()
    runtime = create_transport_runtime(transport_entry(instance_id="m", rank=0, handle_rank=1))

    assert runtime.quiesce() is None
    assert calls == ["m"]
    assert len(list(tmp_path.glob("xpool.transport-observer.*.json"))) == 1


def test_snapshot_write_failure_warns_without_blocking_quiesce(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observer I/O failure does not turn successful native quiesce into failure."""

    quiesced: list[str] = []
    monkeypatch.setattr(
        AtnAgentTransportRuntime,
        "quiesce",
        lambda owner: quiesced.extend(resource.instance_id for resource in owner.resources),
    )
    monkeypatch.setattr(
        xpool.devkit.transport_observer,
        "get_runtime_role",
        lambda: xpool.devkit.transport_observer.RuntimeRole.ATNAGENT,
    )
    monkeypatch.setattr(
        xpool.devkit.transport_observer.xpool.native.devkit.transport_observer,
        "read",
        lambda: atnagent_snapshot(),
    )

    def fail_write(**kwargs: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(xpool.devkit.transport_observer, "write_transport_snapshot", fail_write)
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.transport_observer.install()
    runtime = create_transport_runtime(transport_entry(instance_id="m", rank=0, handle_rank=1))

    with caplog.at_level("WARNING", logger="xpool.devkit.transport_observer"):
        assert runtime.quiesce() is None

    assert "failed to record xpool transport observer snapshot" in caplog.text
    assert quiesced == ["m"]


def test_native_quiesce_failure_prevents_snapshot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Resident drain propagates before unsafe trace reads."""

    def fail_quiesce(transport_runtime: AtnAgentTransportRuntime) -> None:
        raise RuntimeError("native quiesce failed")

    reads: list[str] = []
    monkeypatch.setattr(AtnAgentTransportRuntime, "quiesce", fail_quiesce)
    monkeypatch.setattr(
        xpool.devkit.transport_observer,
        "get_runtime_role",
        lambda: xpool.devkit.transport_observer.RuntimeRole.ATNAGENT,
    )
    monkeypatch.setattr(
        xpool.devkit.transport_observer.xpool.native.devkit.transport_observer,
        "read",
        lambda: reads.append("read") or atnagent_snapshot(),
    )
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.transport_observer.install()
    runtime = create_transport_runtime(transport_entry(instance_id="m", rank=0, handle_rank=1))

    with pytest.raises(RuntimeError, match="native quiesce failed"):
        runtime.quiesce()

    assert reads == []


def create_transport_runtime(*resources: AtnAgentTransportArenaState) -> AtnAgentTransportRuntime:
    """Return a transport runtime populated with supplied transport entries."""

    runtime = AtnAgentTransportRuntime(
        client=cast(XpoolClient, SimpleNamespace()),
        cuda_device=0,
        local_rank=0,
        publisher=ProcessRef(abi_version=ABI_VERSION, pid=1),
    )
    runtime.entries = {resource.instance_id: resource for resource in resources}
    return runtime


def transport_observer_config(outdir: Path) -> XpoolConfig:
    """Return a config with native transport observation enabled."""

    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(outdir),
        },
    )
