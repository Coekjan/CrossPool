from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.devkit.common.transport_observer
import xpool.native
from tests.harness.config import install_test_config
from tests.harness.runtime.atnagent import transport_entry
from xpool.abi import ABI_VERSION, DpPaddingMode, FfnResultCode, FfnResultHandoff, XPoolForwardMode
from xpool.config import XpoolConfig
from xpool.devkit.common.transport_observer import write_transport_snapshot
from xpool.runtime.atnagent import AtnTransportCatalog, AtnTransportEntry
from xpool.service.client import XpoolClient
from xpool.service.wire import ProcessRef
from xpool.transport import TransportArenaHandle

production_quiesce = AtnTransportCatalog.quiesce


@pytest.fixture
def reset_transport_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the production catalog quiesce method around each test."""

    monkeypatch.setattr(xpool.devkit.common.transport_observer, "installed", False)
    monkeypatch.setattr(AtnTransportCatalog, "quiesce", production_quiesce)
    yield


pytestmark = pytest.mark.usefixtures(reset_transport_observer.__name__, "reset_global_config")


def trace_record(**overrides: int) -> SimpleNamespace:
    """Return one complete acknowledged mailbox trace record."""

    values = {
        "trace_id": 1,
        "payload_rows": 3,
        "layer_ordinal": 2,
        "forward_mode": int(XPoolForwardMode.DECODE),
        "result_handoff": int(FfnResultHandoff.REPLICATED_FULL),
        "dp_padding_mode": int(DpPaddingMode.NONE),
        "result_code": int(FfnResultCode.OK),
        "staging_started": 100,
        "staging_completed": 110,
        "published": 120,
        "published_observed": 130,
        "execution_started": 140,
        "execution_admitted": 150,
        "execution_completed": 160,
        "evaluated": 170,
        "evaluated_observed": 180,
        "output_copied": 190,
        "acknowledged": 200,
        "closed": 0,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def trace_snapshot(*records: SimpleNamespace) -> xpool.native.TransportTraceSnapshot:
    """Return a typed test view of one native snapshot."""

    return cast(
        xpool.native.TransportTraceSnapshot,
        SimpleNamespace(sequence=len(records), dropped=0, records=records),
    )


def test_write_transport_snapshot_serializes_structured_records(tmp_path: Path) -> None:
    """Observer output derives semantic values and phases from one mailbox trace."""

    install_test_config(config=transport_observer_config(tmp_path))
    path = write_transport_snapshot(
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
        "acknowledged": 1,
        "closed": 0,
        "incomplete": 0,
    }
    record = payload["records"][0]
    assert record["forward_mode"] == "decode"
    assert record["result_handoff"] == "replicated_full"
    assert record["dp_padding_mode"] == "none"
    assert record["result_code"] == "ok"
    assert record["durations_ns"]["acknowledged_total"] == 100
    assert payload["phase_summary"]["execution"]["median_ns"] == 20


def test_write_transport_snapshot_classifies_closed_record(tmp_path: Path) -> None:
    """A Staging-to-Closed request is terminal rather than incomplete."""

    record = trace_record(
        result_code=int(FfnResultCode.SHUTDOWN),
        staging_completed=0,
        published=0,
        published_observed=0,
        execution_started=0,
        execution_admitted=0,
        execution_completed=0,
        evaluated=0,
        evaluated_observed=0,
        output_copied=0,
        acknowledged=0,
        closed=110,
    )
    install_test_config(config=transport_observer_config(tmp_path))
    path = write_transport_snapshot(
        instance_id="model/0",
        rank=0,
        handle=TransportArenaHandle("00" * 64),
        snapshot=trace_snapshot(record),
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["record_counts"] == {
        "retained": 1,
        "acknowledged": 0,
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

    def fake_quiesce(catalog: AtnTransportCatalog) -> None:
        calls.extend(resource.instance_id for resource in catalog.resources)

    monkeypatch.setattr(AtnTransportCatalog, "quiesce", fake_quiesce)
    monkeypatch.setattr(
        xpool.devkit.common.transport_observer.xpool.native.transport,
        "read_trace",
        lambda handle: trace_snapshot(),
    )
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.common.transport_observer.install()
    catalog = transport_catalog(transport_entry(instance_id="m", rank=0, handle_rank=1))

    assert catalog.quiesce() is None
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
        AtnTransportCatalog,
        "quiesce",
        lambda catalog: quiesced.extend(resource.instance_id for resource in catalog.resources),
    )
    monkeypatch.setattr(
        xpool.devkit.common.transport_observer.xpool.native.transport,
        "read_trace",
        lambda handle: trace_snapshot(),
    )

    def fail_write(**kwargs: object) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(xpool.devkit.common.transport_observer, "write_transport_snapshot", fail_write)
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.common.transport_observer.install()
    catalog = transport_catalog(transport_entry(instance_id="m", rank=0, handle_rank=1))

    with caplog.at_level("WARNING", logger="xpool.devkit.common.transport_observer"):
        assert catalog.quiesce() is None

    assert "Failed to record xpool transport observer snapshot" in caplog.text
    assert quiesced == ["m"]


def test_native_quiesce_failure_prevents_snapshot_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed Resident drain propagates before unsafe trace reads."""

    def fail_quiesce(catalog: AtnTransportCatalog) -> None:
        raise RuntimeError("native quiesce failed")

    reads: list[str] = []
    monkeypatch.setattr(AtnTransportCatalog, "quiesce", fail_quiesce)
    monkeypatch.setattr(
        xpool.devkit.common.transport_observer.xpool.native.transport,
        "read_trace",
        lambda handle: reads.append(handle) or trace_snapshot(),
    )
    install_test_config(config=transport_observer_config(tmp_path))
    xpool.devkit.common.transport_observer.install()
    catalog = transport_catalog(transport_entry(instance_id="m", rank=0, handle_rank=1))

    with pytest.raises(RuntimeError, match="native quiesce failed"):
        catalog.quiesce()

    assert reads == []


def transport_catalog(*resources: AtnTransportEntry) -> AtnTransportCatalog:
    """Return a catalog populated with supplied transport entries."""

    catalog = AtnTransportCatalog(
        client=cast(XpoolClient, SimpleNamespace()),
        cuda_device=0,
        local_rank=0,
        publisher=ProcessRef(abi_version=ABI_VERSION, pid=1),
    )
    catalog.entries = {resource.instance_id: resource for resource in resources}
    return catalog


def transport_observer_config(outdir: Path) -> XpoolConfig:
    """Return a config with native transport observation enabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_TRANSPORT_OBSERVER_OUTDIR": str(outdir),
        },
    )
