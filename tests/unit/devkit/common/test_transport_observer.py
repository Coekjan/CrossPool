from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

import xpool.devkit.common.transport_observer as transport_observer
from tests.harness.runtime.atnagent import transport_arena_resource
from xpool.abi import TransportArenaHandle, TransportTraceRecord, TransportTraceSnapshot
from xpool.config import XpoolConfig, init_global_config
from xpool.devkit.common.transport_observer import write_transport_snapshot

production_destroy = transport_observer.AtnArenaResource.destroy


@pytest.fixture(autouse=True)
def reset_transport_observer(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore the production arena destruction method around each test."""

    monkeypatch.setattr(transport_observer, "installed", False)
    monkeypatch.setattr(transport_observer.AtnArenaResource, "destroy", production_destroy)
    yield


def test_write_transport_snapshot_serializes_structured_records(tmp_path) -> None:
    """Observer output derives phases from structured ABI trace records."""

    record = TransportTraceRecord(
        trace_id=1,
        slot=2,
        num_tokens=3,
        request_begin=100,
        slot_claimed=110,
        input_staged=120,
        request_published=130,
        atnagent_dequeued=140,
        descriptor_granted=150,
        executor_begin=160,
        executor_end=180,
        result_published=190,
        result_observed=200,
        output_copied=210,
        slot_recycled=220,
    )
    snapshot = TransportTraceSnapshot(sequence=1, dropped=0, records=(record,))

    init_global_config(config=transport_observer_config(tmp_path))
    path = write_transport_snapshot(
        instance_id="model/0",
        rank=0,
        handle=TransportArenaHandle("00" * 64),
        snapshot=snapshot,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["sequence"] == 1
    assert payload["dropped"] == 0
    assert payload["records"][0]["trace_id"] == 1
    assert payload["records"][0]["durations_ns"]["total"] == 120
    assert payload["phase_summary"]["executor"]["median_ns"] == 20


def test_install_records_snapshot_around_production_destroy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed observer writes the snapshot returned by arena destruction."""

    snapshot = TransportTraceSnapshot(sequence=0, dropped=0, records=())
    calls: list[str] = []

    def fake_destroy(resource: transport_observer.AtnArenaResource) -> TransportTraceSnapshot:
        calls.append(resource.instance_id)
        return snapshot

    monkeypatch.setattr(transport_observer.AtnArenaResource, "destroy", fake_destroy)
    init_global_config(config=transport_observer_config(tmp_path))
    transport_observer.install()
    resource = transport_arena_resource(instance_id="m", rank=0, handle_rank=1)

    assert resource.destroy() is snapshot
    assert calls == ["m"]
    assert len(list(tmp_path.glob("xpool.transport-observer.*.json"))) == 1


def test_snapshot_write_failure_warns_after_successful_destroy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Observer I/O failure does not turn successful native cleanup into failure."""

    snapshot = TransportTraceSnapshot(sequence=0, dropped=0, records=())
    monkeypatch.setattr(
        transport_observer.AtnArenaResource,
        "destroy",
        lambda resource: snapshot,
    )
    monkeypatch.setattr(
        transport_observer,
        "write_transport_snapshot",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    init_global_config(config=transport_observer_config(tmp_path))
    transport_observer.install()
    resource = transport_arena_resource(instance_id="m", rank=0, handle_rank=1)

    with caplog.at_level("WARNING", logger="xpool.devkit.common.transport_observer"):
        assert resource.destroy() is snapshot

    assert "Failed to record xpool transport observer snapshot" in caplog.text


def test_native_destroy_failure_propagates_without_snapshot_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observer preserves native destruction failures and performs no output."""

    def fail_destroy(resource: transport_observer.AtnArenaResource) -> TransportTraceSnapshot:
        raise RuntimeError("native drain failed")

    writes: list[object] = []
    monkeypatch.setattr(transport_observer.AtnArenaResource, "destroy", fail_destroy)
    monkeypatch.setattr(
        transport_observer,
        "write_transport_snapshot",
        lambda **kwargs: writes.append(kwargs),
    )
    init_global_config(config=transport_observer_config(tmp_path))
    transport_observer.install()
    resource = transport_arena_resource(instance_id="m", rank=0, handle_rank=1)

    with pytest.raises(RuntimeError, match="native drain failed"):
        resource.destroy()

    assert writes == []


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
