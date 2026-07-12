from __future__ import annotations

import json

import pytest

from xpool.service.wire import ReadinessScope, ReadinessSnapshot

pytestmark = pytest.mark.usefixtures("reset_global_config")


def config_record(records: list[dict[str, object]], name: str) -> dict[str, object]:
    for record in records:
        if record["name"] == name:
            return record
    raise AssertionError(f"missing config record for {name}")


def json_config_records(capsys: pytest.CaptureFixture[str]) -> list[dict[str, object]]:
    captured = capsys.readouterr()
    assert captured.err == ""
    records = json.loads(captured.out)
    assert isinstance(records, list)
    return [record for record in records if isinstance(record, dict)]


def ready_snapshot(*, ready: bool) -> ReadinessSnapshot:
    instance_status = "online" if ready else "offline"
    return ReadinessSnapshot.model_validate(
        {
            "ready": ready,
            "mps_status": "online" if ready else "offline",
            "scopes": {"atn": ready},
            "cuda_devices": [0, 1],
            "atnagents": [{"pid": 100, "status": "online", "cuda_device": 0}],
            "instances": [
                {
                    "pid": 200 if ready else None,
                    "status": instance_status,
                    "instance_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
                    "cuda_device": 0,
                    "rank": 0,
                }
            ],
        }
    )


def client_class(readiness: ReadinessSnapshot) -> type:
    class FakeXpoolClient:
        def __init__(self) -> None:
            return None

        def close(self) -> None:
            return None

        def readiness(self, scopes: tuple[ReadinessScope, ...] = ()) -> ReadinessSnapshot:
            return readiness

    return FakeXpoolClient
