from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from tests.harness.native.fabric.topology import run_fabric_topology
from tests.harness.runner.child import PythonChildProcess
from xpool.abi import TensorDType, XPoolForwardMode
from xpool.fabric import FabricUid


def test_topology_rolls_back_started_participants_after_later_start_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    first = cast(
        PythonChildProcess,
        SimpleNamespace(
            name="atnagent-0",
            close=lambda: events.append("close:atnagent-0"),
        ),
    )

    def start(
        cls: type[PythonChildProcess],
        name: str,
        target: object,
        spec: object,
        *,
        log_path: Path,
    ) -> PythonChildProcess:
        del cls, target, spec, log_path
        events.append(f"start:{name}")
        if name == "atnagent-0":
            return first
        raise RuntimeError("second participant failed")

    def terminate_all(
        cls: type[PythonChildProcess],
        processes: tuple[PythonChildProcess, ...],
    ) -> None:
        del cls
        events.extend(f"terminate:{process.name}" for process in processes)

    monkeypatch.setattr(PythonChildProcess, "start", classmethod(start))
    monkeypatch.setattr(PythonChildProcess, "terminate_all", classmethod(terminate_all))

    uid = cast(FabricUid, SimpleNamespace(value="fabric-uid"))
    with pytest.raises(RuntimeError, match="second participant failed"):
        run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_count=1,
            forward_modes=(XPoolForwardMode.DECODE,),
            dtype=TensorDType.FP32,
        )

    assert events == [
        "start:atnagent-0",
        "start:ffnagent-1",
        "terminate:atnagent-0",
        "close:atnagent-0",
    ]
