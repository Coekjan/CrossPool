from __future__ import annotations

import time
import weakref
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from xpool.fabric import FABRIC_UID_HEX_LENGTH, FabricPlan, FabricUid
from xpool.memory import FfnExecutionContract
from xpool.native import RuntimeRole
from xpool.native.ffn import ForwardMode, LayerKind
from xpool.runtime.ffnagent.execution import OwnedExecutionResource
from xtest.harness.native.fabric import participant, topology
from xtest.harness.native.fabric.protocol import (
    FabricParticipantCommand,
    FabricParticipantSpec,
    FabricParticipantStopped,
)
from xtest.harness.native.fabric.topology import run_fabric_topology
from xtest.harness.native.ffn.qualification import execution_fabric_plan
from xtest.harness.support.config import TEST_EXECUTION_CONTRACT


def test_topology_rejects_rows_above_selected_mode_capacity(tmp_path: Path) -> None:
    uid = FabricUid("0" * FABRIC_UID_HEX_LENGTH)
    with pytest.raises(ValueError, match="selected forward mode"):
        run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            decode_payload_row_capacity=2,
            prefill_payload_row_capacity=4,
            payload_rows=(4,),
        )


@pytest.mark.parametrize("failure_site", ("plan", "join", "stop"))
def test_participant_stops_before_plan_or_retains_owner_after_plan(
    monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    class OwnerRetained(BaseException):
        pass

    spec = FabricParticipantSpec(
        role=RuntimeRole.FFNAGENT,
        device=0,
        environment={},
        pe=1,
        atnagent_count=1,
        ffnagent_count=1,
        execution_tp_size=1,
        execution_layer_count=1,
        decode_payload_row_capacity=1,
        prefill_payload_row_capacity=4,
        payload_dtype=torch.bfloat16,
        executor_lane_count=1,
        forward_modes=(ForwardMode.DECODE,),
        layer_kind=LayerKind.DENSE,
        hidden_size=16,
        intermediate_size=32,
    )
    owners: list[weakref.ReferenceType[torch.Tensor]] = []
    sent: list[object] = []

    def prepare():
        tensor = torch.empty(1)
        owners.append(weakref.ref(tensor))
        return TEST_EXECUTION_CONTRACT, (
            OwnedExecutionResource(
                owner=tensor,
                region=participant.xpool.native.ffnagent.CapturedResourceRange("table", 0x1000, 0x1000, 4, True),
            ),
        )

    def receive():
        if failure_site == "plan":
            raise RuntimeError("Plan exchange failed")
        if failure_site == "stop":
            return FabricParticipantCommand.STOP
        return execution_fabric_plan(
            uid="a" * FABRIC_UID_HEX_LENGTH,
            atnagent_count=1,
            ffnagent_count=1,
            ffn_tp_size=1,
            executor_lane_count=1,
            instance_count=1,
            execution_kind=LayerKind.DENSE,
            ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,),
            payload_dtype=torch.bfloat16,
        )

    def join(*args: object) -> None:
        raise RuntimeError("join failed")

    def retain(seconds: float) -> None:
        assert owners[0]() is not None
        assert sent == [TEST_EXECUTION_CONTRACT]
        raise OwnerRetained

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(participant.xpool.native, "initialize", lambda *args: None)
    monkeypatch.setattr(participant.xpool.native.fabric, "join", join)
    monkeypatch.setattr(participant.capture_workspace, "prepare_execution_sources", prepare)
    monkeypatch.setattr(participant, "time", SimpleNamespace(sleep=retain))
    connection = cast(Connection, SimpleNamespace(send=sent.append, recv=receive))
    if failure_site == "join":
        with pytest.raises(OwnerRetained):
            participant.run_fabric_participant(connection, spec)
    elif failure_site == "plan":
        with pytest.raises(RuntimeError, match="Plan exchange failed"):
            participant.run_fabric_participant(connection, spec)
    else:
        participant.run_fabric_participant(connection, spec)
        assert sent == [TEST_EXECUTION_CONTRACT, FabricParticipantStopped()]
        assert owners[0]() is None


@pytest.mark.parametrize("failure", ("partial_dispatch", "unconfirmed_exit"))
def test_topology_retains_mps_after_dispatch_or_without_exit_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    class OwnerRetained(BaseException):
        pass

    def retain(seconds: float) -> None:
        raise OwnerRetained

    def send(message: object) -> None:
        if isinstance(message, FabricPlan):
            raise RuntimeError("Plan send failed")
        assert message is FabricParticipantCommand.STOP

    def wait(*, timeout_seconds: float) -> None:
        raise TimeoutError("child still alive")

    def fail_plan(**arguments: object) -> None:
        raise RuntimeError("Plan construction failed")

    child = SimpleNamespace(
        process=SimpleNamespace(pid=1, is_alive=lambda: True),
        start=lambda: None,
        send=send,
        receive=lambda expected, **kwargs: (
            TEST_EXECUTION_CONTRACT if expected is FfnExecutionContract else FabricParticipantStopped()
        ),
        wait=wait,
        close=lambda: pytest.fail("unconfirmed child must stay owned"),
    )
    monkeypatch.setattr(topology, "get_task_root", lambda: None)
    monkeypatch.setattr(
        topology, "visible_uuids", lambda: tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(2))
    )
    monkeypatch.setattr(
        topology,
        "MpsScope",
        lambda endpoint: SimpleNamespace(
            endpoint=endpoint,
            start=lambda: None,
            stop=lambda **kwargs: pytest.fail("MPS must stay owned"),
        ),
    )
    monkeypatch.setattr(topology, "PythonChildProcess", lambda *args, **kwargs: child)
    monkeypatch.setattr(topology, "time", SimpleNamespace(monotonic=time.monotonic, sleep=retain))
    if failure == "unconfirmed_exit":
        monkeypatch.setattr(topology, "execution_fabric_plan", fail_plan)
    with pytest.raises(OwnerRetained):
        run_fabric_topology(
            FabricUid("a" * FABRIC_UID_HEX_LENGTH),
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
        )
