from __future__ import annotations

from collections import deque
from http import HTTPStatus
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.native
from tests.harness.support.config import reset_global_config
from tests.harness.support.kv import kv_capacity_profile
from tests.harness.support.service.daemon import (
    atnagent_registration,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    register_fabric_world,
    request,
)
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId, FabricGenerationPhase
from xpool.service.daemon.control import ControlPlane
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.daemon.kv import KvCapacityPolicy
from xpool.utils.procs import ProcUniqId

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


class FakeDaemonCapacityChannel:
    """In-memory policy seam retaining complete publications for assertions."""

    def __init__(self) -> None:
        self.name = "/xpool-kv-policy-test"
        self.backing_reports: list[object | None] = [SimpleNamespace(prepared_sequence=0, backed_bundles=1)]
        self.device_reports: list[object | None] = [SimpleNamespace(total_bytes=100_000, free_bytes=50_000)]
        self.pressure_reports: list[object | None] = [None]
        self.commands: list[tuple[int, xpool.native.kv.KvCapacityCommand]] = []
        self.closed = False

    def read_backing_reports(self) -> list[object | None]:
        return self.backing_reports

    def read_device_memory(self) -> list[object | None]:
        return self.device_reports

    def read_pressure_reports(self) -> list[object | None]:
        return self.pressure_reports

    def publish_command(self, group_index: int, command: xpool.native.kv.KvCapacityCommand) -> None:
        self.commands.append((group_index, command))

    def close(self) -> None:
        self.closed = True


def capacity_policy_world(bundle_bytes_by_instance: dict[str, int]) -> tuple[ControlPlane, FabricGenerationState]:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": instance_id, "path": f"/models/{instance_id}"} for instance_id in bundle_bytes_by_instance
            ],
        }
    )
    app = create_app(config)
    assert request(app, "POST", "/atnagent/register", json=atnagent_registration(cuda_device=0)).is_success
    assert request(
        app,
        "POST",
        "/ffnagent/register",
        json=ffnagent_registration(cuda_device=1, model_ids=tuple(bundle_bytes_by_instance)),
    ).is_success
    for instance_id, bundle_bytes in bundle_bytes_by_instance.items():
        registration = instance_registration(instance_id=instance_id)
        profile = kv_capacity_profile().model_copy(update={"bundle_bytes": bundle_bytes})
        registration["kv_capacity"] = profile.model_dump(mode="json")
        assert request(app, "POST", "/instance/register", json=registration).is_success
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None
    return control, fabric


def test_initial_policy_freezes_pool_then_activates_prepared_target() -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    _, _, _, plan = register_fabric_world(app)
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None
    channel = FakeDaemonCapacityChannel()
    policy = KvCapacityPolicy(
        generation=plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[None],
        pressure_sequences=[0],
        pressure_queue=deque(),
    )

    policy.step(control.registrations, fabric)

    assert policy.pool_capacity_bytes == (49_096,)
    assert len(channel.commands) == 1
    group_index, command = channel.commands[-1]
    assert group_index == 0
    assert (command.sequence, command.target_bundles, command.active_bundles) == (1, 8, 1)

    channel.backing_reports = [SimpleNamespace(prepared_sequence=1, backed_bundles=8)]
    policy.step(control.registrations, fabric)

    _, command = channel.commands[-1]
    assert len(channel.commands) == 2
    assert (command.sequence, command.target_bundles, command.active_bundles) == (1, 8, 8)


def test_tp_group_activates_only_after_every_rank_prepares_the_same_target() -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    for path, payload in (
        ("/atnagent/register", atnagent_registration(cuda_device=0)),
        ("/atnagent/register", atnagent_registration(cuda_device=1)),
        ("/ffnagent/register", ffnagent_registration(cuda_device=2, model_ids=("m",))),
        ("/instance/register", instance_registration(instance_id="m", rank=0, atn_tp_size=2)),
        ("/instance/register", instance_registration(instance_id="m", rank=1, atn_tp_size=2)),
    ):
        assert request(app, "POST", path, json=payload).status_code == HTTPStatus.NO_CONTENT
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None

    channel = FakeDaemonCapacityChannel()
    channel.backing_reports = [
        SimpleNamespace(prepared_sequence=0, backed_bundles=1),
        SimpleNamespace(prepared_sequence=0, backed_bundles=1),
    ]
    channel.device_reports = [
        SimpleNamespace(total_bytes=100_000, free_bytes=50_000),
        SimpleNamespace(total_bytes=100_000, free_bytes=50_000),
    ]
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[None, None],
        pressure_sequences=[0, 0],
        pressure_queue=deque(),
        pool_capacity_bytes=(8 * 4096, 8 * 4096),
    )

    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1
    group_index, command = channel.commands[-1]
    assert group_index == 0
    assert (command.sequence, command.target_bundles, command.active_bundles) == (1, 8, 1)

    channel.backing_reports[0] = SimpleNamespace(prepared_sequence=1, backed_bundles=8)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1

    channel.backing_reports[1] = SimpleNamespace(prepared_sequence=1, backed_bundles=8)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 2
    _, command = channel.commands[-1]
    assert (command.sequence, command.target_bundles, command.active_bundles) == (1, 8, 8)


def test_capacity_channel_discovery_is_generation_scoped() -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    _, _, _, plan = register_fabric_world(app)

    response = request(app, "GET", f"/kv/capacity-channel/{plan.generation.format()}")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "generation": plan.model_dump(mode="json")["generation"],
        "name": "/xpool-kv-test",
    }
    stale = FabricGenerationId(high=plan.generation.high, low=plan.generation.low + 1)
    assert request(app, "GET", f"/kv/capacity-channel/{stale.format()}").status_code == HTTPStatus.NOT_FOUND
    assert request(app, "GET", "/kv/capacity-channel/not-a-generation").status_code == HTTPStatus.NOT_FOUND


def test_service_policy_reclaims_then_grows_one_transition_per_step() -> None:
    control, fabric = capacity_policy_world({"borrower": 8192, "donor": 4096})

    channel = FakeDaemonCapacityChannel()
    channel.backing_reports = [
        SimpleNamespace(prepared_sequence=1, backed_bundles=2),
        SimpleNamespace(prepared_sequence=1, backed_bundles=4),
    ]
    channel.pressure_reports = [SimpleNamespace(sequence=1, active_bundles=2), None]
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=2, active_bundles=2),
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=4, active_bundles=4),
        ],
        pressure_sequences=[0, 0],
        pressure_queue=deque(),
        pool_capacity_bytes=(8 * 4096,),
    )

    policy.step(control.registrations, fabric)
    assert [(index, command.target_bundles, command.active_bundles) for index, command in channel.commands] == [
        (1, 3, 3)
    ]

    policy.step(control.registrations, fabric)
    assert [(index, command.target_bundles, command.active_bundles) for index, command in channel.commands] == [
        (1, 3, 3),
        (1, 2, 2),
    ]

    channel.backing_reports[1] = SimpleNamespace(prepared_sequence=3, backed_bundles=2)
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].target_bundles) == (0, 3)

    channel.backing_reports[0] = SimpleNamespace(prepared_sequence=2, backed_bundles=3)
    policy.step(control.registrations, fabric)
    assert channel.commands[-1][1].active_bundles == 3

    policy.step(control.registrations, fabric)
    channel.pressure_reports[0] = SimpleNamespace(sequence=2, active_bundles=None)
    policy.step(control.registrations, fabric)


def test_service_policy_bypasses_an_infeasible_fifo_head() -> None:
    control, fabric = capacity_policy_world({"large": 8192, "small": 4096})
    channel = FakeDaemonCapacityChannel()
    channel.backing_reports = [
        SimpleNamespace(prepared_sequence=1, backed_bundles=1),
        SimpleNamespace(prepared_sequence=1, backed_bundles=2),
    ]
    channel.pressure_reports = [
        SimpleNamespace(sequence=1, active_bundles=1),
        SimpleNamespace(sequence=1, active_bundles=1),
    ]
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=1, active_bundles=1),
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=1, active_bundles=1),
        ],
        pressure_sequences=[0, 0],
        pressure_queue=deque(),
        pool_capacity_bytes=(5 * 4096,),
    )

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 2)]


def test_actual_ceiling_retry_yields_preference_to_the_next_borrower() -> None:
    control, fabric = capacity_policy_world({"preferred": 4096, "peer": 4096})
    channel = FakeDaemonCapacityChannel()
    channel.backing_reports = [
        SimpleNamespace(prepared_sequence=1, backed_bundles=1),
        SimpleNamespace(prepared_sequence=1, backed_bundles=2),
    ]
    channel.pressure_reports = [
        SimpleNamespace(sequence=1, active_bundles=1),
        SimpleNamespace(sequence=1, active_bundles=2),
    ]
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=1, active_bundles=1),
            xpool.native.kv.KvCapacityCommand(sequence=1, target_bundles=2, active_bundles=2),
        ],
        pressure_sequences=[0, 0],
        pressure_queue=deque(),
        pool_capacity_bytes=(3 * 4096,),
    )

    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].target_bundles) == (1, 1)

    channel.backing_reports[1] = SimpleNamespace(prepared_sequence=2, backed_bundles=1)
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].target_bundles) == (0, 2)

    channel.backing_reports[0] = SimpleNamespace(prepared_sequence=2, backed_bundles=2)
    policy.step(control.registrations, fabric)
    assert channel.commands[-1][1].active_bundles == 2

    channel.pressure_reports[0] = SimpleNamespace(sequence=2, active_bundles=2)
    command_count = len(channel.commands)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == command_count + 1
    assert (channel.commands[-1][0], channel.commands[-1][1].target_bundles) == (0, 1)


def test_terminal_generation_retires_capacity_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    config = XpoolConfig.from_mapping(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    register_fabric_world(app)
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None
    channel = FakeDaemonCapacityChannel()
    control.kv_capacity_policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonCapacityChannel, channel),
        commands=[None],
        pressure_sequences=[0],
        pressure_queue=deque(),
    )
    fabric.phase = FabricGenerationPhase.STOPPED
    monkeypatch.setattr(ProcUniqId, "is_alive", lambda self: False)

    control.retire_terminal_generation()

    assert channel.closed
    assert control.kv_capacity_policy is None
    assert control.fabric_controller.generation is None
