from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.native
import xpool.service.daemon.kv
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
from xpool.fabric import FabricGenerationId
from xpool.service.daemon.control import ControlPlane
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.daemon.kv import KvCapacityPolicy

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


class FakeDaemonControlChannel:
    """In-memory typed publication surface for daemon policy tests."""

    def __init__(self, *, pool_count: int, slot_count: int) -> None:
        self.name = "/xpool-kv-policy-test"
        self.initial_backing: list[int | None] = [1] * slot_count
        self.device_reports: list[object | None] = [
            SimpleNamespace(total_bytes=100_000, free_bytes=50_000) for _ in range(pool_count)
        ]
        self.demands: list[xpool.native.kv.KvCapacityDemand | None] = [None] * slot_count
        self.completions: list[xpool.native.kv.KvCapacityCompletion | None] = [None] * slot_count
        self.ceilings: list[tuple[int, int]] = []
        self.commands: list[tuple[int, xpool.native.kv.KvCapacityCommand]] = []
        self.closed = False

    def read_initial_backing(self) -> list[int | None]:
        return self.initial_backing

    def read_device_memory(self) -> list[object | None]:
        return self.device_reports

    def publish_service_ceiling(self, group_index: int, bundles: int) -> None:
        self.ceilings.append((group_index, bundles))

    def publish_command(self, group_index: int, command: xpool.native.kv.KvCapacityCommand) -> None:
        self.commands.append((group_index, command))

    def read_demands(self) -> list[xpool.native.kv.KvCapacityDemand | None]:
        return self.demands

    def read_completions(self) -> list[xpool.native.kv.KvCapacityCompletion | None]:
        return self.completions

    def close(self) -> None:
        self.closed = True


def capacity_policy_world(
    bundle_bytes_by_instance: dict[str, int],
    *,
    atn_tp_size: int = 1,
    atn_dp_size: int = 1,
) -> tuple[ControlPlane, FabricGenerationState]:
    atn_world_size = atn_tp_size * atn_dp_size
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": list(range(atn_world_size))},
            "ffn": {"devices": [atn_world_size]},
            "models": [
                {"id": instance_id, "path": f"/models/{instance_id}"} for instance_id in bundle_bytes_by_instance
            ],
        }
    )
    app = create_app(config)
    for device in range(atn_world_size):
        assert request(app, "POST", "/atnagent/register", json=atnagent_registration(cuda_device=device)).is_success
    assert request(
        app,
        "POST",
        "/ffnagent/register",
        json=ffnagent_registration(cuda_device=atn_world_size, model_ids=tuple(bundle_bytes_by_instance)),
    ).is_success
    for instance_id, bundle_bytes in bundle_bytes_by_instance.items():
        for rank in range(atn_world_size):
            registration = instance_registration(
                instance_id=instance_id,
                rank=rank,
                atn_tp_rank=rank % atn_tp_size,
                atn_tp_size=atn_tp_size,
                atn_dp_rank=rank // atn_tp_size,
                atn_dp_size=atn_dp_size,
            )
            profile = kv_capacity_profile().model_copy(update={"bundle_bytes": bundle_bytes})
            registration["kv_capacity"] = profile.model_dump(mode="json")
            assert request(app, "POST", "/instance/register", json=registration).is_success
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None
    return control, fabric


def service_policy(
    control: ControlPlane,
    fabric: FabricGenerationState,
    channel: FakeDaemonControlChannel,
    *,
    active_by_group: dict[int, int],
    pool_capacity_bytes: int,
) -> KvCapacityPolicy:
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonControlChannel, channel),
    )
    policy.pool_capacity_bytes = (pool_capacity_bytes,) * len(channel.device_reports)
    for group_index, _, _ in policy.capacity_groups(fabric):
        policy.active_bundles[group_index] = active_by_group[group_index]
        policy.command_sequences[group_index] = 1
        policy.applied_sequences[group_index] = 1
        policy.service_ceilings[group_index] = 8
        channel.demands[group_index] = xpool.native.kv.KvCapacityDemand(
            evaluated_sequence=1,
            requested_bundles=None,
            deadline_monotonic_ns=None,
        )
    return policy


def complete_operation(
    policy: KvCapacityPolicy,
    fabric: FabricGenerationState,
    channel: FakeDaemonControlChannel,
    group_index: int,
) -> None:
    operation = policy.operations[group_index]
    assert operation is not None
    location = next(
        (instance_index, dp_rank)
        for candidate, instance_index, dp_rank in policy.capacity_groups(fabric)
        if candidate == group_index
    )
    for partition_index in policy.group_partition_indices(fabric, *location):
        channel.completions[partition_index] = xpool.native.kv.KvCapacityCompletion(
            sequence=operation.command.sequence,
            backed_bundles=operation.command.target_bundles,
        )


def demand(sequence: int, bundles: int | None, deadline_ns: int | None) -> xpool.native.kv.KvCapacityDemand:
    """Construct one coherent daemon-policy demand snapshot."""

    return xpool.native.kv.KvCapacityDemand(
        evaluated_sequence=sequence,
        requested_bundles=bundles,
        deadline_monotonic_ns=deadline_ns,
    )


def test_initial_policy_freezes_pools_and_waits_for_every_tp_completion() -> None:
    control, fabric = capacity_policy_world({"model": 4096}, atn_tp_size=2)
    channel = FakeDaemonControlChannel(pool_count=2, slot_count=2)
    policy = KvCapacityPolicy(
        generation=fabric.plan.generation,
        channel=cast(xpool.native.kv.DaemonControlChannel, channel),
    )

    policy.step(control.registrations, fabric)

    assert channel.ceilings == [(0, 8)]
    assert len(channel.commands) == 1
    group_index, command = channel.commands[0]
    assert group_index == 0
    assert command.sequence == 1
    assert command.target_bundles == 8

    channel.completions[0] = xpool.native.kv.KvCapacityCompletion(sequence=1, backed_bundles=8)
    policy.step(control.registrations, fabric)
    assert policy.active_bundles[0] == 1

    channel.completions[1] = xpool.native.kv.KvCapacityCompletion(sequence=1, backed_bundles=8)
    policy.step(control.registrations, fabric)
    assert policy.active_bundles[0] == 8
    assert policy.operations[0] is None


def test_capacity_channel_discovery_is_generation_scoped() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
    app = create_app(config)
    _, _, _, plan = register_fabric_world(app)

    response = request(app, "GET", f"/kv/control-channel/{plan.generation.format()}")

    assert response.status_code == HTTPStatus.OK
    assert response.json() == {
        "generation": plan.model_dump(mode="json")["generation"],
        "name": "/xpool-kv-test",
    }
    stale = FabricGenerationId(high=plan.generation.high, low=plan.generation.low + 1)
    assert request(app, "GET", f"/kv/control-channel/{stale.format()}").status_code == HTTPStatus.NOT_FOUND
    assert request(app, "GET", "/kv/control-channel/not-a-generation").status_code == HTTPStatus.NOT_FOUND


def test_persistent_demand_issues_one_full_target_growth() -> None:
    control, fabric = capacity_policy_world({"borrower": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=1)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 2},
        pool_capacity_bytes=8 * 4096,
    )
    channel.demands[0] = demand(1, 6, 100)

    policy.step(control.registrations, fabric)
    policy.step(control.registrations, fabric)

    assert [(index, command.sequence, command.target_bundles) for index, command in channel.commands] == [(0, 2, 6)]
    assert policy.active_bundles[0] == 2


def test_pre_deadline_borrower_order_is_earliest_deadline_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world({"later": 4096, "earlier": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=2)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1},
        pool_capacity_bytes=8 * 4096,
    )
    channel.demands[0] = demand(1, 2, 200)
    channel.demands[1] = demand(1, 2, 100)

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 2)]


def test_overdue_borrowers_rotate_by_oldest_completed_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 500)
    control, fabric = capacity_policy_world({"never-served": 4096, "served": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=2)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1},
        pool_capacity_bytes=8 * 4096,
    )
    policy.completed_grant_orders[1] = 0
    channel.demands[0] = demand(1, 2, 100)
    channel.demands[1] = demand(1, 2, 50)

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(0, 2)]


def test_inverse_donor_order_prefers_the_later_active_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 0)
    control, fabric = capacity_policy_world({"borrower": 4096, "later-donor": 4096, "earlier-donor": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=3)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 3, 2: 3},
        pool_capacity_bytes=7 * 4096,
    )
    channel.demands[0] = demand(1, 3, 10)
    channel.demands[1] = demand(1, 4, 300)
    channel.demands[2] = demand(1, 4, 200)

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 1)]


def test_multiple_donors_fund_one_complete_growth_without_overlapping_bytes() -> None:
    control, fabric = capacity_policy_world({"borrower": 4096, "donor-a": 4096, "donor-b": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=3)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 3, 2: 3},
        pool_capacity_bytes=8 * 4096,
    )
    channel.demands[0] = demand(1, 6, 100)

    policy.step(control.registrations, fabric)
    assert policy.pool_free_bytes(control.registrations, fabric) == [4096]
    complete_operation(policy, fabric, channel, 1)
    policy.step(control.registrations, fabric)
    assert channel.commands[-1][0] == 2
    assert channel.commands[-1][1].target_bundles == 1
    complete_operation(policy, fabric, channel, 2)
    policy.step(control.registrations, fabric)
    assert channel.commands[-1][0] == 0
    assert channel.commands[-1][1].target_bundles == 6
    assert policy.pool_free_bytes(control.registrations, fabric) == [0]


def test_changed_borrower_witness_abandons_completed_donor_attempt() -> None:
    control, fabric = capacity_policy_world({"borrower": 4096, "donor": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=2)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 5},
        pool_capacity_bytes=6 * 4096,
    )
    channel.demands[0] = demand(1, 4, 100)

    policy.step(control.registrations, fabric)
    channel.demands[0] = demand(1, 3, 200)
    complete_operation(policy, fabric, channel, 1)
    policy.step(control.registrations, fabric)
    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 2), (0, 3)]


def test_demand_change_waits_for_capacity_evaluation_before_next_operation() -> None:
    control, fabric = capacity_policy_world({"borrower": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=1)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1},
        pool_capacity_bytes=8 * 4096,
    )
    channel.demands[0] = demand(1, 3, 100)
    policy.step(control.registrations, fabric)
    channel.demands[0] = demand(1, 5, 100)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1

    complete_operation(policy, fabric, channel, 0)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1

    channel.demands[0] = demand(2, 5, 100)
    policy.step(control.registrations, fabric)
    assert [(command.sequence, command.target_bundles) for _, command in channel.commands] == [(2, 3), (3, 5)]


def test_infeasible_edf_head_does_not_block_a_fundable_peer() -> None:
    control, fabric = capacity_policy_world({"large": 4096, "small": 4096})
    channel = FakeDaemonControlChannel(pool_count=1, slot_count=2)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1},
        pool_capacity_bytes=4 * 4096,
    )
    channel.demands[0] = demand(1, 9, 10)
    channel.demands[1] = demand(1, 2, 20)

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 2)]


def test_unfundable_overdue_demand_allows_predeadline_growth_on_another_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world({"first": 4096, "second": 4096}, atn_dp_size=2)
    channel = FakeDaemonControlChannel(pool_count=2, slot_count=4)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1, 2: 1, 3: 1},
        pool_capacity_bytes=3 * 4096,
    )
    channel.demands[0] = demand(1, 3, 10)
    channel.demands[1] = demand(1, 2, 100)

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(1, 2)]


def test_unfundable_overdue_demand_blocks_predeadline_growth_on_its_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world({"first": 4096, "second": 4096}, atn_dp_size=2)
    channel = FakeDaemonControlChannel(pool_count=2, slot_count=4)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1, 2: 1, 3: 1},
        pool_capacity_bytes=3 * 4096,
    )
    channel.demands[0] = demand(1, 3, 10)
    channel.demands[2] = demand(1, 2, 100)

    policy.step(control.registrations, fabric)

    assert channel.commands == []


def test_pending_donor_allows_only_disjoint_direct_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world({"first": 4096, "second": 4096}, atn_dp_size=2)
    channel = FakeDaemonControlChannel(pool_count=2, slot_count=4)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1, 2: 3, 3: 1},
        pool_capacity_bytes=4 * 4096,
    )
    channel.demands[0] = demand(1, 3, 10)
    channel.demands[1] = demand(1, 2, 100)

    policy.step(control.registrations, fabric)
    assert [(index, command.target_bundles) for index, command in channel.commands] == [(2, 1)]
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 2 * 4096]

    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(2, 1), (1, 2)]
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 4096]
    assert policy.funding_attempt is not None and policy.funding_attempt.donor_index == 2


def test_pending_donor_does_not_start_another_donor_on_an_unrelated_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xpool.service.daemon.kv, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world({"first": 4096, "second": 4096}, atn_dp_size=2)
    channel = FakeDaemonControlChannel(pool_count=2, slot_count=4)
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: 1, 1: 1, 2: 3, 3: 3},
        pool_capacity_bytes=4 * 4096,
    )
    channel.demands[0] = demand(1, 3, 10)
    channel.demands[1] = demand(1, 3, 20)

    policy.step(control.registrations, fabric)
    policy.step(control.registrations, fabric)

    assert [(index, command.target_bundles) for index, command in channel.commands] == [(2, 1)]
