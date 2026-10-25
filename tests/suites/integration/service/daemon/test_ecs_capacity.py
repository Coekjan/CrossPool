from __future__ import annotations

from http import HTTPStatus
from types import SimpleNamespace
from typing import cast

import pytest

import xpool.native
import xpool.service.daemon.ecs
from xpool.config import XpoolConfig
from xpool.fabric import FabricGenerationId
from xpool.native.ecs import DeviceMemoryReport, QuotaCommand, QuotaCompletion, QuotaDemand
from xpool.service.daemon import ecs
from xpool.service.daemon.control import ControlPlane
from xpool.service.daemon.fabric import FabricGenerationState
from xpool.service.wire import ElasticContextStatePartitionProfile
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config
from xtest.harness.support.ecs import context_state_profile
from xtest.harness.support.service.daemon import (
    atnagent_registration,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    instance_registration,
    register,
    register_fabric_world,
    request,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


class FakeDaemonControlChannel:
    """Policy-facing publications; native tests own concurrent wire visibility."""

    def __init__(self, device_count: int, group_partition_counts: tuple[int, ...]) -> None:
        self.name = "/xpool-ecs-policy-test"
        self.initial_backing: list[int | None] = [4096] * sum(group_partition_counts)
        self.device_reports = [
            cast(DeviceMemoryReport, SimpleNamespace(total_bytes=100_000, free_bytes=50_000))
            for _ in range(device_count)
        ]
        self.demands: list[QuotaDemand | None] = [None] * len(group_partition_counts)
        self.completions: list[QuotaCompletion | None] = [None] * sum(group_partition_counts)
        self.ceilings: list[tuple[int, tuple[int, ...]]] = []
        self.commands: list[tuple[int, QuotaCommand]] = []

    def read_initial_backing(self) -> list[int | None]:
        return self.initial_backing

    def read_device_memory(self) -> list[DeviceMemoryReport | None]:
        return list(self.device_reports)

    def publish_service_ceiling(self, group_index: int, quota_bytes: tuple[int, ...]) -> None:
        self.ceilings.append((group_index, quota_bytes))

    def publish_command(self, group_index: int, command: QuotaCommand) -> None:
        self.commands.append((group_index, command))

    def read_demands(self) -> list[QuotaDemand | None]:
        return self.demands

    def read_completions(self) -> list[QuotaCompletion | None]:
        return self.completions


def capacity_policy_world(
    profiles_by_model: dict[str, ElasticContextStatePartitionProfile | tuple[ElasticContextStatePartitionProfile, ...]],
    *,
    atn_tp_size: int = 1,
    atn_dp_size: int = 1,
    runtime_headroom_by_model: dict[str, int] | None = None,
) -> tuple[ControlPlane, FabricGenerationState]:
    """Register real HTTP membership, retaining each rank's independent byte bounds."""

    world_size = atn_tp_size * atn_dp_size
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": list(range(world_size))},
            "ffn": {"devices": [world_size]},
            "models": [
                {"id": model, "path": f"/models/{model}", "atn_tp_size": atn_tp_size, "atn_dp_size": atn_dp_size}
                for model in profiles_by_model
            ],
        }
    )
    app = create_app(config)
    for device in range(world_size):
        assert register(app, "/atnagent/register", atnagent_registration(device=device)).is_success
    assert register(
        app,
        "/ffnagent/register",
        ffnagent_registration(
            device=world_size,
            model_ids=tuple(profiles_by_model),
        ),
    ).is_success
    for model_id, profiles in profiles_by_model.items():
        for rank in range(world_size):
            profile = profiles[rank] if isinstance(profiles, tuple) else profiles
            registration = instance_registration(
                model_id=model_id,
                rank=rank,
                atn_tp_rank=rank % atn_tp_size,
                atn_tp_size=atn_tp_size,
                atn_dp_rank=rank // atn_tp_size,
                atn_dp_size=atn_dp_size,
                atn_runtime_headroom_bytes=(runtime_headroom_by_model or {}).get(model_id, 0),
            )
            registration["context_state_profile"] = profile.model_dump(mode="json")
            assert register(app, "/instance/register", registration).is_success
    control = app.state.control_plane
    fabric = control.fabric_controller.generation
    assert fabric is not None
    return control, fabric


def service_policy(
    control: ControlPlane,
    fabric: FabricGenerationState,
    channel: FakeDaemonControlChannel,
    *,
    active_by_group: dict[int, tuple[int, ...]],
    pool_capacity_bytes: tuple[int, ...],
) -> ecs.CapacityPolicy:
    """Start policy iteration with completed grants, without replaying startup per scenario."""

    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    policy.pool_capacity_bytes = pool_capacity_bytes
    for group_index, instance_index, dp_rank in policy.capacity_groups(fabric):
        policy.active_quotas[group_index] = active_by_group[group_index]
        policy.command_sequences[group_index] = policy.applied_sequences[group_index] = 1
        policy.service_ceilings[group_index] = policy.service_ceiling(
            control.registrations, fabric, instance_index, dp_rank
        )
        channel.demands[group_index] = QuotaDemand(1, None, None)
    return policy


def complete_operation(
    policy: ecs.CapacityPolicy,
    fabric: FabricGenerationState,
    channel: FakeDaemonControlChannel,
    group_index: int,
    *,
    backed_bytes: tuple[int, ...] | None = None,
) -> None:
    """Publish every partition's actual backing for one outstanding group grant."""

    operation = policy.operations[group_index]
    assert operation is not None
    location = next((instance, dp) for group, instance, dp in policy.capacity_groups(fabric) if group == group_index)
    for index, backing in zip(
        policy.group_partition_indices(fabric, *location),
        operation.command.quota_bytes if backed_bytes is None else backed_bytes,
        strict=True,
    ):
        channel.completions[index] = QuotaCompletion(operation.command.sequence, backing)


def test_initial_policy_waits_for_every_tp_completion_without_eager_backing() -> None:
    control, fabric = capacity_policy_world({str(TEST_MODEL_ID): context_state_profile()}, atn_tp_size=2)
    channel = FakeDaemonControlChannel(2, (2,))
    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    policy.step(control.registrations, fabric)
    assert channel.ceilings == [(0, (32768, 32768))]
    assert [(group, command.sequence, command.quota_bytes) for group, command in channel.commands] == [
        (0, 1, (32768, 32768)),
    ]
    channel.completions[0] = QuotaCompletion(1, 4096)
    policy.step(control.registrations, fabric)
    assert policy.active_quotas[0] == (4096, 4096)
    channel.completions[1] = QuotaCompletion(1, 8192)
    policy.step(control.registrations, fabric)
    assert policy.active_quotas[0] == (32768, 32768)
    assert policy.operations[0] is None


def test_initial_byte_shares_keep_rank_local_floors_caps_and_integer_remainders() -> None:
    control, fabric = capacity_policy_world(
        {
            "test/a": (
                ElasticContextStatePartitionProfile(floor_bytes=4096, reservation_bytes=8192),
                ElasticContextStatePartitionProfile(floor_bytes=8192, reservation_bytes=65536),
            ),
            "test/b": (
                ElasticContextStatePartitionProfile(floor_bytes=8192, reservation_bytes=32768),
                context_state_profile(),
            ),
        },
        atn_tp_size=2,
    )
    channel = FakeDaemonControlChannel(2, (2, 2))
    channel.initial_backing = [4096, 8192, 8192, 4096]
    channel.device_reports = [
        cast(DeviceMemoryReport, SimpleNamespace(total_bytes=100000, free_bytes=value)) for value in (30001, 50001)
    ]
    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    policy.step(control.registrations, fabric)
    assert policy.pool_capacity_bytes == (37289, 57289)
    assert channel.ceilings == [(0, (8192, 53193)), (1, (32768, 32768))]
    assert channel.commands[-1][1].quota_bytes == (8192, 30692)
    complete_operation(policy, fabric, channel, 0, backed_bytes=(4096, 8192))
    policy.step(control.registrations, fabric)
    assert channel.commands[-1][1].quota_bytes == (20692, 26596)
    complete_operation(policy, fabric, channel, 1, backed_bytes=(8192, 4096))
    policy.step(control.registrations, fabric)
    assert policy.pool_free_bytes(control.registrations, fabric) == [8405, 1]


@pytest.mark.parametrize(("headroom", "expected"), [(1000, 49096), (10000, 44096)])
def test_pool_freeze_keeps_larger_reserve(headroom: int, expected: int) -> None:
    control, fabric = capacity_policy_world(
        {str(TEST_MODEL_ID): context_state_profile()},
        runtime_headroom_by_model={str(TEST_MODEL_ID): headroom},
    )
    channel = FakeDaemonControlChannel(1, (1,))
    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    policy.step(control.registrations, fabric)
    assert policy.pool_capacity_bytes == (expected,)


def test_pool_freeze_sums_same_device_instance_headroom() -> None:
    control, fabric = capacity_policy_world(
        {"test/a": context_state_profile(), "test/b": context_state_profile()},
        runtime_headroom_by_model={"test/a": 6000, "test/b": 7000},
    )
    channel = FakeDaemonControlChannel(1, (1, 1))
    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    policy.step(control.registrations, fabric)
    assert policy.pool_capacity_bytes == (45192,)


def test_pool_freeze_rejects_runtime_headroom_that_displaces_floors() -> None:
    control, fabric = capacity_policy_world(
        {str(TEST_MODEL_ID): context_state_profile()},
        runtime_headroom_by_model={str(TEST_MODEL_ID): 52000},
    )
    channel = FakeDaemonControlChannel(1, (1,))
    policy = ecs.CapacityPolicy(fabric.plan.generation, cast(xpool.native.ecs.DaemonControlChannel, channel))
    with pytest.raises(RuntimeError, match="cannot retain all backing floors"):
        policy.step(control.registrations, fabric)
    assert not channel.commands


def test_capacity_channel_discovery_is_generation_scoped() -> None:
    config = XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    _, _, _, plan = register_fabric_world(app)
    response = request(app, "GET", f"/ecs/control-channel/{plan.generation.format()}")
    assert response.status_code == HTTPStatus.OK
    assert response.json() == {"generation": plan.model_dump(mode="json")["generation"], "name": "/xpool-ecs-test"}
    stale = FabricGenerationId(high=plan.generation.high, low=plan.generation.low + 1)
    assert request(app, "GET", f"/ecs/control-channel/{stale.format()}").status_code == HTTPStatus.NOT_FOUND
    assert request(app, "GET", "/ecs/control-channel/not-a-generation").status_code == HTTPStatus.NOT_FOUND


def test_persistent_demand_issues_one_full_target_growth() -> None:
    control, fabric = capacity_policy_world({str(TEST_MODEL_ID): context_state_profile()})
    channel = FakeDaemonControlChannel(1, (1,))
    policy = service_policy(control, fabric, channel, active_by_group={0: (8192,)}, pool_capacity_bytes=(32768,))
    channel.demands[0] = QuotaDemand(1, (24576,), 100)
    policy.step(control.registrations, fabric)
    policy.step(control.registrations, fabric)
    assert [(group, command.sequence, command.quota_bytes) for group, command in channel.commands] == [
        (0, 2, (24576,)),
    ]
    assert policy.active_quotas[0] == (8192,)


def test_pre_deadline_borrower_order_is_earliest_deadline_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world(
        {"test/later": context_state_profile(), "test/earlier": context_state_profile()}
    )
    channel = FakeDaemonControlChannel(1, (1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (4096,), 1: (4096,)}, pool_capacity_bytes=(32768,)
    )
    channel.demands[0] = QuotaDemand(1, (8192,), 200)
    channel.demands[1] = QuotaDemand(1, (8192,), 100)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(1, (8192,))]


def test_overdue_borrowers_rotate_by_oldest_completed_grant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 500)
    control, fabric = capacity_policy_world(
        {"test/never": context_state_profile(), "test/served": context_state_profile()}
    )
    channel = FakeDaemonControlChannel(1, (1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (4096,), 1: (4096,)}, pool_capacity_bytes=(32768,)
    )
    policy.completed_grant_orders[1] = 0
    channel.demands[0] = QuotaDemand(1, (8192,), 100)
    channel.demands[1] = QuotaDemand(1, (8192,), 50)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(0, (8192,))]


def test_inverse_donor_order_prefers_the_later_active_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 0)
    control, fabric = capacity_policy_world(
        {
            "test/borrower": context_state_profile(),
            "test/later": context_state_profile(),
            "test/earlier": context_state_profile(),
        }
    )
    channel = FakeDaemonControlChannel(1, (1, 1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (4096,), 1: (12288,), 2: (12288,)}, pool_capacity_bytes=(28672,)
    )
    channel.demands[0] = QuotaDemand(1, (12288,), 10)
    channel.demands[1] = QuotaDemand(1, (16384,), 300)
    channel.demands[2] = QuotaDemand(1, (16384,), 200)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(1, (4096,))]


def test_multiple_donors_fund_one_complete_growth_without_overlapping_bytes() -> None:
    control, fabric = capacity_policy_world(
        {
            "test/borrower": ElasticContextStatePartitionProfile(floor_bytes=8192, reservation_bytes=65536),
            "test/a": context_state_profile(),
            "test/b": ElasticContextStatePartitionProfile(floor_bytes=12288, reservation_bytes=98304),
        }
    )
    channel = FakeDaemonControlChannel(1, (1, 1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (8192,), 1: (12288,), 2: (36864,)}, pool_capacity_bytes=(65536,)
    )
    channel.demands[0] = QuotaDemand(1, (49152,), 100)
    policy.step(control.registrations, fabric)
    assert policy.pool_free_bytes(control.registrations, fabric) == [8192]
    assert (channel.commands[-1][0], channel.commands[-1][1].quota_bytes) == (1, (4096,))
    complete_operation(policy, fabric, channel, 1)
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].quota_bytes) == (2, (12288,))
    complete_operation(policy, fabric, channel, 2)
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].quota_bytes) == (0, (49152,))
    assert policy.pool_free_bytes(control.registrations, fabric) == [0]


def test_donor_shrinks_only_short_devices_and_credit_waits_for_the_whole_group() -> None:
    control, fabric = capacity_policy_world(
        {"test/a": context_state_profile(), "test/b": context_state_profile()}, atn_tp_size=2
    )
    channel = FakeDaemonControlChannel(2, (2, 2))
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: (4096, 8192), 1: (12288, 16384)},
        pool_capacity_bytes=(16384, 32768),
    )
    channel.demands[0] = QuotaDemand(1, (8192, 12288), 100)
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].quota_bytes) == (1, (8192, 16384))
    channel.completions[2] = QuotaCompletion(2, 4096)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 8192]
    complete_operation(policy, fabric, channel, 1, backed_bytes=(4096, 4096))
    policy.step(control.registrations, fabric)
    assert (channel.commands[-1][0], channel.commands[-1][1].quota_bytes) == (0, (8192, 12288))
    assert policy.active_quotas[1] == (8192, 16384)
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 4096]


def test_changed_borrower_witness_abandons_completed_donor_attempt() -> None:
    control, fabric = capacity_policy_world({"test/a": context_state_profile(), "test/b": context_state_profile()})
    channel = FakeDaemonControlChannel(1, (1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (4096,), 1: (20480,)}, pool_capacity_bytes=(24576,)
    )
    channel.demands[0] = QuotaDemand(1, (16384,), 100)
    policy.step(control.registrations, fabric)
    channel.demands[0] = QuotaDemand(1, (12288,), 200)
    complete_operation(policy, fabric, channel, 1)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(1, (8192,)), (0, (12288,))]


def test_demand_change_waits_for_capacity_evaluation_before_next_operation() -> None:
    control, fabric = capacity_policy_world({str(TEST_MODEL_ID): context_state_profile()})
    channel = FakeDaemonControlChannel(1, (1,))
    policy = service_policy(control, fabric, channel, active_by_group={0: (4096,)}, pool_capacity_bytes=(32768,))
    channel.demands[0] = QuotaDemand(1, (12288,), 100)
    policy.step(control.registrations, fabric)
    channel.demands[0] = QuotaDemand(1, (20480,), 100)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1
    complete_operation(policy, fabric, channel, 0)
    policy.step(control.registrations, fabric)
    assert len(channel.commands) == 1
    channel.demands[0] = QuotaDemand(2, (20480,), 100)
    policy.step(control.registrations, fabric)
    assert [(command.sequence, command.quota_bytes) for _, command in channel.commands] == [
        (2, (12288,)),
        (3, (20480,)),
    ]


def test_infeasible_edf_head_does_not_block_a_fundable_peer() -> None:
    control, fabric = capacity_policy_world(
        {"test/large": context_state_profile(), "test/small": context_state_profile()}
    )
    channel = FakeDaemonControlChannel(1, (1, 1))
    policy = service_policy(
        control, fabric, channel, active_by_group={0: (4096,), 1: (4096,)}, pool_capacity_bytes=(16384,)
    )
    channel.demands[0] = QuotaDemand(1, (36864,), 10)
    channel.demands[1] = QuotaDemand(1, (8192,), 20)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(1, (8192,))]


@pytest.mark.parametrize(("peer_group", "expected"), [(1, [(1, (8192,))]), (2, [])], ids=("disjoint", "same-device"))
def test_unfundable_overdue_demand_protects_only_its_device(
    peer_group: int,
    expected: list[tuple[int, tuple[int, ...]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world(
        {"test/a": context_state_profile(), "test/b": context_state_profile(), "test/c": context_state_profile()},
        atn_dp_size=2,
    )
    channel = FakeDaemonControlChannel(2, (1, 1, 1, 1, 1, 1))
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: (4096,), 1: (4096,), 2: (4096,), 3: (4096,), 4: (12288,), 5: (4096,)},
        pool_capacity_bytes=(24576, 24576),
    )
    # The large holder has not evaluated its current grant. Its unused
    # permission cannot be borrowed yet, though this witness fits the ceiling.
    channel.demands[4] = None
    channel.demands[0] = QuotaDemand(1, (12288,), 10)
    channel.demands[peer_group] = QuotaDemand(1, (8192,), 100)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == expected


def test_pending_donor_allows_only_disjoint_direct_growth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world(
        {"test/a": context_state_profile(), "test/b": context_state_profile()}, atn_dp_size=2
    )
    channel = FakeDaemonControlChannel(2, (1, 1, 1, 1))
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: (4096,), 1: (4096,), 2: (12288,), 3: (4096,)},
        pool_capacity_bytes=(16384, 16384),
    )
    channel.demands[0] = QuotaDemand(1, (12288,), 10)
    channel.demands[1] = QuotaDemand(1, (8192,), 100)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(2, (4096,))]
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 8192]
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(2, (4096,)), (1, (8192,))]
    assert policy.pool_free_bytes(control.registrations, fabric) == [0, 4096]
    assert policy.funding_attempt is not None and policy.funding_attempt.donor_index == 2


def test_pending_donor_does_not_start_another_donor_on_an_unrelated_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.service.daemon.ecs, "perf_counter_ns", lambda: 50)
    control, fabric = capacity_policy_world(
        {"test/a": context_state_profile(), "test/b": context_state_profile()}, atn_dp_size=2
    )
    channel = FakeDaemonControlChannel(2, (1, 1, 1, 1))
    policy = service_policy(
        control,
        fabric,
        channel,
        active_by_group={0: (4096,), 1: (4096,), 2: (12288,), 3: (12288,)},
        pool_capacity_bytes=(16384, 16384),
    )
    channel.demands[0] = QuotaDemand(1, (12288,), 10)
    channel.demands[1] = QuotaDemand(1, (12288,), 20)
    policy.step(control.registrations, fabric)
    policy.step(control.registrations, fabric)
    assert [(group, command.quota_bytes) for group, command in channel.commands] == [(2, (4096,))]
