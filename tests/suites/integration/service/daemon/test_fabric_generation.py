from __future__ import annotations

from http import HTTPStatus

import pytest

import xpool.service.daemon.control
from xpool.config import FfnSchedulingPolicy, XpoolConfig
from xpool.fabric import (
    FabricGenerationId,
    FabricGenerationPhase,
    FabricParticipantPhase,
    FabricPlan,
    FabricRole,
    FifoSchedulerPolicy,
    RandomSchedulerPolicy,
)
from xpool.native import ABI_VERSION
from xpool.service.wire import AgentStartupAdmission, ServingListener
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config
from xtest.harness.support.kv import kv_capacity_profile
from xtest.harness.support.service.daemon import (
    FakeMonotonicClock,
    activate_fabric_world,
    atnagent_registration,
    atnagent_transport_arenas,
    create_app,
    deterministic_daemon_dependencies,
    ffn_profile,
    ffnagent_registration,
    instance_registration,
    process_ref,
    register,
    register_fabric_world,
    report_fabric_phase,
    request,
    start_sleeping_proc,
    stop_proc,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


def invocation_failure(*, origin_pe: int, sequence: int = 1) -> dict[str, object]:
    """Return one valid canonical invocation-failure projection."""

    return {
        "result_code": 2,
        "origin_pe": origin_pe,
        "instance_index": 0,
        "invocation_sequence": sequence,
        "layer_ordinal": 0,
    }


def make_config(payload: dict[str, object], *, scheduler: dict[str, object] | None = None) -> XpoolConfig:
    """Supply the required global latency objective to generation tests."""

    return XpoolConfig.from_mapping(
        payload | {"scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}, **(scheduler or {})}}
    )


def test_registration_and_reports_form_executable_ready_generation() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent, ffnagent, instance, plan = register_fabric_world(app)

    assert isinstance(plan.scheduler, FifoSchedulerPolicy)
    assert plan.scheduler.policy is FfnSchedulingPolicy.FIFO
    assert plan.instance_plans[0].ffn_profile.model_dump(mode="json") == instance["ffn_profile"]
    topology = plan.instance_plans[0].instance_rank_topology
    assert (topology.atn_tp_size, topology.atn_dp_size) == (1, 1)
    assert (
        request(
            app,
            "POST",
            "/atnagent/0/transport-arenas",
            json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )

    heartbeat = request(app, "POST", "/atnagent/0/heartbeat", json=process_ref(atnagent)).json()
    assert heartbeat["fabric_phase"] == FabricGenerationPhase.PREPARING_JOIN
    activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))

    executable = request(app, "GET", "/ready").json()
    assert executable["ready"] is False
    assert executable["fabric_phase"] == FabricGenerationPhase.EXECUTABLE
    assert executable["instances_initialized"] is False
    assert app.state.control_plane.capture_serving_health_targets() is None

    initialized = {
        "owner": process_ref(instance),
        "generation": plan.model_dump(mode="json")["generation"],
        "serving_listener": {"host": "127.0.0.1", "port": 30000},
    }
    assert (
        request(app, "POST", f"/instance/{TEST_MODEL_ID}/initialized?rank=0", json=initialized).status_code
        == HTTPStatus.NO_CONTENT
    )
    readiness = request(app, "GET", "/ready").json()
    assert readiness["ready"] is True
    assert readiness["fabric_phase"] == FabricGenerationPhase.EXECUTABLE
    assert readiness["transport_ready"] is True
    assert readiness["instances_initialized"] is True
    assert readiness["fabric_invocation_failure"] is None
    assert readiness["fabric_owner_failure"] is None
    assert readiness["fabric_control_failure"] is None

    targets = app.state.control_plane.capture_serving_health_targets()
    assert targets is not None
    assert targets.generation == plan.generation
    assert targets.listeners == ((TEST_MODEL_ID, ServingListener(host="127.0.0.1", port=30000)),)
    stale_generation = FabricGenerationId.create()
    while stale_generation == targets.generation:
        stale_generation = FabricGenerationId.create()
    stale_targets = xpool.service.daemon.control.ServingHealthTargets(
        generation=stale_generation,
        listeners=targets.listeners,
    )
    assert not app.state.control_plane.confirm_serving_health(stale_targets)
    assert app.state.control_plane.confirm_serving_health(targets)
    assert not app.state.control_plane.confirm_serving_health(targets)
    assert app.state.control_plane.capture_serving_health_targets() is None


def test_instance_initialized_listener_mismatch_is_atomic() -> None:
    config = make_config(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent0 = atnagent_registration(device=0)
    atnagent1 = atnagent_registration(device=1)
    ffnagent = ffnagent_registration(device=2, model_ids=(str(TEST_MODEL_ID),))
    rank0 = instance_registration(model_id=str(TEST_MODEL_ID), rank=0, atn_tp_size=2)
    rank1 = instance_registration(model_id=str(TEST_MODEL_ID), rank=1, atn_tp_size=2)
    for payload, path in (
        (atnagent0, "/atnagent/register"),
        (atnagent1, "/atnagent/register"),
        (ffnagent, "/ffnagent/register"),
        (rank0, "/instance/register"),
        (rank1, "/instance/register"),
    ):
        assert register(app, path, payload).status_code == HTTPStatus.NO_CONTENT
    plan = FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json())
    activate_fabric_world(app, plan, (atnagent0, 0), (atnagent1, 1), (ffnagent, 2))
    generation = plan.model_dump(mode="json")["generation"]
    rank0_initialized = {
        "owner": process_ref(rank0),
        "generation": generation,
        "serving_listener": {"host": "127.0.0.1", "port": 30000},
    }
    assert (
        request(app, "POST", f"/instance/{TEST_MODEL_ID}/initialized?rank=0", json=rank0_initialized).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(app, "POST", f"/instance/{TEST_MODEL_ID}/initialized?rank=0", json=rank0_initialized).status_code
        == HTTPStatus.NO_CONTENT
    )

    mismatch = request(
        app,
        "POST",
        f"/instance/{TEST_MODEL_ID}/initialized?rank=1",
        json={
            "owner": process_ref(rank1),
            "generation": generation,
            "serving_listener": {"host": "127.0.0.1", "port": 30001},
        },
    )

    assert mismatch.status_code == HTTPStatus.CONFLICT
    assert request(app, "GET", "/ready").json()["instances_initialized"] is False
    assert (
        request(
            app,
            "POST",
            f"/instance/{TEST_MODEL_ID}/initialized?rank=1",
            json={
                **rank0_initialized,
                "owner": process_ref(rank1),
            },
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "GET", "/ready").json()["instances_initialized"] is True


def test_plan_waits_for_every_model_while_transport_publication_is_incremental() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [
                {"id": "test/a", "path": "/models/a"},
                {"id": "test/b", "path": "/models/b"},
            ],
        }
    )
    app = create_app(config)
    atnagent = atnagent_registration(device=0)
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert (
        register(app, "/ffnagent/register", ffnagent_registration(model_ids=("test/a", "test/b"))).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        register(app, "/instance/register", instance_registration(model_id="test/a")).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        request(
            app,
            "POST",
            "/atnagent/0/transport-arenas",
            json=atnagent_transport_arenas(("test/a", 0), publisher=atnagent),
        ).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE

    assert (
        register(app, "/instance/register", instance_registration(model_id="test/b")).status_code
        == HTTPStatus.NO_CONTENT
    )
    plan = FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json())

    assert [instance.ffn_profile.hidden_size for instance in plan.instance_plans] == [4, 4]
    assert request(app, "GET", "/ready").json()["transport_ready"] is False


def test_random_scheduler_seed_is_generated_once_and_persisted_in_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        },
        scheduler={"ffn_policy": "random"},
    )
    generated: list[int] = []

    def generate_seed(bits: int) -> int:
        if bits == 128:
            return 1
        if bits == 64:
            generated.append(17)
            return 17
        raise AssertionError(f"unexpected random width: {bits}")

    monkeypatch.setattr(xpool.service.daemon.control.secrets, "randbits", generate_seed)
    app = create_app(config)
    _, _, _, plan = register_fabric_world(app)

    assert isinstance(plan.scheduler, RandomSchedulerPolicy)
    assert plan.scheduler.seed == 17
    assert FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json()) == plan
    assert generated == [17]


def test_rank_independent_ffn_profile_mismatch_is_rejected_during_registration() -> None:
    config = make_config(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    first = instance_registration(model_id=str(TEST_MODEL_ID), rank=0, atn_tp_size=2)
    second = instance_registration(model_id=str(TEST_MODEL_ID), rank=1, atn_tp_size=2)
    second["ffn_profile"] = ffn_profile(hidden_size=4096)

    assert register(app, "/instance/register", first).status_code == HTTPStatus.NO_CONTENT
    response = register(app, "/instance/register", second)

    assert response.status_code == HTTPStatus.CONFLICT
    assert "ffn_profile disagrees" in response.json()["detail"]["message"]


def test_kv_capacity_geometry_mismatch_is_rejected_during_registration() -> None:
    config = make_config(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    first = instance_registration(model_id=str(TEST_MODEL_ID), rank=0, atn_tp_size=2)
    second = instance_registration(model_id=str(TEST_MODEL_ID), rank=1, atn_tp_size=2)
    second["kv_capacity"] = kv_capacity_profile().model_copy(update={"row_bytes": 2048}).model_dump(mode="json")

    assert register(app, "/instance/register", first).status_code == HTTPStatus.NO_CONTENT
    response = register(app, "/instance/register", second)

    assert response.status_code == HTTPStatus.CONFLICT
    assert "kv capacity geometry disagrees" in response.json()["detail"]["message"]


def test_kv_capacity_geometry_may_differ_between_dp_groups() -> None:
    config = make_config(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m", "atn_dp_size": 2}],
        }
    )
    app = create_app(config)
    first = instance_registration(model_id=str(TEST_MODEL_ID), rank=0, atn_tp_size=1, atn_dp_size=2)
    second = instance_registration(
        model_id=str(TEST_MODEL_ID),
        rank=1,
        atn_tp_rank=0,
        atn_tp_size=1,
        atn_dp_rank=1,
        atn_dp_size=2,
    )
    second["kv_capacity"] = kv_capacity_profile().model_copy(update={"row_bytes": 2048}).model_dump(mode="json")

    assert register(app, "/atnagent/register", atnagent_registration(device=0)).status_code == 204
    assert register(app, "/atnagent/register", atnagent_registration(device=1)).status_code == 204
    assert (
        register(
            app,
            "/ffnagent/register",
            ffnagent_registration(device=2, model_ids=(str(TEST_MODEL_ID),)),
        ).status_code
        == 204
    )
    assert register(app, "/instance/register", first).status_code == 204
    assert register(app, "/instance/register", second).status_code == 204
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.OK


def test_owner_invocation_and_control_failures_are_retained_independently() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent, ffnagent, instance, plan = register_fabric_world(app)
    activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))

    assert (
        request(app, "POST", f"/instance/{TEST_MODEL_ID}/deregister?rank=0", json=process_ref(instance)).status_code
        == HTTPStatus.NO_CONTENT
    )
    assert (
        report_fabric_phase(
            app,
            atnagent,
            plan,
            pe=0,
            phase=FabricParticipantPhase.ACTIVE,
            invocation_failure=invocation_failure(origin_pe=0),
        )
        == HTTPStatus.NO_CONTENT
    )
    assert (
        report_fabric_phase(
            app,
            ffnagent,
            plan,
            pe=1,
            phase=FabricParticipantPhase.ACTIVE,
            invocation_failure=invocation_failure(origin_pe=1),
        )
        == HTTPStatus.NO_CONTENT
    )

    readiness = request(app, "GET", "/ready").json()
    assert readiness["fabric_phase"] == FabricGenerationPhase.ABORTING
    assert readiness["fabric_invocation_failure"] == invocation_failure(origin_pe=0)
    assert readiness["fabric_owner_failure"] == {
        "role": "instance",
        "model_id": str(TEST_MODEL_ID),
        "rank": 0,
        "reason": "exited",
    }
    assert "conflicting canonical invocation failures" in readiness["fabric_control_failure"]


def test_illegal_report_aborts_but_exact_committed_retry_remains_idempotent() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent, ffnagent, _, plan = register_fabric_world(app)
    for registration, pe in ((atnagent, 0), (ffnagent, 1)):
        assert (
            report_fabric_phase(app, registration, plan, pe=pe, phase=FabricParticipantPhase.JOIN_READY)
            == HTTPStatus.NO_CONTENT
        )
    assert report_fabric_phase(app, atnagent, plan, pe=0, phase=FabricParticipantPhase.JOINING) == HTTPStatus.NO_CONTENT

    assert report_fabric_phase(app, atnagent, plan, pe=0, phase=FabricParticipantPhase.ACTIVE) == HTTPStatus.CONFLICT
    assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.ABORTING
    assert report_fabric_phase(app, atnagent, plan, pe=0, phase=FabricParticipantPhase.JOINING) == HTTPStatus.NO_CONTENT


def test_quiesce_requires_current_agent_owner_and_generation() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_proc_id = start_sleeping_proc()
    ffnagent_process, ffnagent_proc_id = start_sleeping_proc()
    try:
        atnagent, _, instance, plan = register_fabric_world(
            app,
            atnagent=atnagent_registration(device=0, pid=atnagent_proc_id.pid),
            ffnagent=ffnagent_registration(device=1, pid=ffnagent_proc_id.pid, model_ids=(str(TEST_MODEL_ID),)),
        )
        generation = plan.model_dump(mode="json")["generation"]

        invalid = request(
            app,
            "POST",
            "/fabric/quiesce",
            json={"owner": process_ref(instance), "generation": generation},
        )
        assert invalid.status_code == HTTPStatus.CONFLICT
        assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.PREPARING_JOIN

        valid_payload = {"owner": process_ref(atnagent), "generation": generation}
        assert request(app, "POST", "/fabric/quiesce", json=valid_payload).status_code == HTTPStatus.NO_CONTENT
        assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.ABORTING
        assert request(app, "POST", "/fabric/quiesce", json=valid_payload).status_code == HTTPStatus.NO_CONTENT
    finally:
        stop_proc(atnagent_process)
        stop_proc(ffnagent_process)


def test_transition_timeout_records_control_failure_and_selects_abort(
    deterministic_daemon_dependencies: FakeMonotonicClock,
) -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent, ffnagent, instance, plan = register_fabric_world(app)
    for registration, pe in ((atnagent, 0), (ffnagent, 1)):
        assert (
            report_fabric_phase(app, registration, plan, pe=pe, phase=FabricParticipantPhase.JOIN_READY)
            == HTTPStatus.NO_CONTENT
        )
    deterministic_daemon_dependencies.advance(
        xpool.service.daemon.control.FABRIC_PHASE_TIMEOUT_S[FabricGenerationPhase.JOINING] + 1.0
    )
    request(app, "POST", "/atnagent/0/heartbeat", json=process_ref(atnagent))
    request(app, "POST", "/ffnagent/1/heartbeat", json=process_ref(ffnagent))
    request(app, "POST", f"/instance/{TEST_MODEL_ID}/heartbeat?rank=0", json=process_ref(instance))

    app.state.control_plane.watchdog()
    readiness = request(app, "GET", "/ready").json()

    assert readiness["fabric_phase"] == FabricGenerationPhase.ABORTING
    assert readiness["fabric_owner_failure"] is None
    assert readiness["fabric_control_failure"] == "Fabric joining transition timed out"


def test_finalized_agent_exit_does_not_convert_cooperative_cleanup_to_abort() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    ffnagent_process, ffnagent_id = start_sleeping_proc()
    instance_process, instance_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
    ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
    try:
        _, _, _, plan = register_fabric_world(app, atnagent=atnagent, ffnagent=ffnagent, instance=instance)
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))
        assert (
            request(
                app,
                "POST",
                "/fabric/quiesce",
                json={"owner": process_ref(atnagent), "generation": plan.model_dump(mode="json")["generation"]},
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        for phase in (
            FabricParticipantPhase.QUIESCED,
            FabricParticipantPhase.DRAINING,
            FabricParticipantPhase.DRAINED,
        ):
            for registration, pe in ((atnagent, 0), (ffnagent, 1)):
                assert report_fabric_phase(app, registration, plan, pe=pe, phase=phase) == HTTPStatus.NO_CONTENT
        assert (
            report_fabric_phase(app, atnagent, plan, pe=0, phase=FabricParticipantPhase.FINALIZED)
            == HTTPStatus.NO_CONTENT
        )

        stop_proc(atnagent_process)
        app.state.control_plane.watchdog()
        readiness = request(app, "GET", "/ready").json()

        assert readiness["fabric_phase"] == FabricGenerationPhase.FINALIZING
        assert readiness["fabric_owner_failure"] is None
    finally:
        for process in (atnagent_process, ffnagent_process, instance_process):
            if process.poll() is None:
                stop_proc(process)


def test_leased_instance_retirement_during_quiesce_does_not_record_owner_failure() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    ffnagent_process, ffnagent_id = start_sleeping_proc()
    instance_process, instance_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
    ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
    try:
        _, _, _, plan = register_fabric_world(app, atnagent=atnagent, ffnagent=ffnagent, instance=instance)
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))
        assert (
            request(
                app,
                "POST",
                "/atnagent/0/transport-arenas",
                json=atnagent_transport_arenas((str(TEST_MODEL_ID), 0), publisher=atnagent),
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert (
            request(
                app,
                "POST",
                f"/instance/{TEST_MODEL_ID}/transport-arena/acquire?rank=0",
                json=process_ref(instance),
            ).status_code
            == HTTPStatus.OK
        )
        assert (
            request(
                app,
                "POST",
                "/fabric/quiesce",
                json={"owner": process_ref(atnagent), "generation": plan.model_dump(mode="json")["generation"]},
            ).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert (
            request(
                app,
                "POST",
                "/atnagent/0/transport-leases/quiesce",
                json=process_ref(atnagent),
            ).status_code
            == HTTPStatus.OK
        )

        assert instance_process.poll() is None
        stop_proc(instance_process)
        app.state.control_plane.watchdog()
        readiness = request(app, "GET", "/ready").json()

        assert readiness["fabric_phase"] == FabricGenerationPhase.QUIESCING
        assert readiness["fabric_owner_failure"] is None
    finally:
        for process in (atnagent_process, ffnagent_process, instance_process):
            if process.poll() is None:
                stop_proc(process)


def test_unrequested_instance_exit_during_quiesce_records_owner_failure() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    ffnagent_process, ffnagent_id = start_sleeping_proc()
    instance_process, instance_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
    ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
    try:
        _, _, _, plan = register_fabric_world(app, atnagent=atnagent, ffnagent=ffnagent, instance=instance)
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))
        assert (
            request(
                app,
                "POST",
                "/fabric/quiesce",
                json={"owner": process_ref(atnagent), "generation": plan.model_dump(mode="json")["generation"]},
            ).status_code
            == HTTPStatus.NO_CONTENT
        )

        stop_proc(instance_process)
        app.state.control_plane.watchdog()
        readiness = request(app, "GET", "/ready").json()

        assert readiness["fabric_phase"] == FabricGenerationPhase.QUIESCING
        assert readiness["fabric_owner_failure"] == {
            "role": "instance",
            "model_id": str(TEST_MODEL_ID),
            "rank": 0,
            "reason": "exited",
        }
    finally:
        for process in (atnagent_process, ffnagent_process, instance_process):
            if process.poll() is None:
                stop_proc(process)


def test_instance_exit_after_activation_completes_cooperative_shutdown() -> None:
    """Retain Instance owner loss while live Fabric PEs finish every barrier."""

    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    ffnagent_process, ffnagent_id = start_sleeping_proc()
    instance_process, instance_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
    ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
    try:
        _, _, _, plan = register_fabric_world(app, atnagent=atnagent, ffnagent=ffnagent, instance=instance)
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))

        stop_proc(instance_process)
        app.state.control_plane.watchdog()
        readiness = request(app, "GET", "/ready").json()
        assert readiness["fabric_phase"] == FabricGenerationPhase.QUIESCING
        assert readiness["fabric_owner_failure"] == {
            "role": "instance",
            "model_id": str(TEST_MODEL_ID),
            "rank": 0,
            "reason": "exited",
        }
        assert readiness["fabric_invocation_failure"] is None
        assert readiness["fabric_control_failure"] is None
        assert atnagent_process.poll() is None
        assert ffnagent_process.poll() is None

        for phase in (
            FabricParticipantPhase.QUIESCED,
            FabricParticipantPhase.DRAINING,
            FabricParticipantPhase.DRAINED,
            FabricParticipantPhase.FINALIZED,
        ):
            for registration, pe in ((atnagent, 0), (ffnagent, 1)):
                assert report_fabric_phase(app, registration, plan, pe=pe, phase=phase) == HTTPStatus.NO_CONTENT

        readiness = request(app, "GET", "/ready").json()
        assert readiness["fabric_phase"] == FabricGenerationPhase.STOPPED
        assert readiness["fabric_owner_failure"]["role"] == "instance"
        assert readiness["fabric_invocation_failure"] is None
    finally:
        for process in (atnagent_process, ffnagent_process, instance_process):
            if process.poll() is None:
                stop_proc(process)


def test_replacement_waits_for_retirement_then_forms_wholly_new_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    old_atnagent_process, old_atnagent_id = start_sleeping_proc()
    old_ffnagent_process, old_ffnagent_id = start_sleeping_proc()
    old_instance_process, old_instance_id = start_sleeping_proc()
    new_atnagent_process, new_atnagent_id = start_sleeping_proc()
    new_ffnagent_process, new_ffnagent_id = start_sleeping_proc()
    new_instance_process, new_instance_id = start_sleeping_proc()
    old_atnagent = atnagent_registration(device=0, pid=old_atnagent_id.pid)
    old_ffnagent = ffnagent_registration(device=1, pid=old_ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    old_instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=old_instance_id.pid)
    new_atnagent = atnagent_registration(device=0, pid=new_atnagent_id.pid)
    new_ffnagent = ffnagent_registration(device=1, pid=new_ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    new_instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=new_instance_id.pid)
    processes = (
        old_atnagent_process,
        old_ffnagent_process,
        old_instance_process,
        new_atnagent_process,
        new_ffnagent_process,
        new_instance_process,
    )
    try:
        _, _, _, old_plan = register_fabric_world(
            app,
            atnagent=old_atnagent,
            ffnagent=old_ffnagent,
            instance=old_instance,
        )
        admission = AgentStartupAdmission(
            pid=new_atnagent_id.pid,
            create_time=new_atnagent_id.create_time,
            abi_version=ABI_VERSION,
            role=FabricRole.ATNAGENT,
            device=0,
        )
        assert (
            request(app, "POST", "/startup/agent", json=admission.model_dump(mode="json")).status_code
            == HTTPStatus.SERVICE_UNAVAILABLE
        )

        assert all(process.poll() is None for process in processes)
        assert FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json()) == old_plan

        for process in (old_atnagent_process, old_ffnagent_process, old_instance_process):
            stop_proc(process)
        app.state.control_plane.watchdog()
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert app.state.control_plane.capture_serving_health_targets() is None

        for role, identity, device, path, registration in (
            (FabricRole.ATNAGENT, new_atnagent_id, 0, "/atnagent/register", new_atnagent),
            (FabricRole.FFNAGENT, new_ffnagent_id, 1, "/ffnagent/register", new_ffnagent),
        ):
            admission = AgentStartupAdmission(
                pid=identity.pid,
                create_time=identity.create_time,
                abi_version=ABI_VERSION,
                role=role,
                device=device,
            )
            assert (
                request(app, "POST", "/startup/agent", json=admission.model_dump(mode="json")).status_code
                == HTTPStatus.NO_CONTENT
            )
            assert request(app, "POST", path, json=registration).status_code == HTTPStatus.NO_CONTENT
            assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert request(app, "POST", "/instance/register", json=new_instance).status_code == HTTPStatus.NO_CONTENT

        new_plan = FabricPlan.model_validate(request(app, "GET", "/fabric/plan").json())
        assert new_plan.generation != old_plan.generation
    finally:
        for process in processes:
            if process.poll() is None:
                stop_proc(process)


@pytest.mark.parametrize(
    ("target_role", "target_pe"),
    [
        pytest.param("atnagent", 0, id="atnagent"),
        pytest.param("ffnagent", 1, id="ffnagent"),
    ],
)
def test_fabric_pe_exit_records_owner_failure_and_selects_fail_stop(
    target_role: str,
    target_pe: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    atnagent_process, atnagent_id = start_sleeping_proc()
    ffnagent_process, ffnagent_id = start_sleeping_proc()
    instance_process, instance_id = start_sleeping_proc()
    atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
    ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
    instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
    try:
        _, _, _, plan = register_fabric_world(app, atnagent=atnagent, ffnagent=ffnagent, instance=instance)
        activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))
        target_process = atnagent_process if target_role == "atnagent" else ffnagent_process
        surviving_agent_id = ffnagent_id if target_role == "atnagent" else atnagent_id
        stop_proc(target_process)

        app.state.control_plane.watchdog()
        readiness = request(app, "GET", "/ready").json()

        assert readiness["fabric_phase"] == FabricGenerationPhase.ABORTING
        assert readiness["fabric_owner_failure"] == {
            "role": target_role,
            "pe": target_pe,
            "reason": "exited",
        }
        assert surviving_agent_id.is_alive() and instance_id.is_alive()
    finally:
        for process in (atnagent_process, ffnagent_process, instance_process):
            if process.poll() is None:
                stop_proc(process)
