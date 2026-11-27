from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict, replace
from http import HTTPStatus
from threading import Event
from typing import Literal

import httpx
import pytest

import xpool.service.daemon.control
from xpool.config import FfnSchedulingPolicy, XpoolConfig
from xpool.fabric import (
    FabricDeploymentPlan,
    FabricGenerationId,
    FabricGenerationPhase,
    FabricParticipantPhase,
    FabricRole,
    FifoSchedulerPolicy,
    RandomSchedulerPolicy,
)
from xpool.native import ABI_VERSION
from xpool.service.daemon.ffn_placement import FfnPlacementResult
from xpool.service.daemon.registration import AtnAgentRegistrationState
from xpool.service.errors import XpoolDaemonError
from xpool.service.wire import AgentStartupAdmission, ProcessRef
from xpool.utils.mps import MpsProbeResult
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import TEST_MODEL_ID, reset_global_config, synthetic_config
from xtest.harness.support.ecs import context_state_profile
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
    assert heartbeat == {
        "generation": plan.model_dump(mode="json")["generation"],
        "fabric_phase": FabricGenerationPhase.PREPARING_JOIN,
    }
    activate_fabric_world(app, plan, (atnagent, 0), (ffnagent, 1))

    executable = request(app, "GET", "/ready").json()
    assert executable["ready"] is False
    assert executable["fabric_phase"] == FabricGenerationPhase.EXECUTABLE
    assert executable["instances_initialized"] is False

    initialized = {
        "owner": process_ref(instance),
        "generation": plan.model_dump(mode="json")["generation"],
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


def test_instance_initialization_requires_each_rank_to_report_current_generation() -> None:
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
    app.state.control_plane.step_fabric_control()
    plan = FabricDeploymentPlan.model_validate(request(app, "GET", "/fabric/plan").json())
    activate_fabric_world(app, plan, (atnagent0, 0), (atnagent1, 1), (ffnagent, 2))
    generation = plan.model_dump(mode="json")["generation"]
    rank0_initialized = {
        "owner": process_ref(rank0),
        "generation": generation,
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
            "generation": asdict(FabricGenerationId.create()),
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
    app.state.control_plane.step_fabric_control()
    assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE

    assert (
        register(app, "/instance/register", instance_registration(model_id="test/b")).status_code
        == HTTPStatus.NO_CONTENT
    )
    app.state.control_plane.step_fabric_control()
    plan = FabricDeploymentPlan.model_validate(request(app, "GET", "/fabric/plan").json())

    assert [instance.ffn_profile.hidden_size for instance in plan.instance_plans] == [4, 4]
    assert request(app, "GET", "/ready").json()["transport_ready"] is False


@pytest.mark.parametrize("outcome", ["publish", "retire", "failure"])
def test_registered_owners_heartbeat_during_planning(
    deterministic_daemon_dependencies: FakeMonotonicClock,
    monkeypatch: pytest.MonkeyPatch,
    outcome: Literal["publish", "retire", "failure"],
) -> None:
    app = create_app(synthetic_config())
    control = app.state.control_plane
    entered = Event()
    release = Event()
    place = xpool.service.daemon.control.place_ffn_models
    failure_message = "injected placement failure"

    def delayed_placement(**kwargs) -> FfnPlacementResult:
        assert outcome != "retire", "a retired candidate must skip placement"
        entered.set()
        assert release.wait(timeout=5)
        if outcome == "failure":
            raise RuntimeError(failure_message)
        return place(**kwargs)

    monkeypatch.setattr(xpool.service.daemon.control, "place_ffn_models", delayed_placement)
    if outcome == "retire":
        load = xpool.service.daemon.control.architecture.load

        def delayed_compilation(**kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return load(**kwargs)

        monkeypatch.setattr(xpool.service.daemon.control.architecture, "load", delayed_compilation)
    atnagent = atnagent_registration(device=0)
    ffnagent = ffnagent_registration()
    instance = instance_registration()
    assert register(app, "/atnagent/register", atnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/ffnagent/register", ffnagent).status_code == HTTPStatus.NO_CONTENT
    assert register(app, "/instance/register", instance).status_code == HTTPStatus.NO_CONTENT

    with ThreadPoolExecutor(max_workers=1) as executor:
        planning = executor.submit(control.step_fabric_control)
        try:
            assert entered.wait(timeout=5)
            assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
            for _ in range(4):
                deterministic_daemon_dependencies.advance(5)
                for path, owner in (
                    ("/atnagent/0/heartbeat", atnagent),
                    ("/ffnagent/1/heartbeat", ffnagent),
                    (f"/instance/{TEST_MODEL_ID}/heartbeat?rank=0", instance),
                ):
                    assert request(app, "POST", path, json=process_ref(owner)).status_code == HTTPStatus.OK
            if outcome == "retire":
                control.begin_close()
        finally:
            release.set()
        if outcome == "failure":
            with pytest.raises(RuntimeError, match=failure_message):
                planning.result(timeout=5)
        else:
            planning.result(timeout=5)

    control.watchdog()
    match outcome:
        case "retire":
            assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
        case "publish":
            assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.PREPARING_JOIN
        case "failure":
            assert request(app, "GET", "/ready").json()["fabric_control_failure"] == failure_message
            response = request(app, "GET", "/fabric/plan")
            assert response.status_code == HTTPStatus.CONFLICT
            assert response.json()["detail"]["message"] == failure_message


def test_registration_rejects_replacement_if_a_plan_is_published_during_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = create_app(synthetic_config())
    control = app.state.control_plane
    process, owner = start_sleeping_proc()
    forming = Event()
    validating = Event()
    publish = Event()
    commit = Event()
    create_policy = xpool.service.daemon.control.ecs.CapacityPolicy.create
    is_alive = ProcUniqId.is_alive

    def hold_plan_before_publication(config: XpoolConfig, generation: FabricGenerationId):
        forming.set()
        assert publish.wait(timeout=5)
        return create_policy(config, generation)

    def hold_replacement_probe(identity: ProcUniqId) -> bool:
        if identity == owner and forming.is_set():
            validating.set()
            assert commit.wait(timeout=5)
        return is_alive(identity)

    monkeypatch.setattr(
        xpool.service.daemon.control.ecs.CapacityPolicy, "create", staticmethod(hold_plan_before_publication)
    )
    monkeypatch.setattr(ProcUniqId, "is_alive", hold_replacement_probe)
    try:
        for path, payload in (
            ("/atnagent/register", atnagent_registration(device=0)),
            ("/ffnagent/register", ffnagent_registration()),
            ("/instance/register", instance_registration(pid=owner.pid)),
        ):
            assert register(app, path, payload).status_code == HTTPStatus.NO_CONTENT
        with ThreadPoolExecutor(max_workers=2) as executor:
            plan = executor.submit(control.step_fabric_control)
            try:
                assert forming.wait(timeout=5)
                stop_proc(process)
                replacement = executor.submit(request, app, "POST", "/instance/register", json=instance_registration())
                assert validating.wait(timeout=5)
                publish.set()
                plan.result(timeout=5)
                assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.OK
            finally:
                publish.set()
                commit.set()
            assert replacement.result(timeout=5).status_code == HTTPStatus.CONFLICT
    finally:
        stop_proc(process)

    readiness = request(app, "GET", "/ready").json()
    assert readiness["instances"][0]["pid"] == owner.pid
    assert readiness["fabric_phase"] == FabricGenerationPhase.ABORTING


def test_participant_owner_failure_from_retired_generation_is_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(synthetic_config())
    control = app.state.control_plane
    atnagent, _, _, plan = register_fabric_world(app)
    generation = control.fabric_controller.generation
    replacement = replace(
        generation,
        plan=plan.model_copy(
            update={"generation": FabricGenerationId(high=plan.generation.high ^ 1, low=plan.generation.low)}
        ),
        participants={},
    )
    validating, release = Event(), Event()

    def fail_owner_check(registration: AtnAgentRegistrationState, owner: ProcessRef, *, context: str) -> None:
        validating.set()
        assert release.wait(timeout=5)
        raise XpoolDaemonError("conflict", "former owner exited while validation was in flight")

    monkeypatch.setattr(AtnAgentRegistrationState, "validate_process_ref", fail_owner_check)
    with ThreadPoolExecutor(max_workers=1) as executor:
        report = executor.submit(
            report_fabric_phase, app, atnagent, plan, pe=0, phase=FabricParticipantPhase.JOIN_READY
        )
        try:
            assert validating.wait(timeout=5)
            with control.lock:
                control.fabric_controller.generation = replacement
        finally:
            release.set()
        assert report.result(timeout=5) == HTTPStatus.SERVICE_UNAVAILABLE

    assert replacement.phase is FabricGenerationPhase.PREPARING_JOIN
    assert replacement.control_failure is None
    assert not replacement.participants


def test_agent_startup_invalidates_in_flight_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(synthetic_config())
    control = app.state.control_plane
    previous, owner = start_sleeping_proc()
    forming, release = Event(), Event()
    create_policy = xpool.service.daemon.control.ecs.CapacityPolicy.create

    def hold_policy(config: XpoolConfig, generation: FabricGenerationId):
        forming.set()
        assert release.wait(timeout=5)
        return create_policy(config, generation)

    monkeypatch.setattr(xpool.service.daemon.control.ecs.CapacityPolicy, "create", staticmethod(hold_policy))
    try:
        assert register(app, "/atnagent/register", atnagent_registration(device=0)).status_code == HTTPStatus.NO_CONTENT
        assert (
            register(app, "/ffnagent/register", ffnagent_registration(pid=owner.pid)).status_code
            == HTTPStatus.NO_CONTENT
        )
        assert register(app, "/instance/register", instance_registration()).status_code == HTTPStatus.NO_CONTENT
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(control.step_fabric_control)
            try:
                assert forming.wait(timeout=5)
                stop_proc(previous)
                replacement = ProcUniqId.current()
                admission = AgentStartupAdmission(
                    role=FabricRole.FFNAGENT,
                    device=1,
                    abi_version=ABI_VERSION,
                    pid=replacement.pid,
                    create_time=replacement.create_time,
                )
                assert (
                    request(app, "POST", "/startup/agent", json=admission.model_dump(mode="json")).status_code
                    == HTTPStatus.NO_CONTENT
                )
            finally:
                release.set()
            pending.result(timeout=5)
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
        assert (
            request(app, "POST", "/ffnagent/register", json=ffnagent_registration(pid=replacement.pid)).status_code
            == HTTPStatus.NO_CONTENT
        )
        control.step_fabric_control()
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.OK
    finally:
        stop_proc(previous)


def test_mps_observation_does_not_delay_fabric_owner_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    app = create_app(synthetic_config())
    control = app.state.control_plane
    process, owner = start_sleeping_proc()
    probing = Event()
    release = Event()

    def hold_probe() -> MpsProbeResult:
        probing.set()
        assert release.wait(timeout=5)
        return MpsProbeResult(False, None, "controller unavailable")

    monkeypatch.setattr(control.mps_scope, "probe", hold_probe)
    monkeypatch.setattr(control, "close", lambda: None)

    async def observe() -> None:
        async with app.router.lifespan_context(app):
            try:
                assert await asyncio.to_thread(probing.wait, 5)
                await asyncio.to_thread(stop_proc, process)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://daemon"
                ) as client:
                    async with asyncio.timeout(3):
                        while True:
                            readiness = (await client.get("/ready")).json()
                            if readiness["fabric_phase"] == FabricGenerationPhase.ABORTING:
                                break
                            await asyncio.sleep(0.01)
                    assert readiness["mps_status"] == "online"
                    assert readiness["fabric_owner_failure"]["reason"] == "exited"
            finally:
                release.set()

    try:
        register_fabric_world(app, atnagent=atnagent_registration(device=0, pid=owner.pid))
        asyncio.run(observe())
    finally:
        stop_proc(process)


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
    assert FabricDeploymentPlan.model_validate(request(app, "GET", "/fabric/plan").json()) == plan
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


@pytest.mark.parametrize("atn_dp_size", [1, 2])
def test_attention_byte_bounds_may_differ_between_ranks(atn_dp_size: int) -> None:
    atn_tp_size = 2 // atn_dp_size
    config = make_config(
        {
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m", "atn_dp_size": atn_dp_size}],
        }
    )
    app = create_app(config)
    first = instance_registration(model_id=str(TEST_MODEL_ID), rank=0, atn_tp_size=atn_tp_size, atn_dp_size=atn_dp_size)
    second = instance_registration(
        model_id=str(TEST_MODEL_ID),
        rank=1,
        atn_tp_rank=1 % atn_tp_size,
        atn_tp_size=atn_tp_size,
        atn_dp_rank=1 // atn_tp_size,
        atn_dp_size=atn_dp_size,
    )
    second["context_state_profile"] = (
        context_state_profile()
        .model_copy(update={"floor_bytes": 8192, "reservation_bytes": 65536})
        .model_dump(mode="json")
    )

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
    app.state.control_plane.step_fabric_control()
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
    with ExitStack() as stack:
        atnagent_process, atnagent_proc_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_proc_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
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
    with ExitStack() as stack:
        atnagent_process, atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
        instance_process, instance_id = start_sleeping_proc()
        stack.callback(stop_proc, instance_process)
        atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
        ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
        instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
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


def test_leased_instance_retirement_during_quiesce_does_not_record_owner_failure() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    with ExitStack() as stack:
        atnagent_process, atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
        instance_process, instance_id = start_sleeping_proc()
        stack.callback(stop_proc, instance_process)
        atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
        ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
        instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
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


def test_unrequested_instance_exit_during_quiesce_records_owner_failure() -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    with ExitStack() as stack:
        atnagent_process, atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
        instance_process, instance_id = start_sleeping_proc()
        stack.callback(stop_proc, instance_process)
        atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
        ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
        instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
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
    with ExitStack() as stack:
        atnagent_process, atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
        instance_process, instance_id = start_sleeping_proc()
        stack.callback(stop_proc, instance_process)
        atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
        ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
        instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
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
    with ExitStack() as stack:
        old_atnagent_process, old_atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, old_atnagent_process)
        old_ffnagent_process, old_ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, old_ffnagent_process)
        old_instance_process, old_instance_id = start_sleeping_proc()
        stack.callback(stop_proc, old_instance_process)
        new_atnagent_process, new_atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, new_atnagent_process)
        new_ffnagent_process, new_ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, new_ffnagent_process)
        new_instance_process, new_instance_id = start_sleeping_proc()
        stack.callback(stop_proc, new_instance_process)
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
        assert FabricDeploymentPlan.model_validate(request(app, "GET", "/fabric/plan").json()) == old_plan

        for process in (old_atnagent_process, old_ffnagent_process, old_instance_process):
            stop_proc(process)
        app.state.control_plane.watchdog()
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE

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

        app.state.control_plane.step_fabric_control()
        new_plan = FabricDeploymentPlan.model_validate(request(app, "GET", "/fabric/plan").json())
        assert new_plan.generation != old_plan.generation


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
) -> None:
    config = make_config(
        {
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        }
    )
    app = create_app(config)
    with ExitStack() as stack:
        atnagent_process, atnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, atnagent_process)
        ffnagent_process, ffnagent_id = start_sleeping_proc()
        stack.callback(stop_proc, ffnagent_process)
        instance_process, instance_id = start_sleeping_proc()
        stack.callback(stop_proc, instance_process)
        atnagent = atnagent_registration(device=0, pid=atnagent_id.pid)
        ffnagent = ffnagent_registration(device=1, pid=ffnagent_id.pid, model_ids=(str(TEST_MODEL_ID),))
        instance = instance_registration(model_id=str(TEST_MODEL_ID), pid=instance_id.pid)
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
        surviving_registration, surviving_pe = (ffnagent, 1) if target_role == "atnagent" else (atnagent, 0)
        for phase in (
            FabricParticipantPhase.QUIESCED,
            FabricParticipantPhase.DRAINING,
            FabricParticipantPhase.DRAINED,
        ):
            assert (
                report_fabric_phase(app, surviving_registration, plan, pe=surviving_pe, phase=phase)
                == HTTPStatus.NO_CONTENT
            )
        app.state.control_plane.watchdog()
        assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.ABORTING
        stop_proc(instance_process)
        app.state.control_plane.watchdog()
        assert request(app, "GET", "/ready").json()["fabric_phase"] == FabricGenerationPhase.STOPPED
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.OK
        assert surviving_agent_id.is_alive()
        stop_proc(ffnagent_process if target_role == "atnagent" else atnagent_process)
        app.state.control_plane.watchdog()
        assert request(app, "GET", "/fabric/plan").status_code == HTTPStatus.SERVICE_UNAVAILABLE
