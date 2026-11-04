from __future__ import annotations

import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import cast

import pytest
import torch

import xpool.native
from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.memory import ExecutionResourceDeclaration, ExecutionResourceRequirement
from xpool.runtime.ffnagent import execution, preparation, registry, weights
from xpool.runtime.ffnagent.memory_profile import corpus
from xtest.harness.support.config import TEST_EXECUTION_CONTRACT, synthetic_config


@dataclass
class CaptureStream:
    failure: RuntimeError | None = None
    address: int = 7
    destroyed: bool = False

    def wait_stream(self, stream: object) -> None:
        pass

    def synchronize(self) -> None:
        if self.failure is not None:
            raise self.failure

    def destroy(self) -> None:
        self.destroyed = True


@dataclass
class CaptureGraph:
    failure: RuntimeError | None = None
    resets: int = 0

    def reset(self) -> None:
        if self.failure is not None:
            raise self.failure
        self.resets += 1

    def raw_cuda_graph(self) -> int:
        return 1


@dataclass
class PoolSnapshot:
    segment_bytes: tuple[int, ...]

    def snapshot(self) -> list[dict[str, int]]:
        return [{"total_size": value} for value in self.segment_bytes]


def test_pool_prediction_error_is_reported_without_releasing_backing(caplog: pytest.LogCaptureFixture) -> None:
    owner = registry.LaneCaptureResources()
    owner.pool = cast(torch.cuda.MemPool, PoolSnapshot((64, 128)))
    with caplog.at_level("INFO", logger=registry.__name__):
        owner.report_reservation(191)
    assert "estimated_bytes=191 actual_bytes=192 error_bytes=1" in caplog.text
    assert owner.pool is not None
    assert not owner.sealed


@pytest.fixture(autouse=True)
def release_source_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xpool.native.cuda.blas, "release_workspaces", lambda address: None)


@pytest.mark.parametrize("failure_site", ("primary", "control", "observation"))
def test_failed_capture_pair_stays_owned_until_explicit_release(
    monkeypatch: pytest.MonkeyPatch, failure_site: str
) -> None:
    owner = registry.LaneCaptureResources()
    stream = CaptureStream()
    graphs: list[CaptureGraph] = []
    owner.stream_owner = cast(xpool.native.cuda.OwnedCudaStream, stream)
    owner.stream = cast(torch.cuda.ExternalStream, stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "use_mem_pool", lambda pool: nullcontext())

    def graph_factory(*, keep_graph: bool) -> CaptureGraph:
        graph = CaptureGraph()
        graphs.append(graph)
        return graph

    monkeypatch.setattr(torch.cuda, "CUDAGraph", graph_factory)
    monkeypatch.setattr(torch.cuda, "graph", lambda graph, **kwargs: nullcontext())

    # The fake pool only supplies the external handle consumed by graph capture.
    @dataclass
    class Pool:
        id: tuple[int, int] = (1, 1)

        def snapshot(self) -> list[dict[str, int]]:
            return []

    owner.pool = cast(torch.cuda.MemPool, Pool())
    original = RuntimeError("capture failed after warmup")

    primary_launches = 0
    control_launches = 0

    def primary_launch() -> None:
        nonlocal primary_launches
        primary_launches += 1
        if failure_site == "primary" and primary_launches == 2:
            raise original

    def control_launch() -> None:
        nonlocal control_launches
        control_launches += 1
        if failure_site == "control" and control_launches == 2:
            raise original

    def source_snapshot(
        stream: torch.cuda.Stream, contract: registry.FfnExecutionContract
    ) -> tuple[xpool.native.cuda.CaptureAllocation, ...]:
        if failure_site == "observation":
            raise original
        return ()

    monkeypatch.setattr(registry.capture_workspace, "snapshot_resources", source_snapshot)
    monkeypatch.setattr(
        xpool.native.cuda.graph, "used_resource_regions", lambda graph, allocations: range(len(allocations))
    )

    with pytest.raises(RuntimeError) as caught:
        owner.capture_pair(
            primary_launch=primary_launch,
            control_launch=control_launch,
            local_allocations=(),
            execution_contract=TEST_EXECUTION_CONTRACT,
            external_ranges=(),
        )

    assert caught.value is original
    assert all(graph.resets == 0 for graph in graphs)
    assert owner.capture_graphs == graphs
    assert len(graphs) == (2 if failure_site == "control" else 1)
    owner.release_capture_resources()
    assert all(graph.resets == 1 for graph in graphs)
    assert not owner.capture_graphs
    assert owner.pool is not None
    assert not stream.destroyed
    assert owner.sealed
    stream.failure = RuntimeError("sealed construction must not be synchronized again")
    owner.close()
    assert owner.pool is None
    assert stream.destroyed


@pytest.mark.parametrize("failure_site", (None, "construction", "before_trim", "trim", "after_trim"))
def test_registry_completion_releases_locals_and_retains_service_backing(
    monkeypatch: pytest.MonkeyPatch, failure_site: str | None
) -> None:
    config = synthetic_config()
    experiments, _ = corpus.calibration_experiments(config)
    experiment = experiments[0]
    plan = experiment.fabric_plan(
        "a" * FABRIC_UID_HEX_LENGTH, config=config, ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,)
    )
    external = torch.empty(1)
    external_ref = weakref.ref(external)
    service_weight = torch.empty(1)
    weight_ref = weakref.ref(service_weight)
    installation = registry.FfnExecutionRegistry(
        external_resources=(
            execution.OwnedExecutionResource(
                owner=external,
                region=xpool.native.ffnagent.CapturedResourceRange("table", 0x1000, 0x1000, 4, True),
            ),
        ),
        layer_weights=((cast(weights.FfnLayerWeights, service_weight),),),
    )
    del external
    del service_weight
    temporary_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    backing_refs: list[weakref.ReferenceType[PoolSnapshot]] = []
    events: list[str] = []
    original = RuntimeError(f"{failure_site} failed")

    def construct(owner: registry.FfnExecutionRegistry, **arguments: object) -> None:
        events.append("construction")
        temporary = torch.empty(1)
        backing = PoolSnapshot((128,))
        lane = registry.LaneCaptureResources()
        lane.pool = cast(torch.cuda.MemPool, backing)
        lane.capture_tensors.append(temporary)
        owner.capture_lanes.append(lane)
        temporary_refs.append(weakref.ref(temporary))
        backing_refs.append(weakref.ref(backing))
        if failure_site == "construction":
            raise original
        lane.capture_tensors.clear()
        lane.sealed = True

    def synchronize(device: int) -> None:
        assert device == 1
        assert temporary_refs[0]() is None
        assert backing_refs[0]() is not None
        phase = "after_trim" if "trim" in events else "before_trim"
        events.append(phase)
        if failure_site == phase:
            raise original

    def trim() -> None:
        assert temporary_refs[0]() is None
        assert backing_refs[0]() is not None
        assert installation.capture_lanes[0].sealed
        events.append("trim")
        if failure_site == "trim":
            raise original

    monkeypatch.setattr(registry.FfnExecutionRegistry, "construct_execution", construct)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", trim)
    materialize = partial(
        installation.materialize,
        fabric_plan=plan,
        model_specs=experiment.model_specs,
        ffnagent_index=0,
    )
    if failure_site is None:
        materialize()
    else:
        with pytest.raises(RuntimeError) as caught:
            materialize()
        assert caught.value is original
    phases = ["construction", "before_trim", "trim", "after_trim"]
    assert events == (phases if failure_site is None else phases[: phases.index(failure_site) + 1])
    assert backing_refs[0]() is not None
    assert external_ref() is not None
    assert weight_ref() is not None
    assert installation.capture_lanes[0].pool is backing_refs[0]()
    if failure_site == "construction":
        assert installation.capture_lanes[0].capture_tensors
        assert not installation.capture_lanes[0].sealed
    with pytest.raises(RuntimeError, match="cannot be retried"):
        materialize()
    installation.close()
    assert external_ref() is None
    assert weight_ref() is None


def test_empty_participant_installs_once_and_seals_before_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    config = synthetic_config(ffn_devices=(1, 2))
    experiments, _ = corpus.calibration_experiments(config)
    experiment = next(item for item in experiments if all(model.ffn_tp_size == 1 for model in item.models))
    plan = experiment.fabric_plan(
        "a" * FABRIC_UID_HEX_LENGTH,
        config=config,
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
    )
    installation = registry.FfnExecutionRegistry(
        external_resources=(), layer_weights=tuple(tuple(None for _ in spec.layers) for spec in experiment.model_specs)
    )
    events: list[str] = []

    def install(descriptor: xpool.native.ffnagent.ExecutionInstallDescriptor) -> None:
        assert isinstance(descriptor, xpool.native.ffnagent.ExecutionInstallDescriptor)
        events.append("install")

    def synchronize(device: int) -> None:
        assert device == 2
        assert len(installation.capture_lanes) == plan.executor_lane_count
        assert all(lane.sealed for lane in installation.capture_lanes)
        assert all(lane.pool is None and lane.stream_owner is None for lane in installation.capture_lanes)
        events.append("synchronize")

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("trim"))
    monkeypatch.setattr(xpool.native.ffnagent, "install_execution", install)
    installation.materialize(
        fabric_plan=plan,
        model_specs=experiment.model_specs,
        ffnagent_index=1,
    )
    assert events == ["install", "synchronize", "trim", "synchronize"]


@pytest.fixture
def external_declaration() -> ExecutionResourceDeclaration:
    return ExecutionResourceDeclaration(
        requirement=ExecutionResourceRequirement(
            key="table",
            request_bytes=(1024,),
            logical_extent_bound_bytes=1024,
            exact_logical_bytes=1024,
            alignment_bytes=256,
        ),
        aliases=("table",),
        uses=("dense",),
    )


@pytest.mark.parametrize(("key", "extent"), (("table", 1025), ("other", 1024)))
def test_external_owner_cannot_outgrow_or_change_its_prospective_resource(
    key: str, extent: int, external_declaration: ExecutionResourceDeclaration
) -> None:
    resource = execution.OwnedExecutionResource(
        owner=object(),
        region=xpool.native.ffnagent.CapturedResourceRange(key, 0x1000, 0x1000, extent, True),
    )
    installation = registry.FfnExecutionRegistry(layer_weights=(), external_resources=(resource,))
    contract = TEST_EXECUTION_CONTRACT.model_copy(update={"shared_resources": (external_declaration,)})
    plan = preparation.ExecutionResourcePlan(lane_resources=(), shared_resources=())
    with pytest.raises(ValueError, match="prospective"):
        installation.validate_external_resources(resource_plan=plan, contract=contract)


@pytest.mark.parametrize("shared_read_only", (False, True))
def test_external_owner_requires_matching_prepared_budget_and_read_only_sharing(
    shared_read_only: bool, external_declaration: ExecutionResourceDeclaration
) -> None:
    resource = execution.OwnedExecutionResource(
        owner=object(),
        region=xpool.native.ffnagent.CapturedResourceRange("table", 0x1000, 0x1000, 512, shared_read_only),
    )
    installation = registry.FfnExecutionRegistry(layer_weights=(), external_resources=(resource,))
    plan = preparation.ExecutionResourcePlan(lane_resources=(), shared_resources=())
    with pytest.raises(ValueError, match="no matching prospective requirement"):
        installation.validate_external_resources(resource_plan=plan, contract=TEST_EXECUTION_CONTRACT)
    contract = TEST_EXECUTION_CONTRACT.model_copy(update={"shared_resources": (external_declaration,)})
    if not shared_read_only:
        with pytest.raises(ValueError, match="sharing disagrees"):
            installation.validate_external_resources(resource_plan=plan, contract=contract)
        return
    installation.validate_external_resources(resource_plan=plan, contract=contract)
    selected = preparation.ExecutionResourcePlan(
        lane_resources=(),
        shared_resources=(preparation.ExecutionResourceGroup(external_declaration.requirement, (0,)),),
    )
    assert installation.external_ranges(selected, 0) == (resource.region,)
    assert installation.external_ranges(plan, 0) == ()


def test_selected_external_resource_requires_a_retained_owner(
    external_declaration: ExecutionResourceDeclaration,
) -> None:
    experiments, _ = corpus.calibration_experiments(synthetic_config())
    model = experiments[0].models[0]
    profile = model.instance_plan().ffn_profile
    layout = preparation.prepare_ffn_layer(
        model.spec, layer_ordinal=0, payload_dtype=profile.payload_dtype, tp_size=model.ffn_tp_size, tp_rank=0
    )
    signatures = layout.execution_signatures(profile=profile)
    contract = TEST_EXECUTION_CONTRACT.model_copy(update={"shared_resources": (external_declaration,)})
    plan = preparation.ExecutionResourcePlan.from_signatures(signatures, contract=contract)
    installation = registry.FfnExecutionRegistry(layer_weights=(), external_resources=())
    with pytest.raises(ValueError, match="no retained owner"):
        installation.validate_external_resources(resource_plan=plan, contract=contract)


@pytest.mark.parametrize("change", ("address", "size", "missing"))
def test_fixed_source_backing_cannot_change_between_captures(monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    lane = registry.LaneCaptureResources()
    lane.stream = cast(torch.cuda.ExternalStream, CaptureStream())
    snapshots = [
        (xpool.native.cuda.CaptureAllocation("source", 0x1000, 512),),
        ()
        if change == "missing"
        else (
            xpool.native.cuda.CaptureAllocation(
                "source", 0x2000 if change == "address" else 0x1000, 1024 if change == "size" else 512
            ),
        ),
    ]
    monkeypatch.setattr(registry.capture_workspace, "snapshot_resources", lambda *args: snapshots.pop(0))
    lane.snapshot_source_resources(TEST_EXECUTION_CONTRACT)
    with pytest.raises(ValueError, match="source cache"):
        lane.snapshot_source_resources(TEST_EXECUTION_CONTRACT)


@pytest.mark.parametrize("failure_site", ("stream", "graph", "source_maps"))
def test_failed_capture_release_preserves_remaining_owners(failure_site: str, monkeypatch: pytest.MonkeyPatch) -> None:
    owner = registry.LaneCaptureResources()
    original = RuntimeError("cleanup failed")
    stream = CaptureStream(failure=original if failure_site == "stream" else None)
    graph = CaptureGraph(failure=original if failure_site == "graph" else None)
    tensor = torch.empty(1)
    owner.stream_owner = cast(xpool.native.cuda.OwnedCudaStream, stream)
    owner.pool = cast(torch.cuda.MemPool, object())
    owner.capture_graphs.append(cast(torch.cuda.CUDAGraph, graph))
    owner.capture_tensors.append(tensor)

    def release_maps(address: int) -> None:
        if failure_site == "source_maps":
            raise original

    monkeypatch.setattr(xpool.native.cuda.blas, "release_workspaces", release_maps)
    with pytest.raises(RuntimeError) as caught:
        owner.release_capture_resources()

    assert caught.value is original
    assert owner.stream_owner is stream
    assert owner.pool is not None
    assert owner.capture_tensors == [tensor]
    assert not owner.sealed
    assert not stream.destroyed
    stream.failure = None
    graph.failure = None
    monkeypatch.setattr(xpool.native.cuda.blas, "release_workspaces", lambda address: None)
    owner.release_capture_resources()
    assert graph.resets == 1
    assert not owner.capture_graphs
    assert not owner.capture_tensors
    assert owner.pool is not None
    owner.close()
    assert owner.pool is None
    assert stream.destroyed
