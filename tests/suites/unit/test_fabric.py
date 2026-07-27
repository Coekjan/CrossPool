"""Behavior tests for generation-scoped FFN fabric contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from xpool.abi import TensorDType
from xpool.fabric import (
    FabricGeneration,
    FabricGenerationPhase,
    FabricModelPlan,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnLayerKind,
    FfnLayerSpec,
    FfnWorkload,
    FifoSchedulerPlan,
    RandomSchedulerPlan,
)


def workload(*, max_decode_rows: int = 4, max_prefill_rows: int = 8) -> FfnWorkload:
    """Build one valid rank-independent workload for contract behavior tests."""

    return FfnWorkload(
        model_config_digest="a" * 64,
        dtype=TensorDType.BF16,
        hidden_size=2048,
        layers=(
            FfnLayerSpec(layer_id=0, kind=FfnLayerKind.DENSE),
            FfnLayerSpec(layer_id=1, kind=FfnLayerKind.SPARSE),
        ),
        max_decode_rows=max_decode_rows,
        max_prefill_rows=max_prefill_rows,
    )


def fabric_plan(generation: FabricGeneration) -> FabricPlan:
    """Build one valid fabric plan for digest behavior tests."""

    contract = workload()
    return FabricPlan(
        generation=generation,
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(pe=0, role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(pe=1, role=FabricRole.ATNAGENT, cuda_device=1),
            FabricPePlacement(pe=2, role=FabricRole.FFNAGENT, cuda_device=2),
            FabricPePlacement(pe=3, role=FabricRole.FFNAGENT, cuda_device=3),
        ),
        executor_count=2,
        scheduler=FifoSchedulerPlan(),
        models=(FabricModelPlan(workload=contract, atn_tp_size=2, atn_dp_size=1),),
    )


def test_generation_is_nonzero_and_serializes_as_two_words() -> None:
    generation = FabricGeneration.create()

    assert generation.high != 0 or generation.low != 0
    assert fabric_plan(generation).model_dump(mode="json")["generation"] == {
        "high": generation.high,
        "low": generation.low,
    }


def test_generation_round_trips_canonical_text() -> None:
    """Generation text encoding is owned by the Fabric value type."""

    generation = FabricGeneration(high=1, low=2)

    assert generation.format() == "00000000000000010000000000000002"
    assert FabricGeneration.parse(generation.format()) == generation
    with pytest.raises(ValueError, match="32 lowercase hex"):
        FabricGeneration.parse("ABC")
    with pytest.raises(ValueError, match="nonzero"):
        FabricGeneration.parse("0" * 32)


def test_fabric_uid_validates_and_serializes_as_nested_value() -> None:
    uid = FabricUid(value="ab" * 128)
    generation = FabricGeneration(high=1, low=2)
    plan = fabric_plan(generation)
    plan = plan.model_copy(update={"uid": uid})

    assert uid.value == "ab" * 128
    assert plan.model_dump(mode="json")["uid"] == {"value": "ab" * 128}
    assert FabricPlan.model_validate_json(plan.model_dump_json()).uid == uid
    with pytest.raises(ValueError, match="256 lowercase hex"):
        FabricUid(value="ab")
    with pytest.raises(ValueError, match="only lowercase hex"):
        FabricUid(value="AB" * 128)


def test_plan_digest_is_stable_across_generation_replacement() -> None:
    first = fabric_plan(FabricGeneration(high=1, low=2))
    replacement = fabric_plan(FabricGeneration(high=3, low=4)).model_copy(update={"uid": FabricUid(value="cd" * 128)})

    assert first.digest() == replacement.digest()
    assert len(first.digest()) == 64


@pytest.mark.parametrize("mutation", ["executor_count", "models", "pe_placements", "scheduler"])
def test_plan_digest_changes_with_semantic_plan_mutation(mutation: str) -> None:
    plan = fabric_plan(FabricGeneration(high=1, low=2))
    if mutation == "executor_count":
        replacement = plan.model_copy(update={"executor_count": 1})
    elif mutation == "models":
        model = plan.models[0].model_copy(
            update={"workload": plan.models[0].workload.model_copy(update={"max_decode_rows": 5})}
        )
        replacement = plan.model_copy(update={"models": (model,)})
    elif mutation == "pe_placements":
        placements = list(plan.pe_placements)
        placements[0] = placements[0].model_copy(update={"cuda_device": 9})
        replacement = plan.model_copy(update={"pe_placements": tuple(placements)})
    else:
        replacement = plan.model_copy(update={"scheduler": RandomSchedulerPlan(seed=7)})

    assert replacement.digest() != plan.digest()


@pytest.mark.parametrize(
    ("phase", "successor"),
    [
        (FabricGenerationPhase.JOINING, FabricGenerationPhase.EXECUTABLE),
        (FabricGenerationPhase.EXECUTABLE, FabricGenerationPhase.QUIESCING),
        (FabricGenerationPhase.QUIESCING, FabricGenerationPhase.DRAINING),
        (FabricGenerationPhase.DRAINING, FabricGenerationPhase.FINALIZING),
        (FabricGenerationPhase.FINALIZING, FabricGenerationPhase.STOPPED),
        (FabricGenerationPhase.ABORTING, FabricGenerationPhase.STOPPED),
    ],
)
def test_generation_phase_accepts_only_exact_normal_successor(
    phase: FabricGenerationPhase,
    successor: FabricGenerationPhase,
) -> None:
    assert phase.allows(successor)
    assert not phase.allows(phase)


def test_every_live_generation_phase_can_abort() -> None:
    for phase in FabricGenerationPhase:
        assert phase.allows(FabricGenerationPhase.ABORTING) is (
            phase not in {FabricGenerationPhase.ABORTING, FabricGenerationPhase.STOPPED}
        )


@pytest.mark.parametrize(
    ("phase", "successor"),
    [
        (FabricParticipantPhase.JOINING, FabricParticipantPhase.JOINED),
        (FabricParticipantPhase.JOINED, FabricParticipantPhase.ACTIVE),
        (FabricParticipantPhase.ACTIVE, FabricParticipantPhase.QUIESCED),
        (FabricParticipantPhase.QUIESCED, FabricParticipantPhase.DRAINING),
        (FabricParticipantPhase.DRAINING, FabricParticipantPhase.DRAINED),
        (FabricParticipantPhase.DRAINED, FabricParticipantPhase.FINALIZED),
    ],
)
def test_participant_phase_accepts_only_explicit_successor(
    phase: FabricParticipantPhase,
    successor: FabricParticipantPhase,
) -> None:
    assert phase.allows(successor)
    assert not phase.allows(phase)


@pytest.mark.parametrize("field", ["max_decode_rows", "max_prefill_rows"])
def test_workload_requires_positive_capacities(field: str) -> None:
    contract = workload()

    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        FfnWorkload.model_validate({**contract.model_dump(mode="python"), field: 0})


def test_workload_rejects_duplicate_layer_ids() -> None:
    contract = workload()

    with pytest.raises(ValidationError, match="layer ids"):
        FfnWorkload.model_validate(
            {
                **contract.model_dump(mode="python"),
                "layers": [
                    {"layer_id": 0, "kind": FfnLayerKind.DENSE},
                    {"layer_id": 0, "kind": FfnLayerKind.SPARSE},
                ],
            }
        )


def test_plan_rejects_noncanonical_pe_mapping() -> None:
    plan = fabric_plan(FabricGeneration(high=1, low=2))
    placements = list(plan.pe_placements)
    placements[1] = placements[1].model_copy(update={"pe": 7})

    with pytest.raises(ValidationError, match="fabric PEs"):
        FabricPlan.model_validate({**plan.model_dump(mode="python"), "pe_placements": placements})


def test_plan_rejects_interleaved_agent_regions() -> None:
    plan = fabric_plan(FabricGeneration(high=1, low=2))
    placements = (
        plan.pe_placements[0],
        plan.pe_placements[2].model_copy(update={"pe": 1}),
        plan.pe_placements[1].model_copy(update={"pe": 2}),
        plan.pe_placements[3],
    )

    with pytest.raises(ValidationError, match="AtnAgent prefix"):
        FabricPlan.model_validate({**plan.model_dump(mode="python"), "pe_placements": placements})


def test_plan_rejects_combined_attention_tp_by_dp() -> None:
    contract = workload()
    placements = (
        *(FabricPePlacement(pe=pe, role=FabricRole.ATNAGENT, cuda_device=pe) for pe in range(4)),
        FabricPePlacement(pe=4, role=FabricRole.FFNAGENT, cuda_device=4),
    )

    with pytest.raises(ValidationError, match="combined attention TP-by-DP"):
        FabricPlan(
            generation=FabricGeneration(high=1, low=2),
            uid=FabricUid(value="ab" * 128),
            pe_placements=placements,
            executor_count=1,
            scheduler=FifoSchedulerPlan(),
            models=(FabricModelPlan(workload=contract, atn_tp_size=2, atn_dp_size=2),),
        )
