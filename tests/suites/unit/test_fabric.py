from __future__ import annotations

import pytest
import torch
from pydantic import ValidationError

from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricGenerationId,
    FabricGenerationPhase,
    FabricInstancePlan,
    FabricParticipantPhase,
    FabricPePlacement,
    FabricPlan,
    FabricRole,
    FabricUid,
    FfnModelPlan,
    FifoSchedulerPolicy,
    InstanceFfnLayerProfile,
    InstanceFfnProfile,
    InstanceRankTopology,
)
from xpool.native.ffn import LayerKind


def profile(payload_dtype: torch.dtype = torch.bfloat16) -> InstanceFfnProfile:
    """Build one valid rank-independent FFN Profile."""

    return InstanceFfnProfile(
        model_config_digest="a" * 64,
        payload_dtype=payload_dtype,
        hidden_size=4,
        layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
        decode_payload_row_capacity=4,
        prefill_payload_row_capacity=8,
        group_sum_complete_admitted=False,
    )


def fabric_plan() -> FabricPlan:
    """Build one valid co-indexed Fabric Plan."""

    return FabricPlan(
        generation=FabricGenerationId(high=1, low=2),
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=1),
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=2),
        ),
        executor_lane_count=2,
        scheduler=FifoSchedulerPolicy(),
        model_plans=(
            FfnModelPlan(
                model_spec_digest="b" * 64,
                layers=(DenseFfnLayerPlan(ffnagent_indices=(0, 1), local_intermediate_size=4),),
            ),
        ),
        instance_plans=(
            FabricInstancePlan(
                instance_id="m",
                ffn_profile=profile(),
                instance_rank_topology=InstanceRankTopology(
                    atn_tp_size=1,
                    atn_dp_size=1,
                    atnagent_indices=(0,),
                ),
            ),
        ),
    )


def test_generation_round_trips_canonical_text() -> None:
    generation = FabricGenerationId(high=1, low=2)

    assert generation.format() == "00000000000000010000000000000002"
    assert FabricGenerationId.parse(generation.format()) == generation
    with pytest.raises(ValueError, match="32 lowercase hex"):
        FabricGenerationId.parse("ABC")
    with pytest.raises(ValueError, match="nonzero"):
        FabricGenerationId.parse("0" * 32)


def test_fabric_uid_and_plan_round_trip() -> None:
    plan = fabric_plan()

    assert FabricPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.model_plans[0].tp_size == 2
    with pytest.raises(ValueError, match="256 lowercase hex"):
        FabricUid(value="ab")


def test_profile_round_trips_any_canonical_torch_dtype() -> None:
    value = profile(torch.float32)

    assert value.model_dump(mode="python")["payload_dtype"] is torch.float32
    assert value.model_dump(mode="json")["payload_dtype"] == "float32"
    assert InstanceFfnProfile.model_validate_json(value.model_dump_json()) == value


def test_profile_rejects_noncanonical_dtype_name() -> None:
    value = profile().model_dump(mode="json")
    value["payload_dtype"] = "half"

    with pytest.raises(ValidationError, match="canonical unqualified Torch name"):
        InstanceFfnProfile.model_validate(value)


@pytest.mark.parametrize(
    ("phase", "successor"),
    [
        (FabricGenerationPhase.PREPARING_JOIN, FabricGenerationPhase.JOINING),
        (FabricGenerationPhase.JOINING, FabricGenerationPhase.PREPARING_EXECUTION),
        (FabricGenerationPhase.PREPARING_EXECUTION, FabricGenerationPhase.ACTIVATING),
        (FabricGenerationPhase.ACTIVATING, FabricGenerationPhase.EXECUTABLE),
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


@pytest.mark.parametrize(
    ("phase", "successor"),
    [
        (FabricParticipantPhase.JOIN_READY, FabricParticipantPhase.JOINING),
        (FabricParticipantPhase.JOINING, FabricParticipantPhase.JOINED),
        (FabricParticipantPhase.JOINED, FabricParticipantPhase.EXECUTION_READY),
        (FabricParticipantPhase.EXECUTION_READY, FabricParticipantPhase.ACTIVE),
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


def test_profile_rejects_duplicate_layer_ids() -> None:
    value = profile().model_dump(mode="python")
    value["layers"] = [
        {"layer_id": 0, "kind": LayerKind.DENSE},
        {"layer_id": 0, "kind": LayerKind.MOE},
    ]

    with pytest.raises(ValidationError, match="layer ids"):
        InstanceFfnProfile.model_validate(value)


def test_plan_rejects_interleaved_agent_regions() -> None:
    plan = fabric_plan()
    placements = (plan.pe_placements[1], plan.pe_placements[0], plan.pe_placements[2])

    with pytest.raises(ValidationError, match="AtnAgent prefix"):
        FabricPlan.model_validate({**plan.model_dump(mode="python"), "pe_placements": placements})


def test_plan_rejects_non_coindexed_model_and_instance_tables() -> None:
    plan = fabric_plan()

    with pytest.raises(ValidationError, match="co-indexed"):
        FabricPlan.model_validate(
            {**plan.model_dump(mode="python"), "instance_plans": (*plan.instance_plans, *plan.instance_plans)}
        )
