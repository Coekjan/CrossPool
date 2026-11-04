from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from pydantic import ValidationError

from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricDeploymentPlan,
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
from xpool.memory import DeviceMemoryEstimate
from xpool.native.ffn import LayerKind
from xtest.harness.support.config import TEST_EXECUTION_CONTRACT, TEST_MODEL_ID


def profile(payload_dtype: torch.dtype = torch.bfloat16) -> InstanceFfnProfile:
    """Build one valid rank-independent FFN Profile."""

    return InstanceFfnProfile(
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
        ffn_execution_contracts=(TEST_EXECUTION_CONTRACT,) * 2,
        pe_placements=(
            FabricPePlacement(role=FabricRole.ATNAGENT, device=0),
            FabricPePlacement(role=FabricRole.FFNAGENT, device=1),
            FabricPePlacement(role=FabricRole.FFNAGENT, device=2),
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
                model_id=TEST_MODEL_ID,
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


@pytest.mark.parametrize("word", [True, 1.0])
def test_generation_rejects_noninteger_words(word: bool | float) -> None:
    with pytest.raises(ValueError, match="uint64"):
        replace(FabricGenerationId(1, 2), high=word)


def test_fabric_uid_and_plan_round_trip() -> None:
    plan = fabric_plan()

    assert FabricPlan.model_validate_json(plan.model_dump_json()) == plan
    assert plan.model_plans[0].tp_size == 2
    with pytest.raises(ValueError, match="256 lowercase hex"):
        FabricUid(value="ab")


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_profile_round_trips_supported_payload_dtype(dtype: torch.dtype) -> None:
    value = profile(dtype)

    assert value.model_dump(mode="python")["payload_dtype"] is dtype
    assert value.model_dump(mode="json")["payload_dtype"] == str(dtype).removeprefix("torch.")
    assert InstanceFfnProfile.model_validate_json(value.model_dump_json()) == value


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.int64, torch.bool))
def test_profile_rejects_unsupported_payload_precision(dtype: torch.dtype) -> None:
    with pytest.raises(ValidationError, match="BF16 or FP16"):
        profile(dtype)


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


def test_deployment_retains_all_ffn_budgets_through_json() -> None:
    plan = fabric_plan()
    budgets = (
        DeviceMemoryEstimate(retained_bytes=1024, peak_bytes=4096),
        DeviceMemoryEstimate(retained_bytes=2048, peak_bytes=8192),
    )
    deployment = FabricDeploymentPlan(**plan.model_dump(), ffn_memory_estimates=budgets)
    returned = FabricDeploymentPlan.model_validate_json(deployment.model_dump_json())
    assert returned.ffn_memory_estimates == budgets
    assert returned.ffn_execution_contracts == plan.ffn_execution_contracts


@pytest.mark.parametrize("budgets", (None, (), (DeviceMemoryEstimate(retained_bytes=1, peak_bytes=2),)))
def test_deployment_requires_one_budget_per_ffn(budgets: tuple[DeviceMemoryEstimate, ...] | None) -> None:
    values = fabric_plan().model_dump()
    if budgets is not None:
        values["ffn_memory_estimates"] = budgets
    with pytest.raises(ValidationError, match=r"ffn_memory_estimates|co-indexed"):
        FabricDeploymentPlan.model_validate(values)
