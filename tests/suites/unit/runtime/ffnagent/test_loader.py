"""Pure composition tests for Plan-facing FFN weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from tests.harness.support.config import install_test_config, reset_global_config
from xpool.config import XpoolConfig
from xpool.fabric import (
    DenseFfnLayerPlan,
    FabricGenerationId,
    FabricInstancePlan,
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
from xpool.ffn import ActivationKind, DenseFfnSpec, FfnModelSpec, GatedFfnCheckpointKeys
from xpool.native.ffn import LayerKind
from xpool.runtime.ffnagent import loader, weights

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__)


def test_plan_materialization_selects_only_local_layers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = FfnModelSpec(
        model_id="model",
        architecture_name="Qwen3ForCausalLM",
        model_config_digest="a" * 64,
        hidden_size=4,
        activation=ActivationKind.SILU,
        layers=(
            DenseFfnSpec(
                kind=LayerKind.DENSE,
                layer_id=0,
                intermediate_size=8,
                checkpoint=GatedFfnCheckpointKeys(
                    gate_weight_key="gate",
                    up_weight_key="up",
                    down_weight_key="down",
                ),
            ),
        ),
    )
    plan = FabricPlan(
        generation=FabricGenerationId(high=1, low=2),
        uid=FabricUid(value="ab" * 128),
        pe_placements=(
            FabricPePlacement(role=FabricRole.ATNAGENT, cuda_device=0),
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=1),
            FabricPePlacement(role=FabricRole.FFNAGENT, cuda_device=2),
        ),
        executor_lane_count=1,
        scheduler=FifoSchedulerPolicy(),
        model_plans=(
            FfnModelPlan(
                model_spec_digest=spec.digest(),
                layers=(DenseFfnLayerPlan(ffnagent_indices=(0,), local_intermediate_size=8),),
            ),
        ),
        instance_plans=(
            FabricInstancePlan(
                instance_id="model",
                ffn_profile=InstanceFfnProfile(
                    model_config_digest=spec.model_config_digest,
                    payload_dtype=torch.bfloat16,
                    hidden_size=4,
                    layers=(InstanceFfnLayerProfile(layer_id=0, kind=LayerKind.DENSE),),
                    decode_payload_row_capacity=1,
                    prefill_payload_row_capacity=2,
                    group_sum_complete_admitted=False,
                ),
                instance_rank_topology=InstanceRankTopology(
                    atn_tp_size=1,
                    atn_dp_size=1,
                    atnagent_indices=(0,),
                ),
            ),
        ),
    )
    install_test_config(
        XpoolConfig.from_mapping(
            {
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "atn": {"devices": [0]},
                "ffn": {"devices": [1, 2]},
                "models": [{"id": "model", "path": str(tmp_path)}],
            }
        )
    )
    weight = cast(weights.FfnLayerWeights, object())
    observed_requests: list[tuple[loader.LocalLayerWeightRequest, ...]] = []

    def materialize(
        *,
        requests: tuple[loader.LocalLayerWeightRequest, ...],
    ) -> tuple[weights.FfnLayerWeights, ...]:
        observed_requests.append(requests)
        return (weight,) if requests else ()

    monkeypatch.setattr(loader, "materialize_local_layer_weights", materialize)

    assert loader.materialize_layer_weights(
        fabric_plan=plan,
        model_specs=(spec,),
        ffnagent_index=0,
    ) == ((weight,),)
    assert loader.materialize_layer_weights(
        fabric_plan=plan,
        model_specs=(spec,),
        ffnagent_index=1,
    ) == ((None,),)
    assert observed_requests[0] == (
        loader.LocalLayerWeightRequest(
            model_path=tmp_path,
            hidden_size=4,
            payload_dtype=torch.bfloat16,
            router_weight_dtype=None,
            layer=spec.layers[0],
            tp_rank=0,
            tp_size=1,
        ),
    )
    assert observed_requests[1] == ()
