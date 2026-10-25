from __future__ import annotations

import pytest
from pydantic import ValidationError

from xpool.ffn import (
    FfnModelSpec,
    GatedProjectionSpec,
    GateUpOrder,
    MoeFfnSpec,
    PerExpertSource,
    ProjectionSpec,
    RouterAlgorithm,
    RouterProjectionPrecision,
    SiluActivation,
    StackedExpertSource,
    TopKWeightRounding,
    ValuesSourceLayout,
)
from xpool.native.ffn import LayerKind
from xpool.quantization import Mxfp4, Unquantized
from xtest.harness.support.config import TEST_MODEL_ID


def test_checkpoint_tensor_keys_keep_router_and_expert_semantic_order() -> None:
    first, second, shared = (
        GatedProjectionSpec(
            gate=ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.gate"),
            up=ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.up"),
            down=ProjectionSpec(quantization=Unquantized(), values_key=f"{prefix}.down"),
        )
        for prefix in ("first", "second", "shared")
    )
    assert first.tensor_keys() == ("first.gate", "first.up", "first.down")
    layer = MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=RouterAlgorithm.SOFTMAX,
        router_projection_precision=RouterProjectionPrecision.PAYLOAD,
        expert_intermediate_size=32,
        routed_expert_count=2,
        shared_expert_count=1,
        routed_topk=1,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ProjectionSpec(quantization=Unquantized(), values_key="router"),
        router_correction_bias_key="correction",
        routed_experts=PerExpertSource(experts=(first, second)),
        shared_expert=shared,
    )
    expected = (
        "router",
        "correction",
        "first.gate",
        "first.up",
        "first.down",
        "second.gate",
        "second.up",
        "second.down",
        "shared.gate",
        "shared.up",
        "shared.down",
    )
    assert layer.tensor_keys() == expected
    assert layer.model_copy(update={"router_correction_bias_key": None}).tensor_keys() == (expected[:1] + expected[2:])
    with pytest.raises(ValidationError, match="source count"):
        MoeFfnSpec.model_validate({**layer.model_dump(), "routed_expert_count": 3})


def test_stacked_sources_keep_physical_keys_and_complete_semantic_identity() -> None:
    source = StackedExpertSource(
        gate_up=ProjectionSpec(
            quantization=Mxfp4(),
            values_key="experts.gate_up.blocks",
            scales_key="experts.gate_up.scales",
            values_layout=ValuesSourceLayout.SCALE_BLOCKED,
            bias_key="experts.gate_up.bias",
        ),
        down=ProjectionSpec(
            quantization=Mxfp4(),
            values_key="experts.down.blocks",
            scales_key="experts.down.scales",
            values_layout=ValuesSourceLayout.SCALE_BLOCKED,
            bias_key="experts.down.bias",
        ),
        gate_up_order=GateUpOrder.INTERLEAVED,
    )
    spec = FfnModelSpec(
        model_id=TEST_MODEL_ID,
        hidden_size=64,
        activation=SiluActivation(alpha=1.702, limit=7.0, up_offset=1.0),
        layers=(
            MoeFfnSpec(
                kind=LayerKind.MOE,
                layer_id=0,
                router_algorithm=RouterAlgorithm.SOFTMAX,
                router_projection_precision=RouterProjectionPrecision.FP32_LOGITS,
                expert_intermediate_size=96,
                routed_expert_count=32,
                shared_expert_count=0,
                routed_topk=4,
                renormalize=True,
                routed_scaling_factor=1.0,
                router=ProjectionSpec(quantization=Unquantized(), values_key="router.weight", bias_key="router.bias"),
                routed_experts=source,
                shared_expert=None,
            ),
        ),
    )
    expected = (
        "router.weight",
        "router.bias",
        "experts.gate_up.blocks",
        "experts.gate_up.scales",
        "experts.gate_up.bias",
        "experts.down.blocks",
        "experts.down.scales",
        "experts.down.bias",
    )
    assert spec.layers[0].tensor_keys() == expected
    assert FfnModelSpec.model_validate_json(spec.model_dump_json()) == spec
    layer = spec.layers[0]
    assert isinstance(layer, MoeFfnSpec)
    assert layer.topk_weight_rounding is TopKWeightRounding.NONE
    rounded = spec.model_copy(
        update={"layers": (layer.model_copy(update={"topk_weight_rounding": TopKWeightRounding.LOGITS_DTYPE}),)}
    )
    assert FfnModelSpec.model_validate_json(rounded.model_dump_json()) == rounded
    assert rounded.digest() != spec.digest()
    assert hash(source) == hash(StackedExpertSource.model_validate_json(source.model_dump_json()))
    for activation in (
        SiluActivation(alpha=1.7, limit=7.0, up_offset=1.0),
        SiluActivation(alpha=1.702, limit=None, up_offset=1.0),
        SiluActivation(alpha=1.702, limit=7.0, up_offset=0.0),
    ):
        changed = spec.model_copy(update={"activation": activation})
        assert FfnModelSpec.model_validate_json(changed.model_dump_json()) == changed
        assert changed.digest() != spec.digest()

    payload = spec.model_dump(mode="json")
    payload["layers"][0]["routed_experts"]["gate_up"]["values_key"] = "router.weight"
    with pytest.raises(ValidationError, match="globally unique"):
        FfnModelSpec.model_validate(payload)


@pytest.mark.parametrize("parameter", ("alpha", "limit"))
@pytest.mark.parametrize("value", (0, -1, float("inf"), float("nan"), True, "1.702"))
def test_silu_slope_and_limit_require_positive_finite_numbers(parameter: str, value: object) -> None:
    with pytest.raises(ValidationError):
        SiluActivation.model_validate({parameter: value})


@pytest.mark.parametrize("value", (float("inf"), float("nan"), True, "1.0"))
def test_silu_up_offset_requires_a_finite_number(value: object) -> None:
    with pytest.raises(ValidationError):
        SiluActivation.model_validate({"up_offset": value})


@pytest.mark.parametrize("field", ("values_key", "scales_key", "bias_key"))
def test_projection_sources_reject_empty_keys(field: str) -> None:
    with pytest.raises(ValidationError):
        ProjectionSpec.model_validate({"quantization": Unquantized(), "values_key": "weight", field: ""})


def test_blocked_values_require_declared_scales_at_the_projection_boundary() -> None:
    with pytest.raises(ValidationError, match="scales key"):
        ProjectionSpec(quantization=Mxfp4(), values_key="weight", values_layout=ValuesSourceLayout.SCALE_BLOCKED)
