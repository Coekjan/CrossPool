import pytest
import torch

from xpool import ffn
from xpool.native.ffn import LayerKind
from xpool.quantization import Unquantized
from xtest.harness.support.ffn import compare_ffn_router, router_coefficient_outliers, router_selection_bounds


@pytest.fixture
def layer() -> ffn.MoeFfnSpec:
    return ffn.MoeFfnSpec(
        kind=LayerKind.MOE,
        layer_id=0,
        router_algorithm=ffn.RouterAlgorithm.SOFTMAX,
        router_projection_precision=ffn.RouterProjectionPrecision.FP32_LOGITS,
        expert_intermediate_size=4,
        routed_expert_count=17,
        shared_expert_count=0,
        routed_topk=3,
        renormalize=True,
        routed_scaling_factor=1.0,
        router=ffn.ProjectionSpec(quantization=Unquantized(), values_key="router"),
        routed_experts=ffn.StackedExpertSource(
            gate_up=ffn.ProjectionSpec(quantization=Unquantized(), values_key="up"),
            down=ffn.ProjectionSpec(quantization=Unquantized(), values_key="down"),
        ),
        shared_expert=None,
    )


@pytest.mark.parametrize(
    "algorithm, selected, wrong",
    (
        (ffn.RouterAlgorithm.SOFTMAX, (0, 1, 2), (0, 1, 3)),
        (ffn.RouterAlgorithm.SIGMOID, (16, 15, 14), (16, 15, 13)),
    ),
)
def test_router_requires_declared_priority_for_exact_ties(
    layer: ffn.MoeFfnSpec, algorithm: ffn.RouterAlgorithm, selected: tuple[int, ...], wrong: tuple[int, ...]
) -> None:
    layer = layer.model_copy(
        update={
            "router_algorithm": algorithm,
            "router_projection_precision": (
                ffn.RouterProjectionPrecision.FP32
                if algorithm is ffn.RouterAlgorithm.SIGMOID
                else ffn.RouterProjectionPrecision.FP32_LOGITS
            ),
        }
    )
    logits = torch.zeros((1, 17))
    results = [
        compare_ffn_router(
            layer=layer,
            execution_capacity=32,
            logits=logits,
            reference_logits=logits.double(),
            correction_bias=None,
            topk_ids=torch.tensor((ids,), dtype=torch.int32),
            topk_weights=torch.full((1, 3), 1 / 3),
            installed_ids_equal=True,
            installed_weights_equal=True,
            eager_graph_equal=True,
        )
        for ids in (selected, wrong)
    ]
    assert results[0].passed
    assert not results[1].passed
    assert results[1].selection_violation_count == 1
    assert results[0].boundary_ambiguity_count == 0


@pytest.mark.parametrize("failure", ("projection", "coefficient", "installation", "graph"))
def test_router_checks_projection_weights_and_installation_separately(layer: ffn.MoeFfnSpec, failure: str) -> None:
    logits = torch.arange(17, dtype=torch.float32).view(1, -1)
    ids = torch.tensor(((16, 15, 14),), dtype=torch.int32)
    weights = torch.softmax(logits, dim=-1).gather(1, ids.long())
    weights /= weights.sum(dim=-1, keepdim=True)
    result = compare_ffn_router(
        layer=layer,
        execution_capacity=32,
        logits=logits,
        reference_logits=logits.double() + (1 if failure == "projection" else 0),
        correction_bias=None,
        topk_ids=ids,
        topk_weights=weights * (2 if failure == "coefficient" else 1),
        installed_ids_equal=failure != "installation",
        installed_weights_equal=True,
        eager_graph_equal=failure != "graph",
    )
    assert not result.passed
    assert result.projection_outlier_count == (17 if failure == "projection" else 0)
    assert result.coefficient_outlier_count == (3 if failure == "coefficient" else 0)
    assert result.selection_violation_count == 0
    assert result.installed_ids_equal == (failure != "installation")


@pytest.mark.parametrize("ids", (((16, 15, 0),), ((16, 16, 14),), ((17, 15, 14),)))
def test_router_rejects_clearly_worse_duplicate_or_invalid_experts(
    layer: ffn.MoeFfnSpec, ids: tuple[tuple[int, ...], ...]
) -> None:
    logits = torch.arange(17, dtype=torch.float32).view(1, -1)
    result = compare_ffn_router(
        layer=layer,
        execution_capacity=32,
        logits=logits,
        reference_logits=logits.double(),
        correction_bias=None,
        topk_ids=torch.tensor(ids, dtype=torch.int32),
        topk_weights=torch.full((1, 3), 1 / 3),
        installed_ids_equal=True,
        installed_weights_equal=True,
        eager_graph_equal=True,
    )
    assert not result.passed
    assert result.selection_violation_count == 1


def test_sigmoid_correction_store_can_force_an_exact_tie(layer: ffn.MoeFfnSpec) -> None:
    layer = layer.model_copy(
        update={
            "router_algorithm": ffn.RouterAlgorithm.SIGMOID,
            "router_projection_precision": ffn.RouterProjectionPrecision.FP32,
            "routed_expert_count": 2,
            "routed_topk": 1,
        }
    )
    logits = torch.tensor(((0.0, -1.0),))
    correction = torch.full((2,), 2.0**24)
    for expert, expected in ((1, True), (0, False)):
        result = compare_ffn_router(
            layer=layer,
            execution_capacity=1,
            logits=logits,
            reference_logits=logits.double(),
            correction_bias=correction,
            topk_ids=torch.tensor(((expert,),), dtype=torch.int32),
            topk_weights=torch.ones((1, 1)),
            installed_ids_equal=True,
            installed_weights_equal=True,
            eager_graph_equal=True,
        )
        assert result.passed is expected


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
def test_coefficient_comparison_rounds_the_interval_instead_of_granting_one_ulp(dtype: torch.dtype) -> None:
    midpoint = (
        torch.tensor(0.25, dtype=dtype).float()
        + torch.nextafter(torch.tensor(0.25, dtype=dtype), torch.tensor(float("inf"), dtype=dtype)).float()
    ) / 2
    lower = torch.tensor(0.25, dtype=dtype).float()
    upper = torch.nextafter(torch.tensor(0.25, dtype=dtype), torch.tensor(float("inf"), dtype=dtype)).float()
    assert not router_coefficient_outliers(lower, midpoint, rounding_dtype=dtype)
    assert not router_coefficient_outliers(upper, midpoint, rounding_dtype=dtype)
    if dtype is torch.bfloat16:
        reference = torch.tensor(0.24462892273647296)
        assert not router_coefficient_outliers(torch.tensor(0.244140625), reference, rounding_dtype=dtype)
        assert not router_coefficient_outliers(torch.tensor(0.2451171875), reference, rounding_dtype=dtype)
        assert router_coefficient_outliers(upper, lower, rounding_dtype=dtype)
        assert router_coefficient_outliers(torch.tensor(0.2501), lower, rounding_dtype=dtype)
    assert router_coefficient_outliers(torch.tensor(0.5), lower, rounding_dtype=dtype)


def test_router_correction_affects_selection_but_not_coefficients(layer: ffn.MoeFfnSpec) -> None:
    layer = layer.model_copy(
        update={
            "router_algorithm": ffn.RouterAlgorithm.SIGMOID,
            "router_projection_precision": ffn.RouterProjectionPrecision.FP32,
            "routed_expert_count": 3,
            "routed_topk": 2,
        }
    )
    logits = torch.tensor(((0.0, -1.0, 2.0),))
    correction = torch.tensor((3.0, 4.0, -5.0))
    ids = torch.tensor(((1, 0),), dtype=torch.int32)
    activated = torch.sigmoid(logits)
    coefficients = activated.gather(1, ids.long())
    coefficients /= coefficients.sum(dim=-1, keepdim=True)
    wrong = (activated + correction).gather(1, ids.long())
    wrong /= wrong.sum(dim=-1, keepdim=True)
    for values, expected in ((coefficients, True), (wrong, False)):
        result = compare_ffn_router(
            layer=layer,
            execution_capacity=1,
            logits=logits,
            reference_logits=logits.double(),
            correction_bias=correction,
            topk_ids=ids,
            topk_weights=values,
            installed_ids_equal=True,
            installed_weights_equal=True,
            eager_graph_equal=True,
        )
        assert result.passed is expected
        assert result.selection_violation_count == 0


def test_selection_policy_rejects_an_uncovered_softmax_domain(layer: ffn.MoeFfnSpec) -> None:
    logits = torch.arange(17, dtype=torch.float32).view(1, -1) * 1000
    with pytest.raises(ValueError, match="subnormal Softmax cutoff"):
        compare_ffn_router(
            layer=layer,
            execution_capacity=1,
            logits=logits,
            reference_logits=logits.double(),
            correction_bias=None,
            topk_ids=torch.tensor(((16, 15, 14),), dtype=torch.int32),
            topk_weights=torch.tensor(((1.0, 0.0, 0.0),)),
            installed_ids_equal=True,
            installed_weights_equal=True,
            eager_graph_equal=True,
        )
    with pytest.raises(ValueError, match="corrected Softmax"):
        router_selection_bounds(logits, algorithm=ffn.RouterAlgorithm.SOFTMAX, correction_bias=torch.zeros(17))
