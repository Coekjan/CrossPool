import pytest
import torch

import xtest
from xpool.ffn import RouterAlgorithm, RouterProjectionPrecision, SiluActivation, TopKWeightRounding
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import execution, weights
from xpool.runtime.ffnagent.operators import dense, linear, routing


@pytest.fixture(autouse=True)
def ffn_blas_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    # FfnAgent installs this policy before bootstrap; component calls have the same prerequisite.
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":0:0")


@pytest.mark.parametrize(
    "algorithm,precision,weight_rounding",
    (
        (RouterAlgorithm.SOFTMAX, RouterProjectionPrecision.PAYLOAD, TopKWeightRounding.NONE),
        (RouterAlgorithm.SOFTMAX, RouterProjectionPrecision.PAYLOAD, TopKWeightRounding.LOGITS_DTYPE),
        (RouterAlgorithm.SOFTMAX, RouterProjectionPrecision.FP32_LOGITS, TopKWeightRounding.NONE),
        (RouterAlgorithm.SOFTMAX, RouterProjectionPrecision.FP32_LOGITS, TopKWeightRounding.LOGITS_DTYPE),
        (RouterAlgorithm.SIGMOID, RouterProjectionPrecision.FP32, TopKWeightRounding.NONE),
    ),
)
@xtest.requirements(device_count=1)
def test_router_projection_bias_is_fused_before_rounding_and_replays_without_allocation(
    algorithm: RouterAlgorithm,
    precision: RouterProjectionPrecision,
    weight_rounding: TopKWeightRounding,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    torch.manual_seed(41)
    hidden = torch.randn((4, 16), device="cuda", dtype=torch.bfloat16)
    dtype = precision.weight_dtype(payload_dtype=hidden.dtype)
    bias_dtype = hidden.dtype if precision is RouterProjectionPrecision.PAYLOAD else torch.float32
    router = weights.MoeRouterWeights(
        projection=weights.ProjectionWeights(
            values=torch.randn((8, 16), device="cuda", dtype=dtype),
            scales=None,
            bias=torch.linspace(-2, 2, 8, device="cuda", dtype=bias_dtype),
        ),
        correction_bias=None,
    )
    workspace = torch.empty(
        execution.router_workspace_bytes(
            precision,
            payload_dtype=hidden.dtype,
            payload_row_capacity=4,
            hidden_size=16,
            routed_expert_count=8,
            routed_topk=2,
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    ids = torch.empty((4, 2), dtype=torch.int32, device="cuda")
    probabilities = torch.empty((4, 2), dtype=torch.float32, device="cuda")

    def run() -> None:
        routing.compute_routed_topk(
            algorithm,
            precision,
            projection_operation=linear.LinearOperationKind.TORCH,
            weight_rounding=weight_rounding,
            hidden_states=hidden,
            router_weights=router,
            workspace=workspace,
            routed_ids=ids,
            routed_weights=probabilities,
            renormalize=True,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert before == after
    hidden.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert router.projection.bias is not None
    logits = hidden.float() @ router.projection.values.float().t() + router.projection.bias.float()
    if precision is RouterProjectionPrecision.PAYLOAD:
        logits = logits.to(hidden.dtype).float()
    scores = torch.softmax(logits, dim=-1) if algorithm is RouterAlgorithm.SOFTMAX else torch.sigmoid(logits)
    expected_weights, expected_ids = torch.topk(scores, 2, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    if weight_rounding is TopKWeightRounding.LOGITS_DTYPE:
        logits_dtype = hidden.dtype if precision is RouterProjectionPrecision.PAYLOAD else torch.float32
        expected_weights = expected_weights.to(logits_dtype).float()
    torch.testing.assert_close(ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(probabilities, expected_weights, rtol=1e-3, atol=1e-4)


@xtest.requirements(device_count=1)
def test_dense_projection_bias_preserves_fused_rounding_without_capture_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.backends.cuda.matmul, "fp32_precision", "ieee")
    torch.manual_seed(37)
    device = torch.device("cuda", torch.cuda.current_device())
    layer = weights.DenseFfnWeights.allocate(
        gate_up=weights.ProjectionLayout(
            values=TensorLayout((16, 16), torch.bfloat16, 0),
            scales=None,
            bias=TensorLayout((16,), torch.bfloat16, 0),
        ),
        down=weights.ProjectionLayout(
            values=TensorLayout((16, 8), torch.bfloat16, 0),
            scales=None,
            bias=TensorLayout((16,), torch.bfloat16, 0),
        ),
        device=device,
    )
    for projection in (layer.gate_up, layer.down):
        assert projection.bias is not None
        projection.values.copy_(torch.randn_like(projection.values))
        projection.bias.copy_(torch.linspace(-2, 2, 16, dtype=torch.bfloat16, device=device))
    hidden = torch.randn((4, 16), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(hidden)
    workspace = torch.empty(
        execution.dense_workspace_bytes(
            payload_dtype=hidden.dtype,
            row_capacity=4,
            local_intermediate_size=8,
            activation=SiluActivation(),
        ),
        dtype=torch.uint8,
        device=device,
    )

    def run() -> None:
        dense.compute_dense_partial(
            hidden_states=hidden,
            layer_weights=layer,
            workspace=workspace,
            output=output,
            activation=SiluActivation(),
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        before = torch.cuda.memory_stats()["allocation.all.allocated"]
        run()
        after = torch.cuda.memory_stats()["allocation.all.allocated"]
    assert before == after
    hidden.mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    assert layer.gate_up.bias is not None and layer.down.bias is not None
    projected = (hidden.float() @ layer.gate_up.values.float().t() + layer.gate_up.bias.float()).to(hidden.dtype)
    gate, up = projected.float().chunk(2, dim=-1)
    activated = (gate * torch.sigmoid(gate) * up).to(hidden.dtype)
    expected = (activated.float() @ layer.down.values.float().t() + layer.down.bias.float()).to(hidden.dtype)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)
