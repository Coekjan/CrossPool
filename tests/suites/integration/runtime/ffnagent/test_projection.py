import pytest
import torch
from tests.suites.integration.runtime.ffnagent.support import dense_plan, router_plan

import xtest
from xpool import ffn, quantization
from xpool.ffn import GateTransform, RouterAlgorithm, RouterProjectionPrecision, SiluActivation, TopKWeightRounding
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import weights
from xpool.runtime.ffnagent.operators import dense, routing
from xpool.runtime.ffnagent.operators.activation import Activation


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
    plan = router_plan(
        hidden,
        router,
        algorithm=algorithm,
        precision=precision,
        routed_topk=2,
        renormalize=True,
        weight_rounding=weight_rounding,
    )
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = routing.Router(plan, workspace)
    ids, probabilities = operation.ids, operation.weights

    def run() -> None:
        operation.run_into(hidden_states=hidden, router_weights=router)

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
        i_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=ffn.GateUpOrder.CONCATENATED,
            values=TensorLayout((16, 16), torch.bfloat16, 0),
            scales=None,
            bias=TensorLayout((16,), torch.bfloat16, 0),
        ),
        o_proj=weights.ProjectionLayout(
            quantization=quantization.Unquantized(),
            gate_up_order=None,
            values=TensorLayout((16, 8), torch.bfloat16, 0),
            scales=None,
            bias=TensorLayout((16,), torch.bfloat16, 0),
        ),
        device=device,
    )
    for projection in (layer.i_proj, layer.o_proj):
        assert projection.bias is not None
        projection.values.copy_(torch.randn_like(projection.values))
        projection.bias.copy_(torch.linspace(-2, 2, 16, dtype=torch.bfloat16, device=device))
    hidden = torch.randn((4, 16), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(hidden)
    activation = Activation(function=SiluActivation(), gate=GateTransform())
    plan = dense_plan(hidden, layer, activation)
    workspace = torch.empty(plan.workspace.nbytes, device="cuda", dtype=torch.uint8)
    operation = dense.Dense(plan, workspace)

    def run() -> None:
        operation.run_into(hidden_states=hidden, layer_weights=layer, output=output)

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
    assert layer.i_proj.bias is not None and layer.o_proj.bias is not None
    projected = (hidden.float() @ layer.i_proj.values.float().t() + layer.i_proj.bias.float()).to(hidden.dtype)
    gate, up = projected.float().chunk(2, dim=-1)
    activated = (gate * torch.sigmoid(gate) * up).to(hidden.dtype)
    expected = (activated.float() @ layer.o_proj.values.float().t() + layer.o_proj.bias.float()).to(hidden.dtype)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.1)
