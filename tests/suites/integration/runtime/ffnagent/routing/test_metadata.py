from __future__ import annotations

import torch

import xtest
from xpool.runtime.ffnagent.operators import routing


@xtest.requirements(device_count=1)
def test_finalize_moe_routing_replays_one_graph_for_dynamic_live_rows() -> None:
    routed_ids = torch.tensor(
        ((0, 2), (1, 3), (2, 0), (3, 1)),
        device="cuda",
        dtype=torch.int32,
    )
    routed_weights = torch.tensor(
        ((0.75, 0.25), (0.6, 0.4), (0.9, 0.1), (0.55, 0.45)),
        device="cuda",
        dtype=torch.float32,
    )
    final_ids = torch.empty((4, 3), device="cuda", dtype=torch.int32)
    final_weights = torch.empty((4, 3), device="cuda", dtype=torch.float32)
    payload_rows = torch.tensor((2,), device="cuda", dtype=torch.int64)

    def finalize() -> None:
        routing.finalize_moe_routing(
            routed_ids=routed_ids,
            routed_weights=routed_weights,
            final_ids=final_ids,
            final_weights=final_weights,
            payload_rows=payload_rows,
            routed_expert_count=4,
            shared_expert_count=1,
            routed_scaling_factor=2.0,
        )

    finalize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        finalize()

    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        final_ids,
        torch.tensor(((0, 2, 4), (1, 3, 4), (-1, -1, -1), (-1, -1, -1)), device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        final_weights,
        torch.tensor(
            ((0.75, 0.25, 0.5), (0.6, 0.4, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            device="cuda",
            dtype=torch.float32,
        ),
        rtol=0,
        atol=0,
    )

    payload_rows.fill_(3)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(final_ids[2], torch.tensor((2, 0, 4), device="cuda", dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(
        final_weights[2],
        torch.tensor((0.9, 0.1, 0.5), device="cuda", dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(final_ids[3], torch.full((3,), -1, device="cuda", dtype=torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[3], torch.zeros(3, device="cuda"), rtol=0, atol=0)

    routed_weights.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(final_weights[:3, :2], torch.zeros((3, 2), device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[:3, 2], torch.full((3,), 0.5, device="cuda"), rtol=0, atol=0)
    torch.testing.assert_close(final_weights[3], torch.zeros(3, device="cuda"), rtol=0, atol=0)
