from __future__ import annotations

import os
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import safetensors.torch
import torch

import xtest
from xkit.child import PythonChildProcess
from xpool.runtime.ffnagent import operators, weights
from xpool.runtime.ffnagent.models.gpt_oss import GptOssAdapter

ROW_CAPACITIES = tuple(1 << exponent for exponent in range(15))


def router_inputs(expert_count: int) -> tuple[torch.Tensor, weights.MoeRouterWeights]:
    """Use rounded operands that expose cuBLASLt BF16 Split-K reduction."""

    generator = torch.Generator().manual_seed(83)
    hidden = torch.randn(ROW_CAPACITIES[-1], 2880, generator=generator).bfloat16().cuda()
    router = weights.MoeRouterWeights(
        weight=(torch.randn(expert_count, 2880, generator=generator) * 0.1).bfloat16().cuda(),
        correction_bias=None,
        bias=torch.randn(expert_count, generator=generator).bfloat16().cuda(),
    )
    return hidden, router


def run_default_workspace_reference(connection: Connection, spec: tuple[int, Path]) -> None:
    """Keep the original Torch Linear workspace policy in an isolated process."""

    os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    expert_count, output_path = spec
    hidden, router = router_inputs(expert_count)
    results = {}
    for rows in ROW_CAPACITIES:
        logits = torch.nn.functional.linear(hidden[:rows], router.weight, router.bias)
        ids = torch.empty((rows, 4), dtype=torch.int32, device="cuda")
        probabilities = torch.empty((rows, 4), dtype=torch.float32, device="cuda")
        operators.compute_softmax_topk(logits=logits, routed_ids=ids, routed_weights=probabilities, renormalize=True)
        results[f"rows.{rows}.ids"] = ids.cpu()
        results[f"rows.{rows}.weights"] = probabilities.cpu()
    safetensors.torch.save_file(results, output_path)
    connection.send(True)


def run_workspace_free_router(connection: Connection, spec: tuple[int, Path]) -> None:
    """Replay the explicit per-Lane workspace under the FfnAgent cuBLAS policy."""

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":0:0"
    expert_count, reference_path = spec
    hidden, router = router_inputs(expert_count)
    reference = safetensors.torch.load_file(reference_path)
    for rows in ROW_CAPACITIES:
        extent = GptOssAdapter.router_workspace_bytes(
            payload_dtype=torch.bfloat16,
            payload_row_capacity=rows,
            hidden_size=2880,
            routed_expert_count=expert_count,
            routed_topk=4,
        )
        # A Router view can begin inside the shared MoE workspace allocation.
        allocation = torch.empty(extent + 17, dtype=torch.uint8, device="cuda")
        workspace = allocation[17:]
        ids = torch.empty((rows, 4), dtype=torch.int32, device="cuda")
        probabilities = torch.empty((rows, 4), dtype=torch.float32, device="cuda")

        def launch() -> None:
            GptOssAdapter.compute_routed_topk(
                hidden_states=hidden[:rows],
                router_weights=router,
                workspace=workspace,
                routed_ids=ids,
                routed_weights=probabilities,
                renormalize=True,
            )

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()
        allocated = torch.cuda.memory_allocated()
        graph.replay()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == allocated
        torch.testing.assert_close(ids.cpu(), reference[f"rows.{rows}.ids"], rtol=0, atol=0)
        torch.testing.assert_close(probabilities.cpu(), reference[f"rows.{rows}.weights"], rtol=1e-6, atol=1e-7)
    connection.send(True)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("expert_count", (32, 128))
@xtest.requirements(device_count=1)
def test_gpt_oss_router_preserves_original_rounding_with_explicit_workspace(tmp_path: Path, expert_count: int) -> None:
    """Match original Linear for both model geometries through 16K-row replay."""

    reference_path = tmp_path / "reference.safetensors"
    children = (
        PythonChildProcess(
            "default-workspace-router",
            run_default_workspace_reference,
            (expert_count, reference_path),
            log_path=tmp_path / "reference.log",
        ),
        PythonChildProcess(
            "workspace-free-router",
            run_workspace_free_router,
            (expert_count, reference_path),
            log_path=tmp_path / "actual.log",
        ),
    )
    try:
        for child in children:
            child.start()
            assert child.receive(bool, timeout_seconds=120)
            child.wait(timeout_seconds=30)
    finally:
        PythonChildProcess.terminate_all(tuple(child for child in children if child.process.is_alive()))
        for child in children:
            child.close()
