from __future__ import annotations

import torch

import xtest
from xpool import ffn, quantization
from xpool.quantization import TensorLayout
from xpool.runtime.ffnagent import device_memory, weights


@xtest.requirements(device_count=1)
def test_canonical_dense_reservation_covers_inactive_segment_splits() -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    layers: list[weights.DenseFfnWeights] = []
    try:
        for _ in range(24):
            layers.append(
                weights.DenseFfnWeights.allocate(
                    i_proj=weights.ProjectionLayout(
                        quantization=quantization.Unquantized(),
                        gate_up_order=ffn.GateUpOrder.CONCATENATED,
                        values=TensorLayout((4864, 896), torch.bfloat16, 0),
                        scales=None,
                        bias=None,
                    ),
                    o_proj=weights.ProjectionLayout(
                        quantization=quantization.Unquantized(),
                        gate_up_order=None,
                        values=TensorLayout((896, 2432), torch.bfloat16, 0),
                        scales=None,
                        bias=None,
                    ),
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
            )
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved() - before
        budget = sum(
            device_memory.allocator_reservation_bytes((layer.i_proj.values.nbytes, layer.o_proj.values.nbytes))
            for layer in layers
        )

        assert torch.cuda.memory_stats()["inactive_split_bytes.all.current"] > 0
        assert reserved <= budget
    finally:
        layers.clear()
        torch.cuda.empty_cache()
