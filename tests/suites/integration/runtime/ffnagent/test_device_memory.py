from __future__ import annotations

import torch

import xtest
from xpool.runtime.ffnagent import device_memory, weights


@xtest.requirements(device_count=1)
def test_canonical_dense_reservation_covers_inactive_segment_splits() -> None:
    device_memory.ensure_supported_cuda_allocator()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    before = torch.cuda.memory_reserved()
    layers: list[weights.DenseFfnWeights] = []
    try:
        for _ in range(24):
            layers.append(
                weights.DenseFfnWeights.allocate(
                    hidden_size=896,
                    local_intermediate_size=2432,
                    payload_dtype=torch.bfloat16,
                    device=torch.device("cuda", torch.cuda.current_device()),
                )
            )
        torch.cuda.synchronize()
        reserved = torch.cuda.memory_reserved() - before
        budget = sum(
            device_memory.allocator_reservation_bytes((layer.gate_up_weight.nbytes, layer.down_weight.nbytes))
            for layer in layers
        )

        assert torch.cuda.memory_stats()["inactive_split_bytes.all.current"] > 0
        assert reserved <= budget
    finally:
        layers.clear()
        torch.cuda.empty_cache()
