import pytest
import torch
from sglang.srt.layers.moe.topk import TritonKernelTopKOutput, routing

import xtest
from xtest.harness.sglang.reference.ffn_child import read_routing_output
from xtest.harness.support.sglang.runtime import published_sglang_config

pytestmark = pytest.mark.usefixtures(published_sglang_config.__name__)


@xtest.requirements(device_count=1)
def test_reference_reads_actual_expert_sorted_triton_routing_without_recomputing_it() -> None:
    row = torch.arange(16, dtype=torch.float32, device="cuda")
    logits = torch.stack((row, row.flip(0), torch.roll(row, 5)))
    carrier = TritonKernelTopKOutput(*routing(logits, 2))

    expert_ids, weights = read_routing_output(carrier)

    selected, expected_ids = torch.topk(logits.cpu(), 2, dim=-1)
    torch.testing.assert_close(expert_ids, expected_ids.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(weights, torch.softmax(selected, dim=-1), rtol=1e-6, atol=1e-7)
