import pytest
import torch

from xpool.quantization import Mxfp4, QuantizationSpec, TensorLayout, Unquantized
from xpool.runtime.ffnagent.weights import ProjectionLayout


@pytest.mark.parametrize("scheme,expected", ((Unquantized(), (4096, 128)), (Mxfp4(), (1024, 64, 128))))
def test_projection_layout_counts_physical_storages_in_allocation_order(
    scheme: QuantizationSpec,
    expected: tuple[int, ...],
) -> None:
    encoded = scheme.weight_layout(logical_shape=(32, 64), dtype=torch.bfloat16)
    layout = ProjectionLayout(values=encoded.values, scales=encoded.scales, bias=TensorLayout((32,), torch.float32, 0))

    assert layout.allocation_bytes() == expected
    assert ProjectionLayout(values=encoded.values, scales=encoded.scales, bias=None).allocation_bytes() == expected[:-1]
