"""Common SGLang FFN shim scalar contracts."""

from __future__ import annotations

import pytest

from xpool.fabric import FfnLayerKind
from xpool.integrations.sglang.shim import FfnShimModule


@pytest.mark.parametrize("layer_id", [-1, True, 1.5])
def test_ffn_shim_rejects_invalid_layer_id(layer_id: object) -> None:
    with pytest.raises(ValueError, match="layer_id"):
        FfnShimModule(
            layer_id=layer_id,  # ty: ignore[invalid-argument-type]
            hidden_size=4,
            layer_kind=FfnLayerKind.DENSE,
        )


@pytest.mark.parametrize("hidden_size", [0, -1, True, 1.5])
def test_ffn_shim_rejects_invalid_hidden_size(hidden_size: object) -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        FfnShimModule(
            layer_id=0,
            hidden_size=hidden_size,  # ty: ignore[invalid-argument-type]
            layer_kind=FfnLayerKind.DENSE,
        )
