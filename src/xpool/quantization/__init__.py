"""Model-independent weight encodings and source-configuration resolution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated

from pydantic import Field
from transformers.quantizers.quantizers_utils import should_convert_module

from xpool.quantization.base import QuantizationScheme, TensorLayout, TensorSlice, WeightLayout, WeightSlices
from xpool.quantization.mxfp4 import Mxfp4
from xpool.quantization.unquantized import Unquantized

__all__ = [
    "Mxfp4",
    "QuantizationScheme",
    "QuantizationSpec",
    "TensorLayout",
    "TensorSlice",
    "Unquantized",
    "WeightLayout",
    "WeightSlices",
    "resolve",
]

type QuantizationSpec = Annotated[Unquantized | Mxfp4, Field(discriminator="kind")]


def resolve(config: Mapping[str, object] | None, *, module_name: str) -> QuantizationSpec:
    """Resolve a source projection's encoding, honoring Transformers exclusions.

    Args:
        config: Original quantization metadata, absent for ordinary floating models.
        module_name: Source-qualified projection path, independent of Model ID.

    Returns:
        An immutable encoding declaration; excluded components use ``Unquantized``.

    Raises:
        ValueError: If the format, module name or exclusion declaration is invalid.
    """

    if not isinstance(module_name, str) or not module_name:
        raise ValueError("quantization requires a nonempty source module name")
    if config is None:
        return Unquantized()
    method = config.get("quant_method")
    if method != "mxfp4":
        raise ValueError(f"unsupported weight quantization method: {method!r}")

    patterns = None
    declared = config.get("modules_to_not_convert")
    if declared is not None:
        if not isinstance(declared, list):
            raise ValueError("modules_to_not_convert must be a list of strings")
        patterns = []
        for pattern in declared:
            if not isinstance(pattern, str):
                raise ValueError("modules_to_not_convert must contain only strings")
            patterns.append(pattern)
    return Mxfp4() if should_convert_module(module_name, patterns) else Unquantized()
