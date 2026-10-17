"""Lossless Torch dtype representation at JSON boundaries."""

from typing import Annotated

import torch
from pydantic import PlainSerializer, PlainValidator, WithJsonSchema


def decode_dtype(value: object) -> torch.dtype:
    """Accept a Torch dtype or its canonical unqualified name, rejecting aliases."""

    if isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        dtype = getattr(torch, value, None)
        if isinstance(dtype, torch.dtype) and str(dtype) == f"torch.{value}":
            return dtype
    raise ValueError("payload dtype must be a torch.dtype or its canonical unqualified Torch name")


def encode_dtype(value: torch.dtype) -> str:
    """Encode a Torch dtype as its canonical unqualified JSON name."""

    return str(value).removeprefix("torch.")


type TorchDtype = Annotated[
    torch.dtype,
    PlainValidator(decode_dtype),
    PlainSerializer(encode_dtype, return_type=str, when_used="json"),
    WithJsonSchema({"type": "string"}),
]
