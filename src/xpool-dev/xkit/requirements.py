"""Shared external-resource declarations for test and benchmark programs."""

from __future__ import annotations

from typing import Annotated

from pydantic import ConfigDict, Field
from pydantic.dataclasses import dataclass

from xpool.model import ModelId

__all__ = ["ResourceRequirements"]


@dataclass(config=ConfigDict(extra="forbid", strict=True), frozen=True, slots=True)
class ResourceRequirements:
    """External resources required by one concrete tooling parameter row.

    Model IDs name local checkpoints, not externally served targets. The consuming
    tool owns configuration resolution, resource leasing and unavailable-resource
    policy. Local checkpoints require configuration. A deployment owns its MPS
    controller rather than declaring an externally running controller.
    """

    device_count: Annotated[int, Field(ge=0)]
    requires_config: bool
    model_ids: tuple[ModelId, ...]

    def __post_init__(self) -> None:
        if self.model_ids and not self.requires_config:
            raise ValueError("model-weight resource requirements must also require config")
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("resource requirement model IDs must be unique")
