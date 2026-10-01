"""Shared external-resource declarations for test and benchmark programs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self, cast

from pydantic import JsonValue

from xpool.model import ModelId

__all__ = ["ResourceRequirements"]


@dataclass(frozen=True, slots=True)
class ResourceRequirements:
    """External resources required by one concrete tooling parameter row.

    Model IDs name local checkpoints, not externally served targets. The consuming
    tool owns configuration resolution, resource leasing and unavailable-resource
    policy. MPS requires CUDA; local checkpoints require configuration.
    """

    cuda_count: int
    requires_mps: bool
    requires_config: bool
    model_ids: tuple[ModelId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cuda_count, int) or isinstance(self.cuda_count, bool) or self.cuda_count < 0:
            raise ValueError("resource requirement cuda_count must be a nonnegative integer")
        if self.requires_mps and self.cuda_count == 0:
            raise ValueError("MPS resource requirements must also require CUDA")
        if self.model_ids and not self.requires_config:
            raise ValueError("model-weight resource requirements must also require config")
        if any(not isinstance(model_id, ModelId) for model_id in self.model_ids):
            raise ValueError("resource requirement model IDs must be ModelId values")
        if len(self.model_ids) != len(set(self.model_ids)):
            raise ValueError("resource requirement model IDs must be unique")

    @classmethod
    def from_raw(cls, raw: object) -> Self:
        """Strictly parse one requirements JSON object."""

        expected = {"cuda_count", "requires_mps", "requires_config", "model_ids"}
        if not isinstance(raw, dict) or set(raw) != expected:
            raise ValueError(f"resource requirements must contain exactly {sorted(expected)}")
        record = cast(dict[str, object], raw)
        cuda_count = record["cuda_count"]
        requires_mps = record["requires_mps"]
        requires_config = record["requires_config"]
        model_ids = record["model_ids"]
        if not isinstance(cuda_count, int) or isinstance(cuda_count, bool):
            raise ValueError("resource requirement cuda_count must be an integer")
        if not isinstance(requires_mps, bool):
            raise ValueError("resource requirement requires_mps must be a boolean")
        if not isinstance(requires_config, bool):
            raise ValueError("resource requirement requires_config must be a boolean")
        if not isinstance(model_ids, list):
            raise ValueError("resource requirement model_ids must be a string array")
        parsed_model_ids: list[ModelId] = []
        for model_id in model_ids:
            parsed_model_ids.append(ModelId.model_validate(model_id))
        return cls(cuda_count, requires_mps, requires_config, tuple(parsed_model_ids))

    def raw(self) -> dict[str, JsonValue]:
        """Project these requirements to their JSON representation."""

        return {
            "cuda_count": self.cuda_count,
            "requires_mps": self.requires_mps,
            "requires_config": self.requires_config,
            "model_ids": [str(model_id) for model_id in self.model_ids],
        }
