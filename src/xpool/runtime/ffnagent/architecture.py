"""Engine-neutral FFN model semantics and architecture compilation."""

from __future__ import annotations

import math
import pathlib
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar, cast

import torch

from xpool import ffn
from xpool.runtime.ffnagent import checkpoint
from xpool.runtime.ffnagent.weights import MoeRouterWeights
from xpool.utils.discovery import discover_concrete_subclasses

MODELS_PACKAGE = "xpool.runtime.ffnagent.models"
CONFIG_FIELD_MISSING = object()
T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class FfnSourceConfig:
    """Strict access boundary for one parsed model ``config.json`` object."""

    values: Mapping[str, object]

    @property
    def architecture_name(self) -> str:
        """Return the sole nonempty architecture name declared by the source config."""

        architectures = self.get("architectures", list)
        if len(architectures) != 1 or not isinstance(architectures[0], str) or not architectures[0]:
            raise ValueError("architectures must be a one-element array containing a nonempty string")
        return cast(str, architectures[0])

    def get(
        self,
        field_name: str,
        expected_type: type[T],
        *,
        gt: int | float | None = None,
        ge: int | float | None = None,
        lt: int | float | None = None,
        le: int | float | None = None,
    ) -> T:
        """Read one required source field with strict type and numeric constraints."""

        expected_name = (
            "integer" if expected_type is int else "number" if expected_type is float else expected_type.__name__
        )
        value = self.values.get(field_name, CONFIG_FIELD_MISSING)
        if value is CONFIG_FIELD_MISSING:
            raise ValueError(f"{field_name} must be a JSON {expected_name}")
        if expected_type is int:
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be a JSON integer")
        elif expected_type is float:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite JSON number")
            value = float(value)
        elif not isinstance(value, expected_type):
            raise ValueError(f"{field_name} must be a JSON {expected_name}")

        if any(bound is not None for bound in (gt, ge, lt, le)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{field_name} must be numeric to apply bounds")
            if gt is not None and value <= gt:
                raise ValueError(f"{field_name} must be greater than {gt}")
            if ge is not None and value < ge:
                raise ValueError(f"{field_name} must be greater than or equal to {ge}")
            if lt is not None and value >= lt:
                raise ValueError(f"{field_name} must be less than {lt}")
            if le is not None and value > le:
                raise ValueError(f"{field_name} must be less than or equal to {le}")
        return cast(T, value)

    def optional(self, field_name: str, expected_type: type[T], *, allow_none: bool = False) -> T | None:
        """Read an optional source field, preserving missing and null semantics."""

        if field_name not in self.values:
            return None
        if self.values[field_name] is None:
            if allow_none:
                return None
            expected_name = (
                "integer" if expected_type is int else "number" if expected_type is float else expected_type.__name__
            )
            raise ValueError(f"{field_name} must be a JSON {expected_name}")
        return self.get(field_name, expected_type)

    def validate_family_profile(self, *, model_type: str, dtype_field: str) -> None:
        """Validate common model type, activation, and source dtype fields."""

        if self.get("model_type", str) != model_type:
            raise ValueError(f"model_type must equal {model_type!r}")
        if self.get("hidden_act", str) != "silu":
            raise ValueError("hidden_act must equal 'silu'")
        if self.get(dtype_field, str) != "bfloat16":
            raise ValueError(f"{dtype_field} must equal 'bfloat16'")


class FfnModelAdapter(ABC):
    """Compile one strict model-family configuration into FFN semantics."""

    architecture_name: str

    @classmethod
    @abstractmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: FfnSourceConfig,
    ) -> ffn.FfnModelSpec:
        """Compile a parsed model configuration without filesystem access."""


class MoeFfnModelAdapter(FfnModelAdapter):
    """Add caller-owned Router operations for one admitted MoE family."""

    @staticmethod
    @abstractmethod
    def router_weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype:
        """Resolve retained Router precision independently of Expert weights."""

    @staticmethod
    @abstractmethod
    def router_workspace_bytes(
        *,
        payload_dtype: torch.dtype,
        payload_row_capacity: int,
        hidden_size: int,
        routed_expert_count: int,
        routed_topk: int,
    ) -> int:
        """Return the exact caller-owned Router scratch extent."""

    @staticmethod
    @abstractmethod
    def compute_routed_topk(
        *,
        hidden_states: torch.Tensor,
        router_weights: MoeRouterWeights,
        workspace: torch.Tensor,
        routed_ids: torch.Tensor,
        routed_weights: torch.Tensor,
        renormalize: bool,
    ) -> None:
        """Write routed-only ids and weights without allocating Tensor storage."""


def adapters_by_architecture() -> dict[str, type[FfnModelAdapter]]:
    """Discover the one concrete FFN Adapter for each admitted architecture."""

    result: dict[str, type[FfnModelAdapter]] = {}
    for adapter in discover_concrete_subclasses(MODELS_PACKAGE, FfnModelAdapter):
        architecture_name = adapter.architecture_name
        if not isinstance(architecture_name, str) or not architecture_name:
            raise ValueError(
                f"FFN Model Adapter {adapter.__module__}.{adapter.__name__} must declare one nonempty architecture name"
            )
        previous = result.setdefault(architecture_name, adapter)
        if previous is not adapter:
            raise ValueError(
                f"duplicate FFN architecture name {architecture_name!r}: "
                f"{previous.__module__}.{previous.__name__} and {adapter.__module__}.{adapter.__name__}"
            )
    return result


def adapter_for(spec: ffn.FfnModelSpec) -> type[FfnModelAdapter]:
    """Return the fail-closed Adapter selected by one immutable Model Spec."""

    adapter = adapters_by_architecture().get(spec.architecture_name)
    if adapter is None:
        raise ValueError(f"unsupported FFN architecture {spec.architecture_name!r}")
    return adapter


def load(*, model_id: str, model_path: pathlib.Path) -> ffn.FfnModelSpec:
    """Load and compile one model's intrinsic FFN semantics.

    Args:
        model_id: Nonempty CrossPool model identity.
        model_path: Local checkpoint directory resolved through CrossPool config.

    Returns:
        Complete engine-neutral FFN Model Spec.

    Raises:
        RuntimeError: If configuration, architecture discovery, compilation,
            checkpoint authority, or exact FFN namespace coverage fails.
    """

    try:
        config_path = model_path / "config.json"
        config_bytes = config_path.read_bytes()
        model_config = FfnSourceConfig(checkpoint.parse_json_object(config_bytes, source=config_path))
        architecture_name = model_config.architecture_name

        adapter = adapters_by_architecture().get(architecture_name)
        if adapter is None:
            raise ValueError(f"unsupported FFN architecture {architecture_name!r}")
        spec = adapter.compile(
            model_id=model_id,
            model_config=model_config,
        )
        if spec.model_id != model_id or spec.architecture_name != architecture_name:
            raise ValueError("FFN Architecture Adapter returned inconsistent model identity")

        key_view = checkpoint.read_checkpoint_key_view(model_path)
        checkpoint.validate_ffn_coverage(
            spec,
            key_view,
            allow_trailing_ffn_layers=architecture_name == "Glm4MoeLiteForCausalLM",
        )
        return spec
    except Exception as error:
        raise RuntimeError(f"failed to compile FFN model {model_id!r} at {model_path}: {error}") from error


def gated_checkpoint_keys(prefix: str) -> ffn.GatedFfnCheckpointKeys:
    """Construct the canonical gated projection triplet below one FFN prefix."""

    return ffn.GatedFfnCheckpointKeys(
        gate_weight_key=f"{prefix}.gate_proj.weight",
        up_weight_key=f"{prefix}.up_proj.weight",
        down_weight_key=f"{prefix}.down_proj.weight",
    )
