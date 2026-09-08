"""Engine-neutral FFN model semantics and architecture compilation."""

from __future__ import annotations

import hashlib
import math
import pathlib
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping

import torch

from xpool import ffn
from xpool.runtime.ffnagent import checkpoint
from xpool.runtime.ffnagent.weights import MoeRouterWeights
from xpool.utils.discovery import discover_concrete_subclasses

MODELS_PACKAGE = "xpool.runtime.ffnagent.models"
MAIN_FFN_KEY_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>\d+)\.mlp(?:\.|$)")


class FfnModelAdapter(ABC):
    """Compile one strict model-family configuration into FFN semantics."""

    architecture_name: str

    @classmethod
    @abstractmethod
    def compile(
        cls,
        *,
        model_id: str,
        model_config: Mapping[str, object],
        model_config_digest: str,
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
        model_config_digest = hashlib.sha256(config_bytes).hexdigest()
        model_config = checkpoint.parse_json_object(config_bytes, source=config_path)
        architecture_name = require_single_architecture(model_config)

        adapter = adapters_by_architecture().get(architecture_name)
        if adapter is None:
            raise ValueError(f"unsupported FFN architecture {architecture_name!r}")
        spec = adapter.compile(
            model_id=model_id,
            model_config=model_config,
            model_config_digest=model_config_digest,
        )
        if (
            spec.model_id != model_id
            or spec.architecture_name != architecture_name
            or spec.model_config_digest != model_config_digest
        ):
            raise ValueError("FFN Architecture Adapter returned inconsistent model identity")

        key_view = checkpoint.read_checkpoint_key_view(model_path)
        validate_checkpoint_coverage(
            spec,
            key_view,
            allow_trailing_ffn_layers=architecture_name == "Glm4MoeLiteForCausalLM",
        )
        return spec
    except Exception as error:
        raise RuntimeError(f"failed to compile FFN model {model_id!r} at {model_path}: {error}") from error


def require_single_architecture(model_config: Mapping[str, object]) -> str:
    """Return the one strict architecture name in a parsed model config."""

    architectures = model_config.get("architectures")
    if (
        not isinstance(architectures, list)
        or len(architectures) != 1
        or not isinstance(architectures[0], str)
        or not architectures[0]
    ):
        raise ValueError("architectures must be a one-element array containing a nonempty string")
    return architectures[0]


def require_family_profile(
    model_config: Mapping[str, object],
    *,
    architecture_name: str,
    model_type: str,
    dtype_field: str,
) -> None:
    """Validate common architecture, model type, activation, and BF16 fields."""

    if require_single_architecture(model_config) != architecture_name:
        raise ValueError(f"architecture must equal {architecture_name!r}")
    if require_string(model_config, "model_type") != model_type:
        raise ValueError(f"model_type must equal {model_type!r}")
    if require_string(model_config, "hidden_act") != "silu":
        raise ValueError("hidden_act must equal 'silu'")
    if require_string(model_config, dtype_field) != "bfloat16":
        raise ValueError(f"{dtype_field} must equal 'bfloat16'")


def require_integer(
    model_config: Mapping[str, object],
    field_name: str,
    *,
    minimum: int,
) -> int:
    """Read one JSON integer field with an inclusive lower bound."""

    value = model_config.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{field_name} must be a JSON integer greater than or equal to {minimum}")
    return value


def require_boolean(model_config: Mapping[str, object], field_name: str) -> bool:
    """Read one required JSON Boolean field."""

    value = model_config.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a JSON Boolean")
    return value


def require_string(model_config: Mapping[str, object], field_name: str) -> str:
    """Read one required nonempty JSON string field."""

    value = model_config.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a nonempty JSON string")
    return value


def require_positive_number(model_config: Mapping[str, object], field_name: str) -> float:
    """Read one finite positive JSON number field."""

    value = model_config.get(field_name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{field_name} must be a finite positive JSON number")
    return float(value)


def gated_checkpoint_keys(prefix: str) -> ffn.GatedFfnCheckpointKeys:
    """Construct the canonical gated projection triplet below one FFN prefix."""

    return ffn.GatedFfnCheckpointKeys(
        gate_weight_key=f"{prefix}.gate_proj.weight",
        up_weight_key=f"{prefix}.up_proj.weight",
        down_weight_key=f"{prefix}.down_proj.weight",
    )


def validate_checkpoint_coverage(
    spec: ffn.FfnModelSpec,
    key_view: Mapping[str, pathlib.Path],
    *,
    allow_trailing_ffn_layers: bool,
) -> None:
    """Require exact family-owned FFN keys for every main decoder layer."""

    expected_by_layer = {layer.layer_id: set(ffn.checkpoint_keys_for_layer(layer)) for layer in spec.layers}
    all_checkpoint_keys = set(key_view)
    for layer_id, expected_keys in expected_by_layer.items():
        prefix = f"model.layers.{layer_id}.mlp."
        actual_keys = {key for key in all_checkpoint_keys if key.startswith(prefix)}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"FFN checkpoint namespace for layer {layer_id} differs: missing={missing}, extra={extra}")

    if allow_trailing_ffn_layers:
        return
    trailing_keys = []
    for key in all_checkpoint_keys:
        match = MAIN_FFN_KEY_PATTERN.match(key)
        if match is not None and int(match.group("layer_id")) not in expected_by_layer:
            trailing_keys.append(key)
    if trailing_keys:
        raise ValueError(f"checkpoint contains FFN keys outside main decoder layers: {sorted(trailing_keys)}")
