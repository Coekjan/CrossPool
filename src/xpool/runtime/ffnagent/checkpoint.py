"""Strict FFN checkpoint metadata discovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import JsonValue
from safetensors import safe_open

from xpool import ffn

INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"
MAIN_FFN_KEY_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>[0-9]+)\.mlp\.")
DEFAULT_FFN_NAMESPACE = "model.layers.{layer_id}.mlp"


def parse_json_object(payload: bytes, *, source: Path) -> dict[str, JsonValue]:
    """Parse a JSON object while rejecting duplicate members at every level."""

    def unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
        result: dict[str, JsonValue] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r} in {source}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return cast(dict[str, JsonValue], value)


def read_checkpoint_key_view(model_path: Path) -> dict[str, Path]:
    """Read the authoritative checkpoint key-to-file mapping without tensors."""

    index_path = model_path / INDEX_FILENAME
    if index_path.exists():
        index = parse_json_object(index_path.read_bytes(), source=index_path)
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"{index_path} must contain one nonempty object-valued weight_map")

        key_view: dict[str, Path] = {}
        for weight_key, shard_name in weight_map.items():
            if not weight_key:
                raise ValueError(f"{index_path} contains an invalid weight key")
            if not isinstance(shard_name, str) or not shard_name:
                raise ValueError(f"{index_path} contains an invalid shard name for {weight_key!r}")
            if "/" in shard_name or "\\" in shard_name or not shard_name.endswith(".safetensors"):
                raise ValueError(f"{index_path} contains unsafe shard name {shard_name!r}")
            shard_path = model_path / shard_name
            if shard_path.is_symlink() or not shard_path.is_file():
                raise ValueError(f"checkpoint shard {shard_path} is not an existing regular file")
            key_view[weight_key] = shard_path
        return key_view

    canonical_path = model_path / SINGLE_FILE_NAME
    safetensor_files = tuple(
        entry
        for entry in model_path.iterdir()
        if entry.suffix == ".safetensors" and (entry.is_file() or entry.is_symlink())
    )
    if len(safetensor_files) != 1 or safetensor_files[0] != canonical_path:
        raise ValueError(f"{model_path} must contain exactly one unindexed canonical {SINGLE_FILE_NAME} checkpoint")
    if canonical_path.is_symlink() or not canonical_path.is_file():
        raise ValueError(f"checkpoint {canonical_path} is not an existing regular file")

    with safe_open(str(canonical_path), framework="pt", device="cpu") as checkpoint:
        keys = tuple(checkpoint.keys())
    if not keys or any(not isinstance(key, str) or not key for key in keys):
        raise ValueError(f"checkpoint {canonical_path} must contain only nonempty tensor keys")
    return {key: canonical_path for key in keys}


def layer_namespace(spec: ffn.FfnModelSpec, layer_id: int) -> str:
    """Return the layer-relative checkpoint namespace with its identity applied."""

    template = spec.checkpoint_namespace or DEFAULT_FFN_NAMESPACE
    return template.replace("{layer_id}", str(layer_id))


def is_unary_dense_layer(layer: ffn.FfnLayerSpec) -> bool:
    """Return whether one layer declares unary fc1/fc2 projections."""

    return isinstance(layer, ffn.DenseFfnSpec) and isinstance(layer.projections, ffn.UnaryProjectionSpec)


def rebase_projection_keys(
    projection: ffn.ProjectionSpec,
    *,
    old_namespace: str,
    new_namespace: str,
) -> ffn.ProjectionSpec:
    """Rewrite one projection's source keys from one layer namespace to another."""

    def rebase(key: str | None) -> str | None:
        if key is None:
            return None
        prefix = f"{old_namespace}."
        if key.startswith(prefix):
            return f"{new_namespace}.{key[len(prefix) :]}"
        return key

    return projection.model_copy(
        update={
            "values_key": rebase(projection.values_key),
            "scales_key": rebase(projection.scales_key),
            "bias_key": rebase(projection.bias_key),
        }
    )


def rebase_checkpoint_namespace(
    spec: ffn.FfnModelSpec,
    *,
    old_template: str,
    new_template: str,
) -> ffn.FfnModelSpec:
    """Rebind Dense source keys to an equivalent layer namespace variant."""

    def rebase_layer(layer: ffn.FfnLayerSpec) -> ffn.FfnLayerSpec:
        if not isinstance(layer, ffn.DenseFfnSpec):
            return layer
        old_namespace = old_template.replace("{layer_id}", str(layer.layer_id))
        new_namespace = new_template.replace("{layer_id}", str(layer.layer_id))
        projections = layer.projections
        if isinstance(projections, ffn.GatedProjectionSpec):
            projections = projections.model_copy(
                update={
                    "gate": rebase_projection_keys(
                        projections.gate, old_namespace=old_namespace, new_namespace=new_namespace
                    ),
                    "up": rebase_projection_keys(
                        projections.up, old_namespace=old_namespace, new_namespace=new_namespace
                    ),
                    "down": rebase_projection_keys(
                        projections.down, old_namespace=old_namespace, new_namespace=new_namespace
                    ),
                }
            )
        elif isinstance(projections, ffn.UnaryProjectionSpec):
            projections = projections.model_copy(
                update={
                    "fc1": rebase_projection_keys(
                        projections.fc1, old_namespace=old_namespace, new_namespace=new_namespace
                    ),
                    "fc2": rebase_projection_keys(
                        projections.fc2, old_namespace=old_namespace, new_namespace=new_namespace
                    ),
                }
            )
        else:
            return layer
        return layer.model_copy(update={"projections": projections})

    return spec.model_copy(
        update={
            "layers": tuple(rebase_layer(layer) for layer in spec.layers),
            "checkpoint_namespace": new_template,
        }
    )


def resolve_checkpoint_namespace(
    spec: ffn.FfnModelSpec,
    key_view: Mapping[str, Path],
) -> ffn.FfnModelSpec:
    """Select the declared namespace or its optional top-level ``model.`` variant.

    Some OPT conversions store keys as ``decoder.layers...`` while others store
    ``model.decoder.layers...``. SGLang normalizes the former at load time; the
    FFN Spec must select the variant that exists before coverage and reads.
    """

    template = spec.checkpoint_namespace
    if template is None:
        return spec
    candidates = [template]
    if template.startswith("model."):
        candidates.append(template[len("model.") :])
    for candidate in candidates:
        candidate_spec = (
            spec
            if candidate == template
            else rebase_checkpoint_namespace(spec, old_template=template, new_template=candidate)
        )
        if all(key in key_view for layer in candidate_spec.layers for key in layer.tensor_keys()):
            return candidate_spec
    return spec


def trailing_ffn_key_pattern(spec: ffn.FfnModelSpec) -> re.Pattern[str]:
    """Derive the trailing-FFN key pattern from the declared namespace."""

    template = spec.checkpoint_namespace or DEFAULT_FFN_NAMESPACE
    escaped = re.escape(template).replace(re.escape("{layer_id}"), r"(?P<layer_id>[0-9]+)")
    if any(is_unary_dense_layer(layer) for layer in spec.layers):
        return re.compile(rf"^{escaped}\.(?:fc1|fc2)\.")
    return re.compile(rf"^{escaped}\.")


def validate_ffn_coverage(
    spec: ffn.FfnModelSpec,
    key_view: Mapping[str, Path],
    *,
    allow_trailing_ffn_layers: bool,
) -> None:
    """Require exact family-owned FFN keys for every main decoder layer."""

    expected_by_layer = {layer.layer_id: set(layer.tensor_keys()) for layer in spec.layers}
    all_checkpoint_keys = set(key_view)
    for layer in spec.layers:
        layer_id = layer.layer_id
        expected_keys = expected_by_layer[layer_id]
        prefix = f"{layer_namespace(spec, layer_id)}."
        if is_unary_dense_layer(layer):
            actual_keys = {
                key for key in all_checkpoint_keys if key.startswith(f"{prefix}fc1.") or key.startswith(f"{prefix}fc2.")
            }
        else:
            actual_keys = {key for key in all_checkpoint_keys if key.startswith(prefix)}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise ValueError(f"FFN checkpoint namespace for layer {layer_id} differs: missing={missing}, extra={extra}")

    if allow_trailing_ffn_layers:
        return
    pattern = trailing_ffn_key_pattern(spec)
    trailing_keys = []
    for key in all_checkpoint_keys:
        match = pattern.match(key)
        if match is not None and int(match.group("layer_id")) not in expected_by_layer:
            trailing_keys.append(key)
    if trailing_keys:
        raise ValueError(f"checkpoint contains FFN keys outside main decoder layers: {sorted(trailing_keys)}")
