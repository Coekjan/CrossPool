"""Strict FFN checkpoint metadata discovery."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from safetensors import safe_open

from xpool import ffn

INDEX_FILENAME = "model.safetensors.index.json"
SINGLE_FILE_NAME = "model.safetensors"
MAIN_FFN_KEY_PATTERN = re.compile(r"^model\.layers\.(?P<layer_id>[0-9]+)\.mlp\.")


def parse_json_object(payload: bytes, *, source: Path) -> dict[str, object]:
    """Parse a JSON object while rejecting duplicate members at every level."""

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON member {key!r} in {source}")
            result[key] = value
        return result

    value = json.loads(payload, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return cast(dict[str, object], value)


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
            if not isinstance(weight_key, str) or not weight_key:
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


def validate_ffn_coverage(
    spec: ffn.FfnModelSpec,
    key_view: Mapping[str, Path],
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
