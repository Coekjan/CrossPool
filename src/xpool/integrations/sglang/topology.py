"""SGLang-derived model topology and parallel-policy checks for xpool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.configs.model_config import AttentionArch as SglangAtnArch
from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig

from xpool.config import ConfigError, TopologyError

__all__ = [
    "AtnKind",
    "ModelSpec",
    "ParallelPolicy",
    "SglangModelMetadata",
    "derive_parallel_policy",
    "load_model_spec",
]


class AtnKind(StrEnum):
    """Attention topology kind derived from SGLang model metadata.

    Attributes:
        MLA: Multi-head latent attention, represented as one physical KV lane.
        GQA: Grouped-query attention with fewer KV heads than query heads.
        MHA: Multi-head attention with equal query and KV head counts.
        MQA: Multi-query attention with one KV head.
    """

    MLA = "mla"
    GQA = "gqa"
    MHA = "mha"
    MQA = "mqa"


@dataclass(frozen=True, slots=True)
class SglangModelMetadata:
    """Model shape facts derived through SGLang's config resolution path.

    Attributes:
        family: Hugging Face or SGLang model type used for diagnostics.
        hidden_size: Hidden-state width consumed by each FFN shim.
        num_atn_heads: Total query-head count before attention-side TP.
        num_key_value_heads: Total KV-head count reported by SGLang.
        atn_kind: Derived attention topology family.
        physical_kv_lanes: Physical KV lanes used for xpool attention policy.
    """

    family: str
    hidden_size: int
    num_atn_heads: int
    num_key_value_heads: int
    atn_kind: AtnKind
    physical_kv_lanes: int


class ModelSpec(BaseModel):
    """Model metadata resolved from a local config.json through SGLang."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id whose config.json produced this metadata.")
    family: str = Field(description="SGLang/Hugging Face model family string used for diagnostics.")
    hidden_size: int = Field(ge=1, description="Hidden-state width consumed by each FFN shim call.")
    num_atn_heads: int = Field(ge=1, description="Total query-head count reported by SGLang.")
    num_key_value_heads: int = Field(ge=1, description="Total KV-head count reported by SGLang.")
    atn_kind: AtnKind = Field(description="Attention topology derived from SGLang metadata.")
    physical_kv_lanes: int = Field(ge=1, description="Physical KV lanes used for attention policy derivation.")
    dense_intermediate_size: int | None = Field(
        default=None,
        description="Dense FFN intermediate width from config.json, when present.",
    )
    moe_intermediate_size: int | None = Field(
        default=None,
        description="MoE expert intermediate width from config.json, when present.",
    )
    num_experts: int | None = Field(default=None, description="MoE routed expert count from config.json, if present.")
    raw_config_path: Path = Field(description="Resolved config.json path used to derive this metadata.")


class ParallelPolicy(BaseModel):
    """SGLang-derived parallelism policy and topology audit record.

    The SGLang launcher consumes ``sglang_*`` and ``enable_dp_atn`` directly.
    The remaining fields keep the xpool-side topology derivation visible for
    diagnostics and future transport/FFN executor validation, instead of
    forcing every caller to recompute those facts from raw device lists.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id this topology record applies to.")
    atn_kind: AtnKind = Field(description="Model attention topology used to derive SGLang placement.")
    sglang_tp_size: int = Field(ge=1, description="Expected SGLang attention tensor-parallel degree.")
    sglang_dp_size: int = Field(ge=1, description="Expected SGLang attention data-parallel degree.")
    atn_tp_size: int = Field(ge=1, description="xpool attention tensor-parallel degree retained for topology audits.")
    atn_dp_size: int = Field(ge=1, description="xpool attention data-parallel degree retained for topology audits.")
    ffn_tp_size: int = Field(
        ge=1,
        description="FFN tensor-parallel degree retained for future FFN executor validation.",
    )
    physical_kv_lanes: int = Field(
        ge=1,
        description="Physical KV lanes retained to explain attention TP validation decisions.",
    )
    enable_dp_atn: bool = Field(description="Whether attention data parallelism is active.")


def load_model_spec(model_path: str | Path, *, model_id: str) -> ModelSpec:
    """Load and derive SGLang model metadata for one configured model.

    Args:
        model_path: Absolute model directory or direct ``config.json`` path.
        model_id: Configured model id used in diagnostics.

    Returns:
        Model metadata resolved through SGLang plus FFN shape hints from
        ``config.json``.

    Raises:
        OSError: If ``config.json`` cannot be opened.
        ConfigError: If model metadata cannot be derived safely.
    """

    path = Path(model_path).expanduser()
    config_path = path if path.is_file() else path / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        raw = cast(Mapping[str, object], json.load(config_file))
    return parse_model_config(raw, model_id=model_id, config_path=config_path.resolve())


def parse_model_config(raw: Mapping[str, object], *, model_id: str, config_path: Path) -> ModelSpec:
    """Derive xpool model metadata from raw config.json content.

    Args:
        raw: Parsed JSON mapping from a model ``config.json``.
        model_id: Configured model id used in diagnostics.
        config_path: Resolved path to the JSON file that produced ``raw``.

    Returns:
        xpool model metadata and FFN width hints.

    Raises:
        ConfigError: If SGLang metadata or raw integer fields are invalid.
    """

    sglang = load_sglang_model_metadata(config_path, model_id=model_id)
    for label, value in (
        ("hidden_size", sglang.hidden_size),
        ("num_attention_heads", sglang.num_atn_heads),
        ("num_key_value_heads", sglang.num_key_value_heads),
        ("physical_kv_lanes", sglang.physical_kv_lanes),
    ):
        if value <= 0:
            raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    if sglang.atn_kind is AtnKind.MLA and sglang.physical_kv_lanes != 1:
        raise ConfigError(f"{model_id}: SGLang-derived MLA physical_kv_lanes must be 1")
    if sglang.atn_kind is not AtnKind.MLA and sglang.physical_kv_lanes != sglang.num_key_value_heads:
        raise ConfigError(f"{model_id}: SGLang-derived regular physical_kv_lanes must match num_key_value_heads")
    num_experts = optional_int(raw, "num_experts")
    if num_experts is None:
        num_experts = optional_int(raw, "n_routed_experts")
    return ModelSpec(
        model_id=model_id,
        family=sglang.family,
        hidden_size=sglang.hidden_size,
        num_atn_heads=sglang.num_atn_heads,
        num_key_value_heads=sglang.num_key_value_heads,
        atn_kind=sglang.atn_kind,
        physical_kv_lanes=sglang.physical_kv_lanes,
        dense_intermediate_size=optional_int(raw, "intermediate_size"),
        moe_intermediate_size=optional_int(raw, "moe_intermediate_size"),
        num_experts=num_experts,
        raw_config_path=config_path,
    )


def load_sglang_model_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
    """Resolve model shape through SGLang's own model-config loader.

    Args:
        config_path: Direct ``config.json`` path or model directory path.
        model_id: Configured model id used in diagnostics.

    Returns:
        SGLang-derived model metadata normalized for xpool policy checks.

    Raises:
        ConfigError: If SGLang cannot load the model config or returns invalid
            metadata needed by xpool.
    """

    model_path = config_path.parent if config_path.name == "config.json" else config_path
    try:
        sglang_config = SglangModelConfig(str(model_path), trust_remote_code=True)
    except Exception as exc:
        raise ConfigError(f"{model_id}: SGLang failed to resolve model config at {model_path}") from exc

    family = str(
        getattr(sglang_config.hf_text_config, "model_type", None)
        or getattr(sglang_config.hf_config, "model_type", None)
        or model_id
    )
    hidden_size = positive_runtime_int(getattr(sglang_config, "hidden_size", None), "hidden_size", model_id)
    atn_heads = positive_runtime_int(
        sglang_config.get_total_num_attention_heads(),
        "num_attention_heads",
        model_id,
    )
    kv_heads = positive_runtime_int(sglang_config.get_total_num_kv_heads(), "num_key_value_heads", model_id)

    if sglang_config.attention_arch == SglangAtnArch.MLA:
        for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            positive_runtime_int(getattr(sglang_config, name, None), name, model_id)
        return SglangModelMetadata(
            family=family,
            hidden_size=hidden_size,
            num_atn_heads=atn_heads,
            num_key_value_heads=kv_heads,
            atn_kind=AtnKind.MLA,
            physical_kv_lanes=1,
        )

    match kv_heads:
        case 1:
            atn_kind = AtnKind.MQA
        case _ if kv_heads == atn_heads:
            atn_kind = AtnKind.MHA
        case _:
            atn_kind = AtnKind.GQA
    return SglangModelMetadata(
        family=family,
        hidden_size=hidden_size,
        num_atn_heads=atn_heads,
        num_key_value_heads=kv_heads,
        atn_kind=atn_kind,
        physical_kv_lanes=kv_heads,
    )


def derive_parallel_policy(
    spec: ModelSpec,
    *,
    atn_device_count: int,
    ffn_tp_size: int,
) -> ParallelPolicy:
    """Derive attention and FFN parallelism policy for one SGLang model.

    Args:
        spec: SGLang-derived model metadata.
        atn_device_count: Number of attention CUDA devices in config.
        ffn_tp_size: FFN tensor-parallel degree derived from FFN device count.

    Returns:
        Derived role-local parallelism policy.

    Raises:
        TopologyError: If attention or FFN topology cannot be divided safely.
    """

    if atn_device_count <= 0:
        raise TopologyError("attention device count must be positive")
    if ffn_tp_size <= 0:
        raise TopologyError("ffn_tp_size must be positive")

    if spec.atn_kind is AtnKind.MLA:
        atn_tp_size = 1
    else:
        atn_tp_size = min(spec.num_key_value_heads, atn_device_count)

    for label, width in (
        ("query heads", spec.num_atn_heads),
        ("KV lanes", spec.physical_kv_lanes),
        ("attention device count", atn_device_count),
    ):
        if width % atn_tp_size != 0:
            raise TopologyError(f"{spec.model_id}: attention TP {atn_tp_size} does not divide {label} {width}")
    for label, width in (
        ("dense intermediate size", spec.dense_intermediate_size),
        ("MoE intermediate size", spec.moe_intermediate_size),
    ):
        if width is not None and width % ffn_tp_size != 0:
            raise TopologyError(f"{spec.model_id}: FFN TP {ffn_tp_size} does not divide {label} {width}")
    atn_dp_size = atn_device_count // atn_tp_size
    if atn_dp_size > 1:
        raise TopologyError(
            f"{spec.model_id}: attention DP size {atn_dp_size} is unsupported until FFN reduce-scatter is implemented"
        )
    return ParallelPolicy(
        model_id=spec.model_id,
        atn_kind=spec.atn_kind,
        sglang_tp_size=atn_device_count,
        sglang_dp_size=atn_dp_size,
        atn_tp_size=atn_tp_size,
        atn_dp_size=atn_dp_size,
        ffn_tp_size=ffn_tp_size,
        physical_kv_lanes=spec.physical_kv_lanes,
        enable_dp_atn=False,
    )


def optional_int(raw: Mapping[str, object], key: str) -> int | None:
    """Return an optional integer field from raw model config data.

    Args:
        raw: Raw parsed model config mapping.
        key: Field name to read.

    Returns:
        Parsed integer value, or ``None`` when the field is missing or null.

    Raises:
        ConfigError: If the field exists but is not an integer.
    """

    if key not in raw:
        return None
    value = raw[key]
    if value is None:
        return None
    return coerce_config_int(value, key)


def coerce_config_int(value: object, key: str) -> int:
    """Coerce a model config field to an integer.

    Args:
        value: Raw field value.
        key: Field name used in diagnostics.

    Returns:
        Parsed integer.

    Raises:
        ConfigError: If the value is bool or cannot be parsed as an integer.
    """

    if isinstance(value, bool):
        raise ConfigError(f"model config.json field {key} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError as exc:
            raise ConfigError(f"model config.json field {key} must be an integer") from exc
    raise ConfigError(f"model config.json field {key} must be an integer")


def positive_runtime_int(value: object, label: str, model_id: str) -> int:
    """Return a positive integer reported by SGLang runtime config.

    Args:
        value: Raw value from SGLang ``ModelConfig``.
        label: Diagnostic field name.
        model_id: Configured model id used in diagnostics.

    Returns:
        Positive integer value.

    Raises:
        ConfigError: If the value is missing, non-integer, or not positive.
    """

    if value is None:
        raise ConfigError(f"{model_id}: SGLang-derived {label} is missing")
    parsed = coerce_config_int(value, label)
    if parsed <= 0:
        raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    return parsed
