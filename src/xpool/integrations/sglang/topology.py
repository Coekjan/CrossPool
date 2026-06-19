"""SGLang-derived model topology and parallel-policy checks for xpool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.configs.model_config import ModelConfig as SglangModelConfig

from xpool.config import ConfigError, TopologyError


class AttentionKind(StrEnum):
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
        num_attention_heads: Total query-head count before attention-side TP.
        num_key_value_heads: Total KV-head count reported by SGLang.
        attention_kind: Derived attention topology family.
        physical_kv_lanes: Physical KV lanes used for xpool attention policy.
    """

    family: str
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    attention_kind: AttentionKind
    physical_kv_lanes: int


class ModelSpec(BaseModel):
    """Model metadata resolved from a local config.json through SGLang."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id whose config.json produced this metadata.")
    family: str = Field(description="SGLang/Hugging Face model family string used for diagnostics.")
    hidden_size: int = Field(ge=1, description="Hidden-state width consumed by each FFN shim call.")
    num_attention_heads: int = Field(ge=1, description="Total query-head count reported by SGLang.")
    num_key_value_heads: int = Field(ge=1, description="Total KV-head count reported by SGLang.")
    attention_kind: AttentionKind = Field(description="Attention topology derived from SGLang metadata.")
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
    """SGLang-derived role-local policy for one xpool model instance."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id this policy applies to.")
    attention_kind: AttentionKind = Field(description="Attention topology used by this policy.")
    sglang_tp_size: int = Field(ge=1, description="Expected SGLang attention tensor-parallel degree.")
    sglang_dp_size: int = Field(ge=1, description="Expected SGLang attention data-parallel degree.")
    attention_tp_size: int = Field(ge=1, description="xpool attention tensor-parallel degree.")
    attention_dp_size: int = Field(ge=1, description="xpool attention data-parallel degree.")
    ffn_tp_size: int = Field(ge=1, description="FFN tensor-parallel degree derived from FFN devices.")
    physical_kv_lanes: int = Field(ge=1, description="Physical KV lanes used to validate attention TP.")
    enable_dp_attention: bool = Field(description="Whether attention data parallelism is active.")


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
    return _parse_model_config(raw, model_id=model_id, config_path=config_path.resolve())


def _parse_model_config(raw: Mapping[str, object], *, model_id: str, config_path: Path) -> ModelSpec:
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

    sglang = _load_sglang_model_metadata(config_path, model_id=model_id)
    for label, value in (
        ("hidden_size", sglang.hidden_size),
        ("num_attention_heads", sglang.num_attention_heads),
        ("num_key_value_heads", sglang.num_key_value_heads),
        ("physical_kv_lanes", sglang.physical_kv_lanes),
    ):
        if value <= 0:
            raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    if sglang.attention_kind is AttentionKind.MLA and sglang.physical_kv_lanes != 1:
        raise ConfigError(f"{model_id}: SGLang-derived MLA physical_kv_lanes must be 1")
    if sglang.attention_kind is not AttentionKind.MLA and sglang.physical_kv_lanes != sglang.num_key_value_heads:
        raise ConfigError(f"{model_id}: SGLang-derived regular physical_kv_lanes must match num_key_value_heads")
    num_experts = _optional_int(raw, "num_experts")
    if num_experts is None:
        num_experts = _optional_int(raw, "n_routed_experts")
    return ModelSpec(
        model_id=model_id,
        family=sglang.family,
        hidden_size=sglang.hidden_size,
        num_attention_heads=sglang.num_attention_heads,
        num_key_value_heads=sglang.num_key_value_heads,
        attention_kind=sglang.attention_kind,
        physical_kv_lanes=sglang.physical_kv_lanes,
        dense_intermediate_size=_optional_int(raw, "intermediate_size"),
        moe_intermediate_size=_optional_int(raw, "moe_intermediate_size"),
        num_experts=num_experts,
        raw_config_path=config_path,
    )


def _load_sglang_model_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
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
    hidden_size = _positive_runtime_int(getattr(sglang_config, "hidden_size", None), "hidden_size", model_id)
    attention_heads = _positive_runtime_int(
        sglang_config.get_total_num_attention_heads(),
        "num_attention_heads",
        model_id,
    )
    kv_heads = _positive_runtime_int(sglang_config.get_total_num_kv_heads(), "num_key_value_heads", model_id)

    if sglang_config.attention_arch == AttentionArch.MLA:
        for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
            _positive_runtime_int(getattr(sglang_config, name, None), name, model_id)
        return SglangModelMetadata(
            family=family,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            attention_kind=AttentionKind.MLA,
            physical_kv_lanes=1,
        )

    match kv_heads:
        case 1:
            attention_kind = AttentionKind.MQA
        case _ if kv_heads == attention_heads:
            attention_kind = AttentionKind.MHA
        case _:
            attention_kind = AttentionKind.GQA
    return SglangModelMetadata(
        family=family,
        hidden_size=hidden_size,
        num_attention_heads=attention_heads,
        num_key_value_heads=kv_heads,
        attention_kind=attention_kind,
        physical_kv_lanes=kv_heads,
    )


def derive_parallel_policy(
    spec: ModelSpec,
    *,
    attention_device_count: int,
    ffn_tp_size: int,
) -> ParallelPolicy:
    """Derive attention and FFN parallelism policy for one SGLang model.

    Args:
        spec: SGLang-derived model metadata.
        attention_device_count: Number of attention CUDA devices in config.
        ffn_tp_size: FFN tensor-parallel degree derived from FFN device count.

    Returns:
        Derived role-local parallelism policy.

    Raises:
        TopologyError: If attention or FFN topology cannot be divided safely.
    """

    if attention_device_count <= 0:
        raise TopologyError("attention_device_count must be positive")
    if ffn_tp_size <= 0:
        raise TopologyError("ffn_tp_size must be positive")

    if spec.attention_kind is AttentionKind.MLA:
        attention_tp_size = 1
    else:
        attention_tp_size = min(spec.num_key_value_heads, attention_device_count)
        if spec.num_attention_heads % attention_tp_size != 0:
            raise TopologyError(
                f"{spec.model_id}: attention TP {attention_tp_size} does not divide query heads "
                f"{spec.num_attention_heads}"
            )
        if spec.physical_kv_lanes % attention_tp_size != 0:
            raise TopologyError(
                f"{spec.model_id}: attention TP {attention_tp_size} does not divide KV lanes {spec.physical_kv_lanes}"
            )

    if attention_device_count % attention_tp_size != 0:
        raise TopologyError(
            f"{spec.model_id}: attention TP {attention_tp_size} does not divide attention device count "
            f"{attention_device_count}"
        )
    for label, width in (
        ("dense intermediate size", spec.dense_intermediate_size),
        ("MoE intermediate size", spec.moe_intermediate_size),
    ):
        if width is not None and width % ffn_tp_size != 0:
            raise TopologyError(f"{spec.model_id}: FFN TP {ffn_tp_size} does not divide {label} {width}")
    attention_dp_size = attention_device_count // attention_tp_size
    if attention_dp_size > 1:
        raise TopologyError(
            f"{spec.model_id}: attention data parallelism is not supported by the first xpool shim ABI; "
            f"reduce attention CUDA devices to {attention_tp_size} or add DP-aware shim support"
        )
    return ParallelPolicy(
        model_id=spec.model_id,
        attention_kind=spec.attention_kind,
        sglang_tp_size=attention_device_count,
        sglang_dp_size=attention_dp_size,
        attention_tp_size=attention_tp_size,
        attention_dp_size=attention_dp_size,
        ffn_tp_size=ffn_tp_size,
        physical_kv_lanes=spec.physical_kv_lanes,
        enable_dp_attention=False,
    )


def _optional_int(raw: Mapping[str, object], key: str) -> int | None:
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
    return _coerce_config_int(value, key)


def _coerce_config_int(value: object, key: str) -> int:
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


def _positive_runtime_int(value: object, label: str, model_id: str) -> int:
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
    parsed = _coerce_config_int(value, label)
    if parsed <= 0:
        raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
    return parsed
