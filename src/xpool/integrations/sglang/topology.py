"""SGLang-derived model topology and parallel-policy checks for xpool."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from pydantic import BaseModel, ConfigDict, Field
from sglang.srt.configs import model_config
from sglang.srt.server_args import ServerArgs

from xpool.config import ConfigError, TopologyError

__all__ = [
    "AtnKind",
    "ModelSpec",
    "ParallelPolicy",
    "SglangModelMetadata",
]


class AtnKind(StrEnum):
    """Attention topology kind derived from SGLang model metadata.

    Attributes:
        MLA: Multi-head latent attention.
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
    """

    family: str
    hidden_size: int
    num_atn_heads: int
    num_key_value_heads: int
    atn_kind: AtnKind

    @classmethod
    def load(cls, config_path: Path, *, model_id: str) -> SglangModelMetadata:
        """Resolve model shape through SGLang's own model-config loader.

        Args:
            config_path: Direct ``config.json`` path or model directory path.
            model_id: Configured model id used in diagnostics.

        Returns:
            SGLang-derived model metadata normalized for xpool policy checks.

        Raises:
            ConfigError: If SGLang cannot load the model config or returns invalid metadata.
        """

        model_path = config_path.parent if config_path.name == "config.json" else config_path
        try:
            sglang_config = model_config.ModelConfig(str(model_path), trust_remote_code=True)
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

        if sglang_config.attention_arch == model_config.AttentionArch.MLA:
            for name in ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "v_head_dim"):
                positive_runtime_int(getattr(sglang_config, name, None), name, model_id)
            return cls(
                family=family,
                hidden_size=hidden_size,
                num_atn_heads=atn_heads,
                num_key_value_heads=kv_heads,
                atn_kind=AtnKind.MLA,
            )

        match kv_heads:
            case 1:
                atn_kind = AtnKind.MQA
            case _ if kv_heads == atn_heads:
                atn_kind = AtnKind.MHA
            case _:
                atn_kind = AtnKind.GQA
        return cls(
            family=family,
            hidden_size=hidden_size,
            num_atn_heads=atn_heads,
            num_key_value_heads=kv_heads,
            atn_kind=atn_kind,
        )


class ModelSpec(BaseModel):
    """Model metadata resolved from a local config.json through SGLang."""

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id whose config.json produced this metadata.")
    family: str = Field(description="SGLang/Hugging Face model family string used for diagnostics.")
    hidden_size: int = Field(ge=1, description="Hidden-state width consumed by each FFN shim call.")
    num_atn_heads: int = Field(ge=1, description="Total query-head count reported by SGLang.")
    num_key_value_heads: int = Field(ge=1, description="Total KV-head count reported by SGLang.")
    atn_kind: AtnKind = Field(description="Attention topology derived from SGLang metadata.")
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

    @classmethod
    def load(cls, model_path: str | Path, *, model_id: str) -> ModelSpec:
        """Load and derive metadata for one configured model.

        Args:
            model_path: Absolute model directory or direct ``config.json`` path.
            model_id: Configured model id used in diagnostics.

        Returns:
            Model metadata resolved through SGLang plus FFN shape hints.
        """

        path = Path(model_path).expanduser()
        config_path = path if path.is_file() else path / "config.json"
        with config_path.open("r", encoding="utf-8") as config_file:
            raw = cast(Mapping[str, object], json.load(config_file))
        return cls.from_raw(raw, model_id=model_id, config_path=config_path.resolve())

    @classmethod
    def from_raw(cls, raw: Mapping[str, object], *, model_id: str, config_path: Path) -> ModelSpec:
        """Derive model metadata from parsed ``config.json`` content.

        Args:
            raw: Parsed JSON mapping.
            model_id: Configured model id used in diagnostics.
            config_path: Resolved path that produced ``raw``.

        Returns:
            xpool model metadata and FFN width hints.

        Raises:
            ConfigError: If SGLang metadata or raw integer fields are invalid.
        """

        sglang = SglangModelMetadata.load(config_path, model_id=model_id)
        for label, value in (
            ("hidden_size", sglang.hidden_size),
            ("num_attention_heads", sglang.num_atn_heads),
            ("num_key_value_heads", sglang.num_key_value_heads),
        ):
            if value <= 0:
                raise ConfigError(f"{model_id}: SGLang-derived {label} must be positive")
        num_experts = optional_int(raw, "num_experts")
        if num_experts is None:
            num_experts = optional_int(raw, "n_routed_experts")
        return cls(
            model_id=model_id,
            family=sglang.family,
            hidden_size=sglang.hidden_size,
            num_atn_heads=sglang.num_atn_heads,
            num_key_value_heads=sglang.num_key_value_heads,
            atn_kind=sglang.atn_kind,
            dense_intermediate_size=optional_int(raw, "intermediate_size"),
            moe_intermediate_size=optional_int(raw, "moe_intermediate_size"),
            num_experts=num_experts,
            raw_config_path=config_path,
        )


class ParallelPolicy(BaseModel):
    """SGLang-derived parallelism policy and topology audit record.

    The fields preserve SGLang's resolved TP-by-DP topology for rank binding,
    daemon registration, and diagnostics without inventing FFN weight-sharding
    policy from the number of physical FfnAgents.
    """

    model_config = ConfigDict(extra="forbid")

    model_id: str = Field(description="Configured model id this topology record applies to.")
    atn_kind: AtnKind = Field(description="Model attention topology used to derive SGLang placement.")
    sglang_tp_size: int = Field(ge=1, description="Expected SGLang attention tensor-parallel degree.")
    sglang_dp_size: int = Field(ge=1, description="Expected SGLang attention data-parallel degree.")
    atn_tp_size: int = Field(ge=1, description="xpool attention tensor-parallel degree retained for topology audits.")
    atn_dp_size: int = Field(ge=1, description="xpool attention data-parallel degree retained for topology audits.")
    enable_dp_attention: bool = Field(description="Whether SGLang attention data parallelism is active.")

    @classmethod
    def from_server_args(
        cls,
        spec: ModelSpec,
        server_args: ServerArgs,
        *,
        atnagent_count: int,
    ) -> ParallelPolicy:
        """Validate and retain one resolved SGLang attention topology.

        Args:
            spec: SGLang-derived model metadata.
            server_args: Fully resolved pinned-SGLang launch arguments.
            atnagent_count: Number of configured physical AtnAgents.

        Returns:
            Validated TP-by-DP policy with TP-fastest rank geometry.

        Raises:
            TopologyError: If resolved SGLang topology cannot map bijectively to
                the configured AtnAgents or model head geometry.
        """

        if not isinstance(atnagent_count, int) or isinstance(atnagent_count, bool) or atnagent_count <= 0:
            raise TopologyError("atnagent_count must be a positive integer")
        tp_size = positive_runtime_int(server_args.tp_size, "ServerArgs.tp_size", spec.model_id)
        dp_size = positive_runtime_int(server_args.dp_size, "ServerArgs.dp_size", spec.model_id)
        cp_size = positive_runtime_int(server_args.attn_cp_size, "ServerArgs.attn_cp_size", spec.model_id)
        if tp_size != atnagent_count:
            raise TopologyError(
                f"{spec.model_id}: SGLang TP size {tp_size} must equal configured AtnAgent count {atnagent_count}"
            )
        if cp_size != 1:
            raise TopologyError(f"{spec.model_id}: attention context parallel size must be one, got {cp_size}")
        if tp_size % dp_size != 0:
            raise TopologyError(f"{spec.model_id}: SGLang TP size {tp_size} is not divisible by DP size {dp_size}")
        atn_tp_size = tp_size // dp_size
        atn_dp_size = dp_size
        if atn_tp_size > 1 and atn_dp_size > 1:
            raise TopologyError(
                f"{spec.model_id}: combined attention TP-by-DP is not supported; "
                f"resolved TP={atn_tp_size}, DP={atn_dp_size}"
            )
        expected_dp_attention = atn_dp_size > 1
        if server_args.enable_dp_attention is not expected_dp_attention:
            raise TopologyError(
                f"{spec.model_id}: SGLang enable_dp_attention must be {expected_dp_attention} "
                f"for attention DP size {atn_dp_size}"
            )
        for label, width in (("query heads", spec.num_atn_heads), ("KV heads", spec.num_key_value_heads)):
            if width % atn_tp_size != 0:
                raise TopologyError(f"{spec.model_id}: attention TP {atn_tp_size} does not divide {label} {width}")
        return cls(
            model_id=spec.model_id,
            atn_kind=spec.atn_kind,
            sglang_tp_size=tp_size,
            sglang_dp_size=atn_dp_size,
            atn_tp_size=atn_tp_size,
            atn_dp_size=atn_dp_size,
            enable_dp_attention=expected_dp_attention,
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
