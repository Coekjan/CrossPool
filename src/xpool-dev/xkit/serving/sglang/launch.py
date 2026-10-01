"""Concrete SGLang launch geometry and runtime inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from xkit.serving.sglang.graph import SglangGraphMode
from xpool.config import XpoolConfig
from xpool.model import ModelId

__all__ = ["ServingLaunch", "SglangLaunchModel"]


@dataclass(frozen=True, slots=True)
class SglangLaunchModel:
    """One Instance's graph policy; runtime configuration owns its geometry."""

    model_id: ModelId
    graph_mode: SglangGraphMode

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, ModelId):
            raise ValueError("serving model_id must be a ModelId")
        if not isinstance(self.graph_mode, SglangGraphMode):
            raise ValueError("serving graph_mode must be a supported SGLang graph mode")


@dataclass(frozen=True, slots=True)
class ServingLaunch:
    """Ordered launches covering every effective runtime Instance exactly once."""

    config: XpoolConfig
    environment: Mapping[str, str]
    cwd: Path
    models: tuple[SglangLaunchModel, ...]

    def __post_init__(self) -> None:
        ids = tuple(model.model_id for model in self.models)
        configured = tuple(model.id for model in self.config.models)
        if not ids or len(ids) != len(set(ids)) or set(ids) != set(configured):
            raise ValueError("serving launches must cover all runtime Instances exactly once")
        for model in self.models:
            configured_model = self.config.model_by_id[model.model_id]
            if self.config.atn_tp_size_of(model.model_id) > 1 and configured_model.atn_dp_size > 1:
                raise ValueError("combined attention TP-by-DP is not supported")
