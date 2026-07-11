"""Resource requirement resolution shared by pytest and test harnesses."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TypeVar

import torch
from pydantic import ValidationError

from xpool.config import MissingRequiredConfig, XpoolConfig

R = TypeVar("R")


class RequirementUnavailable(RuntimeError):
    """Raised when a valid test requirement is unavailable on this host."""


class RequirementMisconfigured(RuntimeError):
    """Raised when an explicitly supplied test requirement is invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """An E2E configuration and the environment path that supplied it."""

    path: Path
    config: XpoolConfig


@dataclass(frozen=True, slots=True)
class ResolvedModelWeights:
    """A model identifier and its validated local weight directory."""

    model_id: str
    path: Path


@dataclass(frozen=True, slots=True)
class CudaRequirement:
    """CUDA resources required by one selected test.

    Attributes:
        min_devices: Minimum number of visible CUDA devices.
        bf16: Whether the first required device must support BF16.
    """

    min_devices: int = 1
    bf16: bool = False


class RequirementResolver:
    """Resolve test resources without depending on pytest."""

    def __init__(self) -> None:
        self.cached_config_path: Path | None = None
        self.cached_config: ResolvedConfig | None = None

    def require_cuda(self, requirement: CudaRequirement) -> None:
        """Validate visible CUDA device count and optional BF16 support."""

        if requirement.min_devices < 1:
            raise RequirementMisconfigured("requires_cuda min_devices must be at least 1")
        if not torch.cuda.is_available():
            raise RequirementUnavailable("CUDA is not available")
        device_count = torch.cuda.device_count()
        if device_count < requirement.min_devices:
            raise RequirementUnavailable(
                f"requires {requirement.min_devices} visible CUDA devices, found {device_count}"
            )
        if requirement.bf16 and not torch.cuda.is_bf16_supported():
            raise RequirementUnavailable("CUDA device does not support BF16")

    def require_config(self) -> ResolvedConfig:
        """Load the E2E config named by the exact ``XPOOL_CONFIG`` variable."""

        configured_path = os.environ.get("XPOOL_CONFIG")
        if not configured_path:
            self.cached_config_path = None
            self.cached_config = None
            raise RequirementUnavailable("set XPOOL_CONFIG to an xpool TOML file")
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if self.cached_config_path == path and self.cached_config is not None:
            return self.cached_config
        self.cached_config_path = None
        self.cached_config = None
        if not path.is_file():
            raise RequirementMisconfigured(f"XPOOL_CONFIG does not name a readable file: {path}")
        try:
            resolved = ResolvedConfig(path=path, config=XpoolConfig.from_file(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise RequirementMisconfigured(f"invalid XPOOL_CONFIG file {path}: {exc}") from exc
        self.cached_config_path = path
        self.cached_config = resolved
        return resolved

    def require_model_weights(self, model_id: str) -> ResolvedModelWeights:
        """Resolve and validate one model's weight directory from XpoolConfig."""

        if not model_id:
            raise RequirementMisconfigured("requires_model_weights model_id must be non-empty")
        resolved_config = self.require_config()
        try:
            model_path = resolved_config.config.model_path_of(model_id).expanduser().resolve()
        except MissingRequiredConfig as exc:
            raise RequirementUnavailable(str(exc)) from exc
        if not model_path.is_dir():
            raise RequirementUnavailable(f"model {model_id!r} weight directory is unavailable at {model_path}")
        if not (model_path / "config.json").is_file():
            raise RequirementUnavailable(f"model {model_id!r} has no config.json at {model_path}")
        return ResolvedModelWeights(model_id=model_id, path=model_path)


class RequirementGuard:
    """Translate resolver outcomes into caller-provided skip or failure actions."""

    def __init__(
        self,
        *,
        strict: bool,
        skip: Callable[[str], Never],
        fail: Callable[[str], Never],
    ) -> None:
        self.strict = strict
        self.skip = skip
        self.fail = fail

    def run(self, operation: Callable[[], R]) -> R:
        """Run an operation and translate requirement errors to test outcomes."""

        try:
            return operation()
        except RequirementMisconfigured as exc:
            self.fail(str(exc))
        except RequirementUnavailable as exc:
            if self.strict:
                self.fail(str(exc))
            self.skip(str(exc))
