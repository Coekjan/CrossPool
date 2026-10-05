"""Resource requirement resolution shared by pytest and test harnesses."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TypeVar

import torch
from pydantic import ValidationError

from xkit.config import resolve_model_weights
from xpool.config import XpoolConfig
from xpool.model import ModelId

R = TypeVar("R")


class RequirementUnavailable(RuntimeError):
    """Raised when a valid test requirement is unavailable on this host."""


class RequirementMisconfigured(RuntimeError):
    """Raised when an explicitly supplied test requirement is invalid."""


@dataclass(frozen=True, slots=True)
class ResolvedConfig:
    """A resolved development base and its source TOML path."""

    path: Path
    config: XpoolConfig


class RequirementResolver:
    """Resolve test resources without depending on pytest."""

    def require_devices(self, device_count: int) -> None:
        """Validate the visible device count through the supported driver."""

        if device_count < 1:
            raise RequirementMisconfigured("required device count must be at least 1")
        if not torch.cuda.is_available():
            raise RequirementUnavailable("no supported devices are available")
        visible_count = torch.cuda.device_count()
        if visible_count < device_count:
            raise RequirementUnavailable(f"requires {device_count} visible devices, found {visible_count}")

    def require_config(self) -> ResolvedConfig:
        """Reload XPOOL_CONFIG with registered process-environment inputs."""

        configured_path = os.environ.get("XPOOL_CONFIG")
        if not configured_path:
            raise RequirementUnavailable("set XPOOL_CONFIG to an xpool TOML file")
        path = Path(configured_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        if not path.is_file():
            raise RequirementMisconfigured(f"XPOOL_CONFIG does not name a readable file: {path}")
        try:
            resolved = ResolvedConfig(path=path, config=XpoolConfig.from_file(path, env=os.environ))
        except (OSError, ValueError, ValidationError) as exc:
            raise RequirementMisconfigured(f"invalid XPOOL_CONFIG file {path}: {exc}") from exc
        return resolved

    def require_model_weights(self, config: XpoolConfig, model_id: ModelId) -> Path:
        """Require a checkpoint using the caller's resolved base configuration."""

        try:
            return resolve_model_weights(config, model_id)
        except ValueError as error:
            raise RequirementUnavailable(str(error)) from error


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
