"""Resource requirement resolution shared by pytest and test harnesses."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Never, TypeVar

import torch
from pydantic import ValidationError

from xpool.config import XpoolConfig
from xpool.mps import probe_mps_controller

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
    """Minimum visible CUDA device count required by one selected test."""

    min_devices: int = 1


class RequirementResolver:
    """Resolve test resources without depending on pytest."""

    def require_cuda(self, requirement: CudaRequirement) -> None:
        """Validate the visible CUDA device count."""

        if requirement.min_devices < 1:
            raise RequirementMisconfigured("requires_cuda min_devices must be at least 1")
        if not torch.cuda.is_available():
            raise RequirementUnavailable("CUDA is not available")
        device_count = torch.cuda.device_count()
        if device_count < requirement.min_devices:
            raise RequirementUnavailable(
                f"requires {requirement.min_devices} visible CUDA devices, found {device_count}"
            )

    def require_mps(self) -> None:
        """Require the CUDA MPS controller selected by the process environment."""

        result = probe_mps_controller()
        if not result.online:
            raise RequirementUnavailable(result.diagnostic)

    def require_config(self) -> ResolvedConfig:
        """Load the E2E config named by the exact ``XPOOL_CONFIG`` variable."""

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
            resolved = ResolvedConfig(path=path, config=XpoolConfig.from_file(path))
        except (OSError, ValueError, ValidationError) as exc:
            raise RequirementMisconfigured(f"invalid XPOOL_CONFIG file {path}: {exc}") from exc
        return resolved

    def require_model_weights(self, model_id: str) -> ResolvedModelWeights:
        """Resolve manifest-owned model weights below the configured model root."""

        if not model_id:
            raise RequirementMisconfigured("requires_model_weights model_id must be non-empty")
        resolved_config = self.require_config()
        model_base_uri = resolved_config.config.vendor.model_base_uri
        if model_base_uri is None:
            raise RequirementUnavailable("requires_model_weights needs vendor.model_base_uri in XPOOL_CONFIG")
        model_path = (model_base_uri / model_id).expanduser().resolve()
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
