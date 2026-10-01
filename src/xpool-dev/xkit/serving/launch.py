"""Serving endpoints and complete, source-aware runtime snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import tomli_w

from xkit.serving.cluster import XpoolClusterLaunch
from xpool.config import CONFIG_REGISTRY, ConfigSource, XpoolConfig, XpoolDaemonConfig, get_nested
from xpool.model import ModelId

__all__ = ["ServingEndpoint", "snapshot_cluster_launch"]


@dataclass(frozen=True, slots=True)
class ServingEndpoint:
    """Public endpoint of a completely started Instance, in launch order."""

    model_id: ModelId
    base_url: str


def snapshot_cluster_launch(
    config: XpoolConfig,
    *,
    workdir: Path,
    daemon_port: int,
    environment: Mapping[str, str],
    cwd: Path,
) -> XpoolClusterLaunch:
    """Bind a reserved daemon port and freeze all effective runtime settings.

    Registered ordinary environment overrides are removed from children;
    env-only debug settings retain their effective values. Other environment
    inputs and the invocation working directory preserve their meaning.
    Callers own original configuration provenance. Reloading ``xpool.toml``
    establishes new CONFIG provenance. This function
    neither installs global configuration nor acquires process/GPU resources.
    """

    if isinstance(daemon_port, bool) or not 1 <= daemon_port <= 65_535:
        raise ValueError("serving daemon port must be in 1..65535")
    workdir.mkdir(parents=True, exist_ok=True)
    config_path = (workdir / "xpool.toml").resolve()
    daemon = XpoolDaemonConfig(host=config.daemon.host, port=daemon_port)
    payload = config.to_config_mapping()
    payload["daemon"] = daemon.model_dump(mode="json")
    with config_path.open("x", encoding="utf-8") as output:
        output.write(tomli_w.dumps(payload))

    effective = config.model_dump()
    child_environment = dict(environment)
    for setting in CONFIG_REGISTRY:
        if setting.env_var is None:
            continue
        child_environment.pop(setting.env_var, None)
        if setting.path is None or ConfigSource.CONFIG in setting.allowed_sources:
            continue
        found, value = get_nested(effective, setting.path)
        if found and value is not None:
            child_environment[setting.env_var] = ("1" if value else "0") if isinstance(value, bool) else str(value)
    child_environment["XPOOL_CONFIG"] = str(config_path)
    frozen = XpoolConfig.from_file(config_path, env=child_environment)
    return XpoolClusterLaunch(frozen, config_path, MappingProxyType(child_environment), cwd.resolve())
