"""Construct and install isolated CrossPool configurations for reusable fixtures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

import xpool.config
from tests.harness.runner.pytest_plugin import requirement_resolver_key
from tests.harness.runner.requirements import RequirementGuard
from xpool.config import XpoolConfig

TEST_MODEL_ID = "test-model"


@pytest.fixture
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset process-global configuration around one test."""

    monkeypatch.setattr(xpool.config, "global_config", None)
    yield


@pytest.fixture
def e2e_base_config(request: pytest.FixtureRequest) -> XpoolConfig:
    """Return the externally supplied config after requirement handling."""

    resolver = request.config.stash[requirement_resolver_key]
    guard = RequirementGuard(
        strict=request.config.getoption("--strict-requirements"),
        skip=lambda reason: pytest.skip(reason),
        fail=lambda reason: pytest.fail(reason, pytrace=False),
    )
    return guard.run(resolver.require_config).config


def synthetic_config(
    *,
    model_id: str = TEST_MODEL_ID,
    atn_cuda_devices: tuple[int, ...] = (0,),
    ffn_cuda_devices: tuple[int, ...] = (1,),
) -> XpoolConfig:
    """Return an in-memory config for tests where model identity is incidental."""

    return XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 9810},
            "scheduler": {
                "atn_concurrency": 1,
                "ffn_concurrency": 1,
                "ffn_policy": "fifo",
                "slo": {"ttft_ms": 1000, "tbt_ms": 50},
            },
            "vendor": {"model_base_uri": "/models"},
            "atn": {"devices": list(atn_cuda_devices)},
            "ffn": {"devices": list(ffn_cuda_devices)},
            "models": [{"id": model_id}],
        }
    )


def install_test_config(config: XpoolConfig) -> None:
    """Install an already validated config in isolated test process state."""

    if xpool.config.global_config is config:
        return
    if xpool.config.global_config is not None:
        raise RuntimeError("test attempted to replace an installed global config")
    xpool.config.global_config = config


def source_record(records: tuple[Mapping[str, object], ...], name: str) -> Mapping[str, object]:
    for record in records:
        if record["name"] == name:
            return record
    raise AssertionError(f"missing source record for {name}")


def write_minimal_config(
    path: Path,
    *,
    daemon_host: str = "127.0.0.1",
    daemon_port: int = 9810,
    atn_concurrency: int = 1,
    ffn_concurrency: int = 1,
    model_path: Path | None = None,
    atn_cuda_devices: tuple[int, ...] = (0, 1),
    ffn_cuda_devices: tuple[int, ...] = (2, 3, 4),
) -> Path:
    if path.suffix != ".toml":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "xpool.toml"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = model_path or Path("/models") / TEST_MODEL_ID
    atn_devices = ", ".join(str(device) for device in atn_cuda_devices)
    ffn_devices = ", ".join(str(device) for device in ffn_cuda_devices)
    path.write_text(
        f"""
[daemon]
host = "{daemon_host}"
port = {daemon_port}

[scheduler]
atn_concurrency = {atn_concurrency}
ffn_concurrency = {ffn_concurrency}
slo = {{ ttft_ms = 1000, tbt_ms = 50 }}

[atn]
devices = [{atn_devices}]

[ffn]
devices = [{ffn_devices}]

[[models]]
id = "{TEST_MODEL_ID}"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    return path
