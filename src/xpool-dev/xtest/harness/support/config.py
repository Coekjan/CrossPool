"""Construct and install isolated CrossPool configurations for reusable fixtures."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest
import tomli_w

import xpool.config
from xpool.config import XpoolConfig
from xpool.model import ModelId
from xtest.harness.runner.pytest_plugin import resolved_config_key
from xtest.harness.runner.requirements import ResolvedConfig

TEST_MODEL_ID = ModelId("test/test-model")


def minimal_config(*, env: Mapping[str, str] | None = None) -> XpoolConfig:
    """Construct explicit minimal inputs, preserving registry defaults and env sources."""

    return XpoolConfig.from_mapping(
        {
            "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1]},
            "models": [{"id": str(TEST_MODEL_ID), "path": "/models/m"}],
        },
        env=env,
    )


@pytest.fixture
def reset_global_config(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Reset process-global configuration around one test."""

    monkeypatch.setattr(xpool.config, "global_config", None)
    yield


@pytest.fixture
def e2e_base_config(request: pytest.FixtureRequest) -> ResolvedConfig:
    """Return setup's base; the test must declare configuration requirements."""

    return request.node.stash[resolved_config_key]


def synthetic_config(
    *,
    model_id: ModelId = TEST_MODEL_ID,
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
            "models": [{"id": str(model_id)}],
        }
    )


def install_test_config(config: XpoolConfig) -> None:
    """Install an already validated config in isolated test process state."""

    if xpool.config.global_config is config:
        return
    if xpool.config.global_config is not None:
        raise RuntimeError("test attempted to replace an installed global config")
    xpool.config.global_config = config


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
    model_slo: tuple[int, int] | None = None,
) -> Path:
    """Write fixture TOML to a file or directory and return the resulting path.

    ``model_slo`` supplies TTFT/TBT millisecond targets; omission leaves the
    model-level override absent.
    """

    if path.suffix != ".toml":
        path.mkdir(parents=True, exist_ok=True)
        path = path / "xpool.toml"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
    resolved_model_path = model_path or Path("/models") / TEST_MODEL_ID.relative_path
    model: dict[str, object] = {"id": str(TEST_MODEL_ID), "path": str(resolved_model_path)}
    if model_slo is not None:
        model["slo"] = {"ttft_ms": model_slo[0], "tbt_ms": model_slo[1]}
    payload = {
        "daemon": {"host": daemon_host, "port": daemon_port},
        "scheduler": {
            "atn_concurrency": atn_concurrency,
            "ffn_concurrency": ffn_concurrency,
            "slo": {"ttft_ms": 1000, "tbt_ms": 50},
        },
        "atn": {"devices": list(atn_cuda_devices)},
        "ffn": {"devices": list(ffn_cuda_devices)},
        "models": [model],
    }
    path.write_text(tomli_w.dumps(payload), encoding="utf-8")
    return path
