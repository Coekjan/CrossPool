"""Synthetic package and config inputs for Devkit registry tests."""

from __future__ import annotations

from pathlib import Path

from xpool.config import XpoolConfig


def create_observer_package(tmp_path: Path, name: str, files: dict[str, str]) -> str:
    """Create one importable synthetic observer package tree."""

    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    for relative_path, source in files.items():
        path = package / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")
    return name


def observer_enabled_config(observer_name: str, outdir: Path) -> XpoolConfig:
    """Return a config enabling one observer through its public environment source."""

    env_prefix = f"XPOOL_DEBUG_{observer_name.upper()}"
    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        },
        env={
            f"{env_prefix}_ENABLE": "1",
            f"{env_prefix}_OUTDIR": str(outdir),
        },
    )


def observers_disabled_config() -> XpoolConfig:
    """Return a config with every observer disabled."""

    return XpoolConfig.from_mapping(
        {
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "m", "path": "/models/m"}],
        }
    )
