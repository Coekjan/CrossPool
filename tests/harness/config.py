from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

pytestmark = pytest.mark.usefixtures("reset_global_config")


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
    resolved_model_path = model_path or Path("/models/deepseek-ai/DeepSeek-V2-Lite-Chat")
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

[devices]
atn_cuda_devices = [{atn_devices}]
ffn_cuda_devices = [{ffn_devices}]

[[models]]
id = "deepseek-ai/DeepSeek-V2-Lite-Chat"
path = "{resolved_model_path}"
""".strip(),
        encoding="utf-8",
    )
    return path
