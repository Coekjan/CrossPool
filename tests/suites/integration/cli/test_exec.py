from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import tomli_w

from xpool.utils.mps import MpsEndpoint
from xtest.harness.support.config import synthetic_config


@pytest.mark.parametrize("plugins", [None, "__none__"])
def test_exec_preserves_identity_arguments_and_prepares_only_mps_environment(
    plugins: str | None, tmp_path: Path
) -> None:
    executable = tmp_path / "inspect-environment"
    executable.write_text(
        f"#!{sys.executable}\n"
        + """
import json, os, sys
from cuda.bindings import driver
assert os.getpid() == int(os.environ["EXPECTED_PID"])
assert os.getpgrp() == int(os.environ["EXPECTED_PGID"])
print(json.dumps({"argv": sys.argv[1:], "visibility": os.environ["CUDA_VISIBLE_DEVICES"],
                  "pipe": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
                  "log": os.environ["CUDA_MPS_LOG_DIRECTORY"],
                  "plugins": os.environ.get("SGLANG_PLUGINS"),
                  "driver_status": int(driver.cuCtxGetCurrent()[0]),
                  "engine_imported": "sglang" in sys.modules}))
""",
        encoding="utf-8",
    )
    executable.chmod(0o700)
    config = synthetic_config(atn_devices=(0, 1), ffn_devices=(2, 3))
    config_path = tmp_path / "xpool.toml"
    config_path.write_text(tomli_w.dumps(config.to_config_mapping()), encoding="utf-8")
    program = """
import os, sys, uuid
import xpool.utils.device
from xpool.cli.main import main
os.environ["EXPECTED_PID"] = str(os.getpid())
os.environ["EXPECTED_PGID"] = str(os.getpgrp())
xpool.utils.device.query_uuids_mapping = lambda: {
    index: f"GPU-{uuid.UUID(int=index + 1)}" for index in range(4)
}
raise SystemExit(main(["exec", "--", sys.argv[1], "--help", "--config", "engine.toml", "--opaque=a b", "--"]))
"""
    environment = dict(os.environ)
    environment.update(
        XPOOL_CONFIG=str(config_path),
        CUDA_VISIBLE_DEVICES="3,1,2,0",
        CUDA_MPS_PIPE_DIRECTORY="/external/pipe",
    )
    if plugins is None:
        environment.pop("SGLANG_PLUGINS", None)
    else:
        environment["SGLANG_PLUGINS"] = plugins
    result = subprocess.run(
        [sys.executable, "-c", program, str(executable)],
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    data = json.loads(result.stdout)
    visibility = tuple(f"GPU-00000000-0000-0000-0000-{index:012x}" for index in (4, 2, 3, 1))
    endpoint = MpsEndpoint(visibility[:2])
    assert data["argv"] == ["--help", "--config", "engine.toml", "--opaque=a b", "--"]
    assert data["visibility"] == ",".join(visibility)
    assert data["pipe"] == str(endpoint.pipe_directory)
    assert data["log"] == str(endpoint.log_directory)
    assert data["plugins"] == plugins
    assert data["driver_status"] == 3
    assert not data["engine_imported"]
