from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ISOLATED_NATIVE_CASE_TIMEOUT_S = 120.0
ISOLATED_NATIVE_CASE_SCRIPT = r"""
import importlib.util
import json
import sys
import traceback

from xpool.cext import ensure_xpool_ops_loaded

module_path = sys.argv[1]
case_name = sys.argv[2]
args = json.loads(sys.argv[3])

spec = importlib.util.spec_from_file_location("xpool_isolated_native_case", module_path)
module = importlib.util.module_from_spec(spec)
try:
    assert spec.loader is not None
    spec.loader.exec_module(module)
    ensure_xpool_ops_loaded()
    getattr(module, case_name)(*args)
except BaseException:
    traceback.print_exc()
    sys.exit(1)
"""


def run_isolated_native_case(
    module_path: str | Path,
    case_name: str,
    *args: object,
    timeout_s: float = ISOLATED_NATIVE_CASE_TIMEOUT_S,
) -> None:
    """Run one native test case in a fresh Python process.

    Args:
        module_path: Test module containing the named case function.
        case_name: Module-level case function to invoke.
        *args: JSON-serializable positional arguments for the case.
        timeout_s: Maximum subprocess runtime in seconds.
    """

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            ISOLATED_NATIVE_CASE_SCRIPT,
            str(module_path),
            case_name,
            json.dumps(args),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"isolated native case failed: {case_name}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")


def expected_loopback_rotation(hidden_states: torch.Tensor) -> torch.Tensor:
    """Return the pairwise rotation implemented by both loopback sites."""

    x_values = hidden_states.float()[..., 0::2]
    y_values = hidden_states.float()[..., 1::2]
    output = torch.empty_like(hidden_states.float())
    output[..., 0::2] = (x_values - y_values) / (2.0**0.5)
    output[..., 1::2] = (x_values + y_values) / (2.0**0.5)
    return output.to(hidden_states.dtype)
