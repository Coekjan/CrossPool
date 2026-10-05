"""Launch native tests with libraries from the active Python environment."""

from __future__ import annotations

import os
import sys
from importlib.metadata import distribution
from pathlib import Path
from typing import NoReturn


def native_library_paths() -> tuple[Path, ...]:
    """Return runtime library directories for native Python dependencies."""

    return (
        Path(str(distribution("torch").locate_file("torch/lib"))),
        Path(str(distribution("nvidia-nvshmem-cu13").locate_file("nvidia/nvshmem/lib"))),
    )


def launch(arguments: list[str]) -> NoReturn:
    """Replace this process with one native test executable."""

    if not arguments:
        raise RuntimeError("native CTest launcher requires an executable")
    environment = os.environ.copy()
    try:
        configure_device_visibility(environment)
    except (OSError, RuntimeError, ValueError) as error:
        report_infrastructure_failure(str(error))
        raise SystemExit(125) from error
    if environment.get("XPOOL_CTEST_CANONICAL") == "1":
        print(f"DEVICE ASSIGNMENT devices={environment['CUDA_VISIBLE_DEVICES'] or 'none'}", flush=True)
    paths = tuple(str(path) for path in native_library_paths())
    existing = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join((*paths, *((existing,) if existing else ())))
    os.execvpe(arguments[0], arguments, environment)


def configure_device_visibility(environment: dict[str, str]) -> None:
    """Project one CTest device resource into driver visibility."""

    if environment.get("XPOOL_CTEST_CANONICAL") != "1":
        return
    resource_count = int(environment.get("CTEST_RESOURCE_GROUP_COUNT", "0"))
    if resource_count == 0:
        environment["CUDA_VISIBLE_DEVICES"] = ""
        return
    if resource_count != 1 or environment.get("CTEST_RESOURCE_GROUP_0") != "devices":
        raise RuntimeError("native device test requires exactly one CTest devices resource group")
    resource = environment.get("CTEST_RESOURCE_GROUP_0_DEVICES")
    if resource is None:
        raise RuntimeError("native device test received no CTest device resource")
    fields = dict(field.split(":", maxsplit=1) for field in resource.split(","))
    identifier = fields.get("id")
    if identifier is None or fields.get("slots") != "1":
        raise RuntimeError(f"native device test received an invalid CTest device resource: {resource!r}")
    environment["CUDA_VISIBLE_DEVICES"] = decode_ctest_device_id(identifier)


def decode_ctest_device_id(identifier: str) -> str:
    """Decode one resource identifier written by ``CtestResourceSpec``."""

    if not identifier.startswith("device_"):
        raise ValueError(f"invalid CTest device resource ID: {identifier!r}")
    return "GPU-" + identifier.removeprefix("device_").replace("_", "-")


def report_infrastructure_failure(diagnostic: str) -> None:
    """Persist one launcher failure for canonical CTest classification."""

    print(diagnostic, file=sys.stderr)
    sentinel = os.environ.get("XPOOL_CTEST_INFRASTRUCTURE_SENTINEL")
    if sentinel is not None:
        Path(sentinel).write_text(diagnostic + "\n", encoding="utf-8")


if __name__ == "__main__":
    launch(sys.argv[1:])
