"""Canonical CTest resource specification and execution ownership."""

from __future__ import annotations

import json
import os
import sys
import sysconfig
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Self

from tests.harness.runner.gpu import GpuPool
from tests.harness.runner.supervisor import SupervisedTaskScope, TaskCompletionKind

CTEST_SUITE_TIMEOUT_SECONDS = 1800.0


@dataclass(frozen=True, slots=True)
class CtestResourceSpec:
    """One CTest resource file and its reversible GPU-ID mapping."""

    path: Path
    id_to_uuid: Mapping[str, str]

    @classmethod
    def write(cls, path: Path, gpu_uuids: Sequence[str]) -> Self:
        """Atomically write one single-slot CTest resource per physical GPU."""

        identifiers = {ctest_gpu_id(uuid): uuid for uuid in gpu_uuids}
        if len(identifiers) != len(gpu_uuids):
            raise ValueError("physical GPU UUIDs do not have unique CTest resource IDs")
        payload = {
            "version": {"major": 1, "minor": 0},
            "local": [{"gpus": [{"id": identifier, "slots": 1} for identifier in identifiers]}],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        temporary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary_path.replace(path)
        return cls(path, MappingProxyType(identifiers))


@dataclass(frozen=True, slots=True)
class CtestRunResult:
    """Canonical native-suite outcome and retained evidence paths."""

    result_code: int
    log_path: Path
    junit_path: Path


class CtestSuite:
    """Run the currently installed native CTest manifest with borrowed GPUs."""

    def run(self, *, gpu_pool: GpuPool, run_directory: Path) -> CtestRunResult:
        """Execute CTest with resource scheduling and infrastructure detection."""

        build_directory = current_build_directory()
        manifest = build_directory / "CTestTestfile.cmake"
        if not manifest.is_file():
            raise RuntimeError(f"current native build has no CTest manifest: {build_directory}")
        run_directory.mkdir(parents=True, exist_ok=False)
        resource_spec = CtestResourceSpec.write(run_directory / "resources.json", gpu_pool.uuids)
        log_path = run_directory / "ctest.log"
        junit_path = run_directory / "ctest.xml"
        sentinel_path = run_directory / "infrastructure-failure.txt"
        environment = os.environ.copy()
        environment.update(
            {
                "XPOOL_CTEST_CANONICAL": "1",
                "XPOOL_CTEST_INFRASTRUCTURE_SENTINEL": str(sentinel_path),
            }
        )
        command = [
            "ctest",
            "--test-dir",
            str(build_directory),
            "--output-on-failure",
            "--verbose",
            "--output-junit",
            str(junit_path),
            "--resource-spec-file",
            str(resource_spec.path),
            "--parallel",
            str(len(gpu_pool.uuids)),
        ]
        completion = SupervisedTaskScope.run(
            "ctest",
            command,
            cwd=Path(__file__).resolve().parents[3],
            env=environment,
            log_path=log_path,
            timeout_seconds=CTEST_SUITE_TIMEOUT_SECONDS,
        )
        if sentinel_path.exists():
            return CtestRunResult(2, log_path, junit_path)
        if not junit_path.is_file():
            return CtestRunResult(2, log_path, junit_path)
        if completion.kind is not TaskCompletionKind.EXITED:
            return CtestRunResult(2, log_path, junit_path)
        return CtestRunResult(0 if completion.returncode == 0 else 1, log_path, junit_path)


def ctest_gpu_id(uuid: str) -> str:
    """Encode one canonical GPU UUID as a CTest-safe resource identifier."""

    if not uuid.startswith("GPU-"):
        raise ValueError(f"invalid physical GPU UUID: {uuid!r}")
    return "gpu_" + uuid.removeprefix("GPU-").lower().replace("-", "_")


def current_build_directory() -> Path:
    """Return the scikit-build directory for the active managed interpreter."""

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return Path(__file__).resolve().parents[3] / "build" / f"{tag}-{tag}-{platform}"
