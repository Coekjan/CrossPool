"""Canonical CTest resource specification and execution ownership."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import subprocess
import sys
import sysconfig
import time
import xml.etree.ElementTree
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Self

from pydantic import JsonValue

from xkit.device import DevicePool
from xkit.results import write_json
from xkit.supervisor import SupervisedTaskScope, TaskCompletion, TaskCompletionKind, TaskScopeFailure
from xtest.harness.runner.pytest_report import PytestCaseReport, PytestCaseStatus

CTEST_SUITE_TIMEOUT_SECONDS = 1800.0
logger = logging.getLogger("xtest.ctest")
DEVICE_ASSIGNMENT_PATTERN = re.compile(r"DEVICE ASSIGNMENT devices=([^\s]+)")


@dataclass(frozen=True, slots=True)
class CtestResourceSpec:
    """One CTest resource file and its reversible device-ID mapping."""

    path: Path
    id_to_uuid: Mapping[str, str]

    @classmethod
    def write(cls, path: Path, device_uuids: Sequence[str]) -> Self:
        """Atomically write one single-slot CTest resource per physical device."""

        identifiers = {ctest_device_id(uuid): uuid for uuid in device_uuids}
        if len(identifiers) != len(device_uuids):
            raise ValueError("physical device UUIDs do not have unique CTest resource IDs")
        payload = {
            "version": {"major": 1, "minor": 0},
            "local": [{"devices": [{"id": identifier, "slots": 1} for identifier in identifiers]}],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)
        return cls(path, MappingProxyType(identifiers))


@dataclass(frozen=True, slots=True)
class CtestRunResult:
    """Canonical native-suite outcome and retained evidence paths."""

    result_code: Literal[0, 1, 2]
    log_path: Path
    junit_path: Path
    completion: TaskCompletion
    elapsed_seconds: float


class CtestSuite:
    """Run the currently installed native CTest manifest with borrowed devices."""

    def __init__(self, repository_root: Path) -> None:
        self.repository_root = repository_root

    def inventory(self) -> tuple[str, ...]:
        """Read configured native case names, without executing tests or building."""

        build_directory = current_build_directory(self.repository_root)
        if not (build_directory / "CTestTestfile.cmake").is_file():
            raise ValueError(f"current native build has no CTest manifest: {build_directory}")
        completed = subprocess.run(
            ["ctest", "--test-dir", str(build_directory), "--show-only=json-v1"],
            cwd=self.repository_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
        if completed.returncode:
            raise ValueError(f"native collection failed: {completed.stderr.strip()}")
        payload: JsonValue = json.loads(completed.stdout)
        if not isinstance(payload, dict) or not isinstance(tests := payload.get("tests"), list):
            raise ValueError("CTest inventory requires a tests array")
        names: list[str] = []
        for test in tests:
            if not isinstance(test, dict) or not isinstance(name := test.get("name"), str) or not name:
                raise ValueError("CTest inventory contains an invalid case name")
            names.append(name)
        if len(names) != len(set(names)):
            raise ValueError("CTest inventory contains duplicate case names")
        return tuple(names)

    def run(self, *, device_pool: DevicePool, run_directory: Path) -> CtestRunResult:
        """Execute CTest with resource scheduling and infrastructure detection."""

        build_directory = current_build_directory(self.repository_root)
        manifest = build_directory / "CTestTestfile.cmake"
        if not manifest.is_file():
            raise RuntimeError(f"current native build has no CTest manifest: {build_directory}")
        run_directory.mkdir(parents=True, exist_ok=False)
        resource_spec = CtestResourceSpec.write(run_directory / "resources.json", device_pool.uuids)
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
            str(len(device_pool.uuids)),
        ]
        device_assignments = ",".join(
            f"{device_pool.physical_index_by_uuid[uuid]}:{uuid}" for uuid in device_pool.uuids
        )
        logger.info("cext devices=%s", device_assignments, extra={"status": "RUNNING"})
        started_at = time.monotonic()
        lease = device_pool.try_lease(len(device_pool.uuids))
        if lease is None:
            raise RuntimeError("native suite requires the complete eligible device grant")
        try:
            completion = SupervisedTaskScope.run(
                "ctest",
                command,
                cwd=self.repository_root,
                env=environment,
                log_path=log_path,
                timeout_seconds=CTEST_SUITE_TIMEOUT_SECONDS,
            )
        except TaskScopeFailure:
            # The outer pool retains this grant until domain retirement is proved.
            raise
        except BaseException:
            device_pool.release(lease)
            raise
        device_pool.release(lease)
        if sentinel_path.exists() or not junit_path.is_file() or completion.kind is not TaskCompletionKind.EXITED:
            result_code = 2
        else:
            result_code = 0 if completion.returncode == 0 else 1
        logger.info(
            "cext devices=%s elapsed=%.3fs code=%s log=%s junit=%s",
            device_assignments,
            time.monotonic() - started_at,
            result_code,
            log_path,
            junit_path,
            extra={"status": "PASSED" if result_code == 0 else "FAILED"},
        )
        if junit_path.is_file():
            for case_name, case_devices, elapsed in ctest_case_timings(junit_path, device_pool.physical_index_by_uuid):
                logger.info("  %s devices=%s elapsed=%s", case_name, case_devices, elapsed)
        return CtestRunResult(result_code, log_path, junit_path, completion, time.monotonic() - started_at)


def ctest_case_reports(path: Path) -> tuple[PytestCaseReport, ...]:
    """Retain native JUnit outcomes and available reasons for offline reports."""

    root = xml.etree.ElementTree.parse(path).getroot()
    reports: list[PytestCaseReport] = []
    for element in root.iter("testcase"):
        name = element.get("name")
        if not name:
            raise ValueError("CTest JUnit case is missing its name")
        outcomes = tuple(child for child in element if child.tag in {"failure", "error", "skipped"})
        if len(outcomes) > 1:
            raise ValueError(f"CTest case has multiple outcomes: {name}")
        status = PytestCaseStatus.PASSED
        detail = None
        if outcomes:
            outcome = outcomes[0]
            status = PytestCaseStatus.SKIPPED if outcome.tag == "skipped" else PytestCaseStatus.FAILED
            detail = outcome.get("message") or (outcome.text or "").strip() or None
        elapsed_raw = element.get("time")
        elapsed = float(elapsed_raw) if elapsed_raw is not None else None
        if elapsed is not None and (not math.isfinite(elapsed) or elapsed < 0):
            raise ValueError(f"CTest case has invalid elapsed time: {name}")
        reports.append(PytestCaseReport(name, status, detail, elapsed))
    return tuple(reports)


def ctest_case_timings(path: Path, physical_index_by_uuid: Mapping[str, int]) -> tuple[tuple[str, str, str], ...]:
    """Read native case timing and the launcher's device assignment from JUnit."""

    root = xml.etree.ElementTree.parse(path).getroot()
    cases = []
    for element in root.iter("testcase"):
        output = element.findtext("system-out") or ""
        assignment = DEVICE_ASSIGNMENT_PATTERN.search(output)
        device = assignment.group(1) if assignment else "unavailable"
        if device in physical_index_by_uuid:
            device = f"{physical_index_by_uuid[device]}:{device}"
        cases.append(
            (
                element.get("name", "unnamed"),
                device,
                element.get("time", "unavailable"),
            )
        )
    return tuple(cases)


def ctest_device_id(uuid: str) -> str:
    """Encode one canonical device UUID as a CTest-safe resource identifier."""

    if not uuid.startswith("GPU-"):
        raise ValueError(f"invalid physical device UUID: {uuid!r}")
    return "device_" + uuid.removeprefix("GPU-").lower().replace("-", "_")


def current_build_directory(repository_root: Path) -> Path:
    """Return the scikit-build directory for the active managed interpreter."""

    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    platform = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    return repository_root / "build" / f"{tag}-{tag}-{platform}"
