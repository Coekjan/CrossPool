from __future__ import annotations

import json
import xml.etree.ElementTree
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import tests.harness.runner.ctest
from tests.harness.runner.ctest import (
    CtestResourceSpec,
    CtestSuite,
    ctest_case_timings,
    ctest_gpu_id,
    current_build_directory,
)
from tests.harness.runner.gpu import GpuPool
from tests.harness.runner.supervisor import TaskCompletion, TaskCompletionKind


def test_current_build_directory_is_rooted_at_repository_build() -> None:
    repository_root = Path(__file__).resolve().parents[4]

    assert current_build_directory().parent == repository_root / "build"


def test_ctest_case_timings_read_junit_gpu_assignment(tmp_path: Path) -> None:
    path = tmp_path / "ctest.xml"
    suite = xml.etree.ElementTree.Element("testsuites")
    case = xml.etree.ElementTree.SubElement(suite, "testcase", {"name": "cext.example", "time": "1.25"})
    xml.etree.ElementTree.SubElement(case, "system-out").text = "GPU ASSIGNMENT gpus=GPU-example\n"
    xml.etree.ElementTree.ElementTree(suite).write(path, encoding="utf-8")

    assert ctest_case_timings(path, {"GPU-example": 2}) == (("cext.example", "2:GPU-example", "1.25"),)


def test_ctest_resource_spec_preserves_reversible_gpu_mapping(tmp_path: Path) -> None:
    uuids = (
        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "GPU-11111111-2222-3333-4444-555555555555",
    )

    spec = CtestResourceSpec.write(tmp_path / "resources.json", uuids)
    payload = json.loads(spec.path.read_text(encoding="utf-8"))

    assert spec.id_to_uuid == {ctest_gpu_id(uuid): uuid for uuid in uuids}
    assert payload == {
        "version": {"major": 1, "minor": 0},
        "local": [{"gpus": [{"id": ctest_gpu_id(uuid), "slots": 1} for uuid in uuids]}],
    }


def test_ctest_suite_classifies_ordinary_failure_with_junit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    build_directory = tmp_path / "build"
    build_directory.mkdir()
    (build_directory / "CTestTestfile.cmake").write_text("", encoding="utf-8")
    monkeypatch.setattr(tests.harness.runner.ctest, "current_build_directory", lambda: build_directory)

    def run(name: str, command: list[str], **kwargs: object) -> TaskCompletion:
        assert name == "ctest"
        assert "--verbose" in command
        junit_path = Path(command[command.index("--output-junit") + 1])
        junit_path.write_text("<testsuites/>", encoding="utf-8")
        return TaskCompletion(TaskCompletionKind.EXITED, 8, None)

    monkeypatch.setattr(tests.harness.runner.ctest.SupervisedTaskScope, "run", run)
    pool = cast(
        GpuPool,
        SimpleNamespace(
            uuids=("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
            physical_index_by_uuid={"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": 0},
        ),
    )

    result = CtestSuite().run(gpu_pool=pool, run_directory=tmp_path / "run")

    assert result.result_code == 1


@pytest.mark.parametrize(
    ("completion", "write_junit", "write_sentinel"),
    (
        (TaskCompletion(TaskCompletionKind.TIMED_OUT, None, "timeout"), True, False),
        (TaskCompletion(TaskCompletionKind.LEAKED, 1, "leak"), True, False),
        (TaskCompletion(TaskCompletionKind.INFRASTRUCTURE_FAILED, None, "supervisor"), True, False),
        (TaskCompletion(TaskCompletionKind.EXITED, 0, None), False, False),
        (TaskCompletion(TaskCompletionKind.EXITED, 0, None), True, True),
    ),
)
def test_ctest_suite_classifies_infrastructure_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completion: TaskCompletion,
    write_junit: bool,
    write_sentinel: bool,
) -> None:
    build_directory = tmp_path / "build"
    build_directory.mkdir()
    (build_directory / "CTestTestfile.cmake").write_text("", encoding="utf-8")
    monkeypatch.setattr(tests.harness.runner.ctest, "current_build_directory", lambda: build_directory)

    def run(name: str, command: list[str], **kwargs: object) -> TaskCompletion:
        assert name == "ctest"
        if write_junit:
            Path(command[command.index("--output-junit") + 1]).write_text("<testsuites/>", encoding="utf-8")
        if write_sentinel:
            environment = cast(dict[str, str], kwargs["env"])
            Path(environment["XPOOL_CTEST_INFRASTRUCTURE_SENTINEL"]).write_text("failed\n", encoding="utf-8")
        return completion

    monkeypatch.setattr(tests.harness.runner.ctest.SupervisedTaskScope, "run", run)
    pool = cast(
        GpuPool,
        SimpleNamespace(
            uuids=("GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",),
            physical_index_by_uuid={"GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee": 0},
        ),
    )

    result = CtestSuite().run(gpu_pool=pool, run_directory=tmp_path / "run")

    assert result.result_code == 2
