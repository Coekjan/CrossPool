from pathlib import Path

import pytest

import xtest
from xkit import ResourceRequirements
from xkit.source import resolve_source_path
from xtest.harness.support.config import TEST_MODEL_ID


def test_declarations_preserve_the_callable_and_defer_dynamic_resources() -> None:
    resources = ResourceRequirements(1, True, (TEST_MODEL_ID,))
    observed: list[int] = []

    def required_resources(size: int) -> ResourceRequirements:
        observed.append(size)
        assert size == 16
        return resources

    def program(size: int) -> int:
        return size

    assert xtest.parameterize("size", (16, 32))(xtest.requirements(required_resources)(program)) is program
    assert program(32) == 32
    assert observed == []


def test_resource_record_rejects_noninteger_device_count() -> None:
    with pytest.raises(ValueError, match="valid integer"):
        ResourceRequirements(1.5, False, ())  # ty: ignore[invalid-argument-type]


def test_source_reference_uses_the_catalogues_sibling_suites_tree(tmp_path: Path) -> None:
    assert resolve_source_path(tmp_path / "tests/tests.toml", "e2e.sglang.test_serving") == (
        tmp_path / "tests/suites/e2e/sglang/test_serving.py"
    )
    with pytest.raises(ValueError, match="dotted Python module"):
        resolve_source_path(tmp_path / "tests/tests.toml", "../outside")
