from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import xpool.memory
from xpool.memory import (
    ExecutionResourceDeclaration,
    FfnMemoryCalibration,
    FfnMemoryCalibrationCoefficients,
    XpoolMemoryCalibrationProfile,
)
from xtest.harness.support.config import (
    TEST_CALIBRATION_ENVIRONMENT,
    lane_resource_declaration,
    synthetic_memory_context,
)


@pytest.mark.parametrize(
    ("capacity", "alignment", "aliases"),
    (
        (0, 256, ("scratch",)),
        (1024, 3, ("scratch",)),
        (1024, 256, ("other",)),
        (1024, 256, ("scratch", "scratch")),
    ),
)
def test_resource_declaration_rejects_unusable_backing(capacity: int, alignment: int, aliases: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        lane_resource_declaration(key="scratch", max_bytes=capacity, alignment_bytes=alignment, aliases=aliases)


def profile() -> XpoolMemoryCalibrationProfile:
    """Build one strict Profile matching the installed software."""

    return XpoolMemoryCalibrationProfile(
        memory_context=synthetic_memory_context(),
        ffn=FfnMemoryCalibration(
            executor_lane_count=1,
            minimum_held_out_headroom_bytes=1,
            coefficients=FfnMemoryCalibrationCoefficients(
                base_bytes=1,
                bytes_per_tensor_storage_mib=2,
                bytes_per_tensor_storage_allocation=3,
                bytes_per_dense_graph_capture=4,
                bytes_per_moe_graph_capture=5,
                bytes_per_executor_lane=6,
                bytes_per_compute_branch=7,
                dense_implementation_bytes=8,
                moe_implementation_bytes=9,
                joined_ffnagent_bytes=10,
            ),
        ),
    )


@pytest.mark.parametrize(
    "change",
    (
        {"request_bytes": ()},
        {"request_bytes": (0,)},
        {"alignment_bytes": 512},
        {"exact_logical_bytes": 1025},
        {"request_bytes": (1,), "exact_logical_bytes": 1024},
    ),
)
def test_requirement_rejects_incoherent_contract(change: dict[str, object]) -> None:
    requirement = lane_resource_declaration(
        key="scratch", max_bytes=1024, alignment_bytes=256, aliases=("scratch",)
    ).requirement
    with pytest.raises(ValueError):
        replace(requirement, **change)


def test_profile_json_rejects_string_for_numeric_field() -> None:
    payload = profile().model_dump(mode="json")
    payload["ffn"]["coefficients"]["base_bytes"] = "1"

    with pytest.raises(ValidationError) as error:
        XpoolMemoryCalibrationProfile.model_validate_json(json.dumps(payload))
    assert any(item["loc"][-1] == "base_bytes" for item in error.value.errors())


def test_absent_profile_returns_none() -> None:
    assert xpool.memory.load_memory_calibration_profile(path=None) is None


def test_load_parses_explicit_profile(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(profile().model_dump_json(), encoding="utf-8")
    assert xpool.memory.load_memory_calibration_profile(path=path) == profile()


def test_profile_matching_is_separate_from_file_loading(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    incompatible = profile().model_copy(update={"ffn": profile().ffn.model_copy(update={"executor_lane_count": 2})})
    path.write_text(incompatible.model_dump_json(), encoding="utf-8")
    loaded = xpool.memory.load_memory_calibration_profile(path=path)
    assert loaded == incompatible
    assert loaded is not None
    with pytest.raises(RuntimeError, match="Executor Lanes"):
        loaded.require_compatible(context=incompatible.memory_context, executor_lane_count=1)


@pytest.mark.parametrize("content", (b"{", b"\xff"))
def test_configured_malformed_profile_fails_without_fallback(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "memory.json"
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match="failed to load memory calibration Profile"):
        xpool.memory.load_memory_calibration_profile(path=path)


def test_unreadable_profile_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="failed to load memory calibration Profile"):
        xpool.memory.load_memory_calibration_profile(path=tmp_path / "missing.json")


def test_writer_serializes_profile_to_explicit_path(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text("old", encoding="utf-8")
    expected = profile()
    xpool.memory.write_memory_calibration_profile(expected, path=path)

    assert XpoolMemoryCalibrationProfile.model_validate_json(path.read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("name", "different-device"),
        ("torch_version", "different-version"),
        ("configuration_inputs", (("environment.NVSHMEM_DISABLE_NCCL", "1"),)),
        ("configuration_inputs", (("environment.CUDA_MODULE_LOADING", "EAGER"),)),
        ("configuration_inputs", (("environment.CUBLAS_WORKSPACE_CONFIG", ":16:8"),)),
    ),
)
def test_profile_reuse_requires_matching_participant_facts(field: str, value: object) -> None:
    recorded = profile()
    context = recorded.memory_context
    assert context.participants is not None
    recorded.require_compatible(context=context, executor_lane_count=1)
    changed = TEST_CALIBRATION_ENVIRONMENT.model_copy(update={field: value})
    context = context.model_copy(update={"participants": (changed, *context.participants[1:])})
    with pytest.raises(RuntimeError, match="incompatible with the current Fabric memory context"):
        recorded.require_compatible(context=context, executor_lane_count=1)


def test_profile_requires_complete_participant_observations() -> None:
    recorded = profile()
    context = recorded.memory_context.model_copy(update={"participants": None})
    with pytest.raises(ValidationError, match="complete participant observations"):
        XpoolMemoryCalibrationProfile(memory_context=context, ffn=recorded.ffn)


@pytest.mark.parametrize("uses", ([], ["dense", "dense"], ["unknown"]))
def test_resource_requires_distinct_known_computation_sites(uses: list[str]) -> None:
    declaration = lane_resource_declaration(key="scratch", max_bytes=1024, alignment_bytes=256, aliases=("scratch",))
    values = declaration.model_dump(mode="json")
    values["uses"] = uses
    with pytest.raises(ValidationError):
        ExecutionResourceDeclaration.model_validate(values)
