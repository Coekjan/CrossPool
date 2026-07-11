"""Tests for pytest-independent resource requirement resolution."""

from pathlib import Path
from typing import Never

import pytest

from tests.harness.requirements import (
    CudaRequirement,
    RequirementGuard,
    RequirementMisconfigured,
    RequirementResolver,
    RequirementUnavailable,
)


def write_config(path: Path, model_path: Path) -> None:
    """Write the smallest valid xpool config used by requirement tests."""

    path.write_text(
        "\n".join(
            (
                "[devices]",
                "atn_cuda_devices = [0]",
                "ffn_cuda_devices = [1]",
                "",
                "[[models]]",
                'id = "model-a"',
                f'path = "{model_path}"',
            )
        ),
        encoding="utf-8",
    )


def test_cuda_requirement_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RequirementUnavailable, match="CUDA is not available"):
        RequirementResolver().require_cuda(CudaRequirement())


def test_cuda_requirement_checks_count_and_bf16(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    monkeypatch.setattr("torch.cuda.is_bf16_supported", lambda: False)
    resolver = RequirementResolver()

    resolver.require_cuda(CudaRequirement(min_devices=2))
    with pytest.raises(RequirementUnavailable, match="BF16"):
        resolver.require_cuda(CudaRequirement(bf16=True))
    with pytest.raises(RequirementUnavailable, match="requires 3"):
        resolver.require_cuda(CudaRequirement(min_devices=3))


def test_config_requires_exact_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    with pytest.raises(RequirementUnavailable, match="set XPOOL_CONFIG"):
        RequirementResolver().require_config()


def test_explicit_invalid_config_is_misconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("not = [valid", encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    with pytest.raises(RequirementMisconfigured, match="invalid XPOOL_CONFIG"):
        RequirementResolver().require_config()


def test_config_cache_changes_with_environment_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_model = tmp_path / "first-model"
    second_model = tmp_path / "second-model"
    first_config = tmp_path / "first.toml"
    second_config = tmp_path / "second.toml"
    write_config(first_config, first_model)
    write_config(second_config, second_model)
    resolver = RequirementResolver()

    monkeypatch.setenv("XPOOL_CONFIG", str(first_config))
    first = resolver.require_config()
    assert resolver.require_config() is first
    monkeypatch.setenv("XPOOL_CONFIG", str(second_config))
    second = resolver.require_config()

    assert second.path == second_config
    assert second is not first


def test_model_weights_require_directory_and_config_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_path = tmp_path / "model"
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, model_path)
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    resolver = RequirementResolver()

    with pytest.raises(RequirementUnavailable, match="weight directory"):
        resolver.require_model_weights("model-a")
    model_path.mkdir()
    with pytest.raises(RequirementUnavailable, match=r"config\.json"):
        resolver.require_model_weights("model-a")
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert resolver.require_model_weights("model-a").path == model_path


def test_requirement_guard_skips_unavailable_and_always_fails_misconfiguration() -> None:
    outcomes: list[str] = []

    def skip(reason: str) -> Never:
        outcomes.append(f"skip:{reason}")
        raise LookupError

    def fail(reason: str) -> Never:
        outcomes.append(f"fail:{reason}")
        raise LookupError

    default_guard = RequirementGuard(strict=False, skip=skip, fail=fail)
    strict_guard = RequirementGuard(strict=True, skip=skip, fail=fail)

    with pytest.raises(LookupError):
        default_guard.run(lambda: (_ for _ in ()).throw(RequirementUnavailable("cuda")))
    with pytest.raises(LookupError):
        strict_guard.run(lambda: (_ for _ in ()).throw(RequirementUnavailable("cuda")))
    with pytest.raises(LookupError):
        default_guard.run(lambda: (_ for _ in ()).throw(RequirementMisconfigured("config")))

    assert outcomes == ["skip:cuda", "fail:cuda", "fail:config"]
