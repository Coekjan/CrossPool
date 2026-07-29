"""Tests for pytest-independent resource requirement resolution."""

from pathlib import Path
from typing import Never

import pytest

from tests.harness.runner.requirements import (
    CudaRequirement,
    RequirementGuard,
    RequirementMisconfigured,
    RequirementResolver,
    RequirementUnavailable,
)


def write_config(path: Path, model_base_uri: Path) -> None:
    """Write the smallest valid xpool config used by requirement tests."""

    path.write_text(
        "\n".join(
            (
                "[vendor]",
                f'model_base_uri = "{model_base_uri}"',
                "",
                "[devices]",
                "atn_cuda_devices = [0]",
                "ffn_cuda_devices = [1]",
                "",
                "[[models]]",
                'id = "external/model-not-owned-by-tests"',
            )
        ),
        encoding="utf-8",
    )


def test_cuda_requirement_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RequirementUnavailable, match="CUDA is not available"):
        RequirementResolver().require_cuda(CudaRequirement())


def test_cuda_requirement_checks_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    resolver = RequirementResolver()

    resolver.require_cuda(CudaRequirement(min_devices=2))
    with pytest.raises(RequirementUnavailable, match="requires 3"):
        resolver.require_cuda(CudaRequirement(min_devices=3))


def test_mps_requirement_uses_controller_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "tests.harness.runner.requirements.probe_mps_controller",
        lambda: type("Probe", (), {"online": False, "diagnostic": "MPS unavailable"})(),
    )

    with pytest.raises(RequirementUnavailable, match="MPS unavailable"):
        RequirementResolver().require_mps()


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


def test_config_reload_observes_same_path_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_model_root = tmp_path / "first-model-root"
    second_model_root = tmp_path / "second-model-root"
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, first_model_root)
    resolver = RequirementResolver()

    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    first = resolver.require_config()
    write_config(config_path, second_model_root)
    second = resolver.require_config()

    assert first.config.vendor.model_base_uri == first_model_root
    assert second.path == config_path
    assert second.config.vendor.model_base_uri == second_model_root


def test_model_weights_require_directory_and_config_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / "model-a"
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, model_root)
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    resolver = RequirementResolver()

    with pytest.raises(RequirementUnavailable, match="weight directory"):
        resolver.require_model_weights("model-a")
    model_path.mkdir(parents=True)
    with pytest.raises(RequirementUnavailable, match=r"config\.json"):
        resolver.require_model_weights("model-a")
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert resolver.require_model_weights("model-a").path == model_path


def test_model_weights_ignore_external_config_model_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / "manifest/model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, model_root)
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    resolved = RequirementResolver().require_model_weights("manifest/model")

    assert resolved.path == model_path


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
