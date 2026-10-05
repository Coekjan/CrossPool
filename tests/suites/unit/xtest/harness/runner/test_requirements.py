from pathlib import Path

import pytest

from xpool.model import ModelId
from xtest.harness.runner.requirements import (
    RequirementMisconfigured,
    RequirementUnavailable,
    require_config,
    require_devices,
    require_model_weights,
)
from xtest.harness.support.config import TEST_MODEL_ID


def write_config(path: Path, model_base_uri: Path) -> None:
    """Write the smallest valid CrossPool config used by requirement tests."""

    path.write_text(
        "\n".join(
            (
                "[vendor]",
                f'model_base_uri = "{model_base_uri}"',
                "",
                "[logging]",
                'level = "warning"',
                "",
                "[scheduler.slo]",
                "ttft_ms = 1000",
                "tbt_ms = 50",
                "",
                "[atn]",
                "devices = [0]",
                "[ffn]",
                "devices = [1]",
                "",
                "[[models]]",
                'id = "external/model-not-owned-by-tests"',
            )
        ),
        encoding="utf-8",
    )


def test_device_requirement_rejects_unavailable_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RequirementUnavailable, match="no supported devices"):
        require_devices(1)


def test_device_requirement_checks_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    require_devices(2)
    with pytest.raises(RequirementUnavailable, match="requires 3"):
        require_devices(3)
    with pytest.raises(RequirementMisconfigured, match="at least 1"):
        require_devices(0)


def test_config_requires_exact_environment_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("XPOOL_CONFIG", raising=False)

    with pytest.raises(RequirementUnavailable, match="set XPOOL_CONFIG"):
        require_config()


def test_explicit_invalid_config_is_misconfigured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "invalid.toml"
    config_path.write_text("not = [valid", encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    with pytest.raises(RequirementMisconfigured, match="invalid XPOOL_CONFIG"):
        require_config()


def test_config_reload_observes_same_path_rewrite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    first_model_root = tmp_path / "first-model-root"
    second_model_root = tmp_path / "second-model-root"
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, first_model_root)

    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    monkeypatch.setenv("XPOOL_LOG_LEVEL", "debug")
    first = require_config()
    write_config(config_path, second_model_root)
    monkeypatch.setenv("XPOOL_LOG_LEVEL", "error")
    second = require_config()

    assert first.config.vendor.model_base_uri == first_model_root
    assert first.config.logging.level == "debug"
    assert second.path == config_path
    assert second.config.vendor.model_base_uri == second_model_root
    assert second.config.logging.level == "error"


def test_model_weights_require_directory_and_config_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / TEST_MODEL_ID.relative_path
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, model_root)
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))
    config = require_config().config

    with pytest.raises(RequirementUnavailable, match="weight directory"):
        require_model_weights(config, TEST_MODEL_ID)
    model_path.mkdir(parents=True)
    with pytest.raises(RequirementUnavailable, match=r"config\.json"):
        require_model_weights(config, TEST_MODEL_ID)
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    assert require_model_weights(config, TEST_MODEL_ID) == model_path


def test_model_weights_resolve_explicit_path_and_unregistered_vendor_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "models"
    model_path = model_root / "manifest/model"
    model_path.mkdir(parents=True)
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    config_path = tmp_path / "xpool.toml"
    write_config(config_path, model_root)
    custom_path = tmp_path / "custom-model"
    custom_path.mkdir()
    (custom_path / "config.json").write_text("{}", encoding="utf-8")
    config_path.write_text(config_path.read_text(encoding="utf-8") + f'\npath = "{custom_path}"\n', encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(config_path))

    config = require_config().config
    assert require_model_weights(config, ModelId("manifest/model")) == model_path
    assert require_model_weights(config, ModelId("external/model-not-owned-by-tests")) == custom_path
