from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import ValidationError

from xpool.config import MissingRequiredConfig, XpoolConfig


def test_duplicate_model_paths_are_rejected() -> None:
    with pytest.raises(ValidationError, match="model paths must be unique"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [
                    {"id": "m0", "path": "/models/m"},
                    {"id": "m1", "path": "/models/m"},
                ],
            },
            cli={},
        )


def test_vendor_model_base_uri_from_config_derives_model_path() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "test/model"}],
        }
    )

    assert config.vendor.model_base_uri == Path("/models")
    assert config.model_path_of("test/model") == Path("/models/test/model")


def test_vendor_model_base_uri_is_config_only(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="xpool.config"):
        config = XpoolConfig.from_mapping(
            {
                "vendor": {"model_base_uri": "/models-from-config"},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "test/model"}],
            },
            env={"XPOOL_VENDOR_MODEL_BASE_URI": "/models-from-env"},
        )

    assert "XPOOL_VENDOR_MODEL_BASE_URI" in caplog.text
    assert config.vendor.model_base_uri == Path("/models-from-config")
    assert config.model_path_of("test/model") == Path("/models-from-config/test/model")


def test_explicit_model_path_overrides_vendor_model_base_uri() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "test/model", "path": "/custom/deepseek"}],
        }
    )

    assert config.models[0].path == Path("/custom/deepseek")
    assert config.model_path_of("test/model") == Path("/custom/deepseek")


def test_model_path_lookup_rejects_unknown_model_id() -> None:
    config = XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": "/models"},
            "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
            "models": [{"id": "test/model"}],
        }
    )

    with pytest.raises(MissingRequiredConfig, match="unknown configured model id"):
        config.model_path_of("unknown-model")


def test_model_path_or_vendor_model_base_uri_is_required() -> None:
    with pytest.raises(ValidationError, match=r"vendor\.model_base_uri"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "test/model"}],
            }
        )


def test_vendor_model_base_uri_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match=r"vendor\.model_base_uri must be absolute"):
        XpoolConfig.from_mapping(
            {
                "vendor": {"model_base_uri": "relative/models"},
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "test/model"}],
            }
        )


def test_model_path_must_be_absolute() -> None:
    with pytest.raises(ValidationError, match="path must be absolute"):
        XpoolConfig.from_mapping(
            {
                "devices": {"atn_cuda_devices": [0], "ffn_cuda_devices": [1]},
                "models": [{"id": "m", "path": "relative/model"}],
            },
            cli={},
        )
