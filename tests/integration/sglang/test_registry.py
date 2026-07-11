from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner

from xpool.integrations.sglang.adapter import SglangHook, SglangModelAdapter
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2Adapter
from xpool.integrations.sglang.registry import (
    MODELS_PACKAGE,
    adapter_classes_in_module,
    discover_sglang_model_adapters,
    sort_and_validate_adapters,
)


def test_registry_discovers_deepseek_adapter() -> None:
    adapters = discover_sglang_model_adapters(MODELS_PACKAGE)

    assert any(isinstance(adapter, DeepseekV2Adapter) for adapter in adapters)


def test_adapter_class_discovery_only_returns_module_owned_subclasses() -> None:
    classes = adapter_classes_in_module(sys.modules[__name__])

    assert RegistryTestAdapter in classes
    assert DeepseekV2Adapter not in classes


def test_registry_rejects_duplicate_adapter_names() -> None:
    with pytest.raises(RuntimeError, match="duplicate SGLang adapter name"):
        sort_and_validate_adapters([DuplicateAdapter(), DuplicateAdapter()])


def test_registry_discovers_adapters_in_model_subpackages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "xpool_registry_probe"
    nested = package / "nested_model"
    nested.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (nested / "__init__.py").write_text("", encoding="utf-8")
    (nested / "adapter.py").write_text(
        """
from collections.abc import Sequence

from sglang.srt.model_executor.model_runner import ModelRunner

from xpool.integrations.sglang.adapter import SglangHook, SglangModelAdapter


class NestedProbeAdapter(SglangModelAdapter):
    name = "nested_probe"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    adapters = discover_sglang_model_adapters("xpool_registry_probe")

    assert [adapter.name for adapter in adapters] == ["nested_probe"]


def test_registry_skips_broken_adapter_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    package = tmp_path / "xpool_registry_fault_probe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "broken.py").write_text("raise ImportError('missing optional adapter dependency')\n", encoding="utf-8")
    (package / "working.py").write_text(
        """
from collections.abc import Sequence

from sglang.srt.model_executor.model_runner import ModelRunner

from xpool.integrations.sglang.adapter import SglangHook, SglangModelAdapter


class WorkingProbeAdapter(SglangModelAdapter):
    name = "working_probe"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False
""".strip(),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with caplog.at_level("WARNING", logger="xpool.integrations.sglang.registry"):
        adapters = discover_sglang_model_adapters("xpool_registry_fault_probe", strict=False)

    assert [adapter.name for adapter in adapters] == ["working_probe"]
    assert "Skipping SGLang adapter module xpool_registry_fault_probe.broken" in caplog.text


def test_registry_strict_discovery_rejects_broken_adapter_modules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "xpool_registry_strict_fault_probe"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "broken.py").write_text("raise ImportError('missing core adapter dependency')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    with pytest.raises(RuntimeError, match=r"xpool_registry_strict_fault_probe\.broken"):
        discover_sglang_model_adapters("xpool_registry_strict_fault_probe")


class RegistryTestAdapter(SglangModelAdapter):
    name = "registry_test"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False


class DuplicateAdapter(SglangModelAdapter):
    name = "duplicate"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False
