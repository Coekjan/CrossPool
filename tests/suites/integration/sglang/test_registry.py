from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner

from xpool.integrations.sglang.adapter import SglangShimAdapter
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2ShimAdapter
from xpool.integrations.sglang.models.qwen3 import Qwen3ShimAdapter
from xpool.integrations.sglang.registry import (
    MODELS_PACKAGE,
    discover_sglang_model_adapters,
    sort_and_validate_adapters,
)


def test_registry_discovers_deepseek_adapter() -> None:
    adapters = discover_sglang_model_adapters(MODELS_PACKAGE)

    assert any(isinstance(adapter, DeepseekV2ShimAdapter) for adapter in adapters)
    assert any(isinstance(adapter, Qwen3ShimAdapter) for adapter in adapters)


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

from xpool.integrations.sglang.adapter import SglangShimAdapter
from xpool.integrations.sglang.hooks.registry import SglangHook
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2ShimAdapter


class NestedProbeAdapter(SglangShimAdapter):
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


class DuplicateAdapter(SglangShimAdapter):
    name = "duplicate"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        return False
