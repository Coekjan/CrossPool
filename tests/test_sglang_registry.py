from __future__ import annotations

import sys
from collections.abc import Sequence

import pytest
from sglang.srt.model_executor.model_runner import ModelRunner

from xpool.integrations.sglang.adapter import SglangHook, SglangModelAdapter
from xpool.integrations.sglang.models.deepseek_v2 import DeepseekV2Adapter
from xpool.integrations.sglang.registry import (
    adapter_classes_in_module,
    sglang_model_adapters,
    sort_and_validate_adapters,
)


def test_registry_discovers_deepseek_adapter() -> None:
    adapters = sglang_model_adapters()

    assert any(isinstance(adapter, DeepseekV2Adapter) for adapter in adapters)


def test_adapter_class_discovery_only_returns_module_owned_subclasses() -> None:
    classes = adapter_classes_in_module(sys.modules[__name__])

    assert _RegistryTestAdapter in classes
    assert DeepseekV2Adapter not in classes


def test_registry_rejects_duplicate_adapter_names() -> None:
    with pytest.raises(RuntimeError, match="duplicate SGLang adapter name"):
        sort_and_validate_adapters([_DuplicateAdapter(), _DuplicateAdapter()])


class _RegistryTestAdapter(SglangModelAdapter):
    name = "registry_test"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        del model_runner
        return False


class _DuplicateAdapter(SglangModelAdapter):
    name = "duplicate"

    def hooks(self) -> Sequence[SglangHook]:
        return ()

    def matches(self, model_runner: ModelRunner) -> bool:
        del model_runner
        return False
