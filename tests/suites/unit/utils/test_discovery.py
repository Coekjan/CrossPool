from __future__ import annotations

from xpool.runtime.ffnagent.architecture import FfnModelAdapter
from xpool.utils.discovery import discover_concrete_subclasses, walk_package_modules

MODELS_PACKAGE = "xpool.runtime.ffnagent.models"


def test_ffn_model_adapter_discovery_is_stable_and_module_owned() -> None:
    """Discovery returns the concrete family compilers in stable order."""

    modules = walk_package_modules(MODELS_PACKAGE)
    adapters = discover_concrete_subclasses(MODELS_PACKAGE, FfnModelAdapter)

    assert tuple(module.name for module in modules) == (
        "xpool.runtime.ffnagent.models.deepseek_v2",
        "xpool.runtime.ffnagent.models.glm4_moe_lite",
        "xpool.runtime.ffnagent.models.qwen2",
        "xpool.runtime.ffnagent.models.qwen3",
        "xpool.runtime.ffnagent.models.qwen3_moe",
    )
    assert tuple(adapter.__module__ for adapter in adapters) == tuple(module.name for module in modules)
    assert all(adapter.__module__ != FfnModelAdapter.__module__ for adapter in adapters)
