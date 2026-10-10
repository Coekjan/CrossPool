from __future__ import annotations

from pathlib import Path

import pytest
import tomli_w
from tests import TEST_CATALOG_PATH

import xkit.device
from xkit.serving.sglang.graph import SglangGraphMode, SglangGraphSettings
from xpool.config import XpoolConfig
from xtest.harness.runner.requirements import ResolvedConfig
from xtest.harness.sglang.catalog import TestCatalog
from xtest.harness.sglang.serving.launch import SERVING_OBSERVER_RECORD_CAPACITY, prepare


def test_prepare_applies_deployment_policy_and_sanitizes_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = TestCatalog.from_file(TEST_CATALOG_PATH)
    case = next(case for case in manifest.serving_cases if len(case.models) == 2)
    base_config = base_e2e_config(manifest, tmp_path)
    inherited = {
        "PATH": "/test/bin",
        "HOME": "/test/home",
        "LOGNAME": "test-user",
        "USER": "test-user",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TMPDIR": "/test/tmp",
        "LD_LIBRARY_PATH": "/test/lib",
        "CUDA_HOME": "/test/cuda",
        "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c",
        "CUDA_MPS_PIPE_DIRECTORY": "/tmp/mps-pipe",
        "CUDA_MPS_LOG_DIRECTORY": "/tmp/mps-log",
    }
    for name, value in inherited.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        xkit.device,
        "query_visible_device_total_memory_bytes",
        lambda: (40 * 1024**3, 40 * 1024**3, 40 * 1024**3),
    )
    monkeypatch.setenv("XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY", "1")
    monkeypatch.setenv("XPOOL_UNDECLARED_POLICY", "bad")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/preload.so")
    monkeypatch.setenv("NCCL_DEBUG", "TRACE")
    monkeypatch.setenv("NVSHMEM_DEBUG", "TRACE")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.invalid")
    monkeypatch.setenv("HF_HOME", "/untrusted/huggingface")
    monkeypatch.setenv("SGLANG_GRPC_PORT", "12345")

    launch = prepare(
        case,
        base_config=base_config,
        workdir=tmp_path / "attempt",
        graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend="breakable"),
    )

    assert launch.config.scheduler.slo == case.deployment_config.slo
    assert launch.config.scheduler.ffn_concurrency == case.executor_lane_count
    assert launch.config.atn.devices == [0]
    assert launch.config.ffn.devices == [1, 2]
    assert [model.id for model in launch.config.models] == [model.model_id for model in launch.models]
    for model in launch.config.models:
        assert launch.config.model_path_of(model.id) == base_config.config.model_path_of(model.id)
        assert model.ffn_tp_size == 1 and model.slo is None
    assert launch.config.vendor == base_config.config.vendor
    assert {name: launch.environment[name] for name in inherited} == inherited
    assert "XPOOL_UNDECLARED_POLICY" not in launch.environment
    for name in ("PYTHONPATH", "LD_PRELOAD", "NCCL_DEBUG", "NVSHMEM_DEBUG", "HTTPS_PROXY", "HF_HOME"):
        assert name not in launch.environment
    assert "SGLANG_GRPC_PORT" not in launch.environment
    assert launch.config.debug.transport_observer.record_capacity == SERVING_OBSERVER_RECORD_CAPACITY
    assert launch.config.debug.fabric_observer.record_capacity == SERVING_OBSERVER_RECORD_CAPACITY
    assert not launch.config.debug.prefill_logit_observer.enable
    assert "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE" not in launch.environment


@pytest.mark.parametrize(
    ("total_memory_bytes", "external_utilization", "expected_utilization"),
    [
        (40 * 1024**3, 0.9, 0.25),
        (80 * 1024**3, 0.9, 0.125),
        (40 * 1024**3, 0.1, 0.1),
    ],
)
def test_prepare_elastic_kv_preserves_absolute_memory_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    total_memory_bytes: int,
    external_utilization: float,
    expected_utilization: float,
) -> None:
    manifest = TestCatalog.from_file(TEST_CATALOG_PATH)
    case = next(case for case in manifest.serving_cases if case.elastic_kv is not None)
    monkeypatch.setattr(
        xkit.device,
        "query_visible_device_total_memory_bytes",
        lambda: (total_memory_bytes,) * case.required_device_count,
    )

    launch = prepare(
        case,
        base_config=base_e2e_config(manifest, tmp_path, atn_device_memory_utilization=external_utilization),
        workdir=tmp_path / "attempt",
        graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend="breakable"),
    )

    assert launch.config.atn.device_memory_utilization == pytest.approx(expected_utilization)


@pytest.mark.parametrize(
    ("visible_memory", "message"),
    [
        ((40 * 1024**3,), "requires 2 visible attention devices"),
        ((40 * 1024**3, 80 * 1024**3), "equal total memory"),
    ],
)
def test_prepare_rejects_incompatible_elastic_kv_gpu_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    visible_memory: tuple[int, ...],
    message: str,
) -> None:
    manifest = TestCatalog.from_file(TEST_CATALOG_PATH)
    case = next(case for case in manifest.serving_cases if case.elastic_kv is not None and case.atnagent_count == 2)
    monkeypatch.setattr(
        xkit.device,
        "query_visible_device_total_memory_bytes",
        lambda: visible_memory,
    )

    with pytest.raises(ValueError, match=message):
        prepare(
            case,
            base_config=base_e2e_config(manifest, tmp_path),
            workdir=tmp_path / "attempt",
            graph_settings=SglangGraphSettings(decode_backend="full", prefill_backend="breakable"),
        )


@pytest.mark.parametrize(
    ("graph_mode", "expected_enable"),
    [
        (SglangGraphMode.EAGER, True),
        (SglangGraphMode.DECODE_FULL, False),
        (SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE, True),
    ],
)
def test_prepare_enables_prefill_logit_observer_for_alignment_modes(
    tmp_path: Path,
    graph_mode: SglangGraphMode,
    expected_enable: bool,
) -> None:
    manifest = TestCatalog.from_file(TEST_CATALOG_PATH)
    case = manifest.serving_cases[0].model_copy(
        update={
            "graph_modes": (
                SglangGraphMode.EAGER,
                SglangGraphMode.DECODE_FULL,
                SglangGraphMode.DECODE_FULL_PREFILL_BREAKABLE,
            )
        }
    )
    base_config = base_e2e_config(manifest, tmp_path)

    launch = prepare(
        case,
        base_config=base_config,
        workdir=tmp_path / "attempt",
        graph_settings=graph_mode.settings(),
    )

    assert launch.config.debug.prefill_logit_observer.enable is expected_enable
    assert ("XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE" in launch.environment) is expected_enable


def test_prepare_keeps_prefill_logit_observer_off_for_routine_serving(tmp_path: Path) -> None:
    manifest = TestCatalog.from_file(TEST_CATALOG_PATH)
    case = manifest.serving_cases[0]

    launch = prepare(
        case,
        base_config=base_e2e_config(manifest, tmp_path),
        workdir=tmp_path / "attempt",
        graph_settings=case.graph_modes[0].settings(),
    )

    assert not launch.config.debug.prefill_logit_observer.enable


def base_e2e_config(
    manifest: TestCatalog,
    tmp_path: Path,
    *,
    atn_device_memory_utilization: float = 0.9,
) -> ResolvedConfig:
    model_base_uri = tmp_path / "models"
    model_ids = sorted({model_id for case in manifest.serving_cases for model_id in case.models})
    path = tmp_path / "runtime.toml"
    path.write_text(
        tomli_w.dumps(
            {
                "vendor": {"model_base_uri": str(model_base_uri)},
                "scheduler": {"slo": {"ttft_ms": 5000, "tbt_ms": 500}},
                "atn": {
                    "devices": [0],
                    "device_memory_utilization": atn_device_memory_utilization,
                },
                "ffn": {"devices": [1, 2]},
                "models": [
                    {
                        "id": str(model_id),
                        "path": str(tmp_path / "custom" / model_id.relative_path),
                        "ffn_tp_size": 1,
                        "slo": {"ttft_ms": 2000, "tbt_ms": 100},
                    }
                    for model_id in model_ids
                ]
                + [{"id": "external/model-not-owned-by-tests"}],
            }
        ),
        encoding="utf-8",
    )
    return ResolvedConfig(path, XpoolConfig.from_file(path, env={}))
