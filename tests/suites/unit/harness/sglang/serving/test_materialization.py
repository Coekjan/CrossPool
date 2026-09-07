"""SGLang E2E launch materialization behavior."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from tests.harness.sglang.manifest import E2E_MANIFEST_PATH, E2eManifest
from tests.harness.sglang.serving.graph import SglangGraphMode
from tests.harness.sglang.serving.launch import materialize
from xpool.config import XpoolConfig


def test_materialize_writes_config_policy_and_sanitizes_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = E2eManifest.load(E2E_MANIFEST_PATH)
    case = next(case for case in manifest.model_serving_cases if len(case.models) == 2)
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
    monkeypatch.setenv("XPOOL_DEBUG_TRANSPORT_OBSERVER_RECORD_CAPACITY", "1")
    monkeypatch.setenv("XPOOL_UNDECLARED_POLICY", "bad")
    monkeypatch.setenv("PYTHONPATH", "/untrusted/python")
    monkeypatch.setenv("LD_PRELOAD", "/untrusted/preload.so")
    monkeypatch.setenv("NCCL_DEBUG", "TRACE")
    monkeypatch.setenv("NVSHMEM_DEBUG", "TRACE")
    monkeypatch.setenv("HTTPS_PROXY", "http://untrusted.invalid")
    monkeypatch.setenv("HF_HOME", "/untrusted/huggingface")
    monkeypatch.setenv("SGLANG_GRPC_PORT", "12345")

    launch = materialize(
        manifest,
        case,
        base_config=base_config,
        workdir=tmp_path / "attempt",
        daemon_port=19810,
        graph_settings=case.graph_modes[0].settings(),
    )

    with launch.config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)
    assert raw["scheduler"] == {
        "atn_concurrency": len(case.models),
        "ffn_concurrency": case.executor_lane_count,
        "ffn_policy": "fifo",
    }
    assert raw["atn"] == {"devices": [0]}
    assert raw["ffn"] == {"devices": [1, 2]}
    assert raw["models"] == [{"id": model.model_id} for model in launch.models]
    assert raw["vendor"] == {"model_base_uri": str(base_config.vendor.model_base_uri)}
    assert "debug" not in raw
    assert launch.config.daemon.port == 19810
    assert {name: launch.environment[name] for name in inherited} == inherited
    assert "XPOOL_UNDECLARED_POLICY" not in launch.environment
    for name in ("PYTHONPATH", "LD_PRELOAD", "NCCL_DEBUG", "NVSHMEM_DEBUG", "HTTPS_PROXY", "HF_HOME"):
        assert name not in launch.environment
    assert "SGLANG_GRPC_PORT" not in launch.environment
    assert launch.config.debug.transport_observer.record_capacity == case.transport_record_capacity
    assert launch.config.debug.fabric_observer.record_capacity == case.fabric_record_capacity
    assert not launch.config.debug.prefill_logit_observer.enable
    assert "XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE" not in launch.environment


@pytest.mark.parametrize(
    ("graph_mode", "expected_enable"),
    [
        (SglangGraphMode.EAGER, True),
        (SglangGraphMode.DECODE_FULL, False),
        (SglangGraphMode.PREFILL_BREAKABLE, True),
    ],
)
def test_materialize_enables_prefill_logit_observer_for_alignment_modes(
    tmp_path: Path,
    graph_mode: SglangGraphMode,
    expected_enable: bool,
) -> None:
    manifest = E2eManifest.load(E2E_MANIFEST_PATH)
    case = next(case for case in manifest.model_serving_cases if len(case.graph_modes) > 1)
    base_config = base_e2e_config(manifest, tmp_path)

    launch = materialize(
        manifest,
        case,
        base_config=base_config,
        workdir=tmp_path / "attempt",
        daemon_port=19810,
        graph_settings=graph_mode.settings(),
    )

    assert launch.config.debug.prefill_logit_observer.enable is expected_enable
    assert ("XPOOL_DEBUG_PREFILL_LOGIT_OBSERVER_ENABLE" in launch.environment) is expected_enable


def test_materialize_rejects_wrong_model_architecture(tmp_path: Path) -> None:
    manifest = E2eManifest.load(E2E_MANIFEST_PATH)
    case = manifest.model_serving_cases[0]
    base_config = base_e2e_config(manifest, tmp_path)
    model = manifest.model(case.models[0].model)
    model_base_uri = base_config.vendor.model_base_uri
    assert model_base_uri is not None
    config_path = model_base_uri / model.model_id / "config.json"
    config_path.write_text('{"architectures":["WrongArchitecture"]}', encoding="utf-8")

    with pytest.raises(ValueError, match=model.architecture):
        materialize(
            manifest,
            case,
            base_config=base_config,
            workdir=tmp_path / "attempt",
            daemon_port=19810,
            graph_settings=case.graph_modes[0].settings(),
        )


def base_e2e_config(manifest: E2eManifest, tmp_path: Path) -> XpoolConfig:
    model_base_uri = tmp_path / "models"
    for model in manifest.models:
        model_path = model_base_uri / model.model_id
        model_path.mkdir(parents=True)
        (model_path / "config.json").write_text(
            f'{{"architectures":["{model.architecture}"]}}',
            encoding="utf-8",
        )
    return XpoolConfig.from_mapping(
        {
            "vendor": {"model_base_uri": str(model_base_uri)},
            "atn": {"devices": [0]},
            "ffn": {"devices": [1, 2]},
            "models": [{"id": "external/model-not-owned-by-tests"}],
        },
        env={},
    )
