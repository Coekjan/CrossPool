"""Task-local config behavior for installed FFN qualification."""

from __future__ import annotations

from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch

from xkit.child import PythonChildFailure
from xpool.config import XpoolConfig
from xpool.fabric import FABRIC_UID_HEX_LENGTH
from xpool.model import ModelId
from xpool.native.ffn import DpRowLayout, ForwardMode, LayerKind, OutputRequirement
from xpool.runtime.transport import InstanceRankTransportProfile
from xtest.harness.native.ffn import topology
from xtest.harness.native.ffn.protocol import FfnInstanceCommand, FfnInstanceSpec, FfnInvocationSpec
from xtest.harness.native.ffn.qualification import execution_fabric_plan
from xtest.harness.native.ffn.topology import ffn_cluster_environment, materialize_ffn_cluster_launch
from xtest.harness.support.config import TEST_MODEL_ID, synthetic_config


def test_materialization_freezes_selected_topology_and_model_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOGNAME", "test-user")
    monkeypatch.setenv("USER", "test-user")
    model_root = tmp_path / "models"
    model_root.mkdir()
    workdir = tmp_path / "run"
    config = XpoolConfig.from_mapping(
        {
            "daemon": {"host": "127.0.0.1", "port": 19000},
            "vendor": {"model_base_uri": str(model_root)},
            "scheduler": {
                "slo": {"ttft_ms": 1000, "tbt_ms": 50},
                "ffn_concurrency": 2,
                "ffn_policy": "random",
                "ffn_random_seed": 17,
            },
            "atn": {"devices": [0, 1]},
            "ffn": {"devices": [2, 3, 4, 5], "loader": {"parallelism": 2}},
            "models": [
                {"id": "test/first", "path": str(tmp_path / "custom"), "ffn_tp_size": 4},
                {"id": "test/second", "ffn_tp_size": 2},
            ],
        },
        env=ffn_cluster_environment(config_path=workdir / "xpool.toml", observer_outdir=workdir / "observers"),
    )

    launch = materialize_ffn_cluster_launch(
        config=config,
        daemon_port=19001,
        workdir=workdir,
    )

    assert launch.config.atn.devices == [0, 1]
    assert launch.config.ffn.devices == [2, 3, 4, 5]
    assert tuple(model.ffn_tp_size for model in launch.config.models) == (4, 2)
    assert launch.config.scheduler.ffn_random_seed == 17
    assert launch.config.ffn.loader.parallelism == 2
    assert launch.config.model_path_of(ModelId("test/first")) == tmp_path / "custom"
    assert launch.config.model_path_of(ModelId("test/second")) == model_root / "test/second"
    assert launch.config.debug.graph_observer.enable and launch.config.debug.fabric_observer.enable
    assert "SGLANG_PLUGINS" not in launch.environment
    assert launch.environment["LOGNAME"] == "test-user"
    assert launch.environment["USER"] == "test-user"


def test_instance_failure_stops_later_invocations_and_retains_the_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class OwnerRetained(BaseException):
        pass

    def retain(seconds: float) -> None:
        raise OwnerRetained

    plan = execution_fabric_plan(
        uid="0" * FABRIC_UID_HEX_LENGTH,
        atnagent_count=1,
        ffnagent_count=1,
        ffn_tp_size=1,
        executor_lane_count=1,
        instance_count=1,
        execution_kind=LayerKind.DENSE,
        payload_dtype=torch.bfloat16,
    )
    profile = plan.instance_plans[0].ffn_profile
    spec = FfnInstanceSpec(
        environment={},
        model_id=TEST_MODEL_ID,
        rank=0,
        ffn_tp_size=1,
        ffn_profile=profile,
        transport=InstanceRankTransportProfile(
            hidden_size=profile.hidden_size,
            payload_row_capacity=1,
            atn_tp_rank=0,
            atn_tp_size=1,
            atn_dp_rank=0,
            atn_dp_size=1,
        ),
        invocations=tuple(
            FfnInvocationSpec(
                case_id=name,
                input_path=tmp_path / f"{name}.input",
                output_path=tmp_path / f"{name}.output",
                layer_ordinal=0,
                forward_mode=ForwardMode.DECODE,
                output_requirement=OutputRequirement.PER_RANK_COMPLETE,
                dp_row_layout=DpRowLayout.NONE,
                dp_rank_payload_rows=None,
            )
            for name in ("first", "second")
        ),
    )
    callbacks: list[Callable[[BaseException], None]] = []
    messages: list[object] = []
    loaded: list[Path] = []
    failure = RuntimeError("background invocation failed")
    runtime = SimpleNamespace(
        instance_index=0,
        wait_for_fabric_executable=lambda: plan,
        attach_arena_from_daemon=lambda: None,
        start_failure_monitor=lambda: None,
        publish_initialized=lambda listener: None,
        wait_for_ready=lambda: None,
        close=lambda: pytest.fail("failed Instance started normal cleanup"),
    )

    def start(*, on_failure: Callable[[BaseException], None], **kwargs: object) -> SimpleNamespace:
        callbacks.append(on_failure)
        return runtime

    def load(path: Path, *, device: str) -> dict[str, object]:
        loaded.append(path)
        return {"hidden_states": SimpleNamespace(to=lambda **kwargs: torch.ones(1, profile.hidden_size))}

    def invoke(hidden_states: torch.Tensor, *args: object) -> torch.Tensor:
        callbacks[0](failure)
        return hidden_states

    monkeypatch.setattr(topology, "os", SimpleNamespace(environ={}))
    monkeypatch.setattr(topology, "init_global_config", synthetic_config)
    monkeypatch.setattr(topology, "normalize_environment", lambda: None)
    monkeypatch.setattr(
        topology,
        "visible_uuids",
        lambda: tuple(f"GPU-00000000-0000-0000-0000-{index:012d}" for index in range(2)),
    )
    monkeypatch.setattr(topology.MpsEndpoint, "require_client", lambda self: None)
    monkeypatch.setattr(topology, "XpoolClient", lambda: SimpleNamespace(check_config=lambda: None, close=lambda: None))
    monkeypatch.setattr(topology.bootstrap, "init", lambda *args: None)
    monkeypatch.setattr(topology.devkit, "install", lambda: None)
    monkeypatch.setattr(topology, "InstanceRankRuntime", SimpleNamespace(start=start))
    monkeypatch.setattr(topology, "context_state_profile", lambda: None)
    monkeypatch.setattr(topology.safetensors.torch, "load_file", load)
    monkeypatch.setattr(topology, "ffn_shim", invoke)
    monkeypatch.setattr(topology, "time", SimpleNamespace(sleep=retain))
    connection = SimpleNamespace(send=messages.append, recv=lambda: FfnInstanceCommand.RUN)
    with pytest.raises(OwnerRetained):
        topology.run_ffn_instance(cast(Connection, connection), spec=spec)

    assert loaded == [spec.invocations[0].input_path]
    assert spec.invocations[0].output_path.is_file()
    assert not spec.invocations[1].output_path.exists()
    assert any(isinstance(message, PythonChildFailure) and str(failure) in message.traceback for message in messages)
