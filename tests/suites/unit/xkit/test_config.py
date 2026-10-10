import argparse
from pathlib import Path

import pytest
import tomli_w
from pydantic import ValidationError

import xkit.config
from xkit.config import ToolConfigRecord, XpoolDevConfig, assemble_config, resolve_model_weights
from xpool.config import ConfigError, ConfigSource
from xpool.model import ModelId


@pytest.fixture
def development_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(xkit.config, "global_config", None)
    for name in ("XKIT_CONFIG", "XPOOL_CONFIG", "XPOOL_CACHE_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_development_cli_preserves_file_defaults_and_boolean_overrides(tmp_path: Path) -> None:
    path = tmp_path / "development.toml"
    path.write_text(
        '[xtest]\ncatalog = "../tests.toml"\nsuites = ["unit"]\nstrict_requirements = true\n',
        encoding="utf-8",
    )
    parser = argparse.ArgumentParser()
    XpoolDevConfig.add_cli_args(parser, names=("xtest_suites", "xtest_strict_requirements"))
    assert vars(parser.parse_args([])) == {}
    config = XpoolDevConfig.from_file(path, cli=vars(parser.parse_args(["--no-strict-requirements"])))
    assert config.xtest.catalog == tmp_path.parent / "tests.toml"
    assert config.xtest.suites == ("unit",)
    assert config.xtest.strict_requirements is False
    sources = {record["name"]: record for record in config.sources}
    assert sources["xtest.catalog"]["source"] is ConfigSource.CONFIG
    assert sources["xtest.strict_requirements"]["source"] is ConfigSource.CLI
    assert XpoolDevConfig.from_file(
        path, cli=vars(parser.parse_args(["--suite", "integration", "--suite", "models"]))
    ).xtest.suites == ("integration", "models")


@pytest.mark.usefixtures("development_environment")
def test_development_snapshot_preserves_settings_and_sources_without_rereading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "runtime.toml"
    runtime.write_text('cache_root = "storage"\n', encoding="utf-8")
    development = tmp_path / "development.toml"
    development.write_text('[xtest]\nsuites = ["unit"]\n', encoding="utf-8")
    monkeypatch.setenv("XPOOL_CONFIG", str(runtime))
    monkeypatch.setenv("XKIT_CONFIG", str(development))
    configured = xkit.config.init_global_config(cli={"width": 4.0, "xbench_report_ppi": 150})
    record = ToolConfigRecord.model_validate_json(configured.record().model_dump_json())
    assert configured.cache_root == tmp_path / "storage"
    assert record.cache_source["source"] is ConfigSource.CONFIG
    assert record.development_config_path == development
    assert record.runtime_config_path == runtime
    assert configured.xbench.report.ttft.width_inches == 4.0
    assert configured.xbench.report.itl.width_inches == 4.0
    assert configured.xbench.report.throughput.width_inches == 4.0

    development.write_text("[malformed", encoding="utf-8")
    runtime.write_text("[malformed", encoding="utf-8")
    monkeypatch.setenv("XPOOL_CACHE_ROOT", str(tmp_path / "different"))
    monkeypatch.setattr(xkit.config, "global_config", None)
    worker = xkit.config.init_global_config(resolved=XpoolDevConfig.from_record(record))
    assert worker.record().model_dump(mode="json") == record.model_dump(mode="json")
    assert worker.xtest.suites == ("unit",)
    assert worker.xbench.report.ppi == 150


@pytest.mark.usefixtures("development_environment")
def test_development_bootstrap_uses_defaults_but_rejects_explicit_invalid_file(
    tmp_path: Path,
) -> None:
    configured = xkit.config.init_global_config()
    assert configured.xtest.suites == ("cext", "unit", "integration", "e2e")
    assert configured.cache_root == (Path.cwd() / ".xpool-cache").resolve()
    assert configured.keep_runs == 20
    assert configured.xbench.repetitions == 1
    with pytest.raises(FileNotFoundError):
        xkit.config.init_global_config(config_path=tmp_path / "missing.toml")


@pytest.mark.parametrize(
    "payload",
    [
        {"keep_runs": True},
        {"xbench": {"repetitions": True}},
        {"xbench": {"repetitions": 0}},
        {"xbench": {"repetitions": 1.5}},
        {"xbench": {"report": {"ppi": 1.5}}},
        {"xbench": {"report": {"ttft": {"columns": True}}}},
        {"xbench": {"report": {"ttft": {"width_inches": float("inf")}}}},
        {"xbench": {"report": {"formats": []}}},
        {"xbench": {"report": {"formats": ["png", "png"]}}},
    ],
)
def test_development_presentation_and_retention_validate_values(payload: dict[str, object]) -> None:
    with pytest.raises((ConfigError, ValidationError)):
        XpoolDevConfig.from_mapping(payload)


def write_base(path: Path) -> Path:
    path.write_text(
        tomli_w.dumps(
            {
                "vendor": {"model_base_uri": str(path.parent / "models")},
                "atn": {"devices": [0], "device_memory_utilization": 0.8},
                "ffn": {
                    "devices": [1, 2],
                    "device_memory_extra_margin_bytes": 4096,
                    "device_memory_calibration": str(path.parent / "calibration.json"),
                    "loader": {"parallelism": 2},
                    "placement": {"parallelism": 3, "timeout_seconds": 90},
                },
                "scheduler": {"slo": {"ttft_ms": 1000, "tbt_ms": 50}},
                "models": [
                    {
                        "id": "test/one",
                        "path": str(path.parent / "custom"),
                        "ffn_tp_size": 2,
                        "slo": {"ttft_ms": 2000, "tbt_ms": 100},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.parametrize("scene_utilization", [None, 0.6])
def test_scene_owns_geometry_and_slo_while_inheriting_machine_policy(
    tmp_path: Path,
    scene_utilization: float | None,
) -> None:
    base = write_base(tmp_path / "runtime.toml")
    scene = tmp_path / "deployment.toml"
    atn: dict[str, object] = {"devices": [0]}
    if scene_utilization is not None:
        atn["device_memory_utilization"] = scene_utilization
    scene.write_text(
        tomli_w.dumps(
            {
                "atn": atn,
                "ffn": {"devices": [1, 2]},
                "scheduler": {"ffn_concurrency": 2, "slo": {"ttft_ms": 500, "tbt_ms": 20}},
                "models": [
                    {"id": model_id, "atn_tp_size": 1, "atn_dp_size": 1, "ffn_tp_size": 1}
                    for model_id in ("test/fallback", "test/one")
                ],
            }
        )
    )
    config = assemble_config(
        (ModelId("test/fallback"), ModelId("test/one")),
        deployment=scene,
        runtime_config=base,
        cli={"logging_level": "debug"},
        env={
            "XPOOL_CONFIG": "/not-selected.toml",
            "XPOOL_LOG_LEVEL": "warning",
            "XPOOL_DEBUG_GRAPH_OBSERVER_ENABLE": "1",
            "XPOOL_DEBUG_GRAPH_OBSERVER_OUTDIR": str(tmp_path / "observers"),
        },
    )
    assert [model.id for model in config.models] == [ModelId("test/fallback"), ModelId("test/one")]
    assert config.model_path_of(ModelId("test/fallback")) == tmp_path / "models/test/fallback"
    assert config.model_path_of(ModelId("test/one")) == tmp_path / "custom"
    for model_id in (ModelId("test/fallback"), ModelId("test/one")):
        checkpoint = config.model_path_of(model_id)
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        assert resolve_model_weights(config, model_id) == checkpoint
    assert config.models[1].ffn_tp_size == 1
    assert config.models[1].atn_tp_size == 1 and config.models[1].atn_dp_size == 1
    assert config.models[1].slo is None
    assert config.scheduler.slo.ttft_ms == 500
    assert config.atn.device_memory_utilization == (0.8 if scene_utilization is None else scene_utilization)
    assert config.ffn.device_memory_calibration == tmp_path / "calibration.json"
    assert config.ffn.device_memory_extra_margin_bytes == 4096
    assert config.ffn.loader.parallelism == 2
    assert config.ffn.placement.timeout_seconds == 90
    assert config.logging.level == "debug" and config.debug.graph_observer.enable
    sources = {record["name"]: record["source"] for record in config.sources}
    assert sources["logging.level"] is ConfigSource.CLI
    assert sources["logging.color"] is ConfigSource.DEFAULT
    assert sources["scheduler.ffn_concurrency"] is ConfigSource.CONFIG


def test_caller_projection_changes_model_policy_without_losing_its_path(tmp_path: Path) -> None:
    base = write_base(tmp_path / "runtime.toml")
    config = assemble_config(
        (ModelId("test/one"),),
        env={"XPOOL_CONFIG": str(base)},
        overrides={
            "models": [{"id": "test/one", "ffn_tp_size": 1, "slo": None}],
            "scheduler": {"slo": {"ttft_ms": 500, "tbt_ms": 20}},
        },
    )
    assert config.model_path_of(ModelId("test/one")) == tmp_path / "custom"
    assert config.models[0].ffn_tp_size == 1 and config.models[0].slo is None
    assert config.scheduler.slo.ttft_ms == 500


def test_deployment_rejects_host_policy_and_unselected_model_overrides(tmp_path: Path) -> None:
    base = write_base(tmp_path / "runtime.toml")
    scene = tmp_path / "deployment.toml"
    scene.write_text('[vendor]\nmodel_base_uri = "/host-specific"\n')
    with pytest.raises(ConfigError, match="portable"):
        assemble_config((ModelId("test/one"),), deployment=scene, runtime_config=base, env={})
    scene.write_text(
        "[atn]\ndevices = [0]\n[ffn]\ndevices = [1, 2]\n[scheduler]\nffn_concurrency = 1\n"
        "slo = { ttft_ms = 1000, tbt_ms = 50 }\n"
        '[[models]]\nid = "test/unknown"\natn_tp_size = 1\natn_dp_size = 1\nffn_tp_size = 1\n'
    )
    with pytest.raises(ConfigError, match="match the selected"):
        assemble_config((ModelId("test/one"),), deployment=scene, runtime_config=base, env={})


def test_complete_scene_replaces_workspace_geometry_on_a_smaller_fleet(tmp_path: Path) -> None:
    base = write_base(tmp_path / "runtime.toml")
    scene = tmp_path / "deployment.toml"
    scene.write_text(
        "[atn]\ndevices = [0]\n[ffn]\ndevices = [1]\n[scheduler]\nffn_concurrency = 1\n"
        "slo = { ttft_ms = 1000, tbt_ms = 50 }\n"
        '[[models]]\nid = "test/one"\natn_tp_size = 1\natn_dp_size = 1\nffn_tp_size = 1\n'
    )
    config = assemble_config(
        (ModelId("test/one"),),
        deployment=scene,
        runtime_config=base,
        env={},
    )
    assert config.atn.devices == [0]
    assert config.ffn.devices == [1]
    assert config.models[0].ffn_tp_size == 1
