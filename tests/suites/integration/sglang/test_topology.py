from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest
from sglang.srt.server_args import ServerArgs

from tests.harness.support.sglang.fakes import server_args
from xpool.config import ConfigError, TopologyError
from xpool.integrations.sglang.topology import (
    AtnKind,
    ModelSpec,
    ParallelPolicy,
    SglangModelMetadata,
)


def test_gqa_atn_policy_retains_resolved_sglang_topology(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        atn_heads=32,
        kv_heads=4,
        atn_kind=AtnKind.GQA,
    )
    model_dir = write_model_config(
        tmp_path / "qwen",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 11010,
        },
    )

    spec = ModelSpec.load(model_dir, model_id="qwen")
    policy = ParallelPolicy.from_server_args(
        spec,
        server_args(tp_size=2),
        atnagent_count=2,
        supports_dp_attention=False,
    )

    assert spec.atn_kind is AtnKind.GQA
    assert policy.worker_world_size == 2
    assert policy.atn_tp_size == 2
    assert policy.atn_dp_size == 1


@pytest.mark.parametrize(
    ("tp_size", "dp_size", "atnagent_count", "expected_atn_tp_size"),
    [(1, 1, 1, 1), (2, 1, 2, 2), (2, 2, 2, 1)],
)
def test_mla_atn_policy_accepts_independent_tp_or_dp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tp_size: int,
    dp_size: int,
    atnagent_count: int,
    expected_atn_tp_size: int,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v2",
        hidden_size=2048,
        atn_heads=16,
        kv_heads=16,
        atn_kind=AtnKind.MLA,
    )
    model_dir = write_model_config(
        tmp_path / "model",
        {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "intermediate_size": 11010,
            "moe_intermediate_size": 1410,
        },
    )
    spec = ModelSpec.load(model_dir, model_id="test-model")

    policy = ParallelPolicy.from_server_args(
        spec,
        server_args(tp_size=tp_size, dp_size=dp_size, enable_dp_attention=dp_size > 1),
        atnagent_count=atnagent_count,
        supports_dp_attention=True,
    )

    assert policy.atn_tp_size == expected_atn_tp_size
    assert policy.atn_dp_size == dp_size


def test_mla_atn_policy_rejects_combined_tp_by_dp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v2",
        hidden_size=2048,
        atn_heads=16,
        kv_heads=16,
        atn_kind=AtnKind.MLA,
    )
    model_dir = write_model_config(
        tmp_path / "model",
        {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
        },
    )
    spec = ModelSpec.load(model_dir, model_id="test-model")

    with pytest.raises(TopologyError, match="combined attention TP-by-DP"):
        ParallelPolicy.from_server_args(
            spec,
            server_args(tp_size=4, dp_size=2, enable_dp_attention=True),
            atnagent_count=4,
            supports_dp_attention=True,
        )


def test_regular_mqa_when_sglang_reports_non_mla(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="synthetic_mqa",
        hidden_size=4096,
        atn_heads=32,
        kv_heads=1,
        atn_kind=AtnKind.MQA,
    )
    model_dir = write_model_config(
        tmp_path / "synthetic-mqa",
        {
            "model_type": "synthetic_mqa",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 1,
            "qk_rope_head_dim": 64,
            "intermediate_size": 11010,
        },
    )

    spec = ModelSpec.load(model_dir, model_id="synthetic-mqa")
    policy = ParallelPolicy.from_server_args(
        spec,
        server_args(),
        atnagent_count=1,
        supports_dp_attention=False,
    )

    assert spec.atn_kind is AtnKind.MQA
    assert policy.atn_tp_size == 1


def test_zero_kv_heads_from_sglang_metadata_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="bad",
        hidden_size=4096,
        atn_heads=32,
        kv_heads=0,
        atn_kind=AtnKind.GQA,
    )
    model_dir = write_model_config(
        tmp_path / "bad",
        {
            "model_type": "bad",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 0,
            "intermediate_size": 11010,
        },
    )

    with pytest.raises(ConfigError, match="num_key_value_heads"):
        ModelSpec.load(model_dir, model_id="bad")


def test_attention_head_divisibility_is_checked_against_resolved_tp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        atn_heads=32,
        kv_heads=4,
        atn_kind=AtnKind.GQA,
    )
    model_dir = write_model_config(
        tmp_path / "bad-ffn",
        {
            "model_type": "qwen2",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "intermediate_size": 4,
        },
    )
    spec = ModelSpec.load(model_dir, model_id="bad-ffn")

    with pytest.raises(TopologyError, match="does not divide query heads"):
        ParallelPolicy.from_server_args(
            spec,
            server_args(tp_size=3),
            atnagent_count=3,
            supports_dp_attention=False,
        )


def test_policy_rejects_sglang_world_that_does_not_cover_atnagents(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen3",
        hidden_size=4096,
        atn_heads=32,
        kv_heads=8,
        atn_kind=AtnKind.GQA,
    )
    model_dir = write_model_config(
        tmp_path / "qwen",
        {
            "model_type": "qwen3",
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
        },
    )
    spec = ModelSpec.load(model_dir, model_id=model_dir.name)

    with pytest.raises(TopologyError, match="must equal configured AtnAgent count"):
        ParallelPolicy.from_server_args(
            spec,
            server_args(tp_size=1),
            atnagent_count=2,
            supports_dp_attention=False,
        )


@pytest.mark.parametrize(
    "args",
    [
        server_args(tp_size=2, dp_size=2, enable_dp_attention=False),
        server_args(tp_size=1, dp_size=1, enable_dp_attention=True),
    ],
)
def test_policy_rejects_inconsistent_dp_attention_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    args: ServerArgs,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v2",
        hidden_size=2048,
        atn_heads=16,
        kv_heads=16,
        atn_kind=AtnKind.MLA,
    )
    model_dir = write_model_config(
        tmp_path / "deepseek",
        {
            "model_type": "deepseek_v2",
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
        },
    )
    spec = ModelSpec.load(model_dir, model_id="deepseek")

    with pytest.raises(TopologyError, match="enable_dp_attention"):
        ParallelPolicy.from_server_args(
            spec,
            args,
            atnagent_count=args.tp_size,
            supports_dp_attention=True,
        )


@pytest.mark.parametrize("atn_kind", [AtnKind.GQA, AtnKind.MLA])
def test_policy_rejects_dp_attention_without_adapter_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    atn_kind: AtnKind,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="unsupported",
        hidden_size=2048,
        atn_heads=16,
        kv_heads=4,
        atn_kind=atn_kind,
    )
    model_dir = write_model_config(tmp_path / "unsupported", {"model_type": "unsupported"})
    spec = ModelSpec.load(model_dir, model_id="unsupported")

    with pytest.raises(TopologyError, match="adapter does not support"):
        ParallelPolicy.from_server_args(
            spec,
            server_args(tp_size=2, dp_size=2, enable_dp_attention=True),
            atnagent_count=2,
            supports_dp_attention=False,
        )


def test_gqa_policy_accepts_dp_attention_with_adapter_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen3_moe",
        hidden_size=2048,
        atn_heads=32,
        kv_heads=4,
        atn_kind=AtnKind.GQA,
    )
    model_dir = write_model_config(tmp_path / "qwen3-moe", {"model_type": "qwen3_moe"})
    spec = ModelSpec.load(model_dir, model_id="qwen3-moe")

    policy = ParallelPolicy.from_server_args(
        spec,
        server_args(tp_size=2, dp_size=2, enable_dp_attention=True),
        atnagent_count=2,
        supports_dp_attention=True,
    )

    assert policy.worker_world_size == 2
    assert policy.atn_tp_size == 1
    assert policy.atn_dp_size == 2


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_gqa_attention_tp_accepts_kv_sharding_and_replication(size: int) -> None:
    spec = ModelSpec(
        model_id="qwen3-moe",
        family="qwen3_moe",
        hidden_size=2048,
        num_atn_heads=32,
        num_key_value_heads=4,
        atn_kind=AtnKind.GQA,
        raw_config_path=Path("/models/qwen3-moe/config.json"),
    )

    spec.validate_attention_tp(size)


@pytest.mark.parametrize("size", [3, 6])
def test_gqa_attention_tp_rejects_invalid_head_geometry(size: int) -> None:
    spec = ModelSpec(
        model_id="qwen3-moe",
        family="qwen3_moe",
        hidden_size=2048,
        num_atn_heads=32,
        num_key_value_heads=4,
        atn_kind=AtnKind.GQA,
        raw_config_path=Path("/models/qwen3-moe/config.json"),
    )

    with pytest.raises(TopologyError):
        spec.validate_attention_tp(size)


def test_mla_attention_tp_ignores_ordinary_kv_head_geometry() -> None:
    spec = ModelSpec(
        model_id="deepseek",
        family="deepseek_v2",
        hidden_size=2048,
        num_atn_heads=16,
        num_key_value_heads=3,
        atn_kind=AtnKind.MLA,
        raw_config_path=Path("/models/deepseek/config.json"),
    )

    spec.validate_attention_tp(2)


def test_model_config_preserves_explicit_zero_num_experts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="dense",
        hidden_size=2048,
        atn_heads=16,
        kv_heads=16,
        atn_kind=AtnKind.MHA,
    )

    spec = ModelSpec.from_raw(
        {"num_experts": 0, "n_routed_experts": 64},
        model_id="dense-model",
        config_path=tmp_path / "config.json",
    )

    assert spec.num_experts == 0


def write_model_config(path: Path, payload: Mapping[str, object]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")
    return path


def patch_sglang_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    family: str,
    hidden_size: int,
    atn_heads: int,
    kv_heads: int,
    atn_kind: AtnKind,
) -> None:
    def load_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=family or model_id,
            hidden_size=hidden_size,
            num_atn_heads=atn_heads,
            num_key_value_heads=kv_heads,
            atn_kind=atn_kind,
        )

    monkeypatch.setattr(SglangModelMetadata, "load", load_metadata)
