from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from xpool.config import ConfigError, TopologyError
from xpool.integrations.sglang import topology as sglang_topology
from xpool.integrations.sglang.topology import (
    AttentionKind,
    SglangModelMetadata,
    derive_parallel_policy,
    load_model_spec,
)


def test_gqa_attention_policy_is_derived_from_sglang_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=4,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=4,
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

    spec = load_model_spec(model_dir, model_id="qwen")
    policy = derive_parallel_policy(spec, attention_device_count=2, ffn_tp_size=1)

    assert spec.attention_kind is AttentionKind.GQA
    assert policy.attention_kind is AttentionKind.GQA
    assert policy.sglang_tp_size == 2
    assert policy.sglang_dp_size == 1
    assert policy.attention_tp_size == 2
    assert policy.attention_dp_size == 1
    assert policy.enable_dp_attention is False


def test_mla_attention_policy_rejects_implicit_attention_dp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="deepseek_v2",
        hidden_size=2048,
        attention_heads=16,
        kv_heads=16,
        attention_kind=AttentionKind.MLA,
        physical_kv_lanes=1,
    )
    model_dir = write_model_config(
        tmp_path / "deepseek-v2-lite-chat",
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
    spec = load_model_spec(model_dir, model_id="deepseek-v2-lite-chat")

    with pytest.raises(TopologyError, match="attention data parallelism is not supported"):
        derive_parallel_policy(spec, attention_device_count=2, ffn_tp_size=1)


def test_regular_mqa_when_sglang_reports_non_mla(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="synthetic_mqa",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=1,
        attention_kind=AttentionKind.MQA,
        physical_kv_lanes=1,
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

    spec = load_model_spec(model_dir, model_id="synthetic-mqa")
    policy = derive_parallel_policy(spec, attention_device_count=1, ffn_tp_size=3)

    assert spec.attention_kind is AttentionKind.MQA
    assert spec.physical_kv_lanes == 1
    assert policy.attention_tp_size == 1


def test_zero_kv_heads_from_sglang_metadata_fail_fast(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="bad",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=0,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=0,
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
        load_model_spec(model_dir, model_id="bad")


def test_ffn_tp_divisibility_is_checked_during_policy_derivation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="qwen2",
        hidden_size=4096,
        attention_heads=32,
        kv_heads=4,
        attention_kind=AttentionKind.GQA,
        physical_kv_lanes=4,
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
    spec = load_model_spec(model_dir, model_id="bad-ffn")

    with pytest.raises(TopologyError, match="does not divide dense intermediate size"):
        derive_parallel_policy(spec, attention_device_count=2, ffn_tp_size=3)


def test_model_config_preserves_explicit_zero_num_experts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patch_sglang_metadata(
        monkeypatch,
        family="dense",
        hidden_size=2048,
        attention_heads=16,
        kv_heads=16,
        attention_kind=AttentionKind.MHA,
        physical_kv_lanes=16,
    )

    spec = sglang_topology._parse_model_config(
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
    attention_heads: int,
    kv_heads: int,
    attention_kind: AttentionKind,
    physical_kv_lanes: int,
) -> None:
    def load_metadata(_config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=family or model_id,
            hidden_size=hidden_size,
            num_attention_heads=attention_heads,
            num_key_value_heads=kv_heads,
            attention_kind=attention_kind,
            physical_kv_lanes=physical_kv_lanes,
        )

    monkeypatch.setattr(sglang_topology, "_load_sglang_model_metadata", load_metadata)
