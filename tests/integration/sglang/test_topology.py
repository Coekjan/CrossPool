from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from xpool.config import ConfigError, TopologyError
from xpool.integrations.sglang import topology as sglang_topology
from xpool.integrations.sglang.topology import (
    AtnKind,
    SglangModelMetadata,
    derive_parallel_policy,
    load_model_spec,
)


def test_gqa_atn_policy_is_derived_from_sglang_metadata(
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
    policy = derive_parallel_policy(spec, atn_device_count=2, ffn_tp_size=1)

    assert spec.atn_kind is AtnKind.GQA
    assert policy.atn_kind is AtnKind.GQA
    assert policy.sglang_tp_size == 2
    assert policy.sglang_dp_size == 1
    assert policy.atn_tp_size == 2
    assert policy.atn_dp_size == 1
    assert policy.enable_dp_atn is False


def test_mla_atn_policy_rejects_dp_attention_until_reduce_scatter_exists(
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
        physical_kv_lanes=1,
    )
    model_dir = write_model_config(
        tmp_path / "deepseek-ai" / "DeepSeek-V2-Lite-Chat",
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
    spec = load_model_spec(model_dir, model_id="deepseek-ai/DeepSeek-V2-Lite-Chat")

    with pytest.raises(TopologyError, match="unsupported until FFN reduce-scatter"):
        derive_parallel_policy(spec, atn_device_count=2, ffn_tp_size=1)


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
    policy = derive_parallel_policy(spec, atn_device_count=1, ffn_tp_size=3)

    assert spec.atn_kind is AtnKind.MQA
    assert spec.physical_kv_lanes == 1
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
        atn_heads=32,
        kv_heads=4,
        atn_kind=AtnKind.GQA,
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
        derive_parallel_policy(spec, atn_device_count=2, ffn_tp_size=3)


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
        physical_kv_lanes=16,
    )

    spec = sglang_topology.parse_model_config(
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
    physical_kv_lanes: int,
) -> None:
    def load_metadata(config_path: Path, *, model_id: str) -> SglangModelMetadata:
        return SglangModelMetadata(
            family=family or model_id,
            hidden_size=hidden_size,
            num_atn_heads=atn_heads,
            num_key_value_heads=kv_heads,
            atn_kind=atn_kind,
            physical_kv_lanes=physical_kv_lanes,
        )

    monkeypatch.setattr(sglang_topology, "load_sglang_model_metadata", load_metadata)
