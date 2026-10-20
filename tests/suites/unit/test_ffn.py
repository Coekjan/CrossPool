from __future__ import annotations

from xpool.ffn import GatedFfnCheckpointKeys, MoeFfnCheckpointKeys


def test_checkpoint_tensor_keys_keep_router_and_expert_semantic_order() -> None:
    first, second, shared = (
        GatedFfnCheckpointKeys(
            gate_weight_key=f"{prefix}.gate",
            up_weight_key=f"{prefix}.up",
            down_weight_key=f"{prefix}.down",
        )
        for prefix in ("first", "second", "shared")
    )
    assert first.tensor_keys() == ("first.gate", "first.up", "first.down")
    keys = MoeFfnCheckpointKeys(
        router_weight_key="router",
        router_correction_bias_key="correction",
        routed_experts=(first, second),
        shared_expert=shared,
    )
    expected = (
        "router",
        "correction",
        "first.gate",
        "first.up",
        "first.down",
        "second.gate",
        "second.up",
        "second.down",
        "shared.gate",
        "shared.up",
        "shared.down",
    )
    assert keys.tensor_keys() == expected
    assert keys.model_copy(update={"router_correction_bias_key": None}).tensor_keys() == (expected[:1] + expected[2:])
