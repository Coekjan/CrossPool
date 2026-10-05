from __future__ import annotations

from pathlib import Path

import pytest

from xpool.fabric import FABRIC_UID_HEX_LENGTH, FabricUid
from xpool.native.ffn import ForwardMode, LayerKind
from xtest.harness.native.fabric.topology import run_fabric_topology


def test_topology_rejects_rows_above_selected_mode_capacity(tmp_path: Path) -> None:
    uid = FabricUid("0" * FABRIC_UID_HEX_LENGTH)
    with pytest.raises(ValueError, match="selected forward mode"):
        run_fabric_topology(
            uid,
            workdir=tmp_path / "topology",
            atnagent_count=1,
            ffnagent_count=1,
            executor_lane_count=1,
            forward_modes=(ForwardMode.DECODE,),
            layer_kind=LayerKind.DENSE,
            decode_payload_row_capacity=2,
            prefill_payload_row_capacity=4,
            payload_rows=(4,),
        )
