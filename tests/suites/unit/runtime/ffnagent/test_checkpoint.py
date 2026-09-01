from __future__ import annotations

from pathlib import Path

import numpy
import pytest
from safetensors.numpy import save_file

from xpool.runtime.ffnagent import checkpoint


def test_single_file_checkpoint_uses_safetensors_header(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "model.safetensors"
    save_file({"model.layers.0.mlp.gate_proj.weight": numpy.zeros((1,), dtype=numpy.float32)}, checkpoint_path)

    assert checkpoint.read_checkpoint_key_view(tmp_path) == {
        "model.layers.0.mlp.gate_proj.weight": checkpoint_path,
    }


def test_unindexed_checkpoint_rejects_ambiguous_files(tmp_path: Path) -> None:
    for name in ("model.safetensors", "other.safetensors"):
        save_file({name: numpy.zeros((1,), dtype=numpy.float32)}, tmp_path / name)

    with pytest.raises(ValueError, match="exactly one unindexed canonical"):
        checkpoint.read_checkpoint_key_view(tmp_path)
