from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.sglang.graph import SglangGraphMode
from tests.harness.sglang.parity import TokenOutput, TokenParityArtifact, assert_token_parity


def artifact(mode_index: int, output_ids: tuple[int, ...] = (1, 2, 3)) -> TokenParityArtifact:
    return TokenParityArtifact(
        group="case",
        graph_settings=tuple(SglangGraphMode)[mode_index].settings(),
        outputs=(TokenOutput("model", output_ids),),
    )


def test_token_parity_artifact_round_trips_and_rejects_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    expected = artifact(0)

    expected.write(path)

    assert TokenParityArtifact.read(path) == expected
    with pytest.raises(FileExistsError):
        expected.write(path)


def test_token_parity_compares_every_mode_with_eager_output() -> None:
    assert_token_parity((artifact(0), artifact(1), artifact(2)))

    with pytest.raises(AssertionError, match="token parity failed"):
        assert_token_parity((artifact(0), artifact(1, (4, 5, 6))))
