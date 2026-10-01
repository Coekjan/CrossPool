from pathlib import Path

import pytest

from xkit.deployment import resolve_deployment_path
from xpool.model import ModelId
from xtest.harness.support.config import TEST_MODEL_ID


def test_deployment_reference_binds_the_distinct_model_set_for_both_catalogues(tmp_path: Path) -> None:
    models = (ModelId("Qwen/Qwen2.5-0.5B"), ModelId("Qwen/Qwen3-0.6B"))
    expected = tmp_path / "configs/deployments/Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B/atn1-ffn1-lanes2.toml"
    assert resolve_deployment_path(tmp_path / "tests/tests.toml", models, "atn1-ffn1-lanes2") == expected
    assert (
        resolve_deployment_path(tmp_path / "benches/benches.toml", (*reversed(models), models[0]), "atn1-ffn1-lanes2")
        == expected
    )


@pytest.mark.parametrize(
    "reference", ["", ".", "..", "../scene", "dir/scene", "dir//scene", "/scene", "dir\\scene", "scene.toml"]
)
def test_deployment_reference_requires_a_filename_stem(tmp_path: Path, reference: str) -> None:
    with pytest.raises(ValueError, match="suffix-free basename"):
        resolve_deployment_path(tmp_path / "tests/tests.toml", (TEST_MODEL_ID,), reference)
