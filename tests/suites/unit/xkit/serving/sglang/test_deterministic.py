from pathlib import Path

import pytest
from pydantic import TypeAdapter

from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xkit.serving.sglang.deterministic import DeterministicConfig, SamplingRecord, SamplingSelection, install
from xtest.harness.support.config import TEST_MODEL_ID


def sampling_record(request_id: str = "request") -> SamplingRecord:
    return SamplingRecord(
        TEST_MODEL_ID,
        request_id,
        {"input_ids": [1, 2], "sampling_params": {"max_new_tokens": 2}},
        2,
        (3, 4),
        {"type": "length", "length": 2},
        None,
    )


def test_inactive_plugin_preserves_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "reference.jsonl"
    path.write_bytes(b"retained evidence")
    monkeypatch.setenv(DETERMINISTIC_MODE.name, "none")
    monkeypatch.setenv(DETERMINISTIC_PATH.name, str(path))
    install()
    assert path.read_bytes() == b"retained evidence"


def test_record_output_is_exclusive_and_replay_is_read_only(tmp_path: Path) -> None:
    path = tmp_path / "reference.jsonl"
    settings = DeterministicConfig.from_environment(
        {DETERMINISTIC_MODE.name: "record", DETERMINISTIC_PATH.name: str(path)}
    )
    settings.prepare()
    contents = TypeAdapter(SamplingRecord).dump_json(sampling_record()) + b"\n"
    path.write_bytes(contents)
    with pytest.raises(FileExistsError):
        settings.prepare()
    DeterministicConfig("replay", path).prepare()
    assert tuple(SamplingRecord.read(path)) == (sampling_record(),)
    assert path.read_bytes() == contents


@pytest.mark.parametrize("malformed", ["partial", "duplicate"])
def test_rejects_incomplete_or_duplicate_records(tmp_path: Path, malformed: str) -> None:
    path = tmp_path / "reference.jsonl"
    encoded = TypeAdapter(SamplingRecord).dump_json(sampling_record())
    path.write_bytes(encoded if malformed == "partial" else (encoded + b"\n") * 2)
    with pytest.raises(ValueError, match=r"incomplete|duplicate"):
        tuple(SamplingRecord.read(path))


@pytest.mark.parametrize(("prefix", "token"), [(1, None), (2, 3), (3, 4), (4, None)])
def test_choices_follow_logical_prefix_and_preserve_lookahead(prefix: int, token: int | None) -> None:
    assert SamplingSelection(0, sampling_record(), prefix, "decode", 0).token_id == token


@pytest.mark.parametrize("environment", [{DETERMINISTIC_MODE.name: "invalid"}, {DETERMINISTIC_MODE.name: "replay"}])
def test_rejects_invalid_activation(environment: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        DeterministicConfig.from_environment(environment)
