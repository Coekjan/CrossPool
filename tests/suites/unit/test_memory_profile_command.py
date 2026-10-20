from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import pytest

from xpool.cli.subcommands import memory_profile
from xpool.runtime.agent import AgentError
from xtest.harness.support.config import synthetic_config


def test_memory_profile_command_publishes_evidence_before_profile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    profile = object()
    evidence = SimpleNamespace(profile=profile)
    written = []
    config = synthetic_config()
    config = config.model_copy(
        update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": tmp_path / "memory.json"})}
    )
    monkeypatch.setattr(memory_profile, "profile_ffn_memory", lambda: evidence)
    monkeypatch.setattr(
        memory_profile,
        "write_memory_profile_evidence",
        lambda value: written.append(value) or tmp_path / "memory.evidence.json",
    )
    monkeypatch.setattr(memory_profile, "write_memory_calibration_profile", written.append)

    result = memory_profile.MemoryProfileCommand().run(argparse.Namespace(), config)

    assert result == 0
    assert written == [evidence, profile]
    output = capsys.readouterr().out
    assert str(tmp_path / "memory.evidence.json") in output
    assert str(tmp_path / "memory.json") in output


@pytest.mark.parametrize("failure_stage", ("evidence", "profile"))
def test_memory_profile_publication_failure_preserves_stage_and_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure_stage: str
) -> None:
    config = synthetic_config()
    profile_path = tmp_path / "memory.json"
    evidence_path = tmp_path / "memory.evidence.json"
    config = config.model_copy(
        update={"ffn": config.ffn.model_copy(update={"device_memory_calibration": profile_path})}
    )
    result = SimpleNamespace(profile=object())
    published = []

    def write_evidence(value: object) -> Path:
        if failure_stage == "evidence":
            raise RuntimeError("evidence publication denied")
        published.append(value)
        return evidence_path

    def write_profile(value: object) -> None:
        if failure_stage == "evidence":
            pytest.fail("runtime Profile must not be published after an evidence failure")
        raise RuntimeError("profile publication denied")

    monkeypatch.setattr(memory_profile, "profile_ffn_memory", lambda: result)
    monkeypatch.setattr(memory_profile, "write_memory_profile_evidence", write_evidence)
    monkeypatch.setattr(memory_profile, "write_memory_calibration_profile", write_profile)
    with pytest.raises(AgentError, match=f"{failure_stage} publication denied") as caught:
        memory_profile.MemoryProfileCommand().run(argparse.Namespace(), config)
    if failure_stage == "profile":
        assert published == [result]
        assert str(profile_path) in str(caught.value)
        assert str(evidence_path) in str(caught.value)
    else:
        assert published == []
