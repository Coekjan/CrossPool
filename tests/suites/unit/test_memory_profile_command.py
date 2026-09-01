from __future__ import annotations

import argparse

import pytest

from tests.harness.support.config import synthetic_config
from xpool.cli.subcommands import memory_profile


def test_memory_profile_command_publishes_only_completed_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    profile = object()
    written = []
    monkeypatch.setattr(memory_profile, "profile_ffn_memory", lambda: profile)
    monkeypatch.setattr(memory_profile, "write_memory_calibration_profile", written.append)

    result = memory_profile.MemoryProfileCommand().run(argparse.Namespace(), synthetic_config())

    assert result == 0
    assert written == [profile]
