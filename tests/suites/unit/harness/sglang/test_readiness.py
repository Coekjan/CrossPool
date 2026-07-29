from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx

from tests.harness.runner.process import OwnedProcessGroup
from tests.harness.sglang.readiness import RESPONSE_EXCERPT_LIMIT, ReadinessEvidence, ReadinessTimeout


def test_readiness_evidence_retains_only_bounded_final_observation(tmp_path: Path) -> None:
    evidence = ReadinessEvidence("server health", "http://127.0.0.1:20000/health")
    response = httpx.Response(503, text="x" * (RESPONSE_EXCERPT_LIMIT + 100))
    process = cast(
        OwnedProcessGroup,
        SimpleNamespace(name="sglang-model", process=SimpleNamespace(poll=lambda: None)),
    )

    evidence.record_response(response)
    evidence.record_error(ConnectionError("connection reset"))
    evidence.finish(elapsed_seconds=1.25, processes=(process,))
    path = tmp_path / "readiness.json"
    evidence.write(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["attempt_count"] == 2
    assert payload["last_error_type"] == "ConnectionError"
    assert len(payload["last_response_excerpt"]) == RESPONSE_EXCERPT_LIMIT
    assert payload["process_statuses"] == {"sglang-model": None}
    assert "schema_version" not in payload


def test_readiness_timeout_carries_the_exact_evidence() -> None:
    evidence = ReadinessEvidence("daemon health", "http://127.0.0.1:20000/health", elapsed_seconds=3.0)

    error = ReadinessTimeout(evidence)

    assert error.evidence is evidence
    assert "3.000s" in str(error)
