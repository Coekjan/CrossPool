"""Bounded readiness evidence for one E2E attempt phase."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from tests.harness.runner.process import OwnedProcessGroup

RESPONSE_EXCERPT_LIMIT = 2_000


@dataclass(slots=True)
class ReadinessEvidence:
    """Final bounded observation from one readiness polling phase."""

    name: str
    url: str
    attempt_count: int = 0
    elapsed_seconds: float = 0.0
    last_error_type: str | None = None
    last_error_message: str | None = None
    last_status_code: int | None = None
    last_response_excerpt: str | None = None
    process_statuses: dict[str, int | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.url:
            raise ValueError("readiness evidence name and URL must be nonempty")

    def record_error(self, error: BaseException) -> None:
        """Replace the final transport or decoding error."""

        self.attempt_count += 1
        self.last_error_type = type(error).__name__
        self.last_error_message = str(error)[:RESPONSE_EXCERPT_LIMIT]

    def record_response(self, response: httpx.Response) -> None:
        """Replace the final HTTP observation."""

        self.attempt_count += 1
        self.last_status_code = response.status_code
        self.last_response_excerpt = response.text[:RESPONSE_EXCERPT_LIMIT]

    def finish(self, *, elapsed_seconds: float, processes: Sequence[OwnedProcessGroup]) -> None:
        """Capture elapsed time and the final process states."""

        self.elapsed_seconds = elapsed_seconds
        self.process_statuses = {process.name: process.process.poll() for process in processes}

    def write(self, path: Path) -> None:
        """Atomically write one versionless readiness record."""

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


class ReadinessTimeout(RuntimeError):
    """Raised when one readiness deadline expires with final evidence."""

    def __init__(self, evidence: ReadinessEvidence) -> None:
        self.evidence = evidence
        super().__init__(
            f"timed out waiting for {evidence.name} after {evidence.elapsed_seconds:.3f}s; evidence={evidence.url}"
        )
