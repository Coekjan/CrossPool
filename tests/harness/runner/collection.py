"""Owner of one isolated pytest collection process."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from tests.harness.runner.plan import TestPlan


class CollectionFailure(RuntimeError):
    """Raised when isolated pytest discovery cannot publish a valid Test Plan."""


@dataclass(frozen=True, slots=True)
class CollectionWorker:
    """Own one isolated pytest collection process and its durable diagnostics."""

    repository_root: Path
    run_directory: Path
    selectors: tuple[str, ...]
    strict_requirements: bool

    def collect(self) -> TestPlan:
        """Run final pytest discovery and strictly parse its atomic output."""

        self.run_directory.mkdir(parents=True, exist_ok=False)
        plan_path = self.run_directory / "test-plan.json"
        log_path = self.run_directory / "collection.log"
        command = [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            *self.selectors,
            f"--xpool-test-plan={plan_path}",
        ]
        if self.strict_requirements:
            command.append("--strict-requirements")
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = str(self.repository_root / ".xpool-cache" / "pycache")
        try:
            completed = subprocess.run(
                command,
                cwd=self.repository_root,
                env=environment,
                capture_output=True,
                check=False,
                text=True,
            )
        except OSError as error:
            raise CollectionFailure(f"failed to start pytest Collection Worker: {error}") from error
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise CollectionFailure(f"pytest Collection Worker exited with code {completed.returncode}; see {log_path}")
        try:
            return TestPlan.read(plan_path)
        except ValueError as error:
            raise CollectionFailure(f"pytest Collection Worker published an invalid Test Plan: {error}") from error
