"""Strict projection of pytest JUnit XML into task result values."""

from __future__ import annotations

import math
import xml.etree.ElementTree
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol


class ExpectedPytestCase(Protocol):
    """Minimum collected-case projection required for JUnit matching."""

    nodeid: str


class PytestCaseStatus(StrEnum):
    """One semantic pytest testcase outcome recovered from JUnit."""

    PASSED = "passed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PytestCaseReport:
    """One expected nodeid and its validated pytest outcome."""

    nodeid: str
    status: PytestCaseStatus
    detail: str | None
    elapsed_seconds: float | None = None

    @classmethod
    def from_elements(cls, nodeid: str, elements: Sequence[xml.etree.ElementTree.Element]) -> PytestCaseReport:
        """Combine phase records into one item, preserving errors and total time."""

        outcomes = tuple(
            child for element in elements for child in element if child.tag in {"failure", "error", "skipped"}
        )
        unknown = tuple(
            child.tag
            for element in elements
            for child in element
            if child.tag not in {"failure", "error", "skipped", "system-out", "system-err", "properties"}
        )
        if unknown:
            raise ValueError(f"JUnit testcase {nodeid!r} contains unknown elements: {unknown}")
        elapsed_seconds = 0.0
        elapsed_complete = True
        for element in elements:
            elapsed_raw = element.get("time")
            if elapsed_raw is None:
                elapsed_complete = False
                continue
            try:
                elapsed = float(elapsed_raw)
            except ValueError as error:
                raise ValueError(f"JUnit testcase {nodeid!r} has invalid elapsed time") from error
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError(f"JUnit testcase {nodeid!r} has invalid elapsed time")
            elapsed_seconds += elapsed
        if not math.isfinite(elapsed_seconds):
            raise ValueError(f"JUnit testcase {nodeid!r} has invalid elapsed time")
        details: list[str] = []
        for outcome in outcomes:
            detail = "\n".join(part for part in (outcome.get("message"), (outcome.text or "").strip()) if part)
            if outcome.tag == "skipped":
                if outcome.get("type") == "pytest.xfail":
                    raise ValueError(f"JUnit testcase {nodeid!r} contains prohibited pytest.xfail outcome")
                if not detail:
                    raise ValueError(f"JUnit skipped testcase {nodeid!r} has no reason")
            if detail:
                details.append(detail)
        if any(outcome.tag in {"failure", "error"} for outcome in outcomes):
            status = PytestCaseStatus.FAILED
        else:
            status = PytestCaseStatus.SKIPPED if outcomes else PytestCaseStatus.PASSED
        return cls(
            nodeid=nodeid,
            status=status,
            detail="\n\n".join(details) or None,
            elapsed_seconds=elapsed_seconds if elapsed_complete else None,
        )


@dataclass(frozen=True, slots=True)
class PytestTaskReport:
    """Exact pytest testcase results parsed from one task-owned JUnit file."""

    cases: tuple[PytestCaseReport, ...]

    @classmethod
    def read(cls, path: Path, expected_cases: tuple[ExpectedPytestCase, ...]) -> PytestTaskReport:
        """Normalize JUnit phase records to expected items in collection order."""

        try:
            root = xml.etree.ElementTree.parse(path).getroot()
        except (OSError, xml.etree.ElementTree.ParseError) as error:
            raise ValueError(f"failed to read pytest JUnit {path}: {error}") from error
        if root.tag != "testsuites":
            raise ValueError("pytest JUnit root must be testsuites")
        suites = tuple(root)
        if len(suites) != 1 or suites[0].tag != "testsuite" or suites[0].get("name") != "pytest":
            raise ValueError("pytest JUnit must contain exactly one pytest testsuite")
        suite = suites[0]
        elements = tuple(child for child in suite if child.tag == "testcase")
        unknown = tuple(
            child.tag for child in suite if child.tag not in {"testcase", "properties", "system-out", "system-err"}
        )
        if unknown:
            raise ValueError(f"pytest JUnit testsuite contains unknown elements: {unknown}")

        expected_by_identity: dict[tuple[str, str], ExpectedPytestCase] = {}
        for case in expected_cases:
            identity = cls.junit_identity(case.nodeid)
            if identity in expected_by_identity:
                raise ValueError(f"expected pytest cases have duplicate JUnit identity: {identity}")
            expected_by_identity[identity] = case

        cls.validate_summary(suite, elements)
        grouped: dict[str, list[xml.etree.ElementTree.Element]] = {}
        for element in elements:
            classname = element.get("classname")
            name = element.get("name")
            if classname is None or name is None:
                raise ValueError("pytest JUnit testcase requires classname and name")
            expected = expected_by_identity.get((classname, name))
            if expected is None:
                raise ValueError(f"pytest JUnit contains unexpected testcase {(classname, name)!r}")
            grouped.setdefault(expected.nodeid, []).append(element)
        missing = tuple(case.nodeid for case in expected_cases if case.nodeid not in grouped)
        if missing:
            raise ValueError(f"pytest JUnit is missing expected testcases: {missing}")

        return cls(tuple(PytestCaseReport.from_elements(case.nodeid, grouped[case.nodeid]) for case in expected_cases))

    @staticmethod
    def junit_identity(nodeid: str) -> tuple[str, str]:
        """Project a pytest nodeid to its xunit2 classname and name."""

        path, bracket, parameters = nodeid.partition("[")
        names = path.split("::")
        names[0] = names[0].replace("/", ".").removesuffix(".py")
        names[-1] += bracket + parameters
        return ".".join(names[:-1]), names[-1]

    @staticmethod
    def validate_summary(
        suite: xml.etree.ElementTree.Element, elements: Sequence[xml.etree.ElementTree.Element]
    ) -> None:
        """Validate raw XML counters before phase records become item results."""

        expected = {
            "tests": len(elements),
            "failures": sum(child.tag == "failure" for element in elements for child in element),
            "errors": sum(child.tag == "error" for element in elements for child in element),
            "skipped": sum(child.tag == "skipped" for element in elements for child in element),
        }
        actual: dict[str, int] = {}
        for name in expected:
            value = suite.get(name)
            try:
                parsed = int(value) if value is not None else -1
            except ValueError as error:
                raise ValueError(f"pytest JUnit summary {name} must be an integer") from error
            if parsed < 0:
                raise ValueError(f"pytest JUnit summary {name} must be nonnegative")
            actual[name] = parsed
        if actual != expected:
            raise ValueError(f"pytest JUnit summary counts disagree with testcase outcomes: {actual}")

    @property
    def failed(self) -> bool:
        """Return whether at least one testcase failed or errored."""

        return any(case.status is PytestCaseStatus.FAILED for case in self.cases)

    def case(self, nodeid: str) -> PytestCaseReport:
        """Return the unique report for one expected nodeid."""

        matches = tuple(case for case in self.cases if case.nodeid == nodeid)
        if len(matches) != 1:
            raise ValueError(f"pytest task report does not contain exactly one result for {nodeid!r}")
        return matches[0]

    def summary(self) -> str:
        """Return stable passed/skipped/failed counts for diagnostics."""

        counts = {status: sum(case.status is status for case in self.cases) for status in PytestCaseStatus}
        return " ".join(f"{status.value}={counts[status]}" for status in PytestCaseStatus)
