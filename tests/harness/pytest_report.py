"""Strict projection of pytest JUnit XML into task result values."""

from __future__ import annotations

import xml.etree.ElementTree
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from tests.harness.test_plan import CollectedTestCase


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

    @classmethod
    def from_element(cls, nodeid: str, element: xml.etree.ElementTree.Element) -> PytestCaseReport:
        """Parse one testcase while ignoring standard nonsemantic attachments."""

        outcomes = tuple(child for child in element if child.tag in {"failure", "error", "skipped"})
        unknown = tuple(
            child.tag
            for child in element
            if child.tag not in {"failure", "error", "skipped", "system-out", "system-err", "properties"}
        )
        if unknown:
            raise ValueError(f"JUnit testcase {nodeid!r} contains unknown elements: {unknown}")
        if len(outcomes) > 1:
            raise ValueError(f"JUnit testcase {nodeid!r} contains multiple outcomes")
        if not outcomes:
            return cls(nodeid=nodeid, status=PytestCaseStatus.PASSED, detail=None)
        outcome = outcomes[0]
        detail = outcome.get("message") or (outcome.text or "").strip() or None
        if outcome.tag == "skipped":
            if outcome.get("type") == "pytest.xfail":
                raise ValueError(f"JUnit testcase {nodeid!r} contains prohibited pytest.xfail outcome")
            if detail is None:
                raise ValueError(f"JUnit skipped testcase {nodeid!r} has no reason")
            return cls(nodeid=nodeid, status=PytestCaseStatus.SKIPPED, detail=detail)
        return cls(nodeid=nodeid, status=PytestCaseStatus.FAILED, detail=detail)


@dataclass(frozen=True, slots=True)
class PytestTaskReport:
    """Exact pytest testcase results parsed from one task-owned JUnit file."""

    cases: tuple[PytestCaseReport, ...]

    @classmethod
    def read(cls, path: Path, expected_cases: tuple[CollectedTestCase, ...]) -> PytestTaskReport:
        """Read one pytest JUnit report and match every expected case exactly."""

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

        expected_by_identity: dict[tuple[str, str], CollectedTestCase] = {}
        for case in expected_cases:
            identity = cls.junit_identity(case.nodeid)
            if identity in expected_by_identity:
                raise ValueError(f"expected pytest cases have duplicate JUnit identity: {identity}")
            expected_by_identity[identity] = case

        parsed: dict[str, PytestCaseReport] = {}
        for element in elements:
            classname = element.get("classname")
            name = element.get("name")
            if classname is None or name is None:
                raise ValueError("pytest JUnit testcase requires classname and name")
            expected = expected_by_identity.get((classname, name))
            if expected is None:
                raise ValueError(f"pytest JUnit contains unexpected testcase {(classname, name)!r}")
            if expected.nodeid in parsed:
                raise ValueError(f"pytest JUnit contains duplicate testcase {expected.nodeid!r}")
            parsed[expected.nodeid] = PytestCaseReport.from_element(expected.nodeid, element)
        missing = tuple(case.nodeid for case in expected_cases if case.nodeid not in parsed)
        if missing:
            raise ValueError(f"pytest JUnit is missing expected testcases: {missing}")

        report = cls(tuple(parsed[case.nodeid] for case in expected_cases))
        report.validate_summary(suite)
        return report

    @staticmethod
    def junit_identity(nodeid: str) -> tuple[str, str]:
        """Project a pytest nodeid to its xunit2 classname and name."""

        path, bracket, parameters = nodeid.partition("[")
        names = path.split("::")
        names[0] = names[0].replace("/", ".").removesuffix(".py")
        names[-1] += bracket + parameters
        return ".".join(names[:-1]), names[-1]

    def validate_summary(self, suite: xml.etree.ElementTree.Element) -> None:
        """Require JUnit summary counters to agree with testcase elements."""

        expected = {
            "tests": len(self.cases),
            "failures": sum(case.status is PytestCaseStatus.FAILED for case in self.cases),
            "errors": 0,
            "skipped": sum(case.status is PytestCaseStatus.SKIPPED for case in self.cases),
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
        if (
            actual["tests"] != expected["tests"]
            or actual["skipped"] != expected["skipped"]
            or actual["failures"] + actual["errors"] != expected["failures"]
        ):
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
