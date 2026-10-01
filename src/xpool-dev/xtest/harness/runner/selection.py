"""Shared source-owned suite selection for inventory and execution."""

from pathlib import Path

from pydantic import ValidationError

from xpool.model import ModelId
from xtest.harness.runner.collection import CollectionWorker
from xtest.harness.runner.plan import TestPlan

SUITE_ORDER = ("cext", "unit", "integration", "e2e")
INTEGRATIONS = ("sglang",)


def select_suites(
    repository_root: Path,
    suites: tuple[str, ...],
    selectors: tuple[str, ...],
    integration: str | None,
) -> tuple[str, ...]:
    """Validate selection using source-owned model suites, without resource probes."""

    selected = suites or SUITE_ORDER
    if len(selected) != len(set(selected)):
        raise ValueError("--suite cannot select the same suite more than once")
    models = tuple(suite for suite in selected if suite not in (*SUITE_ORDER, "models"))
    unknown = []
    for name in models:
        try:
            model_id = ModelId(name)
        except ValidationError:
            unknown.append(name)
            continue
        if not (repository_root / "tests/suites/models" / model_id.relative_path).is_dir():
            unknown.append(name)
    if unknown:
        raise ValueError(f"unknown model suite: {', '.join(unknown)}")
    if models and "models" in selected:
        raise ValueError("--suite models cannot be combined with individual model suites")
    if selectors and all(suite == "cext" for suite in selected):
        raise ValueError("pytest selectors require a Python suite")
    if integration is not None and selectors:
        raise ValueError("--integration cannot be combined with explicit pytest selectors")
    return selected


def suite_selectors(
    suite: str,
    *,
    repository_root: Path,
    model_suites: tuple[str, ...],
    integration: str | None,
) -> tuple[str, ...]:
    """Choose engine-owned files before pytest imports their runtime modules."""

    root = Path(f"tests/suites/models/{suite}" if suite in model_suites else f"tests/suites/{suite}")
    if integration is None or suite == "unit":
        return (root.as_posix(),)
    selected = tuple(
        path.relative_to(repository_root).as_posix()
        for path in sorted((repository_root / root).rglob("test_*.py"))
        if integration_for_file(path, "models" if suite in model_suites else suite) in (None, integration)
    )
    if not selected:
        raise ValueError(f"no tests for integration {integration!r} in suite {suite!r}")
    return selected


def integration_for_file(path: Path, stage: str) -> str | None:
    """Classify engine-specific directories or model-test filenames."""

    for name in INTEGRATIONS:
        if path.name.startswith(f"test_{name}_") or (stage != "models" and name in path.parts):
            return name
    return None


def collect_plan(
    repository_root: Path,
    selected_suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    integration: str | None,
    strict_requirements: bool,
    directory: Path,
    catalogue_path: Path,
) -> TestPlan | None:
    """Collect exactly the selected Python inventory in one isolated worker."""

    python_suites = tuple(suite for suite in selected_suites if suite != "cext")
    if not python_suites:
        return None
    model_suites = tuple(suite for suite in selected_suites if suite not in (*SUITE_ORDER, "models"))
    collection_selectors = selectors or tuple(
        path
        for suite in python_suites
        for path in suite_selectors(
            suite, repository_root=repository_root, model_suites=model_suites, integration=integration
        )
    )
    plan = CollectionWorker(
        repository_root=repository_root,
        run_directory=directory,
        selectors=collection_selectors,
        strict_requirements=strict_requirements,
        catalogue_path=catalogue_path,
    ).collect()
    for case in plan.cases:
        if case.stage.value not in selected_suites and not (
            case.stage.value == "models"
            and any(case.path.startswith(f"tests/suites/models/{model_id}/") for model_id in model_suites)
        ):
            raise ValueError(f"collected test outside selected suites: {case.nodeid}")
        if integration is not None and integration_for_file(Path(case.path), case.stage.value) not in (
            None,
            integration,
        ):
            raise ValueError(f"collected test outside selected integration: {case.nodeid}")
    return plan
