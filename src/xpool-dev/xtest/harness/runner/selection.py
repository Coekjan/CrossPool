"""Suite scope resolved with pytest's parsed collection inputs."""

from pathlib import Path

import pytest
from _pytest.main import resolve_collection_argument
from pydantic import ValidationError

from xkit.config import get_global_config
from xpool.model import ModelId
from xpool.utils.config import ConfigSource
from xtest.harness.runner.collection import CollectionWorker
from xtest.harness.runner.plan import TestPlan

SUITE_ORDER = ("cext", "unit", "integration", "e2e")
selected_suites_key = pytest.StashKey[tuple[str, ...]]()


def select_suites(repository_root: Path, suites: tuple[str, ...]) -> tuple[str, ...]:
    """Validate declared scope using source-owned model suites, without probes."""
    if len(suites) != len(set(suites)):
        raise ValueError("--suite cannot select the same suite more than once")
    models = tuple(suite for suite in suites if suite not in (*SUITE_ORDER, "models"))
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
    if models and "models" in suites:
        raise ValueError("--suite models cannot be combined with individual model suites")
    return suites


def suite_path(suite: str, repository_root: Path) -> Path:
    """Return the source root for a validated Python suite or Model ID."""
    if suite in (*SUITE_ORDER, "models"):
        return repository_root / "tests/suites" / suite
    return repository_root / "tests/suites/models" / ModelId(suite).relative_path


def configure_selection(config: pytest.Config) -> None:
    """Establish scope before imports, using pytest's own positional parser.

    Only runner-owned collection invokes this adapter. Explicit paths define
    scope when no suite was explicitly selected; option-only filters retain the
    configured Python roots. CTest is included only without pytest inputs or
    with an explicit suite selection.
    """
    tool_config = get_global_config()
    selected = select_suites(config.rootpath, tool_config.xtest.suites)
    explicit_suites = any(
        record["name"] == "xtest.suites" and record["source"] is ConfigSource.CLI for record in tool_config.sources
    )
    if not explicit_suites and config.getoption("--xpool-pytest-inputs"):
        selected = tuple(suite for suite in selected if suite != "cext")
    if config.args_source is pytest.Config.ArgsSource.ARGS:
        paths = tuple(
            resolve_collection_argument(
                config.invocation_params.dir,
                arg,
                index,
                as_pypath=config.getoption("pyargs"),
                consider_namespace_packages=config.getini("consider_namespace_packages"),
            ).path.resolve()
            for index, arg in enumerate(config.args)
        )
        if explicit_suites:
            roots = tuple(suite_path(suite, config.rootpath) for suite in selected if suite != "cext")
            for path in paths:
                if not any(path.is_relative_to(root) for root in roots):
                    raise pytest.UsageError(f"explicit pytest path is outside selected suites: {path}")
        else:
            inferred = []
            for path in paths:
                try:
                    parts = path.relative_to(config.rootpath / "tests/suites").parts
                except ValueError as error:
                    raise pytest.UsageError(f"pytest path is outside test suites: {path}") from error
                if not parts or parts[0] not in (*SUITE_ORDER[1:], "models"):
                    raise pytest.UsageError(f"pytest path does not identify a Python suite: {path}")
                suite = "/".join(parts[1:3]) if parts[0] == "models" and len(parts) >= 3 else parts[0]
                if suite not in inferred:
                    inferred.append(suite)
            selected = select_suites(config.rootpath, tuple(inferred))
    else:
        config.args[:] = [str(suite_path(suite, config.rootpath)) for suite in selected if suite != "cext"]
    config.stash[selected_suites_key] = selected


def collect_plan(
    repository_root: Path,
    selected_suites: tuple[str, ...],
    selectors: tuple[str, ...],
    *,
    strict_requirements: bool,
    directory: Path,
    catalogue_path: Path,
) -> TestPlan | None:
    """Collect selected inventory and return the effective scope from pytest."""
    if selected_suites == ("cext",):
        if selectors:
            raise ValueError("CTest-only invocation cannot include pytest inputs")
        return None
    return CollectionWorker(
        repository_root=repository_root,
        run_directory=directory,
        selectors=selectors,
        strict_requirements=strict_requirements,
        catalogue_path=catalogue_path,
        tool_config=get_global_config().record(),
    ).collect()
