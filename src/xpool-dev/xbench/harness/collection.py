"""Isolated source declaration collection shared by benchmark inventory and execution."""

from __future__ import annotations

import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from xbench.harness.serving.case import BenchCase, ResolvedDeployment
from xkit import ResourceRequirements
from xkit.child import PythonChildProcess
from xkit.config import ToolConfigRecord, XpoolDevConfig, get_global_config, init_global_config
from xkit.declaration import Parameterization
from xkit.source import resolve_source_path
from xpool.config import XpoolConfig


@dataclass(frozen=True, slots=True)
class CollectedProgram:
    """One discovered entry and resource declaration for a selected concrete case."""

    module: str
    source_path: Path
    entrypoint: str
    import_roots: tuple[Path, ...]
    requirements: ResourceRequirements


def collect_programs(
    catalogue_path: Path,
    cases: tuple[BenchCase, ...],
    *,
    deployments: tuple[ResolvedDeployment | None, ...] | None = None,
) -> tuple[CollectedProgram, ...]:
    """Import selected source declarations in a fresh child, retaining case order.

    Collection executes resource callbacks but neither benchmark bodies nor input
    preparation. Explicit import roots accompany the returned references so later
    supervised workers can import the same programs from the invocation directory.
    """

    catalogue_path = catalogue_path.expanduser().resolve()
    roots = (catalogue_path.parent / "suites", catalogue_path.parent, catalogue_path.parent.parent)
    with TemporaryDirectory(prefix="xpool-benchmark-inventory-") as temporary:
        child = PythonChildProcess(
            "benchmark-source-collection",
            collect_program_worker,
            (catalogue_path, cases, get_global_config().record(), deployments),
            log_path=Path(temporary) / "collection.log",
            import_paths=roots,
        )
        try:
            child.start()
            # The installed child producer sends this exact tuple through the typed Pipe.
            programs = cast(tuple[CollectedProgram, ...], child.receive(tuple, timeout_seconds=30))
            child.wait(timeout_seconds=30)
            return programs
        finally:
            PythonChildProcess.terminate_all((child,))
            child.close()


def collect_program_worker(
    connection: Connection,
    inputs: tuple[Path, tuple[BenchCase, ...], ToolConfigRecord, tuple[ResolvedDeployment | None, ...] | None],
) -> None:
    """Discover only module-defined catalogue entries and their declared resources."""

    catalogue_path, cases, tool_config, deployments = inputs
    init_global_config(resolved=XpoolDevConfig.from_record(tool_config))
    roots = (catalogue_path.parent / "suites", catalogue_path.parent, catalogue_path.parent.parent)
    programs = []
    for index, case in enumerate(cases):
        source_path = resolve_source_path(catalogue_path, case.module)
        module = importlib.import_module(case.module)
        if module.__file__ is None or Path(module.__file__).resolve() != source_path:
            raise ValueError(f"{case.module}: imported source does not match {source_path}")
        entries = []
        for name, function in vars(module).items():
            if not inspect.isfunction(function) or function.__module__ != module.__name__:
                continue
            declaration = cast(Parameterization | None, getattr(function, "xpool_parameters", None))
            if declaration is not None and declaration.values is None:
                entries.append((name, function, declaration))
        if len(entries) != 1:
            raise ValueError(f"{source_path}: expected one catalogue-bound entry, found {len(entries)}")
        name, function, parameters = entries[0]
        if (
            parameters.argument_names != ("case",)
            or parameters.rows is not None
            or inspect.iscoroutinefunction(function)
        ):
            raise ValueError(f"{source_path}: benchmark entry must bind case to a synchronous function")
        inspect.signature(function).bind(case=case, workdir=Path())
        resources = cast(
            ResourceRequirements | Callable[..., ResourceRequirements] | None,
            getattr(function, "xpool_requirements", None),
        )
        if resources is None:
            raise ValueError(f"{source_path}: benchmark entry requires a resource declaration")
        deployment = None if deployments is None else deployments[index]
        if deployment is not None:
            config = XpoolConfig.model_validate_json(json.dumps(deployment.effective))
            requirements = ResourceRequirements(
                len(config.atn.devices) + len(config.ffn.devices), True, tuple(model.id for model in config.models)
            )
        else:
            requirements = resources if isinstance(resources, ResourceRequirements) else resources(case=case)
        programs.append(CollectedProgram(case.module, source_path, name, roots, requirements))
    connection.send(tuple(programs))
