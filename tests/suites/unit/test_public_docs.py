"""Explicit quality gates for supported Python and native documentation."""

import ast
import inspect
from importlib import import_module
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

import xpool.native

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PYTHON_SOURCE_ROOT = REPOSITORY_ROOT / "src" / "xpool"
PUBLIC_MODEL_MODULES = (
    "xpool.config",
    "xpool.integrations.sglang.topology",
    "xpool.runtime.transport",
    "xpool.service.wire",
)


def python_module_name(path: Path) -> str:
    relative = path.relative_to(PYTHON_SOURCE_ROOT)
    parts = relative.with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(("xpool", *parts))


def literal_exports(module_name: str, tree: ast.Module) -> tuple[str, ...] | None:
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in statement.targets
        ):
            continue
        try:
            exports = ast.literal_eval(statement.value)
        except (ValueError, TypeError, SyntaxError):
            raise AssertionError(f"{module_name}.__all__ must be a literal sequence") from None
        if not isinstance(exports, (list, tuple)) or not all(isinstance(name, str) for name in exports):
            raise AssertionError(f"{module_name}.__all__ must contain only strings")
        return tuple(exports)
    return None


def imported_symbols(module_name: str, tree: ast.Module) -> dict[str, tuple[str, str]]:
    imports: dict[str, tuple[str, str]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.ImportFrom) or statement.level or statement.module is None:
            continue
        for imported in statement.names:
            if imported.name == "*":
                continue
            imports[imported.asname or imported.name] = (statement.module, imported.name)
    return imports


def resolve_declaration(
    module_name: str,
    symbol_name: str,
    trees: dict[str, ast.Module],
    seen: set[tuple[str, str]],
) -> tuple[str, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef] | None:
    key = (module_name, symbol_name)
    if key in seen or module_name not in trees:
        return None
    seen.add(key)
    tree = trees[module_name]
    for statement in tree.body:
        if (
            isinstance(statement, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and statement.name == symbol_name
        ):
            return module_name, statement
    imported = imported_symbols(module_name, tree).get(symbol_name)
    if imported is None:
        return None
    return resolve_declaration(*imported, trees, seen)


def test_python_public_surface_has_declaration_documentation() -> None:
    """Require declaration-site docs for explicit Python class and function exports."""

    trees = {
        python_module_name(path): ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for path in PYTHON_SOURCE_ROOT.rglob("*.py")
    }
    missing: list[str] = []
    for module_name, tree in trees.items():
        exports = literal_exports(module_name, tree)
        if exports is None:
            continue
        for export_name in exports:
            declaration = resolve_declaration(module_name, export_name, trees, set())
            if declaration is None:
                continue
            owner_name, node = declaration
            qualified_name = f"{owner_name}.{node.name}"
            if ast.get_docstring(node, clean=False) is None:
                missing.append(qualified_name)
            if not isinstance(node, ast.ClassDef):
                continue
            for member in node.body:
                if (
                    isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and not member.name.startswith("_")
                    and ast.get_docstring(member, clean=False) is None
                ):
                    missing.append(f"{qualified_name}.{member.name}")

    assert missing == []


def test_public_pydantic_fields_have_descriptions() -> None:
    """Require declaration-site descriptions for public Pydantic fields."""

    missing: list[str] = []
    for module_name in PUBLIC_MODEL_MODULES:
        module = import_module(module_name)
        for name, value in inspect.getmembers(module, inspect.isclass):
            if name.startswith("_") or value.__module__ != module_name or not issubclass(value, BaseModel):
                continue
            for field_name, field in value.model_fields.items():
                if not field.description:
                    missing.append(f"{module_name}.{name}.{field_name}")

    assert missing == []


def test_native_public_surface_has_runtime_documentation() -> None:
    """Require reflection-visible docs on the generated native API surface."""

    modules = [xpool.native]
    visited: set[str] = set()
    missing: list[str] = []
    while modules:
        module = modules.pop()
        if module.__name__ in visited:
            continue
        visited.add(module.__name__)
        if not inspect.getdoc(module):
            missing.append(module.__name__)
        for name, value in inspect.getmembers(module):
            if name.startswith("_"):
                continue
            if isinstance(value, ModuleType):
                if value.__name__.startswith("xpool.native"):
                    modules.append(value)
                continue
            qualified_name = f"{module.__name__}.{name}"
            if inspect.isclass(value):
                if not inspect.getdoc(value):
                    missing.append(qualified_name)
                for member_name, member in vars(value).items():
                    if member_name.startswith("_") or member_name in {"name", "value"}:
                        continue
                    if (inspect.isroutine(member) or inspect.isdatadescriptor(member)) and not inspect.getdoc(member):
                        missing.append(f"{qualified_name}.{member_name}")
            elif callable(value) and not inspect.getdoc(value):
                missing.append(qualified_name)

    assert missing == []
