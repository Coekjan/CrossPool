"""Source resolution shared by runtime and development configuration schemas."""

from __future__ import annotations

import argparse
import logging
import tomllib
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import ClassVar, Literal, Self, TypedDict, cast

from pydantic import BaseModel, ConfigDict, JsonValue, PrivateAttr

__all__ = [
    "ConfigError",
    "ConfigModel",
    "ConfigSetting",
    "ConfigSource",
    "ConfigSourceRecord",
    "MissingRequiredConfig",
    "format_source_record_name",
    "get_nested",
    "set_nested",
]


class ConfigSource(StrEnum):
    """Configuration value source used by the CrossPool setting registry.

    Attributes:
        CLI: Value came from an explicit CrossPool CLI override.
        ENV: Value came from an allowlisted process environment variable.
        CONFIG: Value came from the TOML config file.
        DEFAULT: Value came from a registry default.
        UNSET: Optional value has no direct CLI, environment, TOML, or default source.
    """

    CLI = "cli"
    ENV = "env"
    CONFIG = "config"
    DEFAULT = "default"
    UNSET = "unset"


type ParserName = Literal["bool", "int", "path", "raw", "str"]


class ConfigSourceRecord(TypedDict):
    """Resolved config value and source provenance."""

    name: str
    value: object
    source: ConfigSource


class ConfigError(ValueError):
    """Base error for CrossPool config resolution failures."""


class MissingRequiredConfig(ConfigError):
    """Raised when a required setting has no value from any allowed source."""


@dataclass(frozen=True, slots=True)
class ConfigSetting:
    """Registry entry for one TOML, CLI, or defaulted setting.

    Attributes:
        name: Stable registry key used by CLI overrides and diagnostics.
        path: Concrete schema field path, or ``None`` for bootstrap-only settings.
        parser: Parser name used to normalize raw source values.
        allowed_sources: Sources allowed to provide this setting.
        description: Human-readable setting purpose for generated registry output.
        default: Default value used when ``DEFAULT`` is an allowed source.
        required: Whether missing values are configuration errors.
        cli: CLI flag name when the setting is overrideable from the command line.
        env_var: Environment variable name when the setting is process-env backed.
        cli_action: Repeated-value argparse form when the setting accepts a list.
        cli_choices: Allowed CLI spellings, when restricted by the schema.
    """

    name: str
    path: tuple[str, ...] | None
    parser: ParserName
    allowed_sources: tuple[ConfigSource, ...]
    description: str
    default: object = None
    required: bool = False
    cli: str | None = None
    env_var: str | None = None
    cli_action: Literal["store", "append"] = "store"
    cli_choices: tuple[str, ...] | None = None

    def parse(self, value: object) -> object:
        """Parse one raw value using this setting's declared parser.

        Args:
            value: Raw selected value.

        Returns:
            Parsed config value.

        Raises:
            ConfigError: If the parser rejects the value or is unknown.
        """

        match self.parser:
            case "bool":
                if isinstance(value, bool):
                    return value
                normalized = str(value).strip()
                if normalized == "1":
                    return True
                if normalized == "0":
                    return False
                raise ConfigError(f"expected boolean flag value '0' or '1', got {value!r}")
            case "int":
                try:
                    return int(str(value).strip())
                except ValueError as exc:
                    raise ConfigError(f"expected integer config value for {self.name}, got {value!r}") from exc
            case "path":
                if not isinstance(value, (str, Path)):
                    raise ConfigError(f"expected path config value for {self.name}, got {value!r}")
                return Path(value).expanduser()
            case "raw":
                return value
            case "str":
                return str(value)
            case _:
                raise ConfigError(f"unknown parser for {self.name}: {self.parser}")

    def resolve(
        self,
        payload: Mapping[str, object],
        cli_overrides: Mapping[str, object],
        env: Mapping[str, str],
        *,
        origin: Path | None = None,
    ) -> tuple[object, ConfigSource | None]:
        """Resolve this setting according to CrossPool source precedence.

        Args:
            payload: Config-file payload.
            cli_overrides: Explicit CLI values keyed by setting name.
            env: Allowlisted environment values.
            origin: Base directory for file-sourced relative paths; CLI and
                environment paths use the current working directory.

        Returns:
            Resolved value and source, or ``(None, None)`` when optional and unset.

        Raises:
            MissingRequiredConfig: If this required setting has no value.
            ConfigError: If the selected value cannot be parsed.
        """

        found, value = get_nested(payload, self.path or ())
        if ConfigSource.CLI in self.allowed_sources and self.name in cli_overrides:
            value, source = cli_overrides[self.name], ConfigSource.CLI
        elif ConfigSource.ENV in self.allowed_sources and self.env_var is not None and self.env_var in env:
            value, source = env[self.env_var], ConfigSource.ENV
        elif ConfigSource.CONFIG in self.allowed_sources and found:
            source = ConfigSource.CONFIG
        elif ConfigSource.DEFAULT in self.allowed_sources:
            value, source = self.default, ConfigSource.DEFAULT
        elif self.required:
            raise MissingRequiredConfig(f"missing required config setting: {self.name}")
        else:
            return None, None

        parsed = self.parse(value)
        if self.parser == "path":
            path = cast(Path, parsed)
            directory = origin if source is ConfigSource.CONFIG and origin is not None else Path.cwd()
            parsed = (directory / path).resolve()
        return parsed, source

    def source_records(self, payload: Mapping[str, object]) -> list[ConfigSourceRecord]:
        """Expand this setting's wildcard path into concrete source records.

        Args:
            payload: Original config mapping before overrides are applied.

        Returns:
            Source records for every concrete wildcard path.

        Raises:
            ConfigError: If the payload shape does not match the wildcard path.
        """

        path = self.path or ()
        records: list[ConfigSourceRecord] = []

        def walk(value: object, remaining_path: tuple[str, ...], concrete_path: tuple[str | int, ...]) -> None:
            if not remaining_path:
                records.append(
                    {
                        "name": format_source_record_name(concrete_path),
                        "value": value,
                        "source": ConfigSource.CONFIG,
                    }
                )
                return

            segment = remaining_path[0]
            rest = remaining_path[1:]
            if segment == "*":
                if not isinstance(value, list):
                    raise ConfigError(
                        f"expected list config value at {format_source_record_name(concrete_path)} "
                        f"for wildcard setting {self.name}"
                    )
                for index, item in enumerate(value):
                    walk(item, rest, (*concrete_path, index))
                return

            if not isinstance(value, Mapping):
                raise ConfigError(f"expected mapping config value at {format_source_record_name(concrete_path)}")
            mapping = cast(Mapping[str, object], value)
            if segment not in mapping:
                if "*" in rest:
                    return
                records.append(
                    {
                        "name": format_source_record_name((*concrete_path, segment, *rest)),
                        "value": self.default if ConfigSource.DEFAULT in self.allowed_sources else None,
                        "source": ConfigSource.DEFAULT
                        if ConfigSource.DEFAULT in self.allowed_sources
                        else ConfigSource.UNSET,
                    }
                )
                return
            walk(mapping[segment], rest, (*concrete_path, segment))

        walk(payload, path, ())
        return records


class ConfigModel(BaseModel):
    """Registry-backed root schema with resolved leaf provenance.

    Concrete schemas own their setting registry and environment prefix.
    File reading never installs a process-global value or reads implicit env.
    """

    model_config = ConfigDict(extra="forbid")
    registry: ClassVar[tuple[ConfigSetting, ...]]
    env_prefix: ClassVar[str]
    _sources: tuple[ConfigSourceRecord, ...] = PrivateAttr(default_factory=tuple)

    @classmethod
    def add_cli_args(
        cls,
        parser: argparse.ArgumentParser,
        *,
        names: Sequence[str] | None = None,
    ) -> None:
        """Register explicit-only CLI overrides, optionally selecting setting names.

        Boolean settings expose positive and negative flags; repeated settings
        collect values in invocation order. Defaults remain registry-owned.
        Unknown selection names raise ConfigError.
        """

        if names is not None and (unknown := set(names) - {setting.name for setting in cls.registry}):
            raise ConfigError(f"unknown CLI config settings: {sorted(unknown)}")
        for setting in cls.registry:
            if setting.cli is None or ConfigSource.CLI not in setting.allowed_sources:
                continue
            if names is not None and setting.name not in names:
                continue
            if setting.parser == "bool":
                parser.add_argument(
                    setting.cli,
                    dest=setting.name,
                    default=argparse.SUPPRESS,
                    help=setting.description,
                    action=argparse.BooleanOptionalAction,
                )
            else:
                parser.add_argument(
                    setting.cli,
                    type=int if setting.parser == "int" else str,
                    action=setting.cli_action,
                    choices=setting.cli_choices,
                    dest=setting.name,
                    default=argparse.SUPPRESS,
                    help=setting.description,
                )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        cli: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
    ) -> Self:
        """Load and validate a CrossPool TOML file.

        Args:
            path: Path to the TOML config file.
            cli: Optional CLI-derived setting overrides that take
                precedence over TOML values.
            env: Optional allowlisted environment settings. Only registry
                entries with ``ENV`` in ``allowed_sources`` may read it.

        Returns:
            Validated config object with defaults and CLI overrides resolved.

        Raises:
            OSError: If the file cannot be opened.
            tomllib.TOMLDecodeError: If the file is not valid TOML.
            ConfigError: If registry resolution fails.
            pydantic.ValidationError: If schema validation fails.
        """

        config_path = Path(path).expanduser().resolve()
        with config_path.open("rb") as config_file:
            payload = tomllib.load(config_file)
        return cls.from_mapping(payload, cli=cli, env=env, origin=config_path.parent)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, object],
        *,
        cli: Mapping[str, object] | None = None,
        env: Mapping[str, str] | None = None,
        origin: Path | None = None,
    ) -> Self:
        """Validate an in-memory config mapping.

        Args:
            payload: TOML-like mapping, copied before resolution.
            origin: Declaring directory for TOML path settings, or cwd when absent.
                CLI, environment and default paths always use cwd.
            cli: Optional CLI-derived setting overrides that take
                precedence over mapping values.
            env: Optional allowlisted environment settings. Only registry
                entries with ``ENV`` in ``allowed_sources`` may read it.

        Returns:
            Validated config object.

        Raises:
            ConfigError: If registry resolution fails.
            pydantic.ValidationError: If schema validation fails.

        Side Effects:
            Does not mutate ``payload``.
        """

        source_payload = deepcopy(dict(payload))
        resolved: dict[str, object] = deepcopy(dict(payload))
        effective_cli = cli or {}
        effective_env = env or {}
        if env is not None:
            allowed_env_vars = frozenset(setting.env_var for setting in cls.registry if setting.env_var is not None)
            unknown_env_vars = tuple(
                sorted(
                    name for name in effective_env if name.startswith(cls.env_prefix) and name not in allowed_env_vars
                )
            )
            if unknown_env_vars:
                logging.getLogger(cls.__module__).warning(
                    "ignoring unknown %s environment variables: %s",
                    cls.env_prefix.rstrip("_").lower(),
                    ", ".join(unknown_env_vars),
                )

        for setting in cls.registry:
            if setting.path is None or ConfigSource.CONFIG in setting.allowed_sources:
                continue
            if get_nested(source_payload, setting.path)[0]:
                raise ConfigError(f"config setting {setting.name} does not allow TOML source: {'.'.join(setting.path)}")

        wildcard_parents = {
            setting.path[: setting.path.index("*")]
            for setting in cls.registry
            if setting.path is not None and "*" in setting.path
        }
        sources: list[ConfigSourceRecord] = []
        for setting in cls.registry:
            if setting.path is not None and "*" in setting.path:
                sources.extend(setting.source_records(source_payload))
                continue
            value, source = setting.resolve(source_payload, effective_cli, effective_env, origin=origin)
            if setting.path is not None and setting.path not in wildcard_parents:
                sources.append(
                    {
                        "name": format_source_record_name(setting.path),
                        "value": value,
                        "source": ConfigSource.UNSET if source is None else source,
                    }
                )
            if setting.path is not None and source is not None:
                set_nested(resolved, setting.path, value)

        config = cls.model_validate(resolved)
        config._sources = tuple(sources)
        return config

    @property
    def sources(self) -> tuple[ConfigSourceRecord, ...]:
        """Return immutable source records for resolved leaf config values."""

        return self._sources

    def to_config_mapping(self) -> dict[str, JsonValue]:
        """Snapshot CONFIG-allowed effective values for TOML materialization.

        Registry source permissions, including model wildcard paths, own the
        projection. Optional None values, bootstrap inputs and env-only debug
        values are omitted. Callers retain debug environment and original source
        provenance separately; reloading the snapshot establishes CONFIG sources.
        """

        paths = tuple(
            setting.path
            for setting in self.registry
            if setting.path is not None and ConfigSource.CONFIG in setting.allowed_sources
        )

        def project(value: JsonValue, allowed: tuple[tuple[str, ...], ...]) -> JsonValue:
            if () in allowed:
                return value
            if isinstance(value, dict):
                return {
                    name: project(child, tuple(path[1:] for path in allowed if path and path[0] == name))
                    for name, child in value.items()
                    if any(path and path[0] == name for path in allowed)
                }
            if isinstance(value, list):
                nested = tuple(path[1:] for path in allowed if path and path[0] == "*")
                return [project(child, nested) for child in value]
            raise ConfigError("configuration registry path does not match its schema")

        payload = cast(dict[str, JsonValue], self.model_dump(mode="json", exclude_none=True))
        return cast(dict[str, JsonValue], project(payload, paths))


def format_source_record_name(path: tuple[str | int, ...]) -> str:
    """Format a nested config path for source-report diagnostics.

    Args:
        path: Config path segments, including integer list indexes.

    Returns:
        Dot-and-bracket notation for the path.
    """

    return "".join(
        f"[{segment}]" if isinstance(segment, int) else f"{'.' if index else ''}{segment}"
        for index, segment in enumerate(path)
    )


def get_nested(payload: Mapping[str, object], path: tuple[str, ...]) -> tuple[bool, object]:
    """Read a nested mapping path without conflating missing and null values.

    Args:
        payload: Mapping to traverse.
        path: String path segments.

    Returns:
        Whether the path exists and its value when present.
    """

    cursor: object = payload
    for key in path:
        if not isinstance(cursor, Mapping):
            return False, None
        mapping = cast(Mapping[str, object], cursor)
        if key not in mapping:
            return False, None
        cursor = mapping[key]
    return True, cursor


def set_nested(payload: dict[str, object], path: tuple[str, ...], value: object) -> None:
    """Set a nested config path, creating missing mappings.

    Args:
        payload: Mutable config mapping.
        path: Non-empty path to update.
        value: Resolved value to install.

    Raises:
        ConfigError: If the path is empty or crosses a non-mapping value.

    Side Effects:
        Mutates ``payload`` in place.
    """

    if not path:
        raise ConfigError("cannot set empty config path")
    cursor = payload
    for key in path[:-1]:
        child = cursor.get(key)
        if child is None:
            child = {}
            cursor[key] = child
        if not isinstance(child, dict):
            raise ConfigError(f"cannot override nested config path {'.'.join(path)}")
        cursor = cast(dict[str, object], child)
    cursor[path[-1]] = value
