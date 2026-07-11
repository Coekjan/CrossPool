from __future__ import annotations

from xpool.cli.registry import discover_cli_commands


def test_cli_command_discovery_finds_command_tree() -> None:
    commands = discover_cli_commands()

    command_tree = {(command.parent, command.name): type(command).__name__ for command in commands}

    assert all(type(command).__module__.startswith("xpool.cli.subcommands.") for command in commands)
    assert command_tree == {
        (None, "config"): "ConfigCommand",
        (None, "daemon"): "DaemonCommand",
        (None, "devagent"): "DevagentCommand",
        ("config", "dump"): "ConfigDumpCommand",
        ("daemon", "check"): "DaemonCheckCommand",
        ("daemon", "serve"): "DaemonServeCommand",
    }
