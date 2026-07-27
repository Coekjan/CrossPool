"""Closed synthetic subprocess behaviors for process-lifecycle tests."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from xpool.utils.procs import set_process_title


def main() -> int:
    """Execute one named synthetic process behavior."""

    arguments = parse_arguments()
    match arguments.behavior:
        case "sleep":
            time.sleep(arguments.seconds)
        case "ignore-term":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            time.sleep(arguments.seconds)
        case "ignore-term-ready":
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            print("ready", flush=True)
            time.sleep(arguments.seconds)
        case "spawn-child":
            child = subprocess.Popen(command("sleep", seconds=arguments.seconds))
            print(child.pid, flush=True)
            time.sleep(arguments.seconds)
        case "spawn-ignoring-child":
            child = subprocess.Popen(
                command("ignore-term-ready", seconds=arguments.seconds),
                stdout=subprocess.PIPE,
                text=True,
            )
            assert child.stdout is not None
            if child.stdout.readline() != "ready\n":
                raise RuntimeError("TERM-ignoring child did not report readiness")
            print(child.pid, flush=True)
            time.sleep(arguments.seconds)
        case "spawn-detached-child":
            child = subprocess.Popen(command("sleep", seconds=arguments.seconds), start_new_session=True)
            print(child.pid, flush=True)
        case "spawn-exiting-intermediate":
            intermediate = subprocess.Popen(command("spawn-detached-child", seconds=arguments.seconds))
            intermediate.wait()
        case "report-topology":
            print(os.getpid(), os.getppid(), os.getsid(0), os.getpgid(0), flush=True)
            time.sleep(arguments.seconds)
        case "record-term-ignore":
            output = Path(arguments.output)
            output.write_text("ready\n", encoding="utf-8")

            def record_term(signal_number: int, frame: object) -> None:
                del signal_number, frame
                with output.open("a", encoding="utf-8") as output_file:
                    output_file.write("SIGTERM\n")

            signal.signal(signal.SIGTERM, record_term)
            time.sleep(arguments.seconds)
        case "exit":
            return arguments.code
        case "title":
            set_process_title(arguments.title)
            print("ready", flush=True)
            sys.stdin.readline()
    return 0


def command(
    behavior: str,
    *,
    seconds: float | None = None,
    output: Path | None = None,
) -> list[str]:
    """Build this module's command for an internal child behavior."""

    result = [sys.executable, "-m", "tests.harness.process_probe", behavior]
    if seconds is not None:
        result.extend(("--seconds", str(seconds)))
    if output is not None:
        result.extend(("--output", str(output)))
    return result


def parse_arguments() -> argparse.Namespace:
    """Parse one closed synthetic process behavior."""

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="behavior", required=True)
    for behavior in (
        "sleep",
        "ignore-term",
        "ignore-term-ready",
        "spawn-child",
        "spawn-ignoring-child",
        "spawn-detached-child",
        "spawn-exiting-intermediate",
        "report-topology",
    ):
        subparser = subparsers.add_parser(behavior)
        subparser.add_argument("--seconds", type=float, required=True)
    record_term_parser = subparsers.add_parser("record-term-ignore")
    record_term_parser.add_argument("--seconds", type=float, required=True)
    record_term_parser.add_argument("--output", required=True)
    exit_parser = subparsers.add_parser("exit")
    exit_parser.add_argument("--code", type=int, required=True)
    title_parser = subparsers.add_parser("title")
    title_parser.add_argument("--title", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
