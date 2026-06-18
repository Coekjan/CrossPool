from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

MARKERS = (
    "agents.md",
    "blocked",
    "commit",
    "conflict",
    "failure",
    "git",
    "hook",
    "instruction",
    "memory",
    "pre-commit",
    "preference",
    "review",
    "rule",
    "self-evolve",
    "skill",
    "test",
    "tool",
    "validation",
    "workflow",
    "不要",
    "偏好",
    "工具",
    "应该",
    "约定",
    "规则",
)


@dataclass(frozen=True)
class SessionFile:
    path: Path
    date: dt.date | None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print Codex session excerpts for self-evolve review.",
    )
    parser.add_argument(
        "--sessions-root",
        default=str(Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "sessions"),
        help="Codex sessions directory to scan.",
    )
    parser.add_argument(
        "--since",
        help="UTC or local date/time, for example 2026-06-01. Required only for the first run.",
    )
    parser.add_argument(
        "--last-file",
        default=str(_default_last_file()),
        help="Timestamp file used when --since is omitted.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum excerpts to print.")
    parser.add_argument(
        "--no-update-last",
        action="store_true",
        help="Print excerpts without updating the repo-level last timestamp file.",
    )
    args = parser.parse_args()

    run_started = dt.datetime.now(dt.UTC)
    last_file = Path(args.last_file).expanduser()
    since = _load_since(args.since, last_file)
    sessions_root = Path(args.sessions_root).expanduser()
    if not sessions_root.exists():
        raise SystemExit(f"sessions root does not exist: {sessions_root}")

    printed = 0
    limit_reached = False
    for session_file in _index_session_files(sessions_root):
        if session_file.date is not None and session_file.date < since.date():
            continue
        for line_number, line in enumerate(
            session_file.path.read_text(encoding="utf-8", errors="replace").splitlines(),
            1,
        ):
            excerpt = _session_excerpt(line, since)
            if excerpt is None:
                continue
            print(f"\n## {session_file.path}:{line_number}")
            print(excerpt)
            printed += 1
            if printed >= args.limit:
                limit_reached = True
                break
        if limit_reached:
            break
    if limit_reached:
        print(
            f"self-evolve excerpt limit reached; not updating last file: {last_file}",
            file=sys.stderr,
        )
        return 2
    if args.no_update_last:
        print(f"self-evolve last file not updated due to --no-update-last: {last_file}", file=sys.stderr)
        return 0
    _write_last_file(last_file, run_started)
    print(f"updated self-evolve last file: {last_file}", file=sys.stderr)
    return 0


def _default_last_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == ".codex":
            return parent / "self-evolve-last.txt"
    return Path.cwd() / ".codex" / "self-evolve-last.txt"


def _load_since(value: str | None, last_file: Path) -> dt.datetime:
    if value:
        return _parse_since(value)
    if last_file.exists():
        last_value = last_file.read_text(encoding="utf-8").strip()
        if last_value:
            return _parse_since(last_value)
    raise SystemExit(
        f"no previous self-evolve timestamp found at {last_file}; run once with --since YYYY-MM-DD to initialize it"
    )


def _write_last_file(path: Path, timestamp: dt.datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{_format_timestamp(timestamp)}\n", encoding="utf-8")


def _format_timestamp(timestamp: dt.datetime) -> str:
    return timestamp.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def _parse_since(value: str) -> dt.datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalized)
    except ValueError:
        parsed = dt.datetime.fromisoformat(f"{normalized}T00:00:00")
    return _to_utc(parsed)


def _to_utc(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(dt.UTC)


def _index_session_files(sessions_root: Path) -> list[SessionFile]:
    session_files = [
        SessionFile(path=path, date=_session_date_from_path(path)) for path in sessions_root.rglob("*.jsonl")
    ]
    return sorted(session_files, key=lambda item: (item.date or dt.date.min, str(item.path)))


def _session_date_from_path(path: Path) -> dt.date | None:
    parts = path.parts
    for index, part in enumerate(parts):
        if part != "sessions" or index + 3 >= len(parts):
            continue
        try:
            return dt.date(int(parts[index + 1]), int(parts[index + 2]), int(parts[index + 3]))
        except ValueError:
            return None
    return None


def _session_excerpt(line: str, since: dt.datetime) -> str | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    timestamp = _record_time(record)
    if timestamp is None or timestamp < since:
        return None
    text = _extract_text(record)
    if not text:
        return None
    lowered = text.lower()
    if not any(marker in lowered for marker in MARKERS):
        return None
    return text[:2000]


def _record_time(record: dict[str, object]) -> dt.datetime | None:
    raw = record.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        parsed = dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _to_utc(parsed)


def _extract_text(record: dict[str, object]) -> str | None:
    record_type = record.get("type")
    payload = record.get("payload")
    if not isinstance(payload, dict):
        return None
    if record_type != "response_item":
        return None
    payload_type = payload.get("type")
    if payload_type == "message":
        role = payload.get("role")
        if role not in {"assistant", "user"}:
            return None
        text = _collect_text(payload.get("content"))
        if text.startswith("# AGENTS.md instructions for "):
            return None
        if text:
            return f"{role}\n{text}"
        return None
    if payload_type == "function_call_output":
        output = payload.get("output")
        if isinstance(output, str) and _tool_output_interesting(output):
            return f"command-output\n{_normalize(output)}"
    return None


def _collect_text(value: object) -> str:
    parts: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("text", "input_text", "output_text", "message"):
                value = node.get(key)
                if isinstance(value, str):
                    parts.append(_normalize(value))
            for value in node.values():
                if isinstance(value, dict | list):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return "\n".join(parts)


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _tool_output_interesting(text: str) -> bool:
    lowered = text.lower()
    if "process exited with code 0" in lowered:
        return (
            "installed python" in lowered
            or "skill is valid!" in lowered
            or re.search(r"resolved [0-9]+ packages", lowered) is not None
            or re.search(r"\bversion: [0-9]", lowered) is not None
            or ("passed" in lowered and any(name in lowered for name in ("ruff", "ty check", "pre-commit")))
        )
    return any(marker in lowered for marker in ("error", "failed", "invalid", "no such file", "not found", "warning"))


if __name__ == "__main__":
    raise SystemExit(main())
