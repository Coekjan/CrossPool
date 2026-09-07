from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sys
from pathlib import Path

MARKERS = (
    "agents.md",
    "architecture",
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
    "wrapper",
    "不要",
    "内联",
    "偏好",
    "屎山",
    "单例",
    "架构",
    "工具",
    "应该",
    "行为测试",
    "降智",
    "配置",
    "测试",
    "质量",
    "重构",
    "规则",
)

PREFERENCE_MARKERS = (
    "architecture",
    "behavior test",
    "cache",
    "config",
    "helper",
    "inline",
    "mirror",
    "mock",
    "object",
    "preference",
    "protocol",
    "rule",
    "should",
    "test",
    "wrapper",
    "不",
    "不要",
    "偏好",
    "内联",
    "单例",
    "实现",
    "应该",
    "架构",
    "测试",
    "没必要",
    "直接",
    "行为测试",
    "规则",
    "配置",
    "降智",
    "镜像",
    "约定",
    "质量",
    "屎山",
)


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
        help=(
            "UTC or local date/time, for example 2026-06-01. If omitted, use "
            "the last-file timestamp; --since is required when no timestamp exists."
        ),
    )
    parser.add_argument(
        "--last-file",
        default=str(default_last_file()),
        help="Timestamp file used when --since is omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=("markers", "preferences"),
        default="markers",
        help="Excerpt mode: marker scan across roles or user-correction preference scan.",
    )
    parser.add_argument("--limit", type=int, default=200, help="Maximum excerpts to print.")
    parser.add_argument(
        "--no-update-last",
        action="store_true",
        help="Print excerpts without updating the repo-level last timestamp file.",
    )
    args = parser.parse_args()

    run_started = datetime.datetime.now(datetime.UTC)
    last_file = Path(args.last_file).expanduser()
    since = load_since(args.since, last_file)
    sessions_root = Path(args.sessions_root).expanduser()
    if not sessions_root.exists():
        raise SystemExit(f"sessions root does not exist: {sessions_root}")

    printed = 0
    limit_reached = False
    seen_preference_keys: set[str] = set()
    # A session can resume long after its directory's date; filter each record.
    for session_file in sorted(sessions_root.rglob("*.jsonl")):
        with session_file.open(encoding="utf-8", errors="replace") as lines:
            for line_number, line in enumerate(lines, 1):
                excerpt = session_excerpt(line, since, args.mode)
                if excerpt is None:
                    continue
                if args.mode == "preferences":
                    key = dedup_key(excerpt)
                    if key in seen_preference_keys:
                        continue
                    seen_preference_keys.add(key)
                print(f"\n## {session_file}:{line_number}")
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
    write_last_file(last_file, run_started)
    print(f"updated self-evolve last file: {last_file}", file=sys.stderr)
    return 0


def default_last_file() -> Path:
    for parent in Path(__file__).resolve().parents:
        if parent.name == ".codex":
            return parent / "self-evolve-last.txt"
    return Path.cwd() / ".codex" / "self-evolve-last.txt"


def load_since(value: str | None, last_file: Path) -> datetime.datetime:
    if value:
        return parse_since(value)
    if last_file.exists():
        last_value = last_file.read_text(encoding="utf-8").strip()
        if last_value:
            return parse_since(last_value)
    raise SystemExit(f"no previous self-evolve timestamp found at {last_file}; provide --since")


def write_last_file(path: Path, timestamp: datetime.datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{format_timestamp(timestamp)}\n", encoding="utf-8")


def format_timestamp(timestamp: datetime.datetime) -> str:
    return timestamp.astimezone(datetime.UTC).isoformat().replace("+00:00", "Z")


def parse_since(value: str) -> datetime.datetime:
    normalized = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.datetime.fromisoformat(normalized)
    except ValueError:
        parsed = datetime.datetime.fromisoformat(f"{normalized}T00:00:00")
    return to_utc(parsed)


def to_utc(value: datetime.datetime) -> datetime.datetime:
    if value.tzinfo is None:
        value = value.astimezone()
    return value.astimezone(datetime.UTC)


def session_excerpt(line: str, since: datetime.datetime, mode: str) -> str | None:
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None
    timestamp = record_time(record)
    if timestamp is None or timestamp < since:
        return None
    text = extract_text(record)
    if not text:
        return None
    if mode == "preferences":
        if not text.startswith("user\n"):
            return None
        body = text.removeprefix("user\n")
        if looks_like_review_task(body):
            return None
        lowered_body = body.lower()
        if not any(marker in lowered_body for marker in PREFERENCE_MARKERS):
            return None
        return truncate(f"user-preference\n{body}", 4000)
    lowered = text.lower()
    if not any(marker in lowered for marker in MARKERS):
        return None
    return truncate(text, 2000)


def record_time(record: dict[str, object]) -> datetime.datetime | None:
    raw = record.get("timestamp")
    if not isinstance(raw, str):
        return None
    try:
        parsed = datetime.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return to_utc(parsed)


def extract_text(record: dict[str, object]) -> str | None:
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
        text = collect_text(payload.get("content"))
        if text.startswith("# AGENTS.md instructions for "):
            return None
        if text:
            return f"{role}\n{text}"
        return None
    if payload_type == "function_call_output":
        output = payload.get("output")
        if isinstance(output, str) and tool_output_interesting(output):
            return f"command-output\n{normalize(output)}"
    return None


def collect_text(value: object) -> str:
    parts: list[str] = []

    def visit(node: object) -> None:
        if isinstance(node, dict):
            for key in ("text", "input_text", "output_text", "message"):
                value = node.get(key)
                if isinstance(value, str):
                    parts.append(normalize(value))
            for value in node.values():
                if isinstance(value, dict | list):
                    visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(value)
    return "\n".join(parts)


def normalize(text: str) -> str:
    return " ".join(text.split())


def truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    print(
        f"self-evolve excerpt truncated from {len(text)} to {limit} characters",
        file=sys.stderr,
    )
    return text[:limit]


def dedup_key(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()[:240]


def looks_like_review_task(text: str) -> bool:
    return text.startswith(
        (
            "Pre-commit review for ",
            "Review the current code changes",
            "Self-evolve review for ",
            "<turn_aborted>",
            "You are Axis ",
            "You are the self-evolve reviewer",
        )
    )


def tool_output_interesting(text: str) -> bool:
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
