from __future__ import annotations

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import TracebackType

from xbench.harness.serving.case import ClientBenchCase, ServingEndpoint
from xbench.harness.serving.measure import BenchCaseManifest, RepetitionManifest
from xbench.harness.serving.report import ReportSeries, RetainedAttempt, load_series
from xtest.harness.support.config import TEST_CASE_ID, TEST_MODEL_ID


def load_retained_series(directory: Path, label: str) -> ReportSeries:
    directory = directory.resolve()
    case = BenchCaseManifest.model_validate_json((directory.parent / "case.json").read_bytes())
    checkpoint = RepetitionManifest.from_directory(directory)
    return load_series(RetainedAttempt(directory, case, checkpoint), label)


class FakeServingServer:
    """A real CPU Completions peer with a bounded gate."""

    def __init__(self, *, block_first: bool = False, omit_done: bool = False) -> None:
        self.received: list[str] = []
        self.models: list[str] = []
        self.first_started = threading.Event()
        self.gate = threading.Event()
        self.gate_released = False
        self.block_first = block_first
        self.omit_done = omit_done
        self.frames: tuple[bytes, ...] = completion_frames()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                if self.path != "/v1/completions":
                    self.send_error(404)
                    return
                length = int(self.headers["content-length"])
                payload = json.loads(self.rfile.read(length))
                owner.received.append(payload["prompt"])
                owner.models.append(payload["model"])
                if len(owner.received) == 1:
                    owner.first_started.set()
                    if owner.block_first:
                        owner.gate_released = owner.gate.wait(30)
                frames = list(owner.frames)
                if not owner.omit_done:
                    frames.append(b"data: [DONE]\n\n")
                encodings = self.headers.get("accept-encoding", "").split(",")
                use_gzip = any(encoding.strip() == "gzip" for encoding in encodings)
                if use_gzip:
                    frames = [gzip.compress(b"".join(frames))]
                try:
                    self.send_response(200)
                    self.send_header("content-type", "text/event-stream")
                    self.send_header("transfer-encoding", "chunked")
                    if use_gzip:
                        self.send_header("content-encoding", "gzip")
                    self.end_headers()
                    for frame in frames:
                        self.wfile.write(f"{len(frame):x}\r\n".encode() + frame + b"\r\n")
                        self.wfile.flush()
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    @property
    def endpoint(self) -> ServingEndpoint:
        return ServingEndpoint(base_url=self.url, served_model_name=str(TEST_MODEL_ID))

    def __enter__(self) -> FakeServingServer:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.gate.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)
        assert not self.thread.is_alive()


def completion_frames() -> tuple[bytes, bytes]:
    return (
        b'data: {"choices":[{"text":"","finish_reason":"length"}],'
        b'"usage":{"completion_tokens":3,"prompt_tokens":5}}\r\n\r\n',
        b'data: {"choices":[],"usage":{"completion_tokens":3,"prompt_tokens":5,'
        b'"prompt_tokens_details":{"cached_tokens":4}}}\n\n',
    )


def client_case(tmp_path: Path, endpoint: str, *, future: bool = False) -> ClientBenchCase:
    prompts = tmp_path / "input-prompts.jsonl"
    ids = ("first", "second", "third")
    prompts.write_text("".join(json.dumps({"prompt_id": id, "text": id}) + "\n" for id in ids), encoding="utf-8")
    trace = tmp_path / "input-trace.jsonl"
    trace.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": id,
                    "model_id": str(TEST_MODEL_ID),
                    "arrival_seconds": 1000 if future and id == "third" else 0,
                    "prompt_id": id,
                    "max_new_tokens": 3,
                }
            )
            + "\n"
            for id in ids
        ),
        encoding="utf-8",
    )
    return ClientBenchCase.model_validate_json(
        json.dumps(
            {
                "id": str(TEST_CASE_ID),
                "description": "Observe HTTP admission, streaming timing and supervised cleanup.",
                "mode": "client",
                "max_inflight": 1,
                "warmup_requests_per_target": 0,
                "arrivals": {"kind": "jsonl", "path": str(trace)},
                "targets": [
                    {
                        "model_id": str(TEST_MODEL_ID),
                        "base_url": endpoint,
                        "prompts": {"kind": "jsonl", "path": str(prompts)},
                    }
                ],
            }
        )
    )
