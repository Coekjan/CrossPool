from __future__ import annotations

import logging
import os
import select
import signal
import subprocess
import sys
from pathlib import Path
from time import monotonic, sleep

import xtest
from xkit.task import get_task_root
from xpool.utils.device import visible_uuids
from xpool.utils.mps import MPS_CLEANUP_TIMEOUT_S, MpsEndpoint, MpsScope
from xpool.utils.procs import ProcUniqId
from xpool.utils.sighandler import defer_signal_exceptions

logger = logging.getLogger(__name__)


@xtest.requirements(device_count=1)
def test_owned_client_termination_keeps_peer_and_server_usable(tmp_path: Path) -> None:
    """Qualify actual driver termination with pending work and an independent peer.

    Both clients are owned single-process programs, with no IPC or child creation.
    Unconfirmed cleanup retains the controller and diagnostics for inspection.
    """

    program = """
import os
import sys
import torch

from xpool.utils.mps import MpsEndpoint
from xpool.utils.device import visible_uuids

torch.cuda.set_device(0)
value = torch.ones(1, device="cuda")
torch.cuda.synchronize()
MpsEndpoint(visible_uuids()).require_client()
print("ready", flush=True)
for command in sys.stdin:
    if command.strip() == "work":
        value.add_(1)
        print(value.item(), flush=True)
    elif command.strip() == "busy":
        torch.cuda._sleep(20_000_000_000)
        done = torch.cuda.Event()
        done.record()
        print(f"busy {not done.query()}", flush=True)
    elif command.strip() == "stop":
        torch.cuda.synchronize()
        break
    else:
        raise ValueError(command)
"""
    task = get_task_root()
    proof = None if task is None else task.register_scope()
    scope: MpsScope | None = None
    clients: list[subprocess.Popen[str]] = []
    terminated: set[int] = set()
    cleanup_deadline: float | None = None

    def exchange(client: subprocess.Popen[str], command: str | None) -> str:
        if client.stdout is None or client.stdin is None:
            raise RuntimeError("owned MPS client has no command channel")
        if command is not None:
            client.stdin.write(f"{command}\n")
            client.stdin.flush()
        if not select.select((client.stdout,), (), (), 30.0)[0]:
            raise TimeoutError("owned MPS client did not reply")
        reply = client.stdout.readline().strip()
        if not reply:
            raise RuntimeError(f"owned MPS client exited before replying: {client.poll()}")
        return reply

    try:
        if task is not None:
            task.activate()
        uuids = visible_uuids()[:1]
        with defer_signal_exceptions():
            scope = MpsScope(MpsEndpoint(uuids))
        scope.start()
        environment = dict(os.environ)
        environment.update(scope.endpoint.environment())
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(uuids)
        for name in ("target", "peer"):
            with (tmp_path / f"{name}.log").open("w", encoding="utf-8") as log:
                with defer_signal_exceptions():
                    client = subprocess.Popen(
                        [sys.executable, "-c", program],
                        env=environment,
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=log,
                        text=True,
                    )
                    clients.append(client)
            assert exchange(client, None) == "ready"
        target, peer = clients
        target_identity = ProcUniqId(target.pid)
        servers = scope.observe_servers()
        assert servers
        assert exchange(peer, "work") == "2.0"
        assert exchange(target, "busy") == "busy True"

        cleanup_deadline = monotonic() + MPS_CLEANUP_TIMEOUT_S
        scope.terminate_client(target_identity, deadline=cleanup_deadline)
        terminated.add(target.pid)
        assert target.poll() is None
        target_identity.send_signal(signal.SIGKILL)
        target.wait(timeout=5)
        assert target.returncode == -signal.SIGKILL
        assert scope.observe_servers() == servers
        assert exchange(peer, "work") == "3.0"
        assert scope.probe().online is True
    finally:
        if cleanup_deadline is None:
            cleanup_deadline = monotonic() + MPS_CLEANUP_TIMEOUT_S
        try:
            for client in clients:
                if client.poll() is None and client.stdin is not None:
                    client.stdin.write("stop\n")
                    client.stdin.flush()
            for client in clients:
                client.wait(timeout=max(0.0, cleanup_deadline - monotonic()))
            if any(client.returncode != 0 and client.pid not in terminated for client in clients):
                raise RuntimeError("MPS qualification client cleanup unconfirmed")
            for client in clients:
                if client.stdin is not None:
                    client.stdin.close()
                if client.stdout is not None:
                    client.stdout.close()
            if scope is not None:
                scope.stop(deadline=cleanup_deadline)
            if proof is not None:
                proof.complete()
        except Exception:
            logger.exception("MPS qualification cleanup unconfirmed; retaining controller and diagnostics")
            while True:
                sleep(1)

    assert scope is not None and scope.closed
    assert scope.controller is not None and scope.controller.poll() == 0
    assert not scope.endpoint.directory.exists()
