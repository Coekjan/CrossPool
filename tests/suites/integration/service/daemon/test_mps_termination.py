from __future__ import annotations

import subprocess
import sys
from http import HTTPStatus

import psutil
import pytest

from xpool.native import ABI_VERSION
from xpool.service.wire import MpsClientTermination
from xpool.utils.mps import MpsEndpoint, MpsProbeResult, MpsScope
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import create_app, deterministic_daemon_dependencies, request

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


@pytest.mark.parametrize(
    "termination_error_type",
    [None, RuntimeError, psutil.NoSuchProcess],
    ids=["confirmed", "unconfirmed", "process-exited"],
)
def test_context_termination_requires_exact_identity_and_remains_available_during_close(
    termination_error_type: type[Exception] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    # CPU processes establish real identities without creating device contexts.
    uuids = ("GPU-00000000-0000-0000-0000-000000000001",)
    scope = MpsScope(MpsEndpoint(uuids))
    monkeypatch.setattr(scope, "probe", lambda: MpsProbeResult(True, None, "CPU admission check"))
    calls: list[tuple[ProcUniqId, float]] = []

    def terminate_client(target: ProcUniqId, *, deadline: float) -> None:
        calls.append((target, deadline))
        if termination_error_type is not None:
            raise termination_error_type(target.pid)

    monkeypatch.setattr(scope, "terminate_client", terminate_client)
    root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        identity = ProcUniqId(root.pid)
        app = create_app(synthetic_config())
        control = app.state.control_plane
        control.mps_scope = scope
        payload = MpsClientTermination(
            pid=identity.pid, create_time=identity.create_time, abi_version=ABI_VERSION, deadline=1020.0
        )
        path = "/serving/mps/terminate-client"
        for invalid in (
            payload.model_copy(update={"create_time": identity.create_time + 1}),
            payload.model_copy(update={"abi_version": ABI_VERSION + 1}),
        ):
            assert request(app, "POST", path, json=invalid.model_dump(mode="json")).status_code == HTTPStatus.CONFLICT
        assert not calls
        control.begin_close(deadline=1010.0)
        result = request(app, "POST", path, json=payload.model_dump(mode="json"))
        assert result.status_code == (
            HTTPStatus.NO_CONTENT if termination_error_type is None else HTTPStatus.SERVICE_UNAVAILABLE
        )
        if termination_error_type is not None:
            assert result.json()["detail"]["kind"] == "not_ready"
        assert calls == [(identity, 1010.0)]
        assert root.poll() is None
        assert control.admission_closed and not control.closed
        assert not scope.closed
    finally:
        root.terminate()
        root.wait(timeout=5)
