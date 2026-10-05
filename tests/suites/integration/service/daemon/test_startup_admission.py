from __future__ import annotations

import subprocess
import sys
from http import HTTPStatus

import pytest

from xpool.fabric import FabricRole
from xpool.native import ABI_VERSION
from xpool.service.wire import AgentStartupAdmission
from xpool.utils.mps import MpsEndpoint, MpsProbeResult, MpsScope
from xpool.utils.procs import ProcUniqId
from xtest.harness.support.config import reset_global_config, synthetic_config
from xtest.harness.support.service.daemon import (
    atnagent_registration,
    create_app,
    deterministic_daemon_dependencies,
    ffnagent_registration,
    request,
)

pytestmark = pytest.mark.usefixtures(reset_global_config.__name__, deterministic_daemon_dependencies.__name__)


@pytest.mark.parametrize("role", [FabricRole.ATNAGENT, FabricRole.FFNAGENT])
def test_agent_admission_binds_exact_startup_and_configured_device(
    role: FabricRole, monkeypatch: pytest.MonkeyPatch
) -> None:
    uuids = ("GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002")
    scope = MpsScope(MpsEndpoint(uuids[:1]))
    monkeypatch.setattr(scope, "probe", lambda: MpsProbeResult(True, None, "CPU admission check"))
    root = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        identity = ProcUniqId(root.pid)
        selected_device = 0 if role is FabricRole.ATNAGENT else 1
        payload = AgentStartupAdmission(
            pid=identity.pid,
            create_time=identity.create_time,
            abi_version=ABI_VERSION,
            role=role,
            device=selected_device,
        )
        app = create_app(synthetic_config())
        control = app.state.control_plane
        control.mps_scope = None
        control.device_uuids = None
        assert (
            request(app, "POST", "/startup/agent", json=payload.model_dump(mode="json")).status_code
            == HTTPStatus.SERVICE_UNAVAILABLE
        )
        control.mps_scope = scope
        control.device_uuids = uuids
        registration_path = "/atnagent/register" if role is FabricRole.ATNAGENT else "/ffnagent/register"
        registration = (
            atnagent_registration(device=selected_device, pid=identity.pid)
            if role is FabricRole.ATNAGENT
            else ffnagent_registration(device=selected_device, pid=identity.pid)
        )
        assert request(app, "POST", registration_path, json=registration).status_code == HTTPStatus.CONFLICT
        response = request(app, "POST", "/startup/agent", json=payload.model_dump(mode="json"))
        assert response.status_code == HTTPStatus.NO_CONTENT
        assert response.content == b""
        assert request(app, "POST", registration_path, json=registration).status_code == HTTPStatus.NO_CONTENT
        assert (
            request(app, "POST", "/startup/agent", json=payload.model_dump(mode="json")).status_code
            == HTTPStatus.NO_CONTENT
        )
        for conflicting in (
            payload.model_copy(update={"create_time": identity.create_time + 1}),
            payload.model_copy(update={"abi_version": ABI_VERSION + 1}),
            payload.model_copy(update={"device": 1 - selected_device}),
        ):
            assert (
                request(app, "POST", "/startup/agent", json=conflicting.model_dump(mode="json")).status_code
                == HTTPStatus.CONFLICT
            )
        control.admission_closed = True
        assert (
            request(app, "POST", "/startup/agent", json=payload.model_dump(mode="json")).status_code
            == HTTPStatus.SERVICE_UNAVAILABLE
        )
    finally:
        if root.poll() is None:
            root.terminate()
            root.wait(timeout=5)
