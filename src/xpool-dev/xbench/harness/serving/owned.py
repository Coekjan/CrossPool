"""Start and retire an owned Xpool deployment for one benchmark attempt."""

from __future__ import annotations

import os
import platform
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

from xbench.harness.serving.case import OwnedBenchCase, ResolvedDeployment, ServingEndpoint
from xbench.harness.serving.environment import (
    ENVIRONMENT_KEYS,
    ServingCommand,
    ServingEnvironment,
    ServingSoftware,
    capture_serving_environment,
    package_versions,
)
from xbench.harness.serving.workload import PreparedWorkload
from xkit.envs import DETERMINISTIC_MODE, DETERMINISTIC_PATH
from xkit.serving.sglang.launch import ServingLaunch, SglangLaunchModel
from xkit.serving.sglang.system import XpoolServingSystem
from xpool.config import XpoolConfig


@dataclass(slots=True)
class OwnedServingAdapter:
    case: OwnedBenchCase
    deployment: ResolvedDeployment
    errors: list[str]
    system: XpoolServingSystem = field(default_factory=XpoolServingSystem)

    @asynccontextmanager
    async def start(self, *, workdir: Path, startup_timeout_seconds: float) -> AsyncIterator[ServingEnvironment]:
        config = XpoolConfig.model_validate(self.deployment.effective)
        environment = dict(os.environ)
        DETERMINISTIC_MODE.clear(env=environment)
        DETERMINISTIC_PATH.clear(env=environment)
        launch = ServingLaunch(
            config=config,
            environment=environment,
            cwd=self.deployment.cwd,
            models=tuple(
                SglangLaunchModel(target.model_id, target.graph_mode.settings(), served_model_name=str(target.model_id))
                for target in self.case.targets
            ),
        )
        try:
            await self.system.start_async(launch, workdir=workdir, startup_timeout_seconds=startup_timeout_seconds)
            effective = self.system.launch
            endpoints = {
                endpoint.model_id: ServingEndpoint(base_url=endpoint.base_url, served_model_name=str(endpoint.model_id))
                for endpoint in self.system.endpoints
            }
            uuids = tuple(effective.environment["CUDA_VISIBLE_DEVICES"].split(","))
            software = ServingSoftware(
                python=platform.python_version(),
                prefix=sys.prefix,
                packages=package_versions(("sglang", "torch"), self.errors),
            )
            commands = tuple(
                ServingCommand(
                    argv=server.command,
                    environment={
                        name: effective.environment[name] for name in ENVIRONMENT_KEYS if name in effective.environment
                    },
                    software=software,
                )
                for server in self.system.servers
            )
            yield capture_serving_environment(
                endpoints,
                uuids,
                commands=commands,
                errors=self.errors,
                role_device_uuids={
                    "atn": tuple(uuids[index] for index in config.atn.devices),
                    "ffn": tuple(uuids[index] for index in config.ffn.devices),
                },
                runtime_config=effective.config_path,
            )
        finally:
            print("xbench phase=shutdown", file=sys.stderr)
            await self.system.close_async()

    async def validate_workload(self, workload: PreparedWorkload) -> None:
        deadline = self.system.startup_deadline
        if deadline is None:
            raise RuntimeError("owned request-limit check requires the serving startup deadline")
        prompts = {(prompt.model_id, prompt.prompt_id): prompt for prompt in workload.prompts}
        for server in self.system.servers:
            model_id = server.model.model_id
            limits = await server.request_limits(context_length=workload.model_contexts[model_id], deadline=deadline)
            for request in (*workload.requests, *workload.warmup):
                if request.model_id != model_id:
                    continue
                input_tokens = prompts[model_id, request.prompt_id].input_tokens
                if input_tokens is None:
                    raise ValueError(f"{model_id}: prepared request {request.request_id!r} lacks input length")
                try:
                    limits.validate_request(input_tokens, request.max_new_tokens)
                except ValueError as error:
                    raise ValueError(f"{model_id}: request {request.request_id!r}: {error}") from error

    def check_alive(self) -> None:
        self.system.check_alive()
