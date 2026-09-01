from __future__ import annotations

import types
import typing
from collections.abc import Callable
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
import safetensors.torch
import torch

from tests.harness.runner.child import PythonChildProcess
from tests.harness.sglang.reference import ffn, ffn_protocol


def install_fake_child(
    monkeypatch: pytest.MonkeyPatch,
    *,
    output_factory: Callable[[ffn_protocol.FfnReferenceCaseSpec], dict[str, torch.Tensor]],
    completed_count: int | None = None,
    receive_error: BaseException | None = None,
) -> dict[str, object]:
    """Replace the owning module's child dependency with one in-memory control seam."""

    state: dict[str, object] = {
        "alive": False,
        "closed": False,
        "terminated": False,
        "timeouts": [],
        "job": None,
    }

    def start(
        cls: type[PythonChildProcess],
        name: str,
        target: Callable[[Connection, ffn_protocol.FfnReferenceJob], None],
        spec: ffn_protocol.FfnReferenceJob,
        *,
        log_path: Path,
    ) -> PythonChildProcess:
        del cls, target
        assert name == "sglang-ffn-reference"
        assert log_path == spec.workdir / "child.log"
        state["job"] = spec
        state["alive"] = receive_error is not None

        def receive(expected: type[object], *, timeout_seconds: float) -> object:
            assert expected is ffn_protocol.FfnReferenceCompleted
            typing.cast(list[float], state["timeouts"]).append(timeout_seconds)
            if receive_error is not None:
                raise receive_error
            for case in spec.cases:
                with case.output_path.open("xb") as output_file:
                    output_file.write(safetensors.torch.save(output_factory(case)))
            return ffn_protocol.FfnReferenceCompleted(
                case_count=len(spec.cases) if completed_count is None else completed_count
            )

        def wait(*, timeout_seconds: float) -> None:
            typing.cast(list[float], state["timeouts"]).append(timeout_seconds)

        def close() -> None:
            state["closed"] = True

        return typing.cast(
            PythonChildProcess,
            types.SimpleNamespace(
                process=types.SimpleNamespace(is_alive=lambda: state["alive"]),
                receive=receive,
                wait=wait,
                close=close,
            ),
        )

    def terminate_all(
        cls: type[PythonChildProcess],
        processes: tuple[PythonChildProcess, ...],
    ) -> None:
        del cls
        assert len(processes) == 1
        state["terminated"] = True
        state["alive"] = False

    monkeypatch.setattr(PythonChildProcess, "start", classmethod(start))
    monkeypatch.setattr(PythonChildProcess, "terminate_all", classmethod(terminate_all))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    return state


def make_case(case_id: str = "case", layer_id: int = 0) -> ffn.SglangFfnReferenceCase:
    """Construct one small valid CPU case."""

    return ffn.SglangFfnReferenceCase(
        case_id=case_id,
        layer_id=layer_id,
        hidden_states=torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )


def test_runner_rejects_invalid_requests_before_creating_workdir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    with pytest.raises(ValueError, match="timeout_seconds"):
        ffn.SglangFfnReferenceRunner(workdir=tmp_path / "timeout", timeout_seconds=0)

    invalid_batches = (
        (),
        (make_case(""),),
        (make_case("duplicate"), make_case("duplicate", 1)),
        (make_case(layer_id=-1),),
        (
            ffn.SglangFfnReferenceCase(
                case_id="rank",
                layer_id=0,
                hidden_states=torch.ones(4),
            ),
        ),
        (
            ffn.SglangFfnReferenceCase(
                case_id="integer",
                layer_id=0,
                hidden_states=torch.ones((1, 4), dtype=torch.int32),
            ),
        ),
    )
    for index, cases in enumerate(invalid_batches):
        workdir = tmp_path / f"invalid-{index}"
        runner = ffn.SglangFfnReferenceRunner(workdir=workdir, timeout_seconds=5)
        with pytest.raises(ValueError):
            runner.run(model_path=model_path, tensor_parallel_size=1, cases=cases)
        assert not workdir.exists()

    runner = ffn.SglangFfnReferenceRunner(workdir=tmp_path / "placement", timeout_seconds=5)
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        runner.run(model_path=model_path, tensor_parallel_size=3, cases=(make_case(),))
    with pytest.raises(ValueError, match="model_path"):
        runner.run(model_path=tmp_path / "missing", tensor_parallel_size=1, cases=(make_case(),))


def test_runner_preserves_order_normalizes_inputs_and_is_single_use(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    def output_factory(spec: ffn_protocol.FfnReferenceCaseSpec) -> dict[str, torch.Tensor]:
        tensors = safetensors.torch.load_file(spec.input_path)
        hidden_states = tensors["hidden_states"]
        assert hidden_states.device.type == "cpu"
        assert hidden_states.is_contiguous()
        assert not hidden_states.requires_grad
        return {"hidden_states": hidden_states + spec.layer_id}

    state = install_fake_child(monkeypatch, output_factory=output_factory)
    first_input = torch.arange(12, dtype=torch.float32).reshape(3, 4).T.requires_grad_()
    cases = (
        ffn.SglangFfnReferenceCase("second", 2, first_input),
        ffn.SglangFfnReferenceCase("first", 1, torch.ones((2, 4))),
    )
    workdir = tmp_path / "reference"
    runner = ffn.SglangFfnReferenceRunner(workdir=workdir, timeout_seconds=5)

    results = runner.run(model_path=model_path, tensor_parallel_size=2, cases=cases)

    assert tuple(result.case_id for result in results) == ("second", "first")
    assert tuple(result.layer_id for result in results) == (2, 1)
    torch.testing.assert_close(results[0].output, first_input.detach().contiguous() + 2)
    torch.testing.assert_close(results[1].output, torch.full((2, 4), 2.0))
    assert all(result.routing is None for result in results)
    assert typing.cast(ffn_protocol.FfnReferenceJob, state["job"]).tensor_parallel_size == 2
    assert state["closed"] is True
    assert state["terminated"] is False
    timeouts = typing.cast(list[float], state["timeouts"])
    assert len(timeouts) == 2 and 0 < timeouts[1] <= timeouts[0] <= 5
    for index in range(2):
        assert {path.name for path in (workdir / "cases" / f"case-{index:04d}").iterdir()} == {
            "input.safetensors",
            "output.safetensors",
        }

    with pytest.raises(FileExistsError):
        runner.run(model_path=model_path, tensor_parallel_size=1, cases=(make_case(),))


def test_runner_projects_moe_routing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()

    def output_factory(spec: ffn_protocol.FfnReferenceCaseSpec) -> dict[str, torch.Tensor]:
        hidden_states = safetensors.torch.load_file(spec.input_path)["hidden_states"]
        return {
            "hidden_states": hidden_states,
            "topk_ids": torch.tensor([[3, 1], [2, 0], [1, 3]], dtype=torch.int32),
            "topk_weights": torch.full((3, 2), 0.5, dtype=torch.float32),
        }

    install_fake_child(monkeypatch, output_factory=output_factory)
    result = ffn.SglangFfnReferenceRunner(
        workdir=tmp_path / "reference",
        timeout_seconds=5,
    ).run(model_path=model_path, tensor_parallel_size=1, cases=(make_case(),))[0]

    assert result.routing is not None
    assert result.routing.topk_ids.dtype == torch.int32
    assert result.routing.topk_weights.dtype == torch.float32
    assert result.routing.topk_ids.tolist() == [[3, 1], [2, 0], [1, 3]]


def test_runner_terminates_failed_child_without_returning_partial_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    state = install_fake_child(
        monkeypatch,
        output_factory=lambda spec: {"hidden_states": safetensors.torch.load_file(spec.input_path)["hidden_states"]},
        receive_error=RuntimeError("child failed"),
    )
    runner = ffn.SglangFfnReferenceRunner(workdir=tmp_path / "reference", timeout_seconds=5)

    with pytest.raises(RuntimeError, match="child failed"):
        runner.run(model_path=model_path, tensor_parallel_size=1, cases=(make_case(),))

    assert state["terminated"] is True
    assert state["closed"] is True


@pytest.mark.parametrize(
    ("completed_count", "output_factory", "message"),
    [
        (
            0,
            lambda spec: {"hidden_states": safetensors.torch.load_file(spec.input_path)["hidden_states"]},
            "completed 0",
        ),
        (
            None,
            lambda spec: {"wrong": safetensors.torch.load_file(spec.input_path)["hidden_states"]},
            "invalid tensor keys",
        ),
    ],
)
def test_runner_rejects_incomplete_or_malformed_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_count: int | None,
    output_factory: Callable[[ffn_protocol.FfnReferenceCaseSpec], dict[str, torch.Tensor]],
    message: str,
) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    install_fake_child(
        monkeypatch,
        output_factory=output_factory,
        completed_count=completed_count,
    )
    runner = ffn.SglangFfnReferenceRunner(workdir=tmp_path / "reference", timeout_seconds=5)

    with pytest.raises(RuntimeError, match=message):
        runner.run(model_path=model_path, tensor_parallel_size=1, cases=(make_case(),))
