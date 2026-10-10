"""Resource-fit admission and proven-empty retirement for supervised tool tasks."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic, sleep

from xkit.device import DeviceLease, DevicePool
from xkit.supervisor import (
    SupervisedTaskScope,
    TaskCompletion,
    TaskCompletionKind,
    TaskScopeFailure,
    TaskScopeState,
    TaskStartFailure,
    TaskSupervisionFailure,
    drain_unprotected_subreaper_descendants,
)

__all__ = ["ActiveTask", "TaskScheduler"]


@dataclass(frozen=True, slots=True)
class ActiveTask[T]:
    """Retained scope and optional device grant for a caller-owned task value."""

    task: T
    scope: SupervisedTaskScope
    lease: DeviceLease | None
    started_at: float
    device_assignments: str


class TaskScheduler[T]:
    """Borrow a pool and supervise tasks; callers own priorities and verdicts.

    Completion callbacks run after verified retirement and lease return. They
    may submit follow-up work or stop admission. Admission closure is monotonic
    across calls to run, so subsequent stages cannot reopen an invocation.
    """

    def __init__(
        self,
        *,
        device_pool: DevicePool | None,
        on_complete: Callable[[ActiveTask[T], TaskCompletion], None],
    ) -> None:
        self.device_pool = device_pool
        self.on_complete = on_complete
        self.pending: list[tuple[T, int]] = []
        self.active: dict[str, ActiveTask[T]] = {}
        self.retained_leases: list[DeviceLease] = []
        self.admission_open = True
        self.cancellation_requested = False

    def submit(self, task: T, *, device_count: int) -> None:
        """Append caller-prioritized work; closed admission ignores follow-ups."""
        if not self.admission_open:
            return
        if device_count and (self.device_pool is None or device_count > len(self.device_pool.uuids)):
            raise ValueError(f"task requires {device_count} devices, exceeding the eligible device pool")
        self.pending.append((task, device_count))

    def stop_admission(self) -> None:
        """Discard pending work and prohibit follow-ups while active scopes drain."""
        self.admission_open = False
        self.pending.clear()

    def request_cancel(self) -> None:
        """Close admission and request fan-out cancellation on the scheduling loop."""
        self.stop_admission()
        self.cancellation_requested = True

    def run(self, start_task: Callable[[T], None]) -> None:
        """Admit every fitting task and poll active scopes until all work retires."""
        while self.pending or self.active:
            if self.cancellation_requested:
                self.cancel_active()
                return
            completed = self.poll()
            launched = self.admit(start_task)
            if not launched and not completed and (self.pending or self.active):
                if not self.active:
                    raise RuntimeError("pending tasks cannot fit the idle device pool")
                sleep(0.05)

    def admit(self, start_task: Callable[[T], None]) -> bool:
        """Backfill by first fit without changing the caller's pending order."""
        launched = False
        while self.admission_open:
            index = next(
                (
                    index
                    for index, (task, count) in enumerate(self.pending)
                    if count == 0 or (self.device_pool is not None and count <= self.device_pool.available_count)
                ),
                None,
            )
            if index is None:
                break
            task = self.pending.pop(index)[0]
            start_task(task)
            launched = True
        return launched

    def start(
        self,
        task: T,
        *,
        name: str,
        device_count: int,
        command: list[str],
        cwd: Path,
        env: dict[str, str],
        log_path: Path,
        timeout_seconds: float | None,
    ) -> ActiveTask[T]:
        """Borrow devices only after launch inputs exist, then retain scope ownership.

        TaskStartFailure proves startup rollback and returns the lease. Any
        other startup failure retains it because the attempted domain is unproven.
        """
        lease = None
        if device_count:
            if self.device_pool is None:
                raise RuntimeError(f"device task {name} has no device pool")
            lease = self.device_pool.try_lease(device_count)
            if lease is None:
                raise RuntimeError(f"scheduler selected device task {name} without capacity")
            env = {**env, "CUDA_VISIBLE_DEVICES": ",".join(lease.uuids)}
        assignments = (
            ",".join(f"{self.device_pool.physical_index_by_uuid[uuid]}:{uuid}" for uuid in lease.uuids)
            if lease is not None and self.device_pool is not None
            else "none"
        )
        started_at = monotonic()
        try:
            scope = SupervisedTaskScope.start(
                name, command, cwd=cwd, env=env, log_path=log_path, timeout_seconds=timeout_seconds
            )
        except TaskStartFailure:
            if lease is not None:
                assert self.device_pool is not None
                self.device_pool.release(lease)
            raise
        except BaseException:
            if lease is not None:
                self.retained_leases.append(lease)
            raise
        running = ActiveTask(task, scope, lease, started_at, assignments)
        self.active[name] = running
        return running

    def poll(self) -> int:
        """Deliver terminal completions only after scope closure and lease return."""
        completed = 0
        for name, running in tuple(self.active.items()):
            completion = running.scope.poll()
            if completion is not None:
                self.retire(name, running, completion)
                completed += 1
        return completed

    def retire(self, name: str, running: ActiveTask[T], completion: TaskCompletion) -> None:
        running.scope.close()
        if running.lease is not None:
            assert self.device_pool is not None
            self.device_pool.release(running.lease)
        del self.active[name]
        self.on_complete(running, completion)

    def cancel_active(self) -> None:
        """Use fan-out/fallback retirement and retain grants after a failed close.

        A retained completion survives domain recovery; cleanup failure never
        fabricates another task verdict. The caller owns failure compensation.
        """
        self.request_cancel()
        running_tasks = tuple(self.active.items())
        failures: list[str] = []
        if running_tasks:
            try:
                SupervisedTaskScope.terminate_all(tuple(running.scope for name, running in running_tasks))
            except TaskScopeFailure as error:
                failures.append(str(error))
        for name, running in running_tasks:
            if running.scope.state not in (TaskScopeState.COMPLETED, TaskScopeState.DRAINED):
                continue
            completion = running.scope.completion or TaskCompletion(
                TaskCompletionKind.INFRASTRUCTURE_FAILED, None, "task cancelled; scope safely drained"
            )
            try:
                self.retire(name, running, completion)
            except TaskSupervisionFailure as error:
                failures.append(str(error))
        try:
            drain_unprotected_subreaper_descendants()
        except TaskScopeFailure as error:
            failures.append(str(error))
        if failures:
            raise TaskScopeFailure("; ".join(failures))

    @property
    def resources_releasable(self) -> bool:
        """Whether every scope and borrowed device grant was safely retired."""
        return (
            not self.active
            and not self.retained_leases
            and (self.device_pool is None or not self.device_pool.active_leases)
        )
