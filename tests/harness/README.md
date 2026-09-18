# Test Harness Architecture

`xtest run` is the repository test composition root. The harness under
this directory implements collection, resource-aware scheduling, supervised
process ownership, and durable result aggregation. Harness modules are reusable
test infrastructure; they must not import collected modules from
`tests/suites/`.

## Execution Pipeline

1. `runner/collection.py` asks pytest to collect concrete parameterized items and
   records their resource markers and scheduling metadata.
2. `runner/plan.py` validates that metadata and builds the typed suite plan.
3. `runner/task.py` groups compatible cases into independently supervised
   tasks. `tests/cli.py` chooses serving-engine files before collection,
   executes CTest, then creates a `SuiteRunner` for Unit, Integration, E2E,
   and explicitly selected Models in that order.
4. GPU tasks are ordered by GPU count and estimated duration, then backfilled
   over idle devices. CPU tasks use the same supervision boundary without a GPU
   lease.
5. `runner/pytest_report.py` and `runner/results.py` classify JUnit and task
   outcomes. SGLang drivers write inference, timing, and observer evidence;
   injected artifact-group adapters aggregate cross-task parity into the final
   suite result.

Direct pytest and CTest commands remain focused debugging interfaces. They do
not reproduce cross-task scheduling or whole-run parity aggregation.

## Process Ownership

The runner owns scheduling, terminal signals, the global child-subreaper
fallback, and GPU leases. Each task runs under one `SupervisedTaskScope`:

```text
runner (fallback subreaper)
└── task supervisor (setsid, normal subreaper)
    └── pytest root (separate process group)
        └── daemon, Agents, SGLang, and further descendants
```

Normal cancellation travels over the typed runner-to-supervisor pipe. The
Supervisor repeatedly discovers exact process identities, sends TERM and then
KILL when deadlines expire, and reaps adopted children. Kernel child state is
the emptiness proof: after the root is reaped, `waitpid` reporting `ECHILD`
means no task child remains. Recursive process enumeration is used for signal
targets and diagnostics, not as proof of emptiness.

A Supervisor-local infrastructure exception becomes
`TaskCompletionKind.INFRASTRUCTURE_FAILED` only after local cleanup proves the
task domain empty; the runner cancels active same-stage tasks, does not admit
pending or later stages, and reports infrastructure error code 2. A
`TaskSupervisorFailed` message becomes
`TaskSupervisionFailure`; until fallback proves the domain empty, the runner
stops scheduling and retains the affected lease. Directly terminating a
Supervisor is owner loss, not a normal cancellation mechanism.

## GPU And Endpoint Ownership

`runner/gpu.py` derives the eligible pool once from startup
`CUDA_VISIBLE_DEVICES` and normalizes it to physical UUIDs. The visible set is
an externally exclusive test allocation; the harness does not coordinate GPUs
with another CrossPool invocation. Every selected device must pass serialized MPS
preflight before GPU work starts. Within one run, a task sees only the UUIDs in
its `GpuPool` lease, and that lease is released only after its complete process
domain is proved empty.

`runner/network.py` reserves endpoints by retaining a listener after a complete
`bind -> listen -> local connect -> accept` qualification. SGLang helpers own
HTTP, NCCL, gRPC, handshake, and ZMQ-derived endpoints as one family. A bindable
but locally unreachable member rejects the family; only a post-cleanup
`EADDRINUSE` is a retryable conflict. Tests must not select ports with
uncoordinated bind-and-close probes.

## Artifacts

Each run lives under `.xpool-cache/test-runs/<run-id>/`; each task has a stable
subdirectory containing its combined process log and JUnit report. E2E tasks
also retain exact inference request/response JSON, graph-mode timing, token
parity, and enabled observer output. Artifacts are written before semantic
validation where possible so a failed request remains diagnosable.

`xtest clean` owns explicit retention cleanup. It keeps the newest 20 inactive
entries by default and accepts `--keep N`, `--all`, and `--dry-run`. Active
runs are protected by locks; unrecognized entries are cleanup candidates because the
result root is dedicated test storage. Run creation and cleanup resolve the
result root once and serialize through `.cleanup.lock` before acquiring or
probing any `.run.lock`. The result root or one of its parents may be a symbolic
link, but symbolic-link entries inside the resolved root are unlinked without
following their targets. Test execution never performs implicit retention
cleanup.

## Module Boundaries

- `runner/` owns collection, planning, scheduling, supervision, GPU leases,
  endpoint allocation, requirements, generic artifact adapters, and results.
- `native/` owns component-scoped native subprocess drivers.
- `support/` owns assertions, fixtures, fakes, and focused reusable setup; it
  does not own scheduling or process topology.
- `sglang/` owns the shared declarative manifest. `sglang/serving/` owns the
  installed-command server topology, attempts, readiness, graph evidence, and
  token parity; `sglang/reference/` owns isolated original-SGLang FFN execution
  and does not import the serving workflow.

Add reusable machinery here only when more than one behavioral test needs it.
Put assertions and concrete scenarios in the lowest suitable suite. Avoid
implicit fixture activation through directory-level `conftest.py` imports.
