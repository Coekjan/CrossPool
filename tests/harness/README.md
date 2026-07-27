# Test Harness Architecture

`python -m tests` is the repository test composition root. The harness under
this directory implements collection, resource-aware scheduling, supervised
process ownership, and durable result aggregation. Harness modules are reusable
test infrastructure; they must not import collected modules from
`tests/suites/`.

## Execution Pipeline

1. `collection.py` asks pytest to collect concrete parameterized items and
   records their resource markers and scheduling metadata.
2. `test_plan.py` validates that metadata and builds the typed suite plan.
3. `execution_task.py` groups compatible cases into independently supervised
   tasks. `runner.py` executes CTest, Unit, Integration, and E2E stages in that
   order, admitting E2E only after Integration succeeds.
4. GPU tasks are ordered by GPU count and estimated duration, then backfilled
   over idle devices. CPU tasks use the same supervision boundary without a GPU
   lease.
5. `pytest_report.py`, `results.py`, and the SGLang parity helpers classify
   JUnit, inference, timing, observer, and parity artifacts into the final suite
   result.

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
task domain empty; unrelated scopes may continue and the suite reports
infrastructure error code 2. `TaskSupervisorFailed` means emptiness is unproven,
so the runner stops scheduling, retains the affected lease, and performs
global fallback cleanup. Directly terminating a Supervisor is owner loss, not
a normal cancellation mechanism.

## GPU And Endpoint Ownership

`gpu.py` derives the eligible pool once from startup `CUDA_VISIBLE_DEVICES`,
normalizes it to physical UUIDs, and holds the whole-run lock. Every selected
device must pass serialized MPS preflight before GPU work starts. A task sees
only the UUIDs in its lease, and that lease is released only after its complete
process domain is proved empty.

`network.py` reserves endpoint families rather than isolated ports. SGLang
helpers materialize HTTP, NCCL, gRPC, and DP-derived endpoints from that owned
family and release them with the supervised task. Tests must not select ports
with uncoordinated bind-and-close probes.

## Artifacts

Each run lives under `.xpool-cache/test-runs/<run-id>/`; each task has a stable
subdirectory containing its combined process log and JUnit report. E2E tasks
also retain exact inference request/response JSON, graph-mode timing, token
parity, and enabled observer output. Artifacts are written before semantic
validation where possible so a failed request remains diagnosable.

`XPOOL_TEST_KEEP_RUNS` bounds recognized historical runs. Active runs are
protected by locks, and unrecognized directories are never cleanup targets.

## Module Boundaries

- Top-level harness modules own collection, planning, scheduling, supervision,
  GPU leases, endpoint allocation, requirements, and result aggregation.
- `native/` owns component-scoped native subprocess setup and assertions.
- `runtime/` and `service/` own focused reusable fixtures for those Python
  subsystems.
- `sglang/` owns the declarative manifest, installed-command server topology,
  probes, graph evidence, observers, and token parity.

Add reusable machinery here only when more than one behavioral test needs it.
Put assertions and concrete scenarios in the lowest suitable suite. Avoid
implicit fixture activation through directory-level `conftest.py` imports.
