# Unified Timeline Observability: Shared Contract

## Goal

Freeze the C0 contract that allows bounded runtime collection and offline
reconstruction to be implemented independently. The target is one single-Host
Transport/Fabric/Graph/Routing timeline with protocol-derived causality,
paired Host/Device clock sampling, and explicit uncertainty and data quality.

This directory specifies target behavior. It does not declare that runtime
collection, clock calibration, or two-Device qualification is implemented.
The independently usable contracts are:

- [Shared contract](shared-contract.md): events, identities, Native interfaces,
  encoding, configuration, lifecycle, attachments, and legacy snapshots.
- [Calibration](calibration.md): deterministic model and numerical contract.
- [Reference fixtures](fixtures/README.md): fixed inputs, byte vectors, expected
  reconstruction, legacy projections, and calibration outputs.

## Baseline

The source baseline is `21554ea78533590ff2f122ce05f91b485d6da156` on `main`.
Current implementation owners are [Devkit](../../designs/devkit.md),
[Transport](../../designs/transport.md), [Fabric](../../designs/fabric.md),
[FFN Execution](../../designs/ffn-execution.md), and
[Control Plane](../../designs/control-plane.md).

Transport and Fabric mutate one bounded Record per local operation and read
snapshots after quiescence or synchronization. Their local trace identifiers
are not cross-process identities. The concrete correlation seam is
`transport::execute_request()` calling
`fabric::atnagent::execute(fabric_arena, transport_arena)`; the Fabric call
constructs the actual InvocationKey before Submission publication.
The InvocationKey comment attributing its sequence to an Instance rank must
be corrected when that path is instrumented; its allocation algorithm remains
production-owned.

Native Graph snapshots describe installed structure. SGLang Graph events use
Host `monotonic_ns()` and currently append and flush each event synchronously.
Routing snapshots expose InvocationKey, layer, and tensors; their internal
headers also retain Lane/Lease. Routing payload overflow currently returns
without a quality counter. These are collection changes required by the target,
not facilities already provided by the baseline.

Current retirement additionally queues PE-wide outbound NVSHMEM quiet after
local producer completion. Collector teardown must respect that boundary, as
well as Instance IPC detach and CUDA completion. Source declarations remain
authoritative for implemented interfaces; proposed declarations below become
authoritative only when implemented.

## Accepted Changes

- Replace mutable Device observation Records with immutable fixed-size events.
  Carry the operation facts known at each Hook; distinguish absent facts from
  valid zero values. Keep production RequestMetadata unchanged.
- Allocate a fresh shared Transport operation identity after every successful
  Mailbox Claim, independently of event capacity. Publish the association at
  actual entry into Fabric Submission preparation, then observe publication
  separately. Preserve rank-local many-to-one and Graph one-to-many relations.
- Add fresh Graph execution identities and bounded, Stream-ordered dynamic
  Context. Separate Host calls, Device activity, templates, and long-lived Lane
  service Graphs. Attach Routing evidence by exact attachment identity.
- Provide one bounded Collector/Writer per process, shared by independently
  enabled observers and unified mode. Add `debug.timeline` through
  `xpool.config`; collection policy is immutable before native initialization
  and Graph Capture.
- Preserve existing Transport/Fabric snapshot APIs and JSON shapes through
  bounded Host reconstruction of the earliest N operations. Report missing
  facts and snapshot-capacity loss in separate quality metadata.
- Use ten-attempt calibration blocks and a positive-slope affine feasible
  region, with two held-out validation attempts. Preserve raw clocks, model
  assumptions, unavailable intervals, and numerical uncertainty.
- Downstream export uses standard native Perfetto protobufs and Trace Processor.
  ClockSnapshot represents an accepted mapping; it supplies no calibration
  evidence by itself. Uncalibrated domains retain separate timing artifacts.

## Resource and Lifecycle Contract

| Resource | Default and accounting |
| --- | --- |
| Device collection | 8 MiB per process/Device, including event slots, control, Context, and sampling |
| Host collection | 32 MiB per process, including queues, staging, Writer-held blocks, metadata, and snapshot views |
| Raw chunk | At most 1 MiB including the 128-byte header |
| Collection and periodic sampling | 100 ms; at most one in-flight sample per sampler |
| Runtime disk | 1 GiB per process, including metadata, temporary files, snapshots, and attachments; reserve 8 MiB for metadata |
| Additional shutdown wait | At most 5 seconds, bounded by the remaining production shutdown deadline |

Routing tensor pools and Graph structural artifacts keep their independent
memory capacities. Added identity fields enter their sizing and admission.
The event budget is not a bound on all Observer memory. Startup rejects a
configuration whose minimum allocations or metadata reserve exceed its budget;
it never silently expands a budget. Report each allocation category separately.

Hooks perform no file I/O and never wait for the Writer, a free slot, or disk
quota. Full collection storage drops new events and counts the loss. Disk
exhaustion stops collection, preserves published files, and records the stop
reason. Atomic same-directory publication provides a recoverable complete-file
prefix under a healthy filesystem; it does not promise power-loss durability.
Timed-out CUDA or IPC users retain their resources until production retirement
proves safe release. Crashed or unpublished tails remain unknown.

## Implementation and Validation

Downstream work starts from the same versioned fixtures: strict raw parsing and
causal reconstruction; deterministic calibration; runtime collection and
snapshot reconstruction; protocol/Graph identities; Routing attachments; paired
sampling; finally native Perfetto/query integration.

Both Native and offline implementations must independently read and write the
valid binary vector, reject the invalid vectors by structured code, reconstruct
the same legacy projections, and produce the exact calibration expectations.
Fixture changes require a coordinated contract version update; never support
two conflicting interpretations of v1.

Focused tests must cover concurrent reservation/commit/reclamation, multiple
CTAs/Streams, IPC identity lifetime, repeated Replay and Context isolation,
partial events, slow Writer, I/O failure, quota exhaustion, and shutdown.
Follow [test conventions](../../../tests/README.md), repository-managed Ruff,
type, and clang-format checks. Native Host tests attempt ASAN/UBSAN with the
required execution approval.

Real acceptance follows [Qualification](../../designs/qualification.md).
Prepare a two-Device batch for user execution: Qwen3-0.6B Dense followed by
DeepSeek-V2-Lite-Chat MoE, Prefill/Decode/Replay, actual attention MPS and direct
FFN processes, idle/serving/stress sampling, bounded-capacity runs, and
observation-disabled comparisons. Retain separate Attempts, physical Device
mapping, configurations, raw events/samples/attachments, checksums, and losses.
Report measured coverage, drift, uncertainty, and overhead without an invented
accuracy or performance threshold. Fixtures cannot substitute for these runs.

## Out of Scope

This C0 delivery contains documents and fixed machine-readable data.
It adds no runtime implementation, parser, generator, or persistent test code.
Full user Request/Batch/Step context, KV causal chains, exact per-participant
observations derived from aggregate conditions, per-CUDA-node profiling,
cross-Host support, other platforms, continuous monitoring, and alerts remain
outside this workstream. Current design documents will absorb implemented
behavior after subsequent runtime implementation and acceptance.
