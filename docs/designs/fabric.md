# Fabric

Fabric forms distributed invocations from AtnAgent submissions, coordinates
FfnAgent execution, and returns the required output form. See
[FFN Execution](ffn-execution.md) for the compute Graphs selected by an admitted
invocation.

## Arena

Every participant maps one symmetric NVSHMEM Fabric arena with the same byte
layout. A compact header identifies the ABI and total layout size. Plan-derived
tables describe Instances, Layers, PEs, Capacities, and delivery relationships.
Dynamic publications, Lane payloads, completion state, failure, and shutdown
state occupy separate typed regions. Traces are process-local Devkit evidence
and are not Fabric arena regions.

Lane payloads use one contiguous native-owned storage region. Each Executor
Lane has exactly two fixed-capacity buffers in buffer-major order; the number of
Lanes is the generation's `ffn_concurrency`. Native layout code alone owns
alignment, offsets, and element-size arithmetic. Python supplies participant
counts and raw maximum payload and routing-element capacities.

Python bindings accept and return `torch.dtype`; the binding seam converts it
to and from `c10::ScalarType`. Transport and Fabric layouts retain that native
type and checked `payload_row_bytes` geometry, so there is no CrossPool-owned
payload-dtype enum or second Device dtype-to-size mapping.

AtnAgent and FfnAgent code use role-specific views over the same storage. Views
change legal access and interpretation; they do not create duplicate buffers or
alternate address maps.

The protocol uses nine typed records: Submission, Admission, Lane Execution,
Input Ready, Routing Metadata Ready, Partial Ready, FfnAgent Completion, Output
Commit, and Output Acknowledgement. Records carry identity and synchronization
facts; payload bytes remain in fixed Lane storage. Monotonic sequences prevent
stale publications from satisfying a current lease.

`RequestMetadata::valid()` owns intrinsic closed-enum validation.
`InstanceRankRuntime::Attachment::submit()` owns contextual hidden width,
payload dtype, CUDA device, row capacity, DP layout, and DP-rank vector-length
validation at the submission seam.

## Invocation sequence

One layer invocation follows this order:

1. every participating AtnAgent publishes a matching submission;
2. the coordinator validates agreement and leases one compatible executor
   lane;
3. the coordinator publishes admission and lane execution;
4. AtnAgents copy their input contribution into the selected lane and publish
   input readiness;
5. each FfnAgent GraphExec consumes the input and executes its local true-TP
   shard;
6. MoE routing metadata is published when the selected layer needs it;
7. FfnAgents publish immutable partial readiness and local completion;
8. the delivery path produces the required AtnAgent-visible output;
9. the coordinator commits the result only after every required contribution
   is complete;
10. AtnAgents consume the committed payload and acknowledge it; and
11. the coordinator releases the lane for its next monotonic lease.

Lane acquisition and release occur per layer. This keeps protocol ownership
simple; per-pass leasing remains deferred until measurement demonstrates that
per-layer overhead matters.

Publication helpers encode ordering. Local payload publication requires the
producer's writes to become visible before its release signal. Remote NVSHMEM
payload publication performs the required remote completion before signalling
the destination. Consumers wait on monotonic publication state and validate
the expected key after observation.

## Delivery

The plan derives one delivery variant for each Instance and layer topology:

- Direct Partial delivers a rank-local partial where the consumer owns the
  remaining reduction.
- Single Complete reduces to one designated AtnAgent when one complete output
  is required.
- Replicated Complete reduces once and makes the complete result visible to
  every required AtnAgent.

Both the fewer-FfnAgent-than-AtnAgent and greater-or-equal topology families are
supported. Delivery is derived from participant counts and the requested
output requirement; it is not a request-time backend choice.

Physical DP-row metadata describes how live rows are partitioned among
AtnAgents. `payload_rows` is the complete live row count. Per-rank physical row
counts are a view of that payload, not a second request size or decode/prefill
mode.

## Scheduler and failures

The Coordinator Scheduler sees ready Instance submissions and available lanes.
The default policy is FIFO. A random policy exists for controlled experiments
and uses the shared CrossPool random utility. Scheduling returns no decision when no
request and Lane pair is currently admissible.

Expected waiting is Device-side and uses the common wait utility. Timeouts,
protocol mismatches, invocation failures, participant loss, and control-plane
failures converge to one canonical generation failure; the first publication
wins. The AtnAgent leader attempts publication, every thread waits for and
observes the canonical result, and the block returns one value. After failure,
no new invocation is admitted and the process tree drains or terminates; there
is no request retry or collective recovery.

Host and Python failure readout exposes only the validated immutable failure
payload. Atomic claim and publication words remain internal Fabric storage and
are not part of the control-plane interface.

The Coordinator inspects submissions, completions, and acknowledgements without
blocking its outer progress loop. An inspection is Pending, Ready, or a Protocol
Mismatch. This is deliberately distinct from a timed wait: inspection has no
deadline, cancellation, sleep interval, or retained wait state.
