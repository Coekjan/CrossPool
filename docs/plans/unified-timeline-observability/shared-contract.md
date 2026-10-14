# Shared Observability Contract v1

This is the target contract linked from the [task plan](README.md). All integers
on disk are little-endian except opaque UUID bytes. Native struct padding,
`size_t`, pointers, CUDA handles, and enum ordinals inferred from C++ layout are
never wire representations. Fixture JSON uses exact integer tokens; consumers
must not round them through IEEE-754 doubles.

## Identity and Metadata

| Identity | Type and scope | Allocation and isolation |
| --- | --- | --- |
| Run | 16-byte UUID | New launch Attempt; never PID or wall time alone |
| Process creation | 16-byte UUID | New OS process creation, even after PID reuse |
| Producer | Positive u64 within Run | Launcher registers one Host source per process and one Device source per process/Device; retries never reuse IDs |
| Generation | Positive u64 metadata reference within Run | Resolves to the exact production FabricGenerationId string |
| Endpoint | `(owner_producer, endpoint_sequence)` | `owner_producer` is the AtnAgent process's Device Producer; the AtnAgent Device source allocates a positive sequence for each new IPC observation region. One region is exported per Transport Arena pairing, matching the existing one-Arena-per-Instance-rank export granularity; it is never shared across TP/DP ranks |
| Operation | `(Run, Endpoint, operation_sequence)` | Positive Device counter after successful Mailbox Claim; independent of event reservations |
| Graph template | `(Run, graph_owner, template_sequence)` | Host source registers a fresh template when capturing/installing; actual CUDA handles are not identity |
| Graph execution | `(Run, graph_owner, execution_sequence)` | Fresh positive sequence per actual invocation of the execution path; Eager has no template |
| Invocation | `(Run, Generation, instance_index, invocation_sequence)` | Uses production InvocationKey; index may be zero, sequence is positive |
| Lane/Lease | `(Run, Generation, lane_index, lease_sequence)` | Uses production assignment; Lane zero is valid, lease is positive |
| Routing attachment | `(Run, attachment_owner, attachment_sequence)` | Device publication allocates sequence independently of event/tensor capacity |
| Clock domain | Positive u64 within Run | Host CLOCK_MONOTONIC reference; Device domain additionally names process, physical UUID, and raw-clock Epoch |
| Event | `(Run, Producer, event_sequence)` | Positive attempted-event counter; never a buffer slot or operation sequence |

Producer IDs and metadata references are registered before Capture and copied
as immutable observation configuration. Endpoint sequences are not local arena
indices. The Endpoint owner is shared by the Instance and AtnAgent, although
their event Producers differ. Each Producer has exactly one clock domain;
one process can have several Producers. Physical UUIDs, process creation,
logical-device mapping, model IDs, Instance topology, scheduler policy,
capabilities, and artifact references live in metadata.

The launcher distributes one Run and the registered Producer/Generation/clock
IDs through the existing preparation/bootstrap path. IPC exports add optional
`observation_region` with exactly version:u32=1, owner_producer:u64,
endpoint_sequence:u64, generation:u64, allocation_bytes:u64, and handle.
Handle uses the existing canonical hexadecimal CUDA IPC handle representation;
it is never persisted in raw timeline metadata. Disabled related observation
exports null and allocates no region. Version or preparation-policy mismatch
fails startup; all participants use the same Native ABI, with no old-worker
fallback.

### Endpoint allocation, export, and attachment

One `observation_region` is allocated per Transport Arena pairing, in the same
`cudaMalloc` call sequence that creates that Arena (the existing
`Arena::create()` allocation path). The AtnAgent process's Device Producer
assigns a fresh positive `endpoint_sequence` immediately before this
allocation; sequences are never reused within a process lifetime. The region's
IPC handle is exported through `cudaIpcGetMemHandle` and encoded with the
existing `xpool::utils::hex::HexValue<cudaIpcMemHandle_t>` convention,
identical to the Transport Arena handle encoding (`Arena::handle()`). It is
published to the Daemon broker as additional sibling fields on the same
descriptor record already used for the Arena handle (the existing
`TransportBroker.publish()` path); no new RPC is introduced. The Instance
fetches and attaches it with `cudaIpcOpenMemHandle` at the same call site
where it already attaches the Transport Arena handle. When
`debug.timeline.enable` is false, the descriptor's `observation_region` field
is null and no CUDA allocation is made; this is the only condition under
which the Instance must tolerate a missing region.

### Endpoint recreation

A new region and a new `endpoint_sequence` are allocated whenever the
underlying Transport Arena is itself recreated: an AtnAgent process restart
(new process-creation identity), or an explicit Arena re-pairing following a
Fabric Generation rollover. `generation` in the descriptor is the Generation
reference current at the moment of this allocation. The prior region's CUDA
allocation and IPC mapping retire only after the Instance has detached
(`cudaIpcCloseMemHandle`) and any Collector or snapshot-reconstruction work
still reading the prior generation's events has drained, following the
general rule above that mappings retire after IPC detach and in-flight work.
The hot descriptor itself carries no "superseded" field; Endpoint lineage
across recreation is a metadata concern (see Metadata revision below), not a
per-export wire concern.

### Leading-word layout and write/read timing

The region's leading-word layout is: `operation_sequence:u64`,
`graph_owner:u64`, `graph_execution:u64`, `graph_template:u64`,
`context_generation:u64`. The first five words occupy 40 bytes; each is
naturally eight-byte aligned. The descriptor advertises the allocation size,
not its private control layout; everything after byte 40 is the allocation
owner's private storage and is never part of the versioned wire format.

Field ownership and write/read timing:

| Word | Writer | Written | Read by | Read after |
| --- | --- | --- | --- | --- |
| `operation_sequence` | Instance | After successful Claim (`try_begin_staging()` returns true), before `publish_request()` | AtnAgent | After `observe()` returns `Published` |
| `graph_owner`, `graph_execution`, `graph_template`, `context_generation` | AtnAgent | After `fabric::atnagent::execute(fabric_arena, transport_arena)` returns and the triggered Graph execution mode (Replay/Eager/none) is known, before `publish_result()` | AtnAgent's own Collector, committing the `TransportFabricAssociated`/Graph events | Synchronously within the same critical section, before `publish_result()`; never deferred to a later decoupled pass |

No new atomic primitive is introduced. `operation_sequence` visibility reuses
the Mailbox's existing release-acquire pair: `publish_request()`'s
release-store and `observe()`'s acquire-load already order the Instance's
write before the AtnAgent's read, the same ordering that already protects
`payload_rows` and `request`. `graph_*` visibility to AtnAgent's own Collector
is plain program order inside the same thread and never crosses a process
boundary, because the writer and the only production-critical reader are the
same AtnAgent thread executing `execute_request()`.

A subsequent successful Claim may update `operation_sequence` only after
`acknowledge()` (Evaluated to Idle) has executed. This follows from the
existing Mailbox state machine: the Instance only calls `try_begin_staging()`
again after observing `Evaluated` and calling `acknowledge()`, and `Evaluated`
is only reachable through AtnAgent's `publish_result()`, which by construction
happens strictly after AtnAgent has already written and consumed this cycle's
`graph_*` words (see the table above). Consequently, by the time the Instance's
next Claim is eligible to overwrite `operation_sequence`, AtnAgent has already
finished every read of this cycle's `graph_*` words, and no additional fence
is required to prevent the next Claim from racing a stale `graph_*` read.

This ordering guarantee holds only for readers inside the critical section
described above (synchronous, same-thread, before `publish_result()`). Any
out-of-band reader that samples `observation_region` outside that window (for
example, an offline diagnostic peek) MUST treat `context_generation` as a
seqlock: read it, read `operation_sequence`/`graph_*`, then re-read
`context_generation`, and discard the sample if the two reads differ.
Production collection never needs this fallback because it never reads
outside the critical section; the rule exists only to make misuse detectable
rather than silently wrong.

Endpoint recreation allocates a new region and sequence. The operation word is
monotonically incremented by the successful claimant using the existing
read-check-increment pattern (`abort_if(prev == UINT64_MAX)` then `+1`, as
already used for `invocation_sequence` and `executor_lease_sequence`); no
collector reservation controls it. No observation state enters production
RequestMetadata or the production Arena allocation. Mappings retire after IPC
detach and in-flight work.

### Private control storage: recommended default layout

The bytes after offset 40 are private and not wire-versioned, but both
implementers must agree on one default layout so B1 and B2 interoperate
without a second interpretation. The recommended default holds only a small
per-Endpoint *relay* scratch area for Graph-association handoff; it is not the
canonical Dynamic Context store (see "Dynamic Context storage" below for why
that store is process-wide, not per-Endpoint).

| Offset (from byte 40) | Field | Type | Purpose |
| --- | --- | --- | --- |
| 0, 8, 16 | `slot_generation`, `resolved_execution_sequence`, `resolved_template_sequence` | u64 each | One relay slot: the Graph identity AtnAgent most recently resolved for this Endpoint's current cycle, before copying it into the public prefix |
| 24 | `resolved_context_generation` | u64 | Matches the Context slot generation this relay entry refers to |
| 32 | `state` | u32 | `Free=0`, `Resolving=1`, `Copied=2` |
| 36 | (padding) | u32 | Zero, keeps the slot eight-byte aligned |

Slot size is 40 bytes. Default relay slot capacity
(`timeline_relay_slot_capacity`) is **2**, allowing one Mailbox cycle's relay
write to overlap the previous cycle's acknowledgement without blocking.
Per-Endpoint private bytes are `2 * 40 = 80`, so one `observation_region`'s
total size is `40 + 80 = 120` bytes before allocation-alignment rounding.

### Dynamic Context storage: process-wide, not per-Endpoint

Context identity `(graph_owner, execution_sequence, template_sequence,
context_generation)` is scoped to the AtnAgent process's Host Producer
(`graph_owner`), not to any single Endpoint: a `lane_service` execution can
serve multiple Endpoints concurrently, and TP/DP activity under one
Invocation is not attributable to exactly one Endpoint. The canonical bounded
Context slot array is therefore a separate, process-wide Device allocation
owned by the Graph Observer, sized once per `graph_owner`, independent of how
many Endpoints that process currently has open. Its recommended default
layout:

| Offset | Field | Type | Purpose |
| --- | --- | --- | --- |
| 0 | `slot_generation` | u64 | Incremented on every reuse; the generation a reader must match |
| 8 | `execution_sequence` | u64 | Owning Graph execution identity |
| 16 | `template_sequence` | u64 | Zero for Eager |
| 24 | `stream_ordinal` | u64 | Process-local monotonic registration ID of the owning Stream; never a raw `CUstream` handle |
| 32 | `state` | u32 | `Free=0`, `Reserved=1`, `Committed=2` |
| 36 | (padding) | u32 | Zero |

Slot size is 40 bytes. Default capacity (`context_slot_capacity`) is **64**
per process, sized for the worst-case concurrent-Stream count across all
Lanes/Eager paths in one AtnAgent process; `context_slots_bytes =
64 * 40 = 2560` bytes. Concurrent Streams use distinct live slots; slot reuse
follows the existing completion-boundary rule. Context overflow increments
`context_unavailable` and leaves association unavailable; it never overwrites
an in-flight slot.

An alternative of folding this array into each `observation_region` (one full
Context array per Endpoint) was evaluated and rejected: it requires either
duplicating `lane_service` Context across every Endpoint it serves or picking
one arbitrary "owning" Endpoint for it, both of which conflict with the
Context identity's process-wide scope. This record is kept only so a future
reviewer does not re-propose it without seeing why it was declined.

### Calibration sampler storage: per-process Device Producer, not inside observation_region

A sampler's identity is `(Run, Host Producer, Device clock_domain_id)` and
must exist for every process with a Device Producer, including FfnAgent
processes that never create an `observation_region` at all (FfnAgent has no
Transport Endpoint). Sampler storage is therefore a separate, non-IPC,
per-process Device allocation, parallel to `observation_region` rather than
inside it:

| Field | Type | Purpose |
| --- | --- | --- |
| `raw_g_ns` | u64 | Written by the short sampling Kernel on the dedicated Stream; read back by the Host via `cudaMemcpyAsync` after completion confirmation |

Default `sampler_slot_capacity` is **1** per process (one clock domain per
Device Producer in the common case), configurable up to a small bound for a
process observing more than one Device. `sampler_bytes =
sampler_slot_capacity * 8`. All other sampler bookkeeping (`in_flight` flag,
last `attempt_sequence`, stashed `H_before`, current Epoch id) is ordinary
Host-process memory, not a Device allocation, and is accounted under the
existing Host collection budget, not the Device budget.

### Device collection budget composition

`debug.timeline.device_bytes` (default 8388608) must cover, per
process/Device:

```text
device_bytes_required =
    device_event_slot_capacity * (256 + 16)                        # event ring payload + control
  + endpoint_capacity * (40 + timeline_relay_slot_capacity * 40)    # observation_region(s)
  + context_slot_capacity * 40                                     # process-wide Dynamic Context
  + sampler_slot_capacity * 8                                      # calibration raw-G buffers
  + 256                                                            # fixed Collector head/tail/epoch control
```

Recommended defaults: `device_event_slot_capacity=16384`,
`endpoint_capacity=8`, `timeline_relay_slot_capacity=2`,
`context_slot_capacity=64`, `sampler_slot_capacity=1`. Substituting these
values:

```text
16384 * 272 = 4,456,448
   8 * 120  =       960
      2,560 =     2,560
          8 =         8
        256 =       256
total        = 4,460,232 bytes
```

This total is comfortably under the 8,388,608-byte default, leaving headroom
for configuration overrides. Startup computes `device_bytes_required` from the
active configuration and rejects the process if it exceeds `device_bytes`,
per the existing rule that startup rejects a configuration whose minimum
allocations exceed its budget.

Every identity/counter exhaustion is explicit: stop observation in the affected
identity domain, set `identity_exhausted`, and preserve production results.
Saturate quality counters with `counter_exhausted`; never wrap or abort a
production kernel because an observation counter exhausted. Exhausted Context
capacity reports unavailable Graph association; it never overwrites an in-flight
Context or blocks the production path waiting for observation storage.

### Metadata revision

Each immutable `metadata.<producer>.<revision>.json` contains exactly:
`contract_version`, `run_id`, `producer_id`, `revision`, `process`,
`clock_domain_id`, `producers`, `generations`, `clock_domains`, `endpoints`,
`graph_templates`, `artifacts`, `capabilities`, and `configuration`.
UUIDs use lowercase canonical text. Process contains `creation_id`, `pid`, and
`runtime_role`. Producer entries include their complete Process, source kind,
and clock domain so remote roles can reconstruct their own snapshots. Array
entries use positive integer `id` keys where applicable;
Endpoint and template entries use their composite identities. References are
immutable across revisions. Import metadata for every referenced remote owner
and identity before publishing a referencing chunk. Graph executions and
operation sequences are dynamic identities, not per-operation dictionary entries.
Endpoint entries optionally carry `supersedes_endpoint_sequence` naming the
prior Endpoint this one replaces under the same `owner_producer`, letting
offline reconstruction stitch Endpoint-recreation lineage without widening the
hot IPC descriptor.

Revision starts at one and increases. Hash the exact UTF-8 file bytes, including
its final newline; chunk headers contain the first 16 bytes of SHA-256.
Metadata is published before its chunks. Revision updates are bounded by Host
and disk budgets. If metadata cannot be published, do not publish undecodable
chunks; report `metadata_unavailable` and stop the affected collection source.
An entirely absent process artifact is a run-level evidence gap. A present chunk
with missing/mismatched required metadata is a structured parse error.

## Event Vocabulary and Boundaries

These IDs are stable wire values. Hook enum values are mapped explicitly.
The name arrays in [the layout fixture](fixtures/layout.json) are normative.

| IDs | Source and ordered names | Meaning |
| --- | --- | --- |
| 100–110 | Transport: RequestStagingStarted, RequestStagingCompleted, RequestPublished, RequestObserved, ExecutionStarted, ExecutionCompleted, ResultPublished, ResultObserved, OutputCopied, ResultAcknowledged, Closed | Existing `TransportProtocolEventKind` boundaries, unchanged |
| 200–205 | AtnAgent Fabric: SubmissionPrepared, SubmissionPublished, AdmissionObserved, InputReadyPublished, OutputCommitObserved, OutputAcknowledgementPublished | Local preparation/publication/observation, not a remote acknowledgement inferred from time |
| 300–307 | Coordinator: Enqueued, Scheduled, AdmissionPublished, LaneExecutionPublished, FfnAgentCompletionsObserved, OutputCommitPublished, OutputAcknowledgementsObserved, LaneReleased | Existing aggregate conditions; they do not contain individual participant observation times |
| 400–408 | FfnAgent: LaneExecutionObserved, InputReadyObserved, RoutingMetadataPublished, RoutingMetadataObserved, ComputeStarted, ComputeCompleted, PartialReadyPublished, PeerPartialsReadyObserved, CompletionPublished | Observed protocol and compute facts of this PE |
| 450 | TransportFabricAssociated | Actual entry into Submission preparation with the real key; does not imply SubmissionPublished |
| 500–503 | GraphCaptureBegin, GraphCaptureEnd, GraphCaptureError, GraphTemplateInstalled | Template lifecycle; Capture is construction activity |
| 504–506 | GraphHostCallBegin, GraphHostCallEnd, GraphHostCallError | Host call boundaries; return does not prove Device completion |
| 507–508 | GraphDeviceActivityBegin, GraphDeviceActivityEnd | Device record points installed in the actual execution path; only the measured activity is enclosed |
| 509–512 | GraphTemplateRetired, GraphInstallBegin, GraphInstallEnd, GraphInstallError | Installation and retirement lifecycle |
| 550 | RoutingPublished | Same real RoutingMetadataPublished boundary, enriched with attachment identity |
| 600–601 | CalibrationSampleSucceeded, CalibrationSampleFailed | Host-published asynchronous sampling outcome, not production-stream synchronization |

Transport events retain their site. Before Fabric entry, rejection, shutdown, or
timeout can produce a Transport terminal event without an association. Missing
association evidence remains `association_missing`; matching metadata values
or nearby timestamps never repairs it. A retained association proves entry;
a missing publication event makes publication unknown. TP/DP contribution
operations remain separate and can share one Invocation.

Graph Host events have execution mode `replay=1`, `eager=2`, or
`lane_service=3`, and phase `decode=1`, `prefill=2`, or `native_ffn=3`.
Capture events have a template identity; calls and Device activities have a
fresh execution identity. Eager carries no template identity. Native Primary
parameterization maps to GraphTemplateInstalled; Lane build registers a
separate `lane_service` template referencing its selectable compute templates.
Existing installation lifecycle Hooks map to GraphInstallBegin/End and retirement.
New dynamic boundaries are installed by their production execution owners.
The registered Graph owner has disjoint execution-sequence ranges: Host-issued
sequences are 1 through 2^63-1; Device-issued activities use 2^63 plus a local
counter in 1 through 2^63-1. Both local counters stop before overflow. This
permits native activity allocation on Device while templates remain Host-owned.
Compute Branch selection records the actual chosen template, Invocation, and
Lane/Lease at GraphDeviceActivityBegin; its End encloses that activity only.
A long-lived Lane service launch is a separate execution, not one launch per
Invocation. Repeated templates do not imply repeated Capture.

Dynamic Context is `(graph_owner, execution_sequence, template_sequence,
context_generation)` in bounded Device storage. Stream-ordered installation
precedes the actual execution, and slot reuse follows its completion boundary.
Capture warmup uses construction-only Context with no production execution ID;
Replay resolves fresh Context from Device storage. Concurrent Streams use
distinct live slots. A Context overflow leaves association unavailable and
increments quality; no stale Capture ID or guessed replacement is allowed.
This storage is process-wide per `graph_owner`, not per `observation_region`;
see "Dynamic Context storage: process-wide, not per-Endpoint" above for its
default layout and capacity.

## Fixed Event Layout

Every record is 256 bytes. The Producer and Run are supplied by the chunk.
All unlisted presence bits and all absent fields are zero. Zero is a valid
timestamp; only legacy projections use zero as their missing-time sentinel.

| Offset | Field | Type |
| --- | --- | --- |
| 0, 8, 16 | event_sequence, raw_time_ns, clock_domain_id | u64 each |
| 24, 28 | kind, presence | u32 each |
| 32 | generation | u64 |
| 40, 48, 56 | endpoint_owner, endpoint_sequence, operation_sequence | u64 each |
| 64, 72, 80 | graph_owner, graph_template, graph_execution | u64 each |
| 88, 96 | instance_index, invocation_sequence | u64 each |
| 104, 112 | lane_index, lease_sequence | u64 each |
| 120, 128 | attachment_owner, attachment_sequence | u64 each |
| 136 | local_trace_id | u64 |
| 144, 152, 160, 168, 176 | layer_ordinal, payload_rows, dp_rank_payload_rows, payload_row_capacity, ready_ticket | u64 each |
| 184, 188, 192, 196, 200 | forward_mode, output_requirement, dp_row_layout, delivery, result_code | u32 each |
| 204, 208, 212 | role, instance_rank, pe | u32, u32, i32 |
| 216, 224, 232, 240, 248 | data0, data1, data2, data3, data4 | u64 each |

Presence bits group fields:

| Bit | Fields made present |
| --- | --- |
| 0 | generation |
| 1 | endpoint_owner, endpoint_sequence |
| 2 | operation_sequence; requires bit 1 |
| 3 | graph_owner, graph_template |
| 4 | graph_owner, graph_execution |
| 5 | instance_index, invocation_sequence; requires bit 0 |
| 6 | lane_index, lease_sequence; requires bit 0 |
| 7 | attachment_owner, attachment_sequence |
| 8 | local_trace_id |
| 9 | layer_ordinal, payload_rows, forward_mode, output_requirement, dp_row_layout |
| 10 | layer_ordinal, payload_rows, output_requirement |
| 11–17 | dp_rank_payload_rows, payload_row_capacity, ready_ticket, delivery, result_code, instance_rank, pe, respectively |

Bit 9 implies bit 10; both bits are set in the canonical request encoding.
Shared fields such as graph_owner are absent only when every containing group
is absent. Identity components are positive except the explicitly zero-based
Instance/Lane/rank/PE indices. Domain references must match the Event's Producer.

Role is always present: `instance=1`, `atnagent=2`, `coordinator=3`,
`ffnagent=4`, `collector=5`. Absent PE stores zero, not minus one.
Preserve FFN wire values from `xpool/ffn.hpp`: Prefill 1, Decode 2, Idle 4;
PerRankComplete 1, GroupSumComplete 2; None/UniformByRank/PackedByRank 0/1/2;
Ok/Shutdown/ProtocolMismatch/Timeout 0/1/2/3. Delivery uses
DirectPartial/SingleComplete/ReplicatedComplete 1/2/3.

Transport records require Generation, Endpoint, Operation, local trace ID,
request facts, and Instance Rank. Fabric records require Generation,
Invocation, local trace ID, invocation facts, and PE. AtnAgent records also
carry request facts and DP rows; FfnAgent records carry Lease, row capacity,
and delivery. Coordinator ticket is present for FIFO scheduling and absent for
random; scheduler policy is immutable metadata so loss cannot turn FIFO into
random. AtnAgent Lease becomes known at AdmissionObserved; Coordinator Lease
at Scheduled. Each later event repeats already known facts. A result becomes
known only when observed/published/closed at that site; never borrow another
site's effective result.

TransportFabricAssociated requires both Operation and Invocation and repeats
request/submission facts. It uses role atnagent. Graph protocol activity carries
its corresponding Operation or Invocation/Lease; Hosting alone does not prove
the relation. RoutingPublished requires Invocation, Lease, attachment identity,
invocation facts, and publisher PE. Identity-bearing fields must resolve to
metadata in the referencing revision. Any not-applicable payload word is zero.

| Kind family | data0…data4 interpretation |
| --- | --- |
| Transport, Fabric, association | All zero |
| Graph | mode, phase, artifact_id (zero if unavailable), context_generation (zero on Host), error_code |
| RoutingPublished | attachment_state, effective_topk, artifact_id (zero until publication), 0, 0 |
| CalibrationSampleSucceeded | H_before, G, H_after, Device clock_domain_id, attempt_sequence |
| CalibrationSampleFailed | H_before, 0, failure_code, Device clock_domain_id, attempt_sequence |

Graph error_code: `none=0`, `capture_failed=1`, `install_failed=2`,
`host_call_failed=3`, `context_unavailable=4`, `device_boundary_unavailable=5`.
Routing attachment_state: `reserved=1`, `not_collected=2`, `capacity_exhausted=3`,
`payload_oversize=4`; final published/missing/I/O-failed status comes from the
attachment index, not by mutating an event. Sample failure_code:
`enqueue_failed=1`, `completion_failed=2`, `timed_out=3`, `shutdown=4`.
Samples use collector role and the Host Producer/domain. Their raw time is
completion-confirmation time on success and outcome-report time on failure.
Sample Device resources remain owned until completion despite timeout.

## Native Collection Boundary

Proposed declarations live under `xpool::devkit::timeline`; implementation
headers follow the repository's native naming convention. Public field layout
above is independent of private ring-slot layout.

```cpp
enum class ReserveStatus : std::uint32_t {
  Reserved = 0, Full = 1, Stopped = 2, IdentityExhausted = 3,
};
enum class CommitStatus : std::uint32_t {
  Committed = 0, StaleToken = 1, AlreadyCommitted = 2,
};
enum class AbandonStatus : std::uint32_t {
  Abandoned = 0, StaleToken = 1, AlreadyCommitted = 2,
};
enum class CollectionErrorCode : std::uint32_t {
  ResourceBudget = 1, Allocation = 2, Transfer = 3, InvalidRelease = 4,
  Io = 5, MetadataUnavailable = 6,
};
struct WriteToken {
  std::uint64_t slot;
  std::uint64_t slot_generation;
  std::uint64_t event_sequence;
};
struct ReserveResult { ReserveStatus status; WriteToken token; };
struct ConsumeToken { std::uint64_t batch; std::uint64_t generation; };
struct PollResult {
  std::optional<ConsumeToken> token;
  std::span<const Event> events;
};

XPOOL_DEVICE_FN ReserveResult try_reserve_device(BufferView buffer);
XPOOL_DEVICE_FN CommitStatus commit_device(BufferView buffer, WriteToken token,
                                           const Event &event);
XPOOL_DEVICE_FN AbandonStatus abandon_device(BufferView buffer, WriteToken token);
ReserveResult try_reserve_host(HostBuffer &buffer);
CommitStatus commit_host(HostBuffer &buffer, WriteToken token, const Event &event);
AbandonStatus abandon_host(HostBuffer &buffer, WriteToken token);
PollResult poll(Collector &collector, std::uint32_t max_events);
void release(Collector &collector, ConsumeToken token);
QualitySnapshot quality(const Collector &collector);
```

`Event` is a trivially copyable field declaration matching the 256-byte layout;
compile-time size/offset assertions guard it. Serialization explicitly encodes
fields. Host APIs throw `CollectionError`, derived from `c10::Error`, with
typed CollectionErrorCode, Producer, optional path, and root cause. Python
bindings preserve those fields and expose enum names as resource_budget,
allocation, transfer, invalid_release, io, and metadata_unavailable. Device
statuses never change production control flow. `abandon` retires an uncommitted
reservation and increments abandoned; it does not publish an Event.
Non-Reserved ReserveResult tokens are all zero and grant no ownership.

Reservation is bounded and nonblocking; Full consumes an attempted-event
sequence and increments dropped_full. Stopped allocates no sequence after a
source has stopped. Disabled mode installs no collection work. A successful
token grants exclusive ownership of one slot generation. Commit verifies the
token, writes the entire Event, then release-publishes completion. Token checks
cannot race reclamation: reserved slots are ineligible for reclamation.
Stale/double commits are observation failures, with explicit counters.

Each process owns a background collection thread that runs during serving,
using `collection_interval_ms` (default 100 ms) to drive collection. It submits
asynchronous Device-to-Host event copies and checks their completion so that
Device slots can be reclaimed promptly rather than only at shutdown. This
reduces capacity pressure but does not guarantee loss-free collection. Writer
file I/O runs outside this thread and must not block its collection or
completion checks; handoff uses bounded Host-owned batches without waiting
for the Writer or allocating an unbounded queue. Staging, in-flight transfers,
queues, and Writer-held batches remain within the Host collection budget.
Host capacity exhaustion must have explicit loss accounting and never cause
premature reuse of a Device slot or a Host batch still owned by the Writer.

Collector acquire-observes completion before starting a Device-to-Host copy;
an asynchronous copy may read only immutable committed slots. It does not
synchronize production Streams. Out-of-order completion is consumable without
waiting for a pending lower sequence. A slot cannot be reused before the full
copy completes and its previous ownership is retired; reclamation does not
wait for file publication. `poll` returns a bounded
Host-owned batch or an empty result; its span stays valid until `release`.
The Writer releases it after publication or a counted discard. A failed Writer
may stop export while the Collector still drains committed slots into counted
post-commit loss. No unbounded retry queue or device reservation leak is allowed.
Teardown preserves unfinished slots until CUDA retirement; their tail is
unknown if no reliable final checkpoint is possible.
Background-thread shutdown follows the same Quiesce/Detach/Drain boundaries
and shutdown deadline. A timeout does not permit releasing resources still
used by a collection thread or in-flight CUDA/IPC work. Disabled observation
starts no collection thread.

## Raw Chunks and Errors

Published path: `chunks/<producer>/<chunk_sequence>.xpool`. The header is
128 bytes, followed by `event_count * 256` bytes, with no trailer or compression.
At most 4095 events fit in a 1 MiB chunk. Chunks are nonempty; empty collection
is represented by metadata/quality only. Chunk sequences start at one.

| Offset | Header field | Type/value |
| --- | --- | --- |
| 0 | magic | Eight bytes `XPOOLC01` |
| 8, 10, 12, 14 | version, header_bytes, event_bytes, flags | u16: 1, 128, 256, 0 |
| 16, 24, 32, 40 | producer, chunk_sequence, metadata_revision, event_count | u64 each |
| 48, 56 | first_event_sequence, last_event_sequence | u64, minimum/maximum |
| 64, 72 | host_window_begin_ns, host_window_end_ns | u64 CLOCK_MONOTONIC collection window |
| 80 | payload_bytes | u64, exactly event_count * 256 |
| 88, 92 | payload_crc32, header_crc32 | u32 each |
| 96 | Run UUID | 16 opaque bytes, canonical UUID byte order |
| 112 | metadata_sha256_prefix | First 16 SHA-256 bytes |

CRC is CRC-32/ISO-HDLC, equivalent to `zlib.crc32(bytes) & 0xffffffff`.
Compute payload CRC over the payload. Compute header CRC over all 128 header
bytes with bytes 92–95 zero and payload CRC already populated. Record order
within a chunk is increasing event_sequence; holes are legal. Chunks can contain
late commits whose sequences precede a previously published chunk's events.
Chunk windows order collection, not Device event time. Reconstruct by identity,
not file order; detect duplicates across chunks.

Reserve exact disk bytes before creating/extending a unique same-directory
temporary file. Close it before atomic rename. Publish its required metadata
first. Keep failed temporary files as evidence charged to quota; never parse
them as complete chunks. The process reserves 8 MiB for bounded metadata/close
summaries; stop before consuming it with event or attachment files. Conversion
outputs go to an explicit separate directory and quota.

Strict parsing rejects the first failing stage in this order: header length;
magic/version/layout/flags; header CRC; declared sizes and file length; metadata
identity/hash; payload CRC; event widths/presence/enums/references/order.
Unknown event kinds fail v1; unsupported versions fail rather than falling back.
Errors use codes `header_truncated`, `bad_magic`, `unsupported_version`,
`invalid_layout`, `header_checksum`, `invalid_length`, `metadata_unavailable`,
`metadata_checksum`, `payload_checksum`, `invalid_event`, `unknown_kind`,
`identity_conflict`, and `event_conflict`. Include path, offset, Producer,
event_sequence when available, and the underlying cause. Missing normal evidence
is quality, not malformed encoding. Never inspect exception message strings
to choose a recovery path.

## Quality, Attachments, and Configuration

Immutable `quality.<producer>.<revision>.json` checkpoints have these fields:
contract_version:u32, run_id:UUID text, producer_id:u64, revision:u64,
host_time_ns:u64, collection_state, export_state, stop_reason,
last_attempted_sequence:u64, counters, pending:u64|null, tail,
counter_exhausted:bool, exact_totals:bool. Revisions start at one. The complete
example in [quality.json](fixtures/quality.json) freezes the field names.
Counters are u64: attempted, committed, dropped_full,
abandoned, stale_token, duplicate_commit, copied, exported, post_commit_lost,
snapshot_operations_attempted, snapshot_operations_dropped,
attachment_attempted, attachment_capacity_exhausted, attachment_payload_oversize,
attachment_not_collected, attachment_publish_failed, sample_attempted,
sample_failed, sample_skipped_busy, context_unavailable. Counters are cumulative,
never summed across
checkpoints. Unknown final pending/tail quantities are null, never zero.
An exact checkpoint satisfies:

```text
attempted = committed + dropped_full + abandoned + pending
exported + post_commit_lost <= committed
exported <= copied <= committed
```

Stage counters count distinct Event identities, not transfer retries.
Stale/double token reports are diagnostics on an existing attempt, not new
attempts. Sequence holes
are evidence gaps; only checkpoints explain their reason. A metadata failure
or counter saturation sets exact_totals=false; checkpoint values remain
lower-bound evidence. Crash recovery sets final totals unknown rather than
claiming an earlier checkpoint is final. QualitySnapshot exposes the same
fields as typed Native values; its counters have these exact names and widths.
Use fixed retained-view known-fact masks to generate per-operation unknown-result
or missing-fact diagnostics; do not accumulate an unbounded runtime issue list.

Collection state is active/stopped/closed; export is active/failed/closed.
Stop reason is null, disk_quota, io_error, metadata_unavailable,
identity_exhausted, or shutdown_deadline. Tail is complete/pending/unknown.
Routing event loss, tensor loss, payload oversize, and file-publication failure
are independent statuses. Dense advertises Routing capability `not_applicable`.

The immutable attachment index resolves each attachment identity to status
published/not_collected/capacity_exhausted/payload_oversize/publish_failed/missing,
relative artifact path, SHA-256, tensor keys, Invocation, Lane/Lease, layer,
publisher, clock domain, raw publication time, and shape/dtypes. Only published
has a usable path/hash. An event whose reserved attachment has no final outcome
is unknown after truncation and missing after a complete closing index.
Do not map by tensor-slot position. Safetensors retains existing generation,
pe, sequence, dropped, records metadata, and adds contract_version, run_id,
producer_id, and per-record identity/time/lease fields inside records. IDs and
weights remain CPU int32/float32, live_rows by effective_topk. Tensor data stays
outside Event storage. Graph structure artifacts retain their existing inventory
and add template references through the artifact index.

`debug.timeline` adds enable=false, outdir=null, device_bytes=8388608,
host_bytes=33554432, chunk_bytes=1048576, collection_interval_ms=100,
disk_bytes=1073741824, metadata_reserve_bytes=8388608, shutdown_wait_ms=5000,
calibration_block_attempts=10, device_event_slot_capacity=16384,
endpoint_capacity=8, timeline_relay_slot_capacity=2, context_slot_capacity=64,
and sampler_slot_capacity=1. Enable and outdir must be set together.
Block attempts is fixed to ten in v1. `device_event_slot_capacity`,
`endpoint_capacity`, `timeline_relay_slot_capacity`, `context_slot_capacity`,
and `sampler_slot_capacity` are positive; startup first evaluates the device
collection budget composition formula against `device_bytes` using these
five values, then validates each remaining positive resource value against
the minimum required allocations and chunk layout; chunk_bytes
cannot exceed 1 MiB and shutdown_wait_ms cannot exceed 5000. Run-wide policy
is installed before workers start. Configuration uses the existing environment
registration/precedence rules and rejects inconsistent participants at preparation.

Unified mode installs all four evidence classes, Context, and calibration;
existing observer flags request their legacy projections using the same storage,
never a second Device collector. Unified outdir owns all artifacts and quota;
an explicitly enabled legacy outdir must equal it, otherwise configuration fails.
Legacy record_capacity controls its own reconstruction cap. Unified mode with
legacy flags disabled need not build those legacy views. Independent mode uses
one process disk ledger across its configured directories, retains its enabled
class capabilities, and installs shared identity facilities needed by those
classes. Calibration and complete four-class association are capabilities of
unified mode. Advertise enabled/disabled/not_applicable/unavailable explicitly.
The existing snapshot APIs remain callable in unified mode. A disabled legacy
flag means no separately budgeted legacy projection is installed; its getter
returns the same unavailable value as an independently disabled observer.

## Legacy Snapshot Reconstruction

Source owners are
[Transport Record](../../../src/cext-include/xpool/devkit/transport_observer.hpp),
[Fabric Record](../../../src/cext-include/xpool/devkit/fabric_observer.hpp),
[Transport JSON](../../../src/xpool/devkit/transport_observer.py), and
[Fabric JSON](../../../src/xpool/devkit/fabric_observer.py).

Allocate a legacy trace ID per original record-begin Hook independently of
event capacity: Instance staging start, AtnAgent RequestObserved, Fabric
SubmissionPrepared, Enqueued, or LaneExecutionObserved. Snapshot sequence counts
these operation attempts; it never counts events. The earliest N local IDs are
eligible for the view (default 8192 per Transport Endpoint or Fabric Producer).
Omitted opening events do not renumber IDs. Later self-describing events can
reconstruct an eligible operation; operations with no retained evidence remain
absent and are classified separately as unobserved. Capacity dropped counts
IDs above N; event loss may make len(records) smaller than sequence - dropped.
Lifetime is Endpoint for Transport and installed Producer/Generation for Fabric.
Endpoint recreation starts a new snapshot domain; offline queries retain both.

| Legacy fields | Reconstruction source |
| --- | --- |
| Transport trace_id, payload_rows, layer_ordinal, forward_mode, output_requirement, dp_row_layout | Event local_trace_id and repeated request facts |
| Transport result_code | Site-local result presence; absent projects ProtocolMismatch sentinel plus unknown_result quality |
| Transport eleven timestamp getters | Corresponding retained event; absent is zero, recorded bitmap remains authoritative |
| Endpoint instance_index/rank | Endpoint metadata; Instance Rank repeats on events |
| Snapshot pid/site/model_id/arena_handle_suffix | Process/Endpoint metadata; preserve exact model scalar, site, and handle suffix |
| Fabric local_trace_id, key, layer_ordinal, payload_rows, output_requirement | Repeated local and Invocation facts |
| AtnAgent DP rows, forward_mode, DP layout | Repeated Submission facts |
| AtnAgent/Coordinator nullable Lane/Lease | Present only if own AdmissionObserved/Scheduled evidence or later own repeated Lease exists |
| Coordinator ready_ticket and scheduler | Repeated FIFO ticket plus immutable scheduler policy |
| FfnAgent Lane/Lease, payload_row_capacity, delivery | Repeated LaneExecution/selection facts |
| Fabric role event timestamps | Corresponding retained kind; absent getters/JSON entries are zero |
| Fabric generation/pe/counts/model_topologies | Generation and Producer metadata |

If required scalar facts for a legacy Record are absent, retain the raw evidence
and omit that Record with `legacy_fact_missing` quality. In particular, FIFO
without an observed ticket cannot be projected as random. Do not fabricate a
FfnAgent capacity or delivery. Read remains valid only after the original
quiescence/synchronization boundary and returns bounded Host-owned values.
Transport reconstruction slots are 256 bytes and Fabric reconstruction slots
are 384 bytes, including identity, known-fact masks, and timestamps. Reserved
slab bytes are `sum(transport_capacity * 256 for open Endpoints) +
fabric_capacity * 384`. Native snapshot result staging adds its actual declared
Record sizes and bounded vector overhead at peak read time. The slots, metadata,
queues, and peak read staging must fit Host budget; startup fails otherwise.

Presence, not the value zero, controls event merging. Merge repeated facts only
when equal; a conflict is an explicit structured error. Identical event identity
and bytes is a duplicate, counted once. Two different event IDs for the same
site/operation/kind with identical time/facts are semantic duplicates; differing
time/facts conflict. Different roles/sites are different observations.
When a phase is missing, later phases remain retained; do not apply the old
Timeline predecessor assertions to incomplete reconstruction.

Legacy durations retain precisely the existing writer's pairs (enumerated in
fixtures/snapshots.json), and require both endpoints, the same raw domain, and
nondecreasing time. Reversed boundaries report temporal_conflict rather than
silently disappearing. Raw timestamp zero cannot be represented as a recorded
legacy timestamp; mark legacy_zero_timestamp and omit its legacy duration while
keeping the raw event. Transport completed means ResultPublished at AtnAgent
or ResultAcknowledged at Instance; closed means Closed; otherwise incomplete.
These are legacy classifications, not an assertion of all-phase completeness.
Phase summary preserves the existing writer's exact indices on sorted values:
min=0, max=n-1, median=(n-1)//2, p95=round((n-1)*0.95), and
p99=round((n-1)*0.99), with Python ties-to-even rounding. The baseline docstring
calls this nearest-rank, but these index rules define compatibility; do not
substitute ceil(p*n)-1.
Loss never creates normal derived slices; queries return complete/partial/
conflict/unavailable with source event references and quality reasons.
