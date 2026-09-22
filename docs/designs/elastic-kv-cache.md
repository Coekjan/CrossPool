# Elastic KV Cache Pooling

CrossPool makes physical KV Cache backing on one attention GPU elastic across
co-located Instances. SGLang continues to own requests, logical token/page
allocation, prefix-cache contents, and eviction order. CrossPool owns stable
virtual storage, the allocator's admitted prefix, generation-scoped capacity
coordination, and physical map/unmap operations.

Elastic pooling is always present in the CrossPool SGLang integration. Each
Instance keeps independent logical KV contents and prefix-cache ownership;
CrossPool shares only physical backing. Reclamation may evict reclaimable
prefix-cache suffixes through SGLang before their physical backing is removed.

## Capacity model

A **KV VMM Backing** is one Instance Rank's stable CUDA virtual reservation for
its SGLang KV pool. It exposes the Tensor views consumed by attention while its
mapped physical prefix may change.

A **KV Page Bundle** is one rank-local compound mapping unit. It covers the same
row interval across every local attention layer and K/V component. Its byte
size therefore depends on that model and rank's KV geometry. Bundle counts are
comparable inside one Capacity Group; physical accounting uses each
partition's own `bundle_bytes`.

A **KV Capacity Group** is `(instance_id, atn_dp_rank)`. Every TP rank in the
group receives the same operation target and switches to the same
allocator-visible bundle prefix after a common readiness vote, while backing
its own physical partition. Different DP ranks have independent request,
prefix-cache, and capacity state.

A **KV Capacity Pool** is the physical byte budget available across all
Instance partitions on one attention GPU. It is frozen once after Graph capture
from the configured device-memory utilization, observed device memory,
already mapped bootstrap backing, and the summed runtime headroom declared by
the GPU's Instance Ranks. The larger of the configured utilization margin and
that runtime headroom remains outside the pool. `scheduler.atn_concurrency` is
reserved for future attention compute admission and is not part of this memory
model.

The SGLang integration enables upstream post-capture KV sizing before
configuration resolution. Each Instance Rank derives its Attention Runtime
Headroom from device capacity and the complement of the resolved
`mem_fraction_static`. When enabled Decode Graph coverage is smaller than an
explicit or model-derived `max_running_requests`, the rank keeps the larger of
that base headroom and SGLang's eager-activation reserve. SGLang retains
ownership of model eligibility for post-capture sizing; CrossPool does not
override it. The daemon sums the resulting immutable declarations per GPU.

Each partition registers immutable geometry: bundle bytes and capacity, minimum
backed bundles, token capacity, CUDA mapping granularity, row bytes, tokens per
row, and token page size. CrossPool derives the usable logical token prefix from
that geometry, so the daemon can account for each model's byte-to-token ratio.

## Storage and allocator integration

The SGLang integration supplies concrete MHA/GQA and MLA KV-pool subclasses.
Each pool reserves one compound virtual range and owns one `KvVmmBacking`.
Mapping and unmapping operate only on complete tail bundles, so Tensor addresses
remain stable and the mapped region is always a contiguous prefix. The pool
drops its Tensor views before releasing the backing at shutdown.

Concrete ordinary and paged allocator subclasses retain SGLang's allocation
behavior while withholding free suffix IDs above the active token capacity.
Growth returns newly admitted free IDs to SGLang. Shrink first withdraws suffix
IDs from allocation; live, locked, or cached suffix IDs delay physical unmap.
SGLang's Unified Radix Cache remains the source of prefix-cache ownership and
eviction order. CrossPool only queries reclaimable IDs inside the active prefix
and asks SGLang to evict through its existing cache operation.

## KV Control Channel

One daemon-owned POSIX shared-memory channel belongs to one Fabric Generation.
Its fixed header contains the native ABI version, byte extent, and pool, group,
and partition counts. The daemon creates and unlinks it; AtnAgents and Instance
Ranks attach role-specific handles and only publish their owned fields.

The regions and writers are:

| Region | Owner | Published state |
| --- | --- | --- |
| Pool entry | AtnAgent | One post-capture `(total_bytes, free_bytes)` observation for its attention GPU. |
| Group entry | Daemon | Immutable service ceiling and one fixed command stream. |
| Group entry | TP leader Instance Rank | Latest persistent capacity demand. |
| Partition entry | Instance Rank | Capture completion, bootstrap backing, and terminal operation completion. |

Commands and partition completions use lock-free eight-byte `(sequence, value)`
cells; sequence zero means unpublished. A partition publishes one completion
only after its logical switch and terminal physical work. Demand has a separate
revision: zero means unpublished, odd means the leader is updating its payload
and deadline, and nonzero even means committed. A reader accepts the payload
and deadline only between two matching even revision reads. The encoded
deadline distinguishes resolved demand from an active absolute monotonic
deadline. Channel storage belongs to one Generation for its entire lifetime.

A command is one immutable group-local sequence and absolute bundle target.
Each partition maps missing growth backing before voting. The larger logical
prefix becomes visible only after every TP rank is ready. For reclaim, each
partition selects currently reclaimable SGLang cache suffix nodes at its
pre-planning boundary. If any partition is not ready, all keep the old logical
capacity and continue ordinary scheduling; the next iteration may retry with
a fresh selection. New Prefill waits during reclaim, while already admitted
chunked Prefill and running Decode can progress. Reclaim waits for admitted
work to finish; admitted requests are never killed. A live suffix remains
outside the command target.

An all-ready vote makes every rank switch before its next batch planning.
Reclaim then evicts the selected cache nodes and waits for prior GPU users
before physical unmap. Every partition publishes terminal completion; the
daemon retires the operation only after all sequence-correlated completions.
TP-one groups switch without a distributed vote. Startup uses the same switch
contract after Graph capture, but obtains its initial command by channel
polling rather than the service-time TP request broadcast. The daemon
serializes unfinished operations before issuing the next command.

The TP leader also publishes the latest persistent **KV Capacity Demand**:
one concrete Prefill request or blocked Decode batch couples its absolute
bundle requirement to the earliest deadline among its affected requests.
Prefill deadlines start at SGLang scheduler receipt plus the effective model
`ttft_ms`; Decode deadlines start at the last token completion, or Prefill
completion before the first Decode token, plus `tbt_ms`. These scheduler-local
timestamps do not include API handling, tokenization, or request IPC. A group
publishes its earliest-deadline unresolved witness with the completed capacity
operation under which it was evaluated. No requested value means demand
resolved. Re-reading leaves demand unchanged, and applying capacity leaves the
demand record intact until scheduling evaluates the new capacity and publishes
feedback for that operation.

## Lifecycle

Startup follows the Generation lifecycle:

1. SGLang constructs the elastic KV pool, reserves its stable virtual range,
   and maps the minimum bootstrap prefix.
2. The Instance Rank registers its KV geometry together with its Transport and
   FFN profiles. Complete membership creates the Generation KV Control Channel.
3. The Instance attaches its partition and group slots; its AtnAgent attaches
   the corresponding pool and partition slots.
4. SGLang builds attention backends and captures its CUDA Graphs against the
   stable virtual addresses.
5. Post-capture finalization synchronizes the CUDA device, returns the
   allocator and backing to the bootstrap floor, publishes capture completion,
   and waits for an initial command.
6. After all local captures complete, each AtnAgent publishes its sole device
   memory observation. The daemon freezes every physical pool against both the
   utilization limit and registered runtime headroom, publishes each group's
   immutable service ceiling, then issues floor-first initial operations
   sequentially.
7. Each rank waits for its initial command and service ceiling, participates
   in the common readiness vote, switches its logical prefix, and publishes
   completion before continuing scheduler construction. The all-rank
   initialization barrier guards System Ready. SGLang's stable request limit
   uses the service ceiling; its allocator exposes only the current active
   prefix.

During service, each SGLang scheduler iteration receives its DP group's command
through SGLang's existing TP broadcast. Before ordinary batch planning, each
rank with an unapplied command participates in its TP readiness vote. A live,
unpaused scheduler continues these iterations while idle, so capacity commands
progress without inference requests. Paused iterations skip the vote. An
unready vote retains the old logical capacity while ordinary scheduling
continues.
Authoritative Prefill or Decode admission failures update the leader's demand.
During reclaim, new Prefill waits while already admitted work can finish; when
drain completes, waiting requests are made eligible for ordinary scheduling
again. A demand remains eligible only while its exact blocked request set and
evaluated operation sequence remain current.

One daemon task owns policy state. It tries overdue borrowers first, rotating
by oldest completed service-time growth grant and breaking ties by deadline
and stable group index. If none can issue an operation, it considers
predeadline borrowers in deadline order only when their physical pools are
disjoint from every eligible overdue borrower's pools. Groups without active demand are
preferred as donors; when all usable donors have demand, later deadlines and
more recent grants are preferred. Each donor retains its immutable floor.
Deadline ordering prioritizes work; admitted work remains non-preemptive, so a
live suffix can delay physical release.

One serialized funding attempt gathers the complete byte shortfall for a
concrete demand witness from unassigned pool bytes and, if necessary, multiple
donors. It grows the borrower once to that witness's full target rather than
publishing unusable intermediate capacities. Donor shrink remains immutable
even if the borrower demand changes while it drains. After donor completion,
the daemon revalidates the exact borrower sequence, target, and deadline before
continuing; stale attempts release their newly unassigned bytes to normal
arbitration. While a donor operation is pending, another group may grow
directly from already unassigned bytes only on pools disjoint from both the
pending borrower and donor. The daemon allows one donor attempt at a time. Only
completed borrower growth advances its grant order.

While an operation is outstanding, every affected physical pool charges the
larger of the starting and target backing. Reclaimed bytes are not credited
until every TP partition completes physical retirement. This conservative
accounting prevents a borrower and donor from owning the same bytes while
allowing unrelated completion and growth already fundable from unassigned bytes
to proceed.

Service-time unmap is asynchronous with GPU execution. Logical suffix
withdrawal and SGLang cache eviction happen first. The Instance records one CUDA
Event on the scheduler's actual execution stream and unmaps only after that
event completes. No newer command is accepted while an operation or retirement
event is unfinished. All allocator, prefix-cache, reconciler, and VMM mutations
occur on the SGLang scheduler thread.

Shutdown first stops KV users and synchronizes the device. The Instance closes
its channel attachment, drops pool views, unmaps its backed prefix, and releases
the virtual range. The daemon retires and unlinks the KV Control Channel with its
Fabric Generation. Participant loss ends the Generation; a later Generation
creates a fresh channel.

## Operational evidence

The AtnAgent records successful KV Control Channel attachment and its one-shot
post-capture memory observation. The daemon emits low-frequency `INFO` records
after authoritative pool and capacity transitions. Polling, repeated demand,
unavailable donor scans, per-request rejection, per-rank command receipt, and
per-bundle VMM work stay below the operational log level. Logs are diagnostic
evidence, not a protocol or correctness API.

| Event | Principal fields | Meaning |
| --- | --- | --- |
| `kv control attached` | device, pool, partition count | The AtnAgent attached its Generation-scoped control surface. |
| `kv memory observed` | device, total bytes, free bytes | Graph Capture is complete and pool sizing has a stable memory observation. |
| `kv pool frozen` | device, total/free/bootstrap bytes, utilization/runtime/reserve/capacity/floor bytes | The post-capture pool is fixed after both reserve bounds. |
| `kv capacity initialized` | group, bundle transition, token transition | Every partition completed the initial group command. |
| `kv capacity growth requested` | group, bundle transition, token transition | Unassigned pool bytes funded the full demand target. |
| `kv capacity transfer requested` | donor and borrower transitions | A donor shrink target was fixed for one borrower's exact demand. |

The exact split between native, integration, ordinary serving, and dedicated
Elastic KV evidence is owned by [Qualification](qualification.md).

Qualification cases provide evidence rather than define a dtype or hardware
allowlist. The integration accepts configurations whose physical layout and
logical prefix-cache seam match the contract. Unified-memory or alternate KV
layouts, KV offload/disaggregation, disabled or non-Unified Radix caches,
non-Python Unified TreeCore, and overlapping startup weight loading require a
different integration seam.
