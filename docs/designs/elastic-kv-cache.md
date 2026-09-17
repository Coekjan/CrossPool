# Elastic KV-cache Pooling

CrossPool makes physical KV-cache backing on one attention GPU elastic across
co-located Instances. SGLang continues to own requests, logical token/page
allocation, prefix-cache contents, and eviction order. CrossPool owns stable
virtual storage, the allocator's admitted prefix, generation-scoped capacity
coordination, and physical map/unmap operations.

Elastic pooling is always present in the CrossPool SGLang integration. It is
not a shared KV cache: one Instance cannot address, reuse, or inspect another
Instance's KV contents. Reclamation may evict reclaimable prefix-cache suffixes
through SGLang before their physical backing is removed.

## Capacity model

A **KV VMM Backing** is one Instance Rank's stable CUDA virtual reservation for
its SGLang KV pool. It exposes the Tensor views consumed by attention while its
mapped physical prefix may change.

A **KV Page Bundle** is one rank-local compound mapping unit. It covers the same
row interval across every local attention layer and K/V component. Its byte
size therefore depends on that model and rank's KV geometry; bundle counts are
comparable inside one Capacity Group, while physical accounting always uses
each partition's own `bundle_bytes`.

A **KV Capacity Group** is `(instance_id, atn_dp_rank)`. Every TP rank in the
group receives the same operation target and switches to the same
allocator-visible bundle prefix after a common readiness vote, while backing
its own physical partition. Different DP ranks have independent request,
prefix-cache, and capacity state.

A **KV Capacity Pool** is the physical byte budget available across all
Instance partitions on one attention GPU. It is frozen once after Graph capture
from the configured device-memory utilization, observed device memory, and
already mapped bootstrap backing. `scheduler.atn_concurrency` is reserved for
future attention compute admission and is not part of this memory model.

Each partition registers immutable geometry: bundle bytes and capacity, minimum
backed bundles, token capacity, CUDA mapping granularity, row bytes, tokens per
row, and token page size. CrossPool derives the usable logical token prefix from
that geometry; the daemon does not assume one model-independent byte-to-token
conversion.

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
deadline. Channel storage is never reset or reused by a later Generation.

A command is one immutable group-local sequence and absolute bundle target.
Each partition maps missing growth backing before voting, but does not expose
the larger logical prefix until every TP rank is ready. For reclaim, each
partition selects currently reclaimable SGLang cache suffix nodes at its
pre-planning boundary. If any partition is not ready, all keep the old logical
capacity and continue ordinary scheduling; the next iteration may retry with
a fresh selection. New Prefill waits during reclaim, while already admitted
chunked Prefill and running Decode can progress. No admitted request is killed
to satisfy reclaim, and a live suffix does not change the command target.

An all-ready vote makes every rank switch before its next batch planning.
Reclaim then evicts the selected cache nodes and waits for prior GPU users
before physical unmap. Every partition publishes terminal completion; the
daemon retires the operation only after all sequence-correlated completions.
TP-one groups switch without a distributed vote. Startup uses the same switch
contract after Graph capture, but obtains its initial command by channel
polling rather than the service-time TP request broadcast. No unfinished
operation is superseded.

The TP leader also publishes the latest persistent **KV Capacity Demand**:
one concrete Prefill request or blocked Decode batch couples its absolute
bundle requirement to the earliest deadline among its affected requests.
Prefill deadlines start at SGLang scheduler receipt plus the effective model
`ttft_ms`; Decode deadlines start at the last token completion, or Prefill
completion before the first Decode token, plus `tbt_ms`. These scheduler-local
timestamps do not include API handling, tokenization, or request IPC. A group
publishes its earliest-deadline unresolved witness with the completed capacity
operation under which it was evaluated. No requested value means demand
resolved. Re-reading does not consume demand, and applying capacity does not
clear it; scheduling must evaluate the new capacity before publishing feedback
for that operation.

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
   memory observation. The daemon freezes every physical pool, publishes each
   group's immutable service ceiling, then issues floor-first initial
   operations sequentially.
7. Each rank waits for its initial command and service ceiling, participates
   in the common readiness vote, switches its logical prefix, and publishes
   completion before continuing scheduler construction. The all-rank
   initialization barrier guards System Ready. SGLang's stable request limit
   uses the service ceiling; its allocator exposes only the current active
   prefix.

During service, each SGLang scheduler iteration receives its DP group's command
through SGLang's existing TP broadcast. Before ordinary batch planning, each
rank with an unapplied command participates in its TP readiness vote; ordinary
no-command and paused iterations do not vote. An unready vote retains the old
logical capacity while ordinary scheduling continues.
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
more recent grants are preferred. No donor crosses its immutable floor.
Deadline ordering does not guarantee SLO satisfaction or bounded reclaim time:
the policy does not preempt admitted work, and a live suffix can delay physical
release.

One serialized funding attempt gathers the complete byte shortfall for a
concrete demand witness from unassigned pool bytes and, if necessary, multiple
donors. It grows the borrower once to that witness's full target rather than
publishing unusable intermediate capacities. Donor shrink remains immutable
even if the borrower demand changes while it drains. After donor completion,
the daemon revalidates the exact borrower sequence, target, and deadline before
continuing; stale attempts release their newly unassigned bytes to normal
arbitration. While a donor operation is pending, another group may grow
directly from already unassigned bytes only on pools disjoint from both the
pending borrower and donor; no second donor attempt starts. A completed
borrower growth, not command publication or donor reclaim, advances its grant
order.

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
Fabric Generation. Participant loss is generation-fatal; no recovery or stale
channel reuse path exists.

## Operational evidence

The daemon emits low-frequency `INFO` records after authoritative transitions:
pool frozen, growth requested, and reclaim requested. It keeps polling,
repeated demand, unavailable donor scans, per-request rejection, per-rank
command receipt, and per-bundle VMM work silent. Logs are diagnostic evidence,
not a protocol or correctness API.

| Event | Principal fields | Meaning |
| --- | --- | --- |
| `kv capacity pool frozen` | `generation`, `device`, `capacity_bytes`, `floor_bytes` | The post-capture physical pool is fixed. |
| `kv capacity growth requested` | group identity, start and target bundles, command sequence | The complete persistent demand target was funded and issued. |
| `kv capacity reclaim requested` | borrower and donor identity, donor start and target bundles, command sequence | A donor received a fixed smaller target; reclamation has not necessarily completed. |

The exact split between native, integration, ordinary serving, and dedicated
Elastic KV evidence is owned by [Qualification](qualification.md).

Qualification cases describe evidence, not a dtype or hardware allowlist. The
integration rejects modes that replace the required physical layout or logical
prefix-cache seam, including unified-memory or alternate KV layouts, KV
offload/disaggregation, disabled or non-Unified Radix caches, non-Python Unified
TreeCore, and overlapping startup weight loading. Lack of qualification alone
does not reject an otherwise structurally compatible configuration.
