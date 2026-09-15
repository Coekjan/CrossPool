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
group receives the same logical target and active bundle counts, but prepares
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

## Capacity channel

One daemon-owned POSIX shared-memory channel belongs to one Fabric Generation.
Its fixed header contains the native ABI version, byte extent, and pool, group,
and partition counts. The daemon creates and unlinks it; AtnAgents and Instance
Ranks attach role-specific handles and only publish their owned fields.

The regions and writers are:

| Region | Owner | Published state |
| --- | --- | --- |
| Pool entry | AtnAgent | One post-capture `(total_bytes, free_bytes)` observation for its attention GPU. |
| Group entry | Daemon | Target and active `CapacityPublication` values. |
| Group entry | TP leader Instance Rank | Pressure or pressure-clear `CapacityPublication`. |
| Partition entry | Instance Rank | Capture completion and actual backing prepared for a command sequence. |

`CapacityPublication` stores one `uint32` sequence and one `uint32` bundle count
as one lock-free eight-byte atomic value. Zero sequence means no group command
or pressure publication. A bootstrap backing report may use sequence zero, so
its positive bundle count marks presence.

The daemon publishes a command's target before its active value, both with
release ordering. An Instance reads target, active, then target again with
acquire ordering and accepts the command only when the target reads match and
both publications carry the same sequence. Backing, pressure, and capture
publications use their single-writer release/acquire edges. Sequence wrap is
forbidden within a Generation; channel storage is never reset or reused by a
later Generation.

Capacity commands distinguish two prefixes:

- `target_bundles` is the physical prefix every TP partition must protect and
  converge toward.
- `active_bundles` is the logical prefix currently exposed to every allocator
  in the Capacity Group.

Growth publishes a larger target while leaving active capacity unchanged. The
daemon advances active capacity only after every TP partition reports the new
target prepared. Shrink publishes the smaller active prefix immediately, then
each partition waits until the suffix is logically free and prior GPU access is
complete before unmapping it.

## Lifecycle

Startup follows the Generation lifecycle:

1. SGLang constructs the elastic KV pool, reserves its stable virtual range,
   and maps the minimum bootstrap prefix.
2. The Instance Rank registers its KV geometry together with its Transport and
   FFN profiles. Complete membership creates the Generation capacity channel.
3. The Instance attaches its partition and group slots; its AtnAgent attaches
   the corresponding pool and partition slots.
4. SGLang builds attention backends and captures its CUDA Graphs against the
   stable virtual addresses.
5. Post-capture finalization synchronizes the CUDA device, returns the
   allocator and backing to the bootstrap floor, publishes capture completion,
   and waits for an initial command.
6. After all local captures complete, each AtnAgent publishes its sole device
   memory observation. The daemon freezes every physical pool, publishes
   floor-first initial targets, waits for every TP partition to prepare them,
   and activates the common logical capacity.
7. Scheduler construction continues only after the Instance has observed the
   initial active command.

During service, each SGLang scheduler iteration receives its DP group's command
through SGLang's existing control broadcast. Before ordinary scheduling, the
Instance applies active logical capacity, prepares growth, and advances safe
shrink. Prefill or Decode admission failures publish one leader-local pressure
edge. Pressure clears when all witnessed rejected requests leave the waiting
queue.

One daemon task serializes policy steps. It processes the pressure queue in FIFO
order and performs at most one capacity transition per step. Free pool bytes
are used first. Otherwise, one donor above its immutable floor is asked to
reclaim one bundle; a later step grows and then activates the borrower. Target,
actual backing, and active capacity remain separately accounted, so a requested
reclaim is not reported as a completed transfer.

Service-time unmap is asynchronous with GPU execution. Logical suffix
withdrawal and SGLang cache eviction happen first. The Instance records one CUDA
Event on the scheduler's actual execution stream and unmaps only after that
event completes. A newer command sequence retires the previous pending event.
All allocator, prefix-cache, reconciler, and VMM mutations occur on the SGLang
scheduler thread.

Shutdown first stops KV users and synchronizes the device. The Instance closes
its channel attachment, drops pool views, unmaps its backed prefix, and releases
the virtual range. The daemon retires and unlinks the capacity channel with its
Fabric Generation. Participant loss is generation-fatal; no recovery or stale
channel reuse path exists.

## Operational evidence

The daemon emits low-frequency `INFO` records after authoritative transitions:
pool frozen, pressure entered or cleared, growth requested, reclaim requested,
and capacity activated. It keeps polling, repeated state, unavailable donor
scans, per-request rejection, per-rank command receipt, and per-bundle VMM work
silent. Logs are diagnostic evidence, not a protocol or correctness API.

| Event | Principal fields | Meaning |
| --- | --- | --- |
| `kv capacity pool frozen` | `generation`, `device`, `capacity_bytes`, `floor_bytes` | The post-capture physical pool is fixed. |
| `kv capacity pressure entered` | `generation`, `instance`, `dp_rank`, `active_bundles`, `pressure_sequence` | A new unresolved admission-pressure edge was consumed. |
| `kv capacity pressure cleared` | `generation`, `instance`, `dp_rank`, `pressure_sequence` | The corresponding rejected requests no longer remain queued. |
| `kv capacity growth requested` | group identity, active and target bundles, command sequence | Free physical bytes allowed a larger target. |
| `kv capacity reclaim requested` | borrower and donor identity, donor capacities and pressure, command sequence | A donor received a smaller target; reclamation has not necessarily completed. |
| `kv capacity activated` | group identity, active bundles, command sequence | Every TP partition prepared the target and it became allocator-visible. |

The exact split between native, integration, ordinary serving, and dedicated
Elastic KV evidence is owned by [Qualification](qualification.md).

Qualification cases describe evidence, not a dtype or hardware allowlist. The
integration rejects modes that replace the required physical layout or logical
prefix-cache seam, including unified-memory or alternate KV layouts, KV
offload/disaggregation, disabled or non-Unified Radix caches, non-Python Unified
TreeCore, and overlapping startup weight loading. Lack of qualification alone
does not reject an otherwise structurally compatible configuration.
