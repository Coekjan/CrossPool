# Elastic Context State

CrossPool manages physical-memory capacity for Elastic Context State (ECS)
across colocated Instances. SGLang owns requests, logical token/page allocation,
prefix-cache contents and eviction policy. CrossPool owns byte permission, stable virtual
storage, sparse physical backing and coordinated allocation/reclamation.

Pooling is always present in the CrossPool SGLang integration. Full attention
and sliding-window attention (SWA) use their real upstream pools and paired
allocator. Their rank-local backings share one Elastic Context State Memory
Quota; cache contents remain private to each Instance.

## Capacity model

The daemon grants an absolute physical-byte allowance to each TP rank in a
Capacity Group, identified by Model and attention DP rank. It sees each rank's
aggregate bootstrap floor and reservation bound. Component pool names, token
geometry, bundle counts and physical IDs stay inside the Instance.

An Instance Rank's `ecs.MemoryQuota` charges every managed backing.
Pending allocations, resident bundles and retiring bundles retain their charges
until checked rollback or physical release confirms that the bytes are free.
Logical free pages alone do not refund bytes. Full and SWA can reuse the same
quota as their demands change.

An Elastic Context State Capacity Pool is the physical budget shared across
Instance partitions on one attention Device. After local Graph captures
complete, the daemon freezes it from observed free memory plus bootstrap
backing, less the larger of:

- the configured unused-memory margin; and
- summed Attention Runtime Headroom from the device's Instance Ranks.

The integration derives runtime headroom from the complement of resolved
`mem_fraction_static`. It also retains SGLang's eager-activation reserve when
Decode Graph coverage is smaller than explicit or model-derived request
concurrency. SGLang retains ownership of post-capture eligibility.
`scheduler.atn_concurrency` is reserved for attention compute admission and is
independent of this memory budget.

Initial grants preserve every floor, split the remaining device budget equally,
and cap each share at its reservation. Capped shares and integer remainders
remain unassigned. Immutable per-rank service ceilings bound later borrowing.
Request concurrency and logical service limits are derived once through the
upstream configurator under those ceilings; current quota changes do not reset
allocators, resize request storage or change Graph capture geometry.

## Storage and local ownership

Each MHA/GQA or MLA child owns a `KvVmmBacking`: one stable compound virtual
range, its Tensor views and an exact inventory of mapped bundle IDs.
A bundle covers the same row interval in every layer and buffer of that
component. Different components can have different bundle costs. Dummy storage
is pinned; other mappings may contain holes.

The integration queries recommended CUDA mapping granularity with the same
allocation properties used to create physical memory. Allocations stay local
to the Instance Rank's selected device. Mapping/unmapping whole bundles
preserves virtual addresses and view strides.

The original `SWAKVPool` and `SWATokenToKVPoolAllocator` retain their Full/SWA
composition and mapping. Concrete elastic token/paged children partition
unowned IDs into resident free IDs and unavailable IDs. A retirement withdraws
all intersecting IDs, including staged free containers. Deferred grouped frees
remain owned until the upstream group closes.

The shared elastic-allocator reset rejects pending retirement, delegates to the
original token or paged reset, then rebuilds the resident/unavailable partition
from actual backing. Each upstream reset retains its own deferred-free rules.

Page planning prefers resident holes, then partially backed candidates in
stable virtual order. It prices the exact union of missing bundle coverage
across all components under one quota. Only successfully backed pages become
deliverable to the original allocators. A later clean miss can leave new
physical mappings as charged reusable residency.

`SglangElasticContextState` assembles the actual root pool, allocator, request pool
and existing configurator. Its ordered `SglangKvComponent` associations bind
each child allocator to its Full or SWA cache component; backings and pool
profiles derive from those allocator owners. The state owns resident-page
observations, eviction-target conversion and cache-candidate selection. The
reconciler retains TP agreement, quota adoption and retirement ordering.
The runner retains the state before transport startup; Scheduler setup binds
the real Python Unified Radix cache. The state maintains no additional request
table or cache index.

Ordinary backing failures perform checked local rollback. Terminal device
errors or uncertain cleanup retain the original failure and resource owners
for the existing [Generation failure boundary](control-plane.md#startup-and-shutdown).

SWA prefix locks cover the page-rounded live window, splitting its boundary
even through already-locked ancestors while retaining upstream release
receipts. Window frees preserve a matchable tail behind the page-aligned cache
insertion boundary and honor any stricter upstream retain floor. Logical
window progression must leave the next stash/rematch's required cache contents
protected.

## TP decisions and allocation

TP ranks agree on logical requests, order, token ranges, cache effects and the
allocation outcome. Their storage IDs, free lists and physical residency remain
rank-local.

At the fixed planning boundary, ranks reduce logical free-page vectors by
minimum. This observation feeds the existing Prefill budget offsets; it does
not overwrite local free lists. Admission prices all component requirements
together under each local byte grant. The full remaining request forecast is
distinct from pre-lock resident preparation and the next batch's actual page
acquisitions. All three requirements use each component's allocator-page unit.
The forecast retains clipped output lookahead and upstream budget offsets;
preparation establishes deliverable input backing rather than reserving future
output storage.

Prefill retains the upstream selector and commit paths. It agrees on rejection
or feasibility even when one rank rejects locally. If a smaller legal chunk
fits, common probes rerun pure selection before tentative commit. Chunk
continuation retains its distinct lock and ownership semantics; no allocatable
chunk leaves that same continuation parked.

Before a new chunk owner acquires its persistent prefix lock, preparation covers
the remaining Full input and a conservative SWA peak. For SWA, the base target
is `min(R, W + 2 * C) + 1` pages: `R` is the complete remaining input's actual
new acquisitions, `W` the rounded-up window and `C` the configured maximum
chunk size in pages. Include already selected work and preserve the engine's
admission allowances. This target can delay first admission; it is not a
minimum-capacity claim or an exclusive lease on those pages.

`ElasticPrefillAdder` owns batch formation; `CapacityReconciler` owns quota
adoption, matched resource decisions and retirement. An empty adder can advance
one local maintenance step for its current candidate, then yield to the outer
loop. Once its selected-request list is nonempty, later candidates can prepare
additional backing or end selection. They preserve the formed batch's backing
through the complete allocation verdict instead of reclaiming it for their
own shortage. Actual logical window frees and same-pool reuse remain valid.

Upstream candidate commit updates tentative selection and real prefix locks;
the allocation journal commits only after the complete TP verdict. A clean
allocation miss restores unpublished selection and acquisitions, retaining
successful backing mappings. The journal does not reverse committed physical
retirement.

Transient preparation misses, local maintenance and pending retirement return
the engine's `OTHER` result, keeping admission open. A stable empty-batch quota
shortage returns `NO_TOKEN` and retains the external witness with its original
deadline. Both ordinary and continuation callers clear that request's external
witness on a transient wait. A later rejected candidate records no demand
containing the formed batch's tentative budget offsets. Adopting a new grant
reopens the engine's full-batch gate.

Decode coordinates full-batch and retraction decisions outside its repeatedly
called, side-effect-free subset checker. Victim order and beam abort semantics
remain upstream-owned. A temporary quota shortage can return ordinary work
to the existing waiting queue.

The complete allocation window is:

```text
tentative selection / Decode staging
  -> alloc_for_extend or alloc_for_decode binds the actual batch
  -> committed window releases and actual new request rows
  -> exact joint backing preparation at the common allocation helper
  -> original child allocations and Full-to-SWA mapping writes
  -> original request-table writes and KV lengths
  -> one TP CPU-group MAX verdict
      READY: commit, then counters / penalties / DP sync / forward
      CAPACITY_MISS: restore unpublished ownership
      FAILED: preserve the original error and fail the Generation
```

There is no collective between child allocations. The common preparation hook
uses actual batch page requirements, including partial-page reuse and beam
rows, rather than the helper's conservative token estimate.

`SglangAllocation` records actual child acquisitions after upstream conditional
sorting. The Full child snapshots every returned mapping location, including
Decode writes that acquire no new page. New request rows, lock receipts, queue
positions, chunk ownership and tentative KV lengths retain their real owners.

A common clean miss restores overwritten mapping entries, invalidates
unpublished request-table tails, returns actual new child pages, releases new
rows/locks and restores tentative metadata. Decode tail invalidation covers
every actual row, including beam rows absent from the Python request list.
Committed prefix matching, window release, retraction and agreed cache eviction
stay committed. Successful new backing remains charged.

The recoverable window ends before post-allocation accounting, DP
synchronization and forward. Decode stages the actual upstream dataclass and
uses new sequence tensors; shared penalty state changes only after common
success. Arbitrary execution or cleanup failures are not clean capacity misses.

## Reclamation and distributed quota policy

Local pressure first retires completely unowned bundles. When cache contents
must change, the TP leader proposes the original policy's exact candidate as an
owned root-to-node Radix key and component effect. Every rank resolves and
validates that same endpoint before an all-ready vote. Full leaf eviction
includes its actual cascades; an SWA-only operation preserves Full contents and
is rejected if it would instead cascade to Full.

Only actual frees establish physical emptiness. The Instance withdraws the
affected allocation IDs, records a CUDA Event on the actual execution stream,
and polls it at later planning boundaries. Confirmed completion allows exact
unmap and byte refund. A local Event result changes local ownership, not the
collective order or another rank's physical release.

Unmatched local eviction helpers retain their requested shortfall for the next
common cache boundary. They cannot independently mutate replicated cache
contents or enter a collective. Reclamation can take several planning rounds;
one cache operation need not satisfy the entire physical shortfall.

For an external grant, all ranks wait for unfinished local retirement and
confirm that actual retained backing fits their target before adopting the
vector. Growth authorizes future allocation rather than eagerly mapping pool
shares. Shrink parks new Prefill while continuation and Decode can progress.
Each rank publishes actual retained bytes, which can be smaller than its quota.

An admitted chunk owner retains the old grant and prevents new physical
retirement and shrink completion through its final input allocation commit.
The allocation journal retains the prior owner across the final chunk's
tentative handoff; existing retirement can still finish. Decode keeps its
next-step checks and ordinary retraction behavior. Future output storage is
not permanently reserved, and live ownership can delay shrink without a
bounded completion time.

The TP leader retains the earliest unresolved admission witness under its
evaluated grant sequence. A Prefill witness covers remaining uncached input,
clipped remaining output and engine budget charges; a Decode witness covers its
blocked request set. Deadlines use scheduler receipt plus `ttft_ms`, or the
last token/Prefill completion plus `tbt_ms`. Reevaluation preserves the original
deadline. Resolution clears both optional requirement and deadline fields.

The daemon uses the existing deadline ordering and donor arbitration:

- Overdue borrowers precede predeadline borrowers. Disjoint pools can fund
  independent work while another transfer is pending.
- Unassigned bytes fund a witness first. Selected donors retain their floors
  and shrink only on devices with a shortage.
- One funding attempt can gather bytes from several donors, then grants the
  borrower's complete target after revalidating its original witness.
- A stale witness releases newly unassigned capacity to normal arbitration.

Until every rank's sequence-correlated completion arrives, the daemon charges
the elementwise maximum of old and target permissions. Local frees and partial
group completion cannot make those bytes available to another borrower.

## Elastic Context State Control Channel

One daemon-owned POSIX shared-memory channel belongs to each Fabric Generation.
It carries a directory of Capacity Groups and their TP partitions; different
groups can have different TP widths. Attach validates the native ABI, extents
and canonical directory.

| Publication | Writer | Value |
| --- | --- | --- |
| Device observation | AtnAgent | Post-capture total/free physical bytes. |
| Service ceiling | Daemon | Immutable per-TP-rank byte bounds. |
| Quota command | Daemon | Sequence and absolute per-rank byte vector. |
| Admission demand | TP leader | Evaluated sequence, optional byte vector and original deadline. |
| Startup backing/capture | Instance Rank | Actual bootstrap bytes and capture completion. |
| Quota completion | Instance Rank | Sequence and actual retained bytes. |

Command, demand and completion snapshots use atomic payload words inside
odd/even revision publications. Readers accept only matching committed
revisions. One-shot startup and ceiling values have explicit ready words;
zero bytes remain valid data. Exact layouts and interfaces belong to native
declarations and generated bindings under `xpool.native.ecs`.

Commands are immutable within a sequence, and an unfinished group command
cannot be superseded. Demand can be replaced or cleared under the same
evaluated sequence. Startup polls the same channel; serving-time commands
travel through SGLang's existing TP request broadcast.

## Lifecycle and evidence

Startup describes every resolved component before mapping, charges their
summed floor, and lends prepared backings to the original pool factory. Instance
registration publishes aggregate byte bounds alongside Transport and FFN
profiles. After capture, ranks publish actual backing and capture completion;
AtnAgent observations allow the daemon to freeze budgets and issue initial
grants. Stable engine limits and the existing all-rank initialization barrier
precede System Ready.

Serving planning progresses commands and retirement even while idle. All
allocator, cache, reconciler and backing mutations remain on the Scheduler
thread. Shutdown stops users and synchronizes before closing the channel,
dropping all component views and releasing exact mappings and virtual ranges.
Participant loss ends the Generation under the existing lifecycle.

AtnAgents log channel attachment and memory observation. The daemon logs frozen
budgets and issued quota transitions. Logs remain diagnostics rather than a
protocol or correctness assertion surface. [Qualification](qualification.md)
owns the native, component, installed serving and resource evidence boundaries.

The integration supports the existing ordinary decoder boundary with Python
Unified Radix and no KV offload. Alternate layouts, transfer/recurrent variants,
speculative execution and mixed chunked Prefill/Decode require a separate
integration design; Full/SWA support does not enable those paths.
