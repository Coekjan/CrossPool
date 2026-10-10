# Test and Benchmark Tooling

The CrossPool development environment provides `xtest` for correctness validation
and `xbench` for multi-model LLM serving measurements. Their shared resource and
process mechanisms live in `xkit`; each tool owns its execution policy and
retained-result semantics.
[Serving Benchmarks](benchmark.md) owns workload, measurement, continuation
and offline report contracts. Performance measurements remain report-only under
[Qualification](qualification.md).

## Installed packages and source inputs

```text
pyproject.toml              Production project and root uv workspace
uv.lock                     Shared dependency lockfile
src/
  xpool/                    Production runtime and configuration
  xpool-dev/
    pyproject.toml          Private xpool-dev project
    xkit/                   Shared resources, processes, storage and serving
    xtest/cli/              Test command
    xtest/harness/          Collection, scheduling, verdicts and qualification
    xbench/cli/             Benchmark command
    xbench/harness/serving/ Serving cases, workloads, observations and reporting
tests/
  tests.toml                Test scenarios and expectations
  suites/                   Executable validation suites
benches/
  benches.toml              Benchmark catalogue
  suites/<family>/          Source-owned benchmark programs and inputs
configs/
  deployments/              Shared portable runtime scenes
```

`xtest` and `xbench` depend on `xkit`, which may use `xpool` configuration and
runtime values. Production `xpool` does not import these tooling packages.
Shared mechanisms do not import tool verdicts, pytest or collected test modules.
Test observers, numerical references, fakes and qualification drivers remain in
`xtest.harness`; they are not benchmark dependencies.

The root [`pyproject.toml`](../../pyproject.toml) owns production `xpool`, its
native build, console command and SGLang plugin. The private
[`xpool-dev` project](../../src/xpool-dev/pyproject.toml) owns `xkit`, `xtest` and
`xbench`, depends on `xpool`, and disables CMake in the pinned scikit-build-core
backend. Its console entries point directly to `xtest.cli:main` and
`xbench.cli:main`. `xpool-dev` is a project directory and distribution name,
not a Python import prefix.

The root uv workspace selects both projects through workspace sources. Root
`uv sync --group dev` installs both editably into one managed virtual environment
using one root lockfile; invoke the tools with `uv run`. The released `xpool`
wheel owns production packages and entry points. Matplotlib belongs to
`xpool-dev`; both projects declare the TOML writer they use.

Test suites and both catalogues remain source inputs rather than wheel contents.
`xtest list/run` require a checkout and applicable development tools. Its report
and cleanup commands operate on retained evidence outside a checkout. `xbench`
accepts an explicit external catalogue; the default `benches/benches.toml` is
relative to the invocation directory, not discovered from installed modules.
Catalogue-relative input paths belong to the declaring file.
Source-owned benchmark programs and prompt/trace inputs belong under
`benches/suites/<family>/`. Installed serving mechanisms live under
`xbench.harness.serving`; the catalogue selects the source module that owns the
scenario's execution.
Random-prompt/Poisson cases declare their generators in the catalogue and need
no dataset directory.

Both commands expose `list`, `run`, `report`, `clean`, `case-gen` and
`config dump`. Exact CLI options and typed interfaces belong to their declarations in
[`xtest.cli`](../../src/xpool-dev/xtest/cli/),
[`xbench.cli`](../../src/xpool-dev/xbench/cli/) and the owning harness modules.
[Test Architecture](../../tests/README.md) owns placement, resource requirements
and execution commands. The [tooling tutorial](../tutorials/tooling.md) owns the
interactive selection, authoring and reporting workflow.

`xkit.config.XpoolDevConfig` owns one process-global development aggregate,
selected by a leaf command's `--config` or `XKIT_CONFIG`. Shared retention policy
lives at the root; test and benchmark policy live in their respective subtrees.
Entry points freeze effective settings and source records in `ToolConfigRecord`
before collection or preparation. Workers install that record without rereading
development files or environment policy. Runtime scene assembly retains its own
file-resolution lifecycle. [Configuration](../configuration.md#development-tools-and-shared-cache)
owns user-facing sources, file setup and path-origin rules.

## Catalogue and source declarations

Both catalogues use named case tables. The table key is a full, canonical
lowercase UUIDv4 validated by [`xkit.case.CaseId`](../../src/xpool-dev/xkit/case.py);
each case has an English description and a dotted source `module`. Serving
experiments use `serving_cases`, while native FFN topology tests use `topology_cases`.
`CaseFamily` owns these category and table names. Both concrete catalogues inherit
`xkit.case.Catalog`, which owns source-file cloning and atomic publication;
their loaders retain schema validation and tool-specific execution selection.
[`xkit.source`](../../src/xpool-dev/xkit/source.py) resolves each module below the
catalogue's sibling `suites/` directory without importing it. Models qualification
keeps its workload and graph expectations as typed source constants and uses
the same portable deployments.

Case identity is authored independently of experiment conditions and remains
permanent. A distinct fixed experiment receives a new UUID; description,
presentation and execution repetition count do not change it. Benchmark
selection accepts a nonempty unique prefix, while persisted references retain
the complete identity. `case-gen` clones one raw declaration, validates it beside
its catalogue before atomic publication, and prints the new UUID, absolute file
and inclusive line range. Existing catalogue text and path origins remain intact.
Authoring assumes one writer. A loaded catalogue remains an immutable snapshot;
a later load sees the appended declaration.

`xtest` and `xbench` export the shared `requirements` and `parameterize`
decorators from [`xkit.declaration`](../../src/xpool-dev/xkit/declaration.py).
They retain metadata on the original function without wrapping execution.
Omitted parameter values bind catalogue cases; explicit Python values support
source-owned case expansion. Test row callbacks produce native pytest
parameters, leaving fixtures, marks and execution with pytest. Ordinary test
input matrices use native `pytest.mark.parametrize`. Resources use
[`ResourceRequirements`](../../src/xpool-dev/xkit/requirements.py), either static
or returned by a callback receiving all concrete parameter values as keyword
arguments, including native pytest parameterization. The shared value validates
nonnegative integer device counts, excluding booleans, and typed, unique Model IDs.
Local checkpoints require configuration. Role-aware owners select MPS during
execution; requirements declare resources rather than controller policy.
Registration performs no catalogue I/O, resource probing or configuration
installation; callbacks
receive neither fixtures nor workdirs.

Repository-owned tests use function-level `xtest.requirements` for complete
external-resource declarations. Module-level pytest policy owns fixtures,
timeouts and other native marks. Case-dependent resources come from the owning
case producer; the owned benchmark E2E binds its selected case to the serving
suite's `requirements_of` with `functools.partial`.
[Test Architecture](../../tests/README.md#requirements) owns authoring examples
and resource policy.

For graph comparisons, collection derives comparison coordinates from
the declaring function's node ID and the original input index before expanding
graph-mode rows. Rows from one input share a comparison group, including when
different inputs reuse a deployment or display label. Initial inventory and task
recollection derive the same coordinates from the frozen declaration. Display
labels use the catalogue ID or deployment basename plus graph mode; pytest owns
duplicate display-ID handling.

Every pytest process loads its complete portable test catalogue once before
collection, including Unit-only sessions. This resolves declarations and portable
deployments, not machine configuration or checkpoints. Actually collected
catalogue-referenced modules must define exactly one catalogue-bound entry,
including modules yielding no test items. Unselected source modules are not
imported for this check. The initial inventory and later task recollection use
the same absolute catalogue path. Each process evaluates a concrete item's
resource declaration once before marker deselection and retains the typed value
in that item's pytest stash. Test-plan construction and resource setup consume
that value directly. The plugin registers and generates resource marks for
native selectors; the function declaration owns resource policy. Initial
collection and task recollection evaluate their own items independently.
Tests without external-resource needs use ordinary pytest functions.

`xbench list` and `run` share isolated source collection. Each selected module
defines one synchronous catalogue-bound entry accepting `case` and `workdir`;
its resource declaration is evaluated without running the entry or preparing
inputs. The discovered function and caller-owned import roots accompany the
supervised worker, which preserves the invocation directory. No function name
or dispatch registry is declared in the catalogue.

The runner prepares fixed workload inputs once per case and owns device admission,
repetition supervision and final evidence sealing. The
[`serving suite program`](../../benches/suites/serving/multi_model.py) owns actual
startup, warmup, measurement and local shutdown, using installed serving and
measurement mechanisms. A worker invokes that entry once per attempt.
Offline reports depend only on retained evidence, not the catalogue or suite
source. Repository pytest uses its normal root import path to exercise these
source programs; suites remain outside installed wheel contents.

## Shared deployment configuration

Both `tests/tests.toml` and owned cases in `benches/benches.toml` reference
independent TOML files by a suffix-free basename such as `atn1-ffn1-lanes2`.
[`xkit.deployment`](../../src/xpool-dev/xkit/deployment.py) resolves that reference
once in either catalogue loader, using the declared model set and the
conventional root `catalogue.resolve().parent.parent / "configs" / "deployments"`.
The loaded case retains the resulting absolute path. Collection loads neither
machine configuration nor model metadata; reports consume retained evidence
without reopening the source deployment. Owned external catalogues follow the
same sibling deployment-tree convention. Portable scenes use the
existing `XpoolConfig` field definitions and value validation. Every scene
declares attention/FFN devices, Executor Lane count and scheduler SLO; the devices
partition contiguous lease-local indices starting at zero. Its model entries
match the selected Model IDs exactly and supply explicit attention TP/DP and
FFN TP. Attention geometry follows the
[complete-World contract](control-plane.md#configuration-and-integration);
FFN TP must fit the declared FfnAgent Fleet. Optional scene fields supply
attention memory utilization and complete model SLO overrides. Graph mode and
tool-specific workloads remain catalogue-owned.

[`xkit.config.DeploymentConfig`](../../src/xpool-dev/xkit/config.py) retains validated
configuration objects and their explicit fields until complete configuration
assembly. Assembly serializes explicit values at the merge boundary. Scene
geometry and SLO replace those inherited from the base; a model without a scene
SLO override uses the scene's global SLO, not a machine-local model override.
Optional attention memory utilization inherits the base when omitted.
Collection consumes the typed portable values directly.

[`xkit.config`](../../src/xpool-dev/xkit/config.py) assembles explicit base, scene and
caller-owned test projections before registry defaults. An explicit
`runtime_config` selects the base ahead of `XPOOL_CONFIG`; registered CLI and
environment precedence follows
[Control Plane](control-plane.md#configuration-and-integration). Machine paths,
calibration and loader/placement policy belong to the complete runtime base.
Matching Model IDs retain their configured checkpoint paths; new IDs use the
vendor-root fallback. Assembly reads raw base TOML and merges explicit layers
before defaults and registered overrides; an effective preflight configuration
cannot replace that input without changing source precedence. Serving tests,
native qualification and owned benchmarks consume the deployment's complete
topology and SLO. Elastic KV tests additionally
project a byte budget using the assigned attention devices' total memory.

For a selected pytest item declaring configuration, resource setup resolves
`XPOOL_CONFIG` with registered process-environment inputs once and retains one
`ResolvedConfig`. Its checkpoint checks use that value's effective configuration;
`e2e_base_config` returns the retained value and requires the test's configuration
declaration. Each later item resolves its own base, and explicit `require_config()`
calls always reload current file and environment inputs. File and environment
changes do not implicitly refresh the retained base. The retained path supplies
scene assembly; the effective configuration supplies base policy and checkpoint
paths. The base is not installed as global runtime configuration; serving startup
owns installation of the assembled runtime configuration.
Device and checkpoint availability are checked during selected-item setup.
MPS preparation and actual membership belong to the runtime owner, not preflight.

Both tools use `xkit.config.resolve_model_weights` to resolve a checkpoint through
the configuration owner and return its path after requiring its directory and
`config.json`; preflight checks availability, not checkpoint loading or integrity.
Pytest checks its resolved base and owns unavailable-resource and strict-requirement
outcome translation. Before executing a nonempty benchmark workload, the runner
enforces the source's declared local
configuration and checkpoint requirements. Owned execution checks its assembled
effective configuration, preserving explicit base precedence. Client execution
loads local configuration only when declared; external endpoints alone imply no
local configuration or checkpoint requirement. Missing or malformed required
resources produce benchmark infrastructure result two. Inventory remains
resource-free.

Scenes are grouped as `<encoded-model-ids>/<layout>.toml`. The directory joins
distinct full model IDs, sorted by canonical spelling and individually encoded
with `ModelId.uri_encode()`, using `+` as the separator. For example,
`Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B` labels one two-model scene; readable
filenames distinguish its layouts. Model selection remains catalogue-owned.

Layout names use `atnN[-dpD]-ffnM[-tpT]-lanesL`, where `N` and `M` are role
device counts, `D` is attention DP, `T` is FFN TP and `L` is the Lane count.
Omit `dpD` for DP1 and `tpT` when TP uses the complete FFN Fleet. Thus `ffn2`
describes TP2, while `ffn2-tp1` preserves a narrower execution group. Distinct
model TP widths use a `+`-joined suffix in sorted Model ID order. Explicit TOML
values own geometry; filenames are descriptive references, not another parser.

Model qualification chooses a minimum feasible scene. Performance comparisons
retain matching mixed and isolated conditions. A few representative small-model
scenes own complex topology coverage. Reuse depends on the complete geometry and
behavior under test, not device count alone. A TP1 group in a two-device FFN Fleet
does not guarantee one model per device or replicated execution; placement still
owns layer assignments.

## Shared process and resource ownership

Each independently scheduled execution uses the same supervised domain:

```text
tool runner (fallback child subreaper, resource owner)
└── task supervisor (isolated session, normal child subreaper)
    └── task root (separate process group)
        └── clients, daemon, Agents, SGLang and further descendants
```

The runner owns admission, terminal signals and tool-local device allocations.
The supervisor owns execution limits, cancellation and complete-domain recovery.
For ordinary unprotected tasks, it signals exact PID/create-time identities
with bounded TERM/KILL and reaps adopted children. Managed resource retirement
uses the actual owners instead of this generic descendant path.

[`xkit.task`](../../src/xpool-dev/xkit/task.py) carries the task root's aggregate
resource proof, independently of its test or benchmark verdict. Before controller
creation or a nested owned launch, the owner retains a cleanup scope and the root
announces `ACTIVE`. Supervisor acknowledgement commits protection before resource
startup. Cancellation reaches the root and existing owners cooperatively.
Protection remains active through invocation teardown; nested owners do not
maintain a parallel controller registry.

The root seals resource creation and sends `CLEANED` only after every actual
owner and nested tool reports verified retirement. That acknowledged transition
permits generic recovery of remaining host descendants. After root exit,
kernel `waitpid` reporting `ECHILD` proves complete-domain retirement. Resource
proof and this empty-domain proof both precede allocation return. Root exit,
EOF, a process snapshot or an endpoint bind alone supplies neither proof.

Missing proof or supervisor loss during a protected task seals scheduling and
retains the allocation and living owners for manual resolution. It does not
trigger device-blind fallback signals. Unprotected runner fallback retains its
existing bounded domain recovery. Multiprocessing infrastructure is protected by
exact identity, including a tracker started before supervision initialization;
an inherited tracker belongs to its ancestor.

`SupervisedTaskScope` accepts a positive finite total deadline or `None`.
`xtest` retains finite task deadlines. Normal benchmark queue drain uses `None`;
startup and exceptional cleanup retain their separate bounds. HTTP request
deadlines are opt-in.
Protected item or aggregate expiry requests cooperative retirement; direct
pytest and unprotected tasks retain their existing bounded expiry actions.
The first cancellation establishes one cleanup envelope without renewal.
If an actual owner's cleanup expires before retirement is confirmed, the task
root seals further resource creation and publishes the original expired deadline
through the existing cancellation channel. This notification supplies no cleanup
proof. A verified local System close still permits another System in the task.
A supervisor-local infrastructure failure can publish a terminal completion
only after verified local cleanup. Unconfirmed cleanup follows the protected
retention or unprotected fallback policy above and records infrastructure
failure. Only the outer resource owner seals cleanup evidence.

Typed Python children retain their Process, Pipe and log owner before `start()`.
Short signal deferral covers creation and caller publication, not readiness
waiting or device work. Startup acknowledgement failures leave the actual child
with its caller for role-aware rollback. Async serving callers retain and join
the actual startup/close Future when an asyncio waiter is cancelled.
Children use a fresh spawn interpreter and an acknowledgement after session/log
setup. Installed callbacks need no source path. Source-only callbacks
receive explicit caller-owned import roots; artifact locations and installed
module parents do not determine a subprocess's working directory.

[`xkit.scheduler.TaskScheduler`](../../src/xpool-dev/xkit/scheduler.py) shares
first-fitting backfill, active-scope polling and proven lease return between
the tools. Each tool owns task compilation, priority and verdict interpretation;
the scheduler imports neither test nor benchmark cases. It admits every fitting
task without a separate jobs limit. Different benchmark cases may overlap;
repetitions of one case remain ordered. Preparation completes before supervised
execution so input reads and replay writes cannot stall task-control polling.

Admission closure is monotonic across stage changes and follow-up repetitions.
Tests stop admission after safely retired infrastructure failures and retain
their ordinary-failure stage gates. Benchmarks continue other cases and later
logical repetitions after safely retired per-case preflight, startup, warmup,
program and measurement failures by default. They retain the failed outcomes;
later repetitions are distinct measurements, not retries of the failed one.
The invocation-only `--fast-fail` flag stops new benchmark admission after a
newly executed failure while active tasks finish naturally. Earlier retained
failures affect the final verdict but do not trigger this flag. Input collection,
workload preparation and artifact-write failures terminate before execution.
User cancellation closes admission and requests cooperative cancellation of all
active scopes. Unconfirmed retirement never returns
a device lease or permits a successor on that domain; it closes admission and
requests cooperative cancellation of other active tasks. Parallel cases share
host resources; a single-case invocation provides experiment isolation within the tool.

### Devices and endpoints

`xkit.device` derives eligibility from startup `CUDA_VISIBLE_DEVICES` through
the shared [runtime device utility](control-plane.md#managed-mps-and-role-preparation).
It reuses physical inventory querying and orders capacity observations by the
selected UUID view without creating contexts. Task visibility contains only its
allocated UUIDs. `DevicePool` schedules those devices in memory within one tool
invocation; operators coordinate independent invocations. Managed endpoint ownership is not
a physical-device exclusion guarantee. Role preparation and controller lifetime
belong to [Control Plane](control-plane.md#startup-and-shutdown).

Daemon-backed deployments use daemon-owned attention MPS and direct FFN execution.
Daemon-free native topology and calibration owners retain their own MPS scope
and participants, retiring clients before stopping their controller. Cases that
do not need MPS run directly. Allocation return follows verified resource
retirement and complete task-domain drain.

Owned benchmark placement uses the shared attention-first consecutive role
blocks within lease-local ordinals `0..N-1`. Physical UUID mapping and logical
role placement are retained separately. Nested benchmark E2E execution stays
within its outer test lease; inner cleanup does not replace the outer supervisor's proof.

`xkit.network` reserves listeners through `bind -> listen -> local connect ->
accept` qualification. `xkit.serving.sglang.endpoints` groups SGLang HTTP, NCCL,
gRPC, handshake and derived ZMQ endpoints into one owned family.
NCCL/TCPStore reservations match its wildcard bind scope; HTTP, gRPC and the DP
handshake retain their declared host. Fixed DP ZMQ reservations cover upstream
wildcard availability checks. Cooperating namespace locks are keyed by port,
and reacquisition uses each reservation's original address. Releasing listeners
for spawn still permits an external bind race; upstream random worker ports
remain engine-owned.
A bindable but unreachable endpoint rejects the family;
during startup, only a post-cleanup `EADDRINUSE` is a retryable conflict.
Serving startup passes one absolute monotonic deadline through daemon and every
endpoint-family allocation. Candidate traversal and each blocking connect/accept
probe consume that same budget; expiration raises `TimeoutError` and rolls back
partial listeners and namespace locks. Safe cleanup retains its separate bounds;
shutdown reacquisition does not inherit an expired startup deadline.

## Shared serving lifecycle

[`xkit.serving`](../../src/xpool-dev/xkit/serving/) owns the CrossPool daemon and
Agent cluster, bound configuration snapshots, serving endpoints and bounded
readiness evidence. Its concrete
[`sglang` owner](../../src/xpool-dev/xkit/serving/sglang/) groups graph policy,
endpoint families, launch declarations, server processes and system lifecycle.
Both tools use this engine implementation; client API adapters own the separate
request/response protocol boundary.

`xkit.serving.sglang.launch` accepts a `ServingLaunch`
containing effective configuration, environment, invocation working directory
and ordered `SglangLaunchModel` declarations before resource binding.
Models cover every effective runtime Instance exactly once;
each attention TP-by-DP geometry agrees with the configured attention devices.
Graph mode remains attention-side policy. Shared command construction uses
`xpool exec -- sglang serve` with attention-local base zero and step one; runtime
preparation normalizes the complete deployment visibility before engine import.

`XpoolServingSystem.start` reserves all endpoint families, starts the daemon and
Agents, starts all Instance servers, then establishes System Ready and every
public HTTP health check under one complete startup deadline. Ordered endpoints
are available only after readiness. Child exit, public HTTP health and that
deadline determine server startup; log lines supply diagnostic evidence.
`check_alive` reports unexpected owned process exits independently of HTTP
request completion. `close` seals and joins startup, retires each server domain,
then signals the retained daemon PID. The daemon coordinates Agent/Fabric and
MPS retirement; the System releases local endpoint resources after verified close.
Concurrent callers share one retained close operation and its first deadline.
The enclosing supervisor still owns complete-domain recovery and emptiness proof.

Failed startup joins ordered rollback before reporting its operation failure or
a confirmed endpoint conflict. A failed deployment verdict alone does not make
verified resource retirement an unconfirmed cleanup. Other endpoint inspection
errors remain failures. Once readiness publishes endpoints, deployment and
inference failures retain their execution verdict even when a released binding
is occupied.
The test probe closes Systems whose resource retirement remains pending.
Unconfirmed retirement retains ownership and prevents replay.

Owned SGLang startup selects `SGLANG_PLUGINS=xpool` and sets `HF_HUB_OFFLINE=1`
and `TRANSFORMERS_OFFLINE=1` in its child environment. The generic command wrapper
preserves that selection and forwards engine arguments unchanged. After reserving
the daemon endpoint, it calls shared `snapshot_cluster_launch` once to bind its
port, write the runtime TOML and freeze child inputs. The running
cluster owns the resulting `XpoolClusterLaunch`; `XpoolServingSystem.launch`
exposes that cluster-owned launch.
Launch snapshots use the configuration-owned serialization contract in
[Control Plane](control-plane.md#configuration-and-integration). The launch
owner binds its reserved daemon endpoint and retains the actual runtime TOML.
Benchmark case evidence owns the original effective values, source provenance
and invocation working directory once per case. The launch removes inherited
registered overrides that could reinterpret the frozen TOML. Env-only debug
inputs retain their effective values.
Snapshotting preserves other caller-supplied environment inputs. Native FFN
qualification uses the same configuration operation with its own environment.
Invocation working directory and resolved model/calibration paths preserve their
meaning. Each owned worker installs its frozen global configuration once.

`xtest` supplies its case projection and observers, then evaluates
qualification evidence. Test preparation supplies launch inputs; `ProbeRun`
retains expected Model IDs, its observer directory, results and qualification
evidence. The running cluster owns the runtime launch.
The test probe validates `/server_info` Graph settings against the shared
settings declaration at its HTTP reading boundary and retains the typed result
before sending its deterministic inference request.
`xbench` supplies the complete case-resolved deployment
and its own workload. Benchmark execution does not enable test observers or
reuse the test probe's fixed requests and verdict policy. The shared installed
SGLang launch retains its bounded policy: CP one, engine random seed zero,
trust-remote-code and error-level logs.

Test observation readers consume the producer-owned declarations described in
[Devkit](devkit.md#serialized-observations). Fabric, Transport and SGLang Graph
file readers validate declared fields at input with strict `TypeAdapter`
validation and retain unknown fields, including nested extensions. A required
nullable field still requires its key; explicit null can represent an incomplete
lifecycle fact. Private semantic assertions use the validated records and own
completeness, ordering and correspondence checks. Native Graph readers reuse
`FfnGraphObserverSnapshot` and its existing validation policy.

## Test inventory, execution and retained verdicts

`xtest list` uses isolated collection to list concrete parameterized node IDs
and declared requirements. Engine filtering precedes imports. Native inventory
comes from the configured CTest manifest. Listing runs no test bodies or fixtures
and acquires no device/MPS resources; native collection prerequisites still apply.

`xtest run` compiles the typed collection plan, performs required resource
preflight and executes CTest, Unit, Integration, E2E and explicitly selected
Models in canonical order. Device tasks are ordered by resource count and estimated
duration and backfilled over idle leases. Test strictness, JUnit classification
and cross-task qualification verdicts remain test-owned.

`run.json` retains required `tool_config`, source-labeled tool software metadata, selections, strictness,
expected cases and task/artifact mapping. Atomic `results.json`
checkpoints retain supervision outcomes, JUnit projections, stage/group verdicts,
timing and nullable final result and cleanup proof. Cases never admitted remain
unexecuted, not passed or skipped. Recording failure is an infrastructure failure.

JUnit phase records are grouped by the collected node ID into one result per
item, in collection order. Any phase failure or error makes the item failed;
the result retains all outcome messages and tracebacks and sums record durations.
A missing record duration leaves the item's duration unavailable. XML summary
counters are validated against raw records and outcomes before this projection;
retained report totals count unique test items.

The collected plan and retained task/artifact mapping own serving comparison
membership. Serving artifacts supply graph settings and model outputs for that
assignment. Numerical samples use Layer-by-Rows labels local to their case's
isolated reference and production workdirs. The runner mapping supplies test
attribution for both evidence types.

`TestRunReport` projects that original outcome without recollection or current
qualification reevaluation. `xtest report` writes JSON/Markdown with retained
run identity, original strictness, failures, skip reasons, durations, artifacts and completeness.
Inactive interrupted runs remain explicitly incomplete. `xtest report --list`
discovers inactive metadata with supported execution settings, without parsing
all samples; report generation validates the complete retained evidence.
Artifact addresses are exact `RUN_ID` values below the configured test-run root.
Reports default to `RUN_ID/report/`; `--output DIR` exports below
`DIR/xtest/RUN_ID/report/`. Missing manifests are unsupported input; logs are
not a replacement verdict protocol. Reporting a failed run successfully does
not change its original result.

## Result storage and validation ownership

`xkit.results.RunStore` owns locks, completion markers and explicit retention.
The three CLI roots reuse `xpool.utils.cli` declaration and dispatch mechanisms;
each tool owns its selections and execution policy. `RunCleanup.print` presents
the shared cleanup decision without changing its retention or removal logic.
Tool-owned verdict/checkpoint writers, readiness evidence, collection plans and
CTest resource files reuse `xkit.results.write_json` for atomic publication.
Their callers assemble domain values and create parent directories. Shared
`write_jsonl` accepts serialized JSON values, exclusively creates a new UTF-8
file and flushes/fsyncs the complete batch. Both writers reject non-finite JSON
numbers. Live measurement recording and partial-evidence recovery retain their
benchmark-owned lifecycles.
Runtime configuration owns the shared `cache_root`, defaulting to
invocation-relative `.xpool-cache`. Tools bootstrap only this setting and its
source record without requiring a complete deployment, then use disjoint
`test-runs/` and `bench-runs/` subtrees. Runtime model/config resolution remains
operation-owned. Memory profiling uses temporary `memory-profile/` workspaces;
its final calibration destination remains explicit. Python bytecode follows
interpreter and user policy rather than tooling overrides. See
[Configuration](../configuration.md#development-tools-and-shared-cache) for
cache sources and path origins. A completion marker
means lifecycle completion, not test or measurement success. Execution performs
no implicit retention cleanup. Default cached measurements and their reports
are Git-ignored.

Both cleanup commands keep twenty inactive runs by default and support explicit
count, all and dry-run selection. Benchmark cleanup removes each selected run
with its attempt-owned reports. Retention uses execution age, not report
publication time. Active entries remain locked. Root creation
and cleanup serialize before entry locking, resolve the root once and permit
symlinked root parents; symlink entries inside the root are unlinked rather than
followed. The result root is dedicated storage, so unrecognized inactive entries
are cleanup candidates.

Shared lifecycle/resource checks are owned once under `xkit`; corresponding tool
self-tests prove CLI wiring, retained outcomes and finalization under
`tests/suites/<layer>/{xtest,xbench}/`. Symmetry matches responsibilities, not
case counts. Real CPU CLI executions use small source tests or local native streaming
servers through the editable `xpool-dev` development installation only where
the installed entry or actual signal/process boundary is under test. Other
command wiring invokes the tool's `cli.main`; collection and worker execution
keep their real boundaries. Client, report and RunStore checks own their
respective behavior. [Test Architecture](../../tests/README.md#placement) owns
the test-local support and rendering strategy. The owned benchmark regression
uses the ordinary E2E
seam with its outer device visibility and finite deadline; it does not replace
product numerical, graph or topology qualification.
