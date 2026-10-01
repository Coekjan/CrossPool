# Test Tooling

The CrossPool development environment provides `xtest` for correctness validation.
Resource and process mechanisms live in `xkit`; `xtest` owns its execution policy
and retained-result semantics. [Qualification](qualification.md) owns acceptance
evidence and the [glossary](../../CONTEXT.md) owns domain terminology.

## Installed packages and source inputs

```text
pyproject.toml              Production project and root uv workspace
uv.lock                     Shared dependency lockfile
src/
  xpool/                    Production runtime and configuration
  xpool-dev/
    pyproject.toml          Private xpool-dev project
    xkit/                   Shared resources, processes, storage and serving
    xtest/cli.py            Test command
    xtest/harness/          Collection, scheduling, verdicts and qualification
tests/
  tests.toml                Test scenarios and expectations
  suites/                   Executable validation suites
configs/
  deployments/              Shared portable runtime scenes
```

`xtest` depends on `xkit`, which may use `xpool` configuration and
runtime values. Production `xpool` does not import these tooling packages.
Shared mechanisms do not import tool verdicts, pytest or collected test modules.
Test observers, numerical references, fakes and qualification drivers remain in
`xtest.harness`.

The root [`pyproject.toml`](../../pyproject.toml) owns production `xpool`, its
native build, console command and SGLang plugin. The private
[`xpool-dev` project](../../src/xpool-dev/pyproject.toml) owns `xkit` and `xtest`,
depends on `xpool`, and disables CMake in the pinned scikit-build-core backend.
Its console entry points directly to `xtest.cli:main`.
`xpool-dev` is a project directory and distribution name,
not a Python import prefix.

The root uv workspace selects both projects through workspace sources. Root
`uv sync --group dev` installs both editably into one managed virtual environment
using one root lockfile; invoke the tools with `uv run`. The released `xpool`
wheel owns production packages and entry points. Both projects declare the
TOML writer they use.

Test suites and the catalogue remain source inputs rather than wheel contents.
`xtest list/run` require a checkout and applicable development tools. Its report
and cleanup commands operate on retained evidence outside a checkout.
Catalogue-relative input paths belong to the declaring file.

The command exposes `list`, `run`, `report` and `clean`. Exact CLI options and
typed interfaces belong to their declarations in
[`xtest.cli`](../../src/xpool-dev/xtest/cli.py) and the owning harness modules.
[Test Architecture](../../tests/README.md) owns placement, resource requirements
and execution commands.

## Catalogue and source declarations

The test catalogue uses named case tables. The table key is the case ID; each case
has an English description and a dotted source `module`. Serving experiments
use `serving_cases`, while native FFN topology tests use `topology_cases`.
[`xkit.source`](../../src/xpool-dev/xkit/source.py) resolves each module below the
catalogue's sibling `suites/` directory without importing it. Models qualification
keeps its workload and graph expectations as typed source constants and uses
the same portable deployments.

`xtest` exports the shared `requirements` and `parameterize`
decorators from [`xkit.declaration`](../../src/xpool-dev/xkit/declaration.py).
They retain metadata on the original function without wrapping execution.
Omitted parameter values bind catalogue cases; explicit Python values support
source-owned parameterization. Test row callbacks produce native pytest
parameters, leaving fixtures, marks and execution with pytest. Resources use
[`ResourceRequirements`](../../src/xpool-dev/xkit/requirements.py), either static
or returned by a callback receiving the concrete declared parameter values as
keyword arguments. Registration performs no catalogue I/O, resource probing or
configuration installation; callbacks receive neither fixtures nor workdirs.

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
the same absolute catalogue path. Resource callbacks become the existing pytest
resource marks before marker deselection; ordinary pytest tests remain valid.

## Shared deployment configuration

`tests/tests.toml` references independent TOML files by a suffix-free basename
such as `atn1-ffn2-lanes2`.
[`xkit.deployment`](../../src/xpool-dev/xkit/deployment.py) resolves that reference
once in the catalogue loader, using the declared model set and the
conventional root `catalogue.resolve().parent.parent / "configs" / "deployments"`.
The loaded case retains the resulting absolute path. Collection loads neither
machine configuration nor model metadata; reports consume retained evidence
without reopening the source deployment. Portable scenes use the
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
vendor-root fallback. Preflight and execution use `model_path_of()` on the same
assembled configuration. Serving tests and native qualification consume the deployment's complete topology
and SLO. Elastic KV tests additionally
project a byte budget using the assigned attention GPUs' total memory.

`xtest` uses `xkit.config.resolve_model_weights` to resolve a checkpoint through
the configuration owner and require its directory and `config.json`. Pytest owns
unavailable-resource and strict-requirement outcome translation. Inventory remains
resource-free.

Scenes are grouped as `<encoded-model-ids>/<layout>.toml`. The directory joins
distinct full model IDs, sorted by canonical spelling and individually encoded
with `ModelId.uri_encode()`, using `+` as the separator. For example,
`Qwen%2FQwen2.5-0.5B+Qwen%2FQwen3-0.6B` labels one two-model scene; readable
filenames distinguish its layouts. Model selection remains catalogue-owned.

## Shared process and resource ownership

Each independently scheduled execution uses the same supervised domain:

```text
tool runner (fallback child subreaper, resource owner)
└── task supervisor (isolated session, normal child subreaper)
    └── task root (separate process group)
        └── clients, daemon, Agents, SGLang and further descendants
```

The runner owns admission, terminal signals and GPU leases. The supervisor owns
normal timeout, cancellation and descendant cleanup. It repeatedly discovers
exact PID/create-time identities, signals TERM and then KILL under bounded
deadlines, and reaps adopted children. After the root exits, kernel `waitpid`
reporting `ECHILD` is the empty-domain proof. A group leader's exit, recursive
process snapshot or endpoint bind check alone cannot authorize GPU release.

Runner fallback handles supervisor or owner loss. Multiprocessing infrastructure
is protected by its exact identity, including a resource tracker started before
supervision initialization; unrelated existing children remain eligible for
cleanup. An inherited tracker belongs to its ancestor. Failure to prove the
owned domain empty stops admission and retains the affected lease.

`SupervisedTaskScope` accepts a positive finite total deadline or `None`.
`xtest` retains finite task deadlines; startup, HTTP requests and exceptional
cleanup retain their separate bounds.
A supervisor-local infrastructure failure can publish a terminal completion
only after local cleanup. Otherwise the runner performs fallback and records
an infrastructure failure. Only the outer resource owner seals cleanup evidence.

Typed Python children use a fresh spawn interpreter and an acknowledgement after
session/log setup. Installed callbacks need no source path. Source-only callbacks
receive explicit caller-owned import roots; artifact locations and installed
module parents do not determine a subprocess's working directory.

### GPUs and endpoints

`xkit.gpu` derives eligibility from startup `CUDA_VISIBLE_DEVICES`, accepts
physical ordinals or full GPU UUIDs, and normalizes leases to physical UUIDs.
Every eligible selected device must execute through the externally managed MPS
controller. Task visibility contains only its leased UUIDs, and release follows
complete domain drain. The externally supplied allocation must be exclusive;
the tools do not coordinate GPUs across invocations or manage MPS compute mode
or controller lifetime.

`xkit.network` reserves listeners through `bind -> listen -> local connect ->
accept` qualification. `xkit.serving.sglang.endpoints` groups SGLang HTTP, NCCL,
gRPC, handshake and derived ZMQ endpoints into one owned family.
A bindable but unreachable endpoint rejects the family;
only a post-cleanup `EADDRINUSE` is a retryable conflict.
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
The test harness uses this engine implementation.

`xkit.serving.sglang.launch` accepts a `ServingLaunch`
containing effective configuration, environment, invocation working directory
and ordered `SglangLaunchModel` declarations before resource binding.
Models cover every effective runtime Instance exactly once;
each attention TP-by-DP geometry agrees with the configured attention devices.
Graph mode remains attention-side policy. Shared command construction derives
the CUDA base and step from the admitted arithmetic attention placement.

`XpoolServingSystem.start` reserves all endpoint families, starts the daemon and
Agents, starts all Instance servers, then establishes System Ready and every
public HTTP health check under one complete startup deadline. Ordered endpoints
are available only after readiness. Child exit, public HTTP health and that
deadline determine server startup; log lines supply diagnostic evidence.
`check_alive` reports unexpected owned process exits independently of HTTP
request completion. `close` stops server
groups before Agents and daemon and releases local endpoint resources; the
enclosing supervisor still owns escaped-descendant recovery and emptiness proof.

SGLang startup sets `SGLANG_PLUGINS=xpool`, `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` in its child environment. After reserving the daemon
endpoint, it calls shared `snapshot_cluster_launch`
once to bind its port, write the runtime TOML and freeze child inputs. The running
cluster owns the resulting `XpoolClusterLaunch`; `XpoolServingSystem.launch`
exposes that cluster-owned launch.
Launch snapshots use the configuration-owned serialization contract in
[Control Plane](control-plane.md#configuration-and-integration). The launch
owner binds its reserved daemon endpoint and retains the actual runtime TOML.
The launch removes inherited
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
The shared installed SGLang launch retains its bounded policy: CP one,
engine random seed zero,
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
and acquires no GPU/MPS resources; native collection prerequisites still apply.

`xtest run` compiles the typed collection plan, performs required resource
preflight and executes CTest, Unit, Integration, E2E and explicitly selected
Models in canonical order. GPU tasks are ordered by resource count and estimated
duration and backfilled over idle leases. Test strictness, JUnit classification
and cross-task qualification verdicts remain test-owned.

`run.json` retains source-labeled tool software metadata, selections, strictness,
expected cases and task/artifact mapping. Atomic `results.json`
checkpoints retain supervision outcomes, JUnit projections, stage/group verdicts,
timing and nullable final result and cleanup proof. Cases never admitted remain
unexecuted, not passed or skipped. Recording failure is an infrastructure failure.

The collected plan and retained task/artifact mapping own serving comparison
membership. Serving artifacts supply graph settings and model outputs for that
assignment. Numerical samples use Layer-by-Rows labels local to their case's
isolated reference and production workdirs. The runner mapping supplies test
attribution for both evidence types.

`TestRunReport` projects that original outcome without recollection or current
qualification reevaluation. `xtest report` writes labeled JSON/Markdown with
original strictness, failures, skip reasons, durations, artifacts and completeness.
Inactive interrupted runs remain explicitly incomplete. Missing manifests are
unsupported input; logs are not a replacement verdict protocol. Reporting a
failed run successfully does not change its original result.

## Result storage and validation ownership

`xkit.results.RunStore` owns locks, completion markers and explicit retention.
`xtest` reuses `xkit.cli` for argument parsing and cleanup presentation,
and owns its selections and execution policy.
Tool-owned verdict/checkpoint writers, readiness evidence, collection plans and
CTest resource files reuse `xkit.results.write_json` for atomic publication.
Their callers assemble domain values and create parent directories. Shared
`write_jsonl` accepts serialized JSON values, exclusively creates a new UTF-8
file and flushes/fsyncs the complete batch. Both writers reject non-finite JSON
numbers. The default root is invocation-relative `.xpool-cache/test-runs`;
`xtest` accepts an explicit root. A completion marker
means lifecycle completion, not test or measurement success. Execution performs
no implicit retention cleanup. Default cached results and their reports are Git-ignored.

The cleanup command keeps twenty inactive runs by default and supports explicit
count, all and dry-run selection. Active entries remain locked. Root creation
and cleanup serialize before entry locking, resolve the root once and permit
symlinked root parents; symlink entries inside the root are unlinked rather than
followed. The result root is dedicated storage, so unrecognized inactive entries
are cleanup candidates.

Shared lifecycle/resource checks are owned once under `xkit`; tool self-tests
prove CLI wiring, retained outcomes and finalization under
`tests/suites/<layer>/xtest/`. Real CPU CLI cycles use small source tests through
the editable `xpool-dev` development installation, including report and cleanup
from another working directory.
