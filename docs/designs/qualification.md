# Qualification

This document defines the evidence required to claim CrossPool readiness. Detailed
suite placement and developer commands live in
[tests/README.md](../../tests/README.md).

## Numerical and graph evidence

Dtype-specific component coverage exercises BF16 and FP16 weight conversion,
Dense and MoE operator execution, captured Graph replay, and native TP1/TP>1
delivery. It compares both admitted dtypes with an FP32 reference using the
existing numerical policy. Routine end-to-end serving uses the effective dtype
resolved for its two small Qwen deployments; component dtype coverage remains
owned by the component suites.

Numerical qualification compares each supported model, using one admitted TP
realization, with an independent SGLang FFN reference. Shared native topology
cases exercise other TP cardinalities with the same topology-test logic and an
unsharded reference. These evidence surfaces are orthogonal: model suites own
model semantics while shared topology suites own delivery cardinalities.

Serving graph qualification has two verdicts. Decode compares Eager and Decode
Full token IDs with eager prefill in both runs. Prefill compares Eager and the
combined Decode Full plus Prefill Breakable mode's first-prefill logits with
full-distribution forward KL while the Graph Observer proves that Breakable
execution actually occurred. Prefill token identity is diagnostic, not a
correctness requirement.

Routine serving E2E proves installed HTTP completion, graph-mode startup and
observed Graph structure, and Transport and Fabric behavior with Qwen3-0.6B
and Qwen2.5-0.5B at attention TP1 and TP2 under the combined graph mode.
Explicit per-model suites own independent FFN numerical reference and serving
graph comparisons, including qualified MoE models. The ordinary suite owns
installed serving for the two small Qwen deployments;
model suites own their explicit qualification whenever an adapter or numerical
contract changes. The [test architecture](../../tests/README.md) owns suite
selection commands and case placement.

## Topology evidence

Native topology qualification covers representative AtnAgent/FfnAgent
cardinality and delivery shapes, repeated lane leases, and explicit
failure-free participant shutdown. One real Dense BF16 structural matrix runs
1,000 consecutive leases for each Direct Partial, Single Complete, and
Replicated Complete delivery under both Decode and Prefill request shapes. It
uses small row capacities and checks real synthetic-reference output, lease and
invocation sequences, delivery selection, failure absence, and clean shutdown.
The separate two-lane 1,000-repetition case remains because it proves that an
idle Lane stays stable while another Lane is repeatedly leased. Scheduler
mechanisms are owned by native and component tests; installed SGLang serving
owns observed cross-model lane overlap. SGLang serving qualification instead
requires its process group to leave no live descendants; it requires
AtnAgent-side Transport snapshots, while an Instance-side snapshot remains
optional evidence produced only by explicit detach.
Its harness readiness message carries
`layer_execution_groups: tuple[tuple[int, ...], ...]` from the admitted
production Fabric Plan. Placement assertions consume that field, while the
Graph Observer remains limited to observed graph structure, binding-site
counts, and lane branches. Every FfnAgent must expose its Lane Graph, while only
FfnAgents assigned at least one layer by the production Plan must expose a
Primary Graph. The controlled E2E Instance entry installs the role-matching
Devkit observers immediately after native bootstrap, matching the production
serving integration lifecycle.

Transport cooperative-launch capacity is a bound on concurrently resident
endpoint blocks. It is not an Executor Lane bound: FfnAgent Lanes own
independent Lane GraphExecs, while Transport retains one block per endpoint and
the Coordinator launches one block.

## Resource evidence

Resource qualification checks real-model startup against the analytic or
calibrated memory estimate. Baseline startup worlds require zero
underprediction for every observed startup stage. Memory-pressure startup
worlds set the operator device-memory margin to zero, hold one raw CUDA reserve
across the complete ordered startup, and require startup to succeed when actual
post-reserve free memory is no greater
than the predicted peak and at most one 16-MiB positioning quantum below it.
The 16-MiB quantum is qualification positioning resolution, not permitted
underprediction or a production margin. Any underprediction reopens the
estimator rather than increasing a hidden or default margin.

Resource qualification runners are evidence-only tools. Production modules,
reusable test harnesses, and permanent test cases own the runtime contracts
they exercise. A resource qualification is rerun only after memory-estimator,
allocation, admission, or qualified-environment changes. Numerical
qualification is rerun only after FFN mathematics, operators, model adapters,
or admitted dtypes change. Performance measurements are report-only diagnostics
while the system is incomplete; they do not define readiness or regression
gates. Required evidence must remain observable through the accepted public and
Devkit boundaries; a missing observation is a design-review input. Production
interfaces, alternate execution paths, and telemetry follow the accepted
qualification design rather than adapting a runner around an absent boundary.

Elastic KV qualification follows the same evidence ownership. Native tests
prove the process-shared KV Control Channel. Integration tests prove physical
VMM behavior and CUDA Event ordering together with allocator, prefix-cache,
command, and daemon-policy behavior. Ordinary subprocess serving E2E proves
mandatory Elastic KV startup under the combined graph mode. The dedicated
Elastic KV E2E observes prefix-cache hit loss after peer pressure, renewed
hits after repopulation, and concurrent request completion by both models on
the same GPU under Decode Full plus Prefill Breakable. Native and integration
tests own capacity commands, terminal completions, and physical map/unmap.
Operational logs are diagnostics rather than a correctness assertion surface.
Qualification cases provide evidence rather than an adapter allowlist. Runtime
compatibility follows the SGLang memory and cache contract, with structural
incompatibility at that seam determining rejection.

Memory underprediction or memory-pressure startup failure invalidates that
qualification result. Preserve the affected case, measured deviation, and raw
evidence, then continue authorized diagnosis and implementation fixes. Obtain a
design decision before changing the estimator contract, production interfaces,
or acceptance criteria. Runner adaptation that requires new telemetry likewise
requires a design decision.

## Acceptance and invalidation

FFN qualification uses installed serving execution. Derived-plan checks remain
supplemental; observer evidence is authoritative when graph or protocol
behavior is the subject of qualification.

During implementation, run focused checks for each affected boundary. Commit
hooks are a separate check of the submitted changes; a resource-eligible suite
pass does not establish strict acceptance. Final acceptance of runtime or
build-semantic changes requires, on one frozen source and native-module identity:

1. canonical build and generated-stub completion;
2. the complete strict-requirements test suite with no unexpected skip;
3. a final documentation and code-quality review; and
4. every qualification whose invalidation trigger was changed by the goal.

Documentation-only changes do not reopen behavioral acceptance when they change
no executable statement, declaration, configuration value, or build behavior.
They rerun the documentation and static quality checks affected by the change.
Any runtime or build-semantic edit invalidates the prior complete-suite verdict
and the focused or qualification evidence whose scope it changes. Reuse valid
evidence for the final source and build; do not repeat checks solely because
another workflow step requests their result. Complete the strict suite on that
final version rather than after every intermediate edit.

Isolated agent-tool changes use focused tool-behavior and static checks; they
do not require GPU qualification or serving-suite execution when runtime,
build behavior, and test acceptance contracts are unchanged.

Native CTest owns invalid Projection behavior; Unit does not duplicate the
native constructor contract.

Static quality gates are Ruff format and lint, ty, whole-tree clang-format,
Doxygen, and the explicit Python/native public-documentation tests. They reuse
the existing toolchain: there is no second include sorter, docstring linter,
comment-density rule, or source-text vocabulary gate. Doxygen permits useful
partial contracts without requiring parameter boilerplate and excludes
`operator==`; all other extracted public Native declarations remain required.
