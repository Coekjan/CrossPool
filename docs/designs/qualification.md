# Qualification

This document defines the evidence required to claim CrossPool readiness. Detailed
suite placement and developer commands live in
[tests/README.md](../../tests/README.md).

## Numerical and graph evidence

Dtype-specific component coverage exercises BF16 and FP16 weight conversion,
Dense and MoE operator execution, captured Graph replay, and native TP1/TP>1
delivery. Operator comparisons use FP32 references; synthetic native references
use FP64 projections with explicit payload stage rounding. Activation, Router
scoring, route weighting and ordered TP reduction remain FP32; weighted routes
and TP partials retain their payload stores. Both use the existing numerical
policies. Standalone activation coverage exercises gated and
ungated SiLU and ReLU, plus gated GELU with exact and tanh formulas. ReLU
coverage compares every FP16 and BF16 bit pattern
with Torch for both in-place and separate outputs. Routine end-to-end serving
uses the effective dtype resolved for its deployments; component dtype coverage
remains owned by the component suites.

Numerical qualification compares each supported model, using one admitted TP
realization, with an independent SGLang FFN reference. Shared native topology
cases exercise other TP cardinalities with the same topology-test logic and an
unsharded reference. These evidence surfaces are orthogonal: model suites own
model semantics while shared topology suites own delivery cardinalities.
For MoE, qualify Router projection, selection and coefficients independently,
then compare Expert outputs with original SGLang consuming production's actual
IDs and weights. Projection uses FP64 references with the declared stores.
Selection uses a fixed FP64 reference policy for cutoff ambiguity and declared
ID ties. It covers uncorrected Softmax with a normal cutoff and stored-FP32
Sigmoid. Corrected Sigmoid requires finite correction bias. Unsupported
numerical domains fail qualification. Coefficient allowances
pass through the declared final weight store. Clearly wrong selections fail,
while justified cutoff ambiguity remains visible.
Dense retains its direct independent output comparison.

Production retires before a standalone Router child reproduces the observed
Capacity, layout and precision and checks exact installed-routing agreement.
That child exits before the SGLang reference starts, isolating CrossPool's
process-global GEMM policy. Router failures stop the reference stage. Raw Router
tensors retain case ID, Layer ID and observed Capacity before numerical comparison.
Each reference TP rank uses the prescribed routes
through a local TopK hook. Shared Expert and scaling conversion belongs to this
reference boundary; the hook is removed after each case. Stages share readonly
CPU tensors and retained files; CUDA resources remain owned by their children.

The result-writing rank retains natural SGLang routing from the same TopK
invocation and reuses the prescribed IDs and weights verified during carrier
replacement. Both sets of weights reflect SGLang's actual storage rounding.
Diagnostics count tokens with different Expert sets independently of slot order
and compare weights for common Experts. These differences do not change either
numerical gate and do not establish a bound on whole-MoE output error under
independently chosen routes. Capturing them adds no FFN forward or model load.

The numerical harness starts every attention TP rank and compares each complete
output with the independent reference at the existing output limits. Attention
DP is one for these cases.
Rank zero writes `ffn-numerical.json`; additional ranks write
`ffn-numerical-rank-<rank>.json`. Router results are shared by the invocation.
FFN qualification stops new invocations after observing a retained failure.
Its owning harness remains responsible for participant retirement.

Model numerical cases explicitly declare a BF16 or FP16 payload dtype; BF16
is the default. Declarations and retained evidence use `torch.dtype` in Python
and reuse the [shared canonical JSON codec](control-plane.md#generation-planning).
Qualification validates the BF16/FP16 restriction separately from that codec's
general dtype round-trip contract.

Each checkpoint retains its own dtype during qualification: BF16 checkpoints
use BF16 and FP16 checkpoints explicitly declare FP16. Family admission of both
dtypes does not require additional forced-FP16 qualification of BF16 checkpoints.
The declaration flows unchanged through deterministic hidden-state inputs,
the independent SGLang reference, the production Instance profile, saved inputs
and numerical evidence. Reference and production outputs must have matching
admitted dtypes. Both dtypes use the same numerical metrics and tolerances.
Checkpoint metadata does not implicitly select the qualification payload dtype.

Serving Graph numerical qualification records an Eager continuation,
then replays its token choices after each candidate computes its own original
distribution. Every request starts cold with reported `cached_tokens=0`, neutral
greedy sampling and eight returned predictions: terminal Prefill followed by
seven real Decode steps. Complete prescribed-token equality establishes the
conditioning input; full-vocabulary forward KL determines the numerical verdict.

| Stage | Reference | Maximum KL per prediction |
| --- | --- | --- |
| Terminal Prefill | Eager Prefill | 0.05 nats |
| Decode | Eager Decode with the candidate's Prefill backend | 0.001 nats |

Matching the Prefill backend separates Decode changes from differences already
present in KV construction. A three-mode case therefore adds Breakable Prefill
with Eager Decode, yielding four serial deployments under one device lease.
Each deployment retires before the next starts. Two-mode cases use the Eager
recording as their Decode reference. Singleton cases retain structural
qualification.
Shared [sampling records](tooling.md#shared-serving-lifecycle) retain the exact
continuation while every deployment builds its own hidden states and KV.
The active qualification plugin matches FA3 MLA Decode arithmetic for the
recording and candidate captures. The numerical verdict applies under these
comparison conditions.

Raw HTTP responses and CPU FP32 logprob matrices remain unchanged. Comparison
validates model identity, complete equal-width vocabularies, finite values and
identical token prefixes, then normalizes and reduces in CPU FP64. Limits are
inclusive, and any failing position fails the case. `numerical.json` reports
the configured Decode policy and server-reported Attention backends per model
and Graph setting. It separates stage verdicts from Eager-to-candidate
end-to-end diagnostics, retaining their discrepancies and limits. Graph checks
select events by Model ID before applying the host-monotonic request window
returned by the serving probe. Each enabled Graph phase requires that model's
capture events from its complete event stream and actual Prefill execution or
Decode replay within its request window.
Full Decode requires one process to replay at least as many steps as the
returned prediction count minus one. Sampler evidence separately covers
every participating TP rank's token prefixes.

Eager still uses CrossPool's Fabric and native FFN Graphs. Stage agreement
qualifies execution consistency on the recorded continuation; independent FFN
references retain their own contracts. Ordinary serving E2E retains free
generation, completion, observed Graph execution and verified cleanup.

Routine serving E2E proves installed HTTP completion, graph-mode startup and
observed Graph structure, and Transport and Fabric behavior at attention TP1
and TP2 under the combined graph mode.
Explicit per-model suites own independent FFN numerical reference and serving
graph comparisons, including qualified MoE models. Model suites own their
explicit qualification whenever an adapter or numerical
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

Compare predicted and observed incremental bytes at weight materialization,
Fabric join, Graph Capture, installation and retained service. Observe the
same [Registry completion boundary](ffn-execution.md#graphtemplate-and-executor-lanes)
as production. Report the baseline, analytic cost, correction and operator
margin separately. Sample outside active capture without extra synchronization
or cache trimming. Record sampling gaps: a sampled maximum can miss a peak.

- **Without a Profile**, require real-model startup and inference with
  sufficient memory. Report observed minus predicted bytes at each stage.
  Near the estimate, an OOM is acceptable only if readiness stays false and
  resources retire safely. Record the actual failure stage and distinguish
  confirmed native finalization and Registry close from complete process
  exit. A successful constrained run must exercise the requested inference mode.
- **With a Profile**, both independent held-out worlds and real-model runs
  must show no sampled underprediction. Set the operator margin to zero and
  hold a raw CUDA reserve on each FFN device through inference and normal
  retirement. At admission require `P <= F <= P + 16 MiB`, where `P` is
  the predicted peak and `F` is fresh free memory. The interval positions
  the pressure test; it does not excuse underprediction.

Qualify Lane workspace ownership through actual capture and installation.
Verify complete pool coverage, Primary/Control agreement, bounded memory-node
access and disjoint scratch across Lanes. Exercise dynamic weight, scale and
bias changes after construction cleanup, failure retention and release after
confirmed native retirement. Component coverage retains tails, idle FFN
participants and zero-Signature installations at their owning boundaries.

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

Elastic Context State qualification follows the same evidence ownership. Native tests
prove the process-shared Elastic Context State Control Channel. Integration tests prove physical
VMM behavior and CUDA Event ordering together with allocator, prefix-cache,
command, and daemon-policy behavior. Ordinary subprocess serving E2E proves
mandatory Elastic Context State startup under the combined graph mode. The dedicated
Elastic Context State E2E observes a confirmed prefix-cache hit decrease after peer
pressure, a positive stable hit after repopulation within the current active
capacity, and concurrent request completion by both models on the same device
under Decode Full plus Prefill Breakable. Six terminal-Prefill comparisons on
the prefix-serving model compare each prompt's initial full-vocabulary logprobs
with cache-hit, post-pressure and restored results. Its eight numerical probes
generate one token each; ordinary, pressure and sustained requests generate
eight. Comparisons reuse the serving generation capture and forward-KL comparator
with the Prefill limit. ECS owns its cache workload and free continuation;
it adds no Decode verdict.
Greedy output IDs remain diagnostic across different Prefill geometries.
Absolute `cached_tokens` counts are diagnostic evidence, not fixed qualification
thresholds. Native and integration tests own capacity commands, terminal
completions, and physical map/unmap. Operational logs are diagnostics rather
than a correctness assertion surface. Qualification cases provide evidence
rather than an adapter allowlist. Runtime compatibility follows the SGLang
memory and cache contract, with structural incompatibility at that seam
determining rejection.

Computation errors, missing ownership and unconfirmed reclamation invalidate
either route. Preserve the failing case, discrepancy and raw evidence.
Changing production interfaces or acceptance criteria requires a design decision.

Managed lifecycle evidence observes actual attention MPS membership and direct
FFN execution, normal serving/daemon retirement, and cancellation at retained
startup boundaries. Controlled termination evidence uses self-owned clients:
one client's confirmed context termination and host exit must leave an
independent peer and server usable. Cache-helper lifecycle evidence does not
substitute for mode-specific model-loading or inference qualification.
Qualify Profile reuse under the effective compute settings in real startup.
[Tooling](tooling.md#devices-and-endpoints) owns launch and cleanup rules.

Resource cleanup, complete-domain exit and the original execution verdict are
separate facts under [Control Plane](control-plane.md#startup-and-shutdown) and
[Tooling](tooling.md#shared-process-and-resource-ownership).

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
do not require device qualification or serving-suite execution when runtime,
build behavior, and test acceptance contracts are unchanged.

Native CTest owns invalid Projection behavior; Unit does not duplicate the
native constructor contract.

Static quality gates are Ruff format and lint, ty, whole-tree clang-format,
Doxygen, and the explicit Python/native public-documentation tests. They reuse
the existing toolchain: there is no second include sorter, docstring linter,
comment-density rule, or source-text vocabulary gate. Doxygen permits useful
partial contracts without requiring parameter boilerplate and excludes
`operator==`; all other extracted public Native declarations remain required.
