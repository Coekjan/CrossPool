# Memory Profiler Refactor

## Goal

Make offline FFN memory calibration reuse production resource construction,
give its experimental processes explicit resource ownership, and preserve the
measurements needed to inspect a generated Profile. Preserve the calibration
method rather than treating structural cleanup as evidence of better accuracy.
Separate source-model compilation from FFN execution selection so synthetic
experiments describe the required execution behavior directly.
Resolve evidenced allocator-reservation and Graph-stage coverage gaps through a
reviewed model-independent budget, retaining zero-underprediction qualification.

## Baseline

Before the refactor, the implementation lived in
[`memory_profile`](../../../src/xpool/runtime/ffnagent/memory_profile/):
`corpus.py` constructed synthetic experiments, `runner.py` started fresh participant
worlds and sampled startup, and `fitting.py` fitted the overhead envelope.
[`xpool.memory`](../../../src/xpool/memory.py) owned the runtime Profile and its
compatibility checks and publication.

The profiler called the production `FfnExecutionRegistry.materialize`.
It separately constructed TP Layer Plans and canonical weight tensors, while the
Registry reconstructed signatures already derived by
`execution.required_execution_signatures` for estimation and placement.

`FfnModelSpec.architecture_name` selected the runtime model adapter. The
corpus bound each synthetic member to a real source architecture to obtain
Router computation, weight precision and workspace sizing. The loader,
estimator, Signature producer and Registry repeated that lookup. Dense execution
already depended on geometry, dtype and activation rather than source
architecture. The MoE paths had three distinct Router implementations:
Softmax with FP32 logits, Softmax with payload-precision logits, and
correction-biased Sigmoid with FP32 operands and logits.

The child configuration initializer reapplied process environment to a
temporary experiment TOML. A configured calibration output could consequently
become an input to the child's estimator. The sampler started before its cleanup
scope covered all fallible preparation. Before any join broadcast, a participant
failure left healthy peers waiting without sending their supported stop
command. Parent cancellation handlers did not protect ordinary spawn children
from terminal signals delivered to the foreground process group.

The final Profile retained compatibility facts, fitted coefficients and the
minimum held-out headroom, but lacked the original per-stage observations
described in the Control Plane document at that time.

## Accepted Changes

### Preserve the calibration method

- Keep the four fixed synthetic samples and their geometry, fitting
  configurations, one held-out experiment, repetition policy, five observation
  points, ten coefficients, feature meanings and fit objective. Keep Dense TP2
  conditional on at least two FfnAgents.
- Keep the runtime `XpoolMemoryCalibrationProfile` JSON format and compatibility
  contract. Its held-out headroom remains diagnostic, not an extra reserve.
- Keep analytic admission available without a configured Profile. An explicitly
  configured invalid runtime Profile remains an error.
- Keep synthetic fitting distinct from real-model Resource Qualification.

### Correct allocator reservation and qualify Graph coverage

Include physical allocator reservation and remaining Graph-stage overhead in
this task's scope. Use the conservative local construction-group bound specified
under [Resource budget correction](#resource-budget-correction), accepting its
additional admission cost rather than predicting cross-group cache reuse.
Keep exact logical Tensor accounting, zero operator margin, and the accepted
pressure interval. Derive allocation budgets from the pinned implementation and
existing constructors; select empirical coverage from execution behavior rather
than Model IDs or source architectures. Attribute Graph costs with the temporary
qualification script and refit the unchanged synthetic method before deciding
whether further coverage changes are necessary.

### Separate source compilation from FFN execution

- Keep source-architecture discovery, configuration validation and checkpoint
  interpretation at the model adapter boundary. Compile the source into FFN
  geometry and explicit execution semantics.
- Make production loading, estimation, Signature construction and execution
  consume those semantics rather than select operations by source architecture
  or synthetic member names.
- Let synthetic samples declare the same execution semantics directly. They
  must not require a real model architecture, a synthetic adapter or a
  calibration-specific execution bypass.
- Give Router computation, precision policy and resource geometry explicit
  owners. Model compilation selects supported semantics; allocation and
  estimation consume the same declared precision and shared sizing formulas.
- Preserve the three existing Router precision paths, input conversion,
  model-declared normalization, workspace formulas and fixed experiment
  coverage. Give Softmax and Sigmoid one optional selection-only correction
  contract; derive selected weights from original activated scores. Preserve
  Signature sharing for equal execution semantics and resource geometry.
- New models reusing existing FFN execution do not require another corpus
  architecture mapping. New execution behavior requires its own support and
  qualification rather than implicit coverage by a model-neutral label.

### Share resource construction

- Give production placement and fixed experimental placement one pure TP Layer
  Plan projection. The solver still selects groups; the corpus still selects its
  fixed experiment groups.
- Give the real checkpoint loader and calibration the same canonical empty weight
  allocation. The loader retains checkpoint validation, read descriptors and
  filling. Calibration retains its synthetic initialization and reads no weights.
- Make the Registry use `required_execution_signatures`, preserving Plan and
  weight validation and its complete capture/install ownership.
- Construct each world's `FfnModelSpec` values once and pass them to
  participants. Model ID names stop serving as an implicit reconstruction
  protocol. Keep sample geometry and explicit execution semantics together.

### Give calibration experiments a typed interface

Construct experiments directly from concrete FFN specifications and named
placement inputs. An immutable `CalibrationModel` owns one sample's specification
and experimental placement; an immutable `CalibrationExperiment` owns the ordered
model combination. Composition and configuration determine the experiment's
behavior. The fitting and held-out collections determine its measurement role.
Configuration snapshots, Fabric Plans and world execution consume the same
experiment value rather than independently interpreting member strings or
positional tuples.

Keep sample construction independent of real source architectures. Reuse
`FfnModelSpec`, `RouterAlgorithm`, `RouterPrecision`, `InstanceRankTopology`
and the existing Plan types instead of defining parallel execution semantics.
Construct the four fixed sample specifications directly; sample names and
Model IDs carry identity rather than select computation. The existing FFN
structure, algorithm and precision enums express computational choices.
Place intrinsic construction and projections on their owning concrete types;
keep device allocation, configuration I/O, process coordination and fitting in
their existing owners.

Preserve the calibration method and resource budget defined above. Keep
complete-fleet configuration distinct from an experiment instance's attention
membership. A smaller experimental membership must not become an invalid
partial-fleet `ModelConfig` topology.

Each evidence world retains its concrete experiment definition once, together
with repetition and ordered participant observations. Fitting and held-out
collections preserve measurement role. The runtime Profile remains a separate
unchanged format.

### Repair experiment ownership

- Make all world participants consume the same explicit experiment configuration,
  with calibration input disabled. The parent retains the selected output path.
- Cover the sampler's complete lifetime with cleanup immediately after acquisition;
  perform pure estimator/ledger preparation before starting it where possible.
- Before any join broadcast, send the existing cooperative stop command and
  confirm participant exit before releasing the UID owner and MPS.
- Mark join as started before the first join send. Once broadcasting starts,
  missing acknowledgements do not authorize pre-join cleanup or blind finalization.
- Preserve the original execution failure independently of cleanup confirmation.
  A failed child with confirmed exit is not itself unconfirmed retirement.
- Make terminal cancellation reach the parent-owned protocol rather than directly
  interrupting device participants, including during spawn startup.
- Retain the existing safety boundary when collective retirement is unknown.
  Reuse the selected deadlines; add no automatic retry or forced process cleanup.
- Retain each phase's minimum observed free bytes rather than all millisecond
  samples. Preserve boundary observations and synchronize competing updates.

### Coordinate failed-world retirement

Separate experiment failure from permission to exit. A failed participant
retains its original exception and live resource references while the parent
coordinates retirement; reporting a failure does not release those owners.

- Before native join begins, ordinary failure permits local nonzero exit and
  parent-owned cooperative stop of the other prepared participants.
- Record entry into native join before the call, join completion only after it
  returns, and finalization only after native retirement returns successfully.
  A started but unconfirmed join does not permit local-only exit or an
  unconditional finalize.
- After confirmed join, an ordinary execution or evidence failure stops the
  measurement, closes the sampler and reports whether the child can participate
  in coordinated retirement. Keep that child available for the parent's
  existing finalize command rather than unwinding its resource owner.
- The parent records failures while collecting the current execution outcomes.
  It may coordinate normal drain/finalize only after every participant's join
  is confirmed and the complete world has reached a retirement boundary. A
  joined acknowledgement establishes that boundary for a participant not sent
  execute. An executing participant must report success or explicitly report
  failure with retirement eligibility. Ordinary OOM or Host evidence failure
  alone proves neither a damaged world nor successful cleanup.
- Send the existing finalize command once to each participant of that eligible
  world. Confirm native retirement and actual process exit before retiring the
  UID owner and MPS. A child with a retained experiment failure exits nonzero
  after successful retirement; the parent keeps that failure separate from the
  cleanup verdict and does not publish a new Profile.
- Uncertain membership, participant loss, a contradictory retirement report or
  failed native retirement retains the relevant living owners. Do not retry a
  native retirement operation, blindly finalize a partial world, or forcibly
  terminate device participants.
- After confirmed finalization, an ordinary Host failure permits nonzero exit;
  historical join does not itself require retaining an already retired owner.
- Reuse `xpool.utils.cuda.is_terminal_device_error` on the original exception
  and its chain for results requiring immediate nonzero process exit. Once
  identified, do not wait for the sampler or perform another device or collective
  cleanup call. This local exit does not authorize peers, the UID owner or MPS
  to retire.

On an exceptional measurement exit, do not call the successful phase's
synchronize-and-sample completion. Preserve the original error independently of
sampler and retirement errors, including their terminal-device classification.
Stopping a sampler prevents further iterations; it cannot cancel a native query
already in progress. A retained failure also keeps its traceback and remaining
local resource references available until confirmed retirement.

Inspect a retained sampler failure independently of whether its thread is still
alive. Check it before beginning another phase, synchronizing or sampling a
phase boundary, or admitting native retirement. An already stopped sampling
thread can still hold the terminal result that determines the owner's next
action. Preserve ordinary sampler shutdown and the original execution error;
do not introduce another background failure manager.

Owner-retention logs must render the exception traceback before entering their
existing wait. Parent-side spawn, pipe or wait exceptions need not be present in
the retained participant-failure value; logging that value alone can therefore
hide the original operation failure. Use the existing logger's `exc_info` with
the cleanup exception to include its operation-error context. Child retirement
and unconfirmed-join logs likewise include their corresponding exception
tracebacks. Preserve the original failure and cleanup verdict; add no parallel
error container or new exit path.

Keep this coordination in the existing profiler runner. Reuse its control
commands, retirement budget and native drain/finalize operations. The first
cancellation or failure starts that retirement budget; subsequent reports do
not renew it. On budget expiry, retain owners whose retirement is unconfirmed.
Do not force exit.

### Preserve CUDA failure facts through shared result checks

Use the public `cuda.bindings.runtime` and `cuda.bindings.driver` interfaces for
their respective calls and status enums. Consolidate the project-owned result
checks in `xpool.utils.cuda`, replacing the profiler's `checked_cuda` and the
KV backing's use of SGLang's `check_drv`. Bindings' example helpers and
cuda-core's private checkers are not production dependencies. Retain normal
module imports; add no subprocess, dependency upgrade or alternate CUDA object
model for this conversion.

Check only the returned status. Each caller unpacks its operation-specific
outputs directly. A raised exception retains the original upstream status enum
and operation name rather than reducing the failure to text. Runtime and Driver
status domains remain distinct. Error construction and classification use
retained metadata without another CUDA query.

Move the existing terminal-result predicate from `utils.mps` to `utils.cuda` and
extend it to the structured Runtime and Driver errors, preserving its Torch
exception support and standard exception-chain traversal. Update all consumers
together. The predicate supplies evidence; existing lifecycle owners still
decide notification, process exit and collective retirement. Participant failure
callbacks remain with those owners, not in a generic utility. Move
`cuda_versions` from `xpool.memory` to this shared CUDA owner; keep the
profiler-only memory measurement in the profiler.

Apply the existing terminal-result rule at the KV backing's diagnostic and
compensation boundaries. A known terminal failure skips subsequent memory
queries, rollback CUDA calls and constructor cleanup before propagation to the
existing worker owner. If a diagnostic query itself reports a terminal result,
propagate that exception with the original allocation failure as its cause;
an ordinary diagnostic failure remains an explanatory note. Ordinary recoverable
failures retain their necessary rollback, resource-progress tracking and cleanup
errors. This changes no distributed retirement command or owner relationship.

Agent and SGLang worker terminal-exit logs include the later exception's
traceback when a different first failure has already been retained. Preserve
that first failure and the existing notification and exit ordering. If the
terminal error is itself the first failure, reuse its already emitted traceback
rather than printing it twice. Use the existing logger and failure flags.

### Use explicit local KV allocation properties

Construct the KV backing's allocation properties with
`make_device_allocation_prop(device.index, handle_types=None)`, selecting
`CU_MEM_HANDLE_TYPE_NONE`. Query the recommended allocation granularity with
`cuMemGetAllocationGranularity` using those properties and check the Driver
status through `check_driver`. Reuse the same properties for `cuMemCreate`.
This replaces the automatic exportability probe through
`get_device_granularity`; retain useful upstream property and view construction.

Construction requires the calling Instance Rank to have selected its current
device. Remove the redundant `torch.cuda.device` context around construction:
its exit queries CUDA even when the selected device is unchanged, so unwinding
a terminal failure would issue another device call before reaching the worker.

Each Instance Rank continues to own its KV storage locally. Capacity coordination
does not export or import that storage. Preserve the stable virtual range,
compound-bundle accounting, mapping/access operations and retirement Events.
Derive geometry from the returned granularity rather than assuming the previous
value. Add no exportability setting or candidate fallback.

### Stop qualification submissions after a retained failure

At each invocation boundary in the FFN Instance qualification harness, inspect
the existing `failure` under `send_lock` and raise it before starting another
input load or device operation. Release the lock before executing an invocation.
The check stops new work after an observed failure; it does not cancel work
already in progress or prevent a failure from arriving just after the check.

An ordinary failure reaches the existing outer exception handler, which retains
the child and resources for parent-owned MPS context termination followed by Host
reaping. Preserve that path, the existing terminal-result immediate-exit rule,
and the current protocol. Do not convert failed execution into normal `CLOSE`
handling.

### Retain minimal calibration evidence

- Preserve the concrete experiment definition, repetition, FfnAgent index and
  phase for each observation before fitting groups equivalent feature rows.
  Record the definition once per world rather than in every phase observation.
  Keep fitting and held-out worlds in separate collections.
- Reuse `DeviceMemoryPoint`, `MemoryObservation` and participant evidence rather
  than creating parallel representations of the same measurement.
- Retain exact ledger bytes, allocator allowance, feature values, observed bytes,
  analytic prediction and calibrated prediction with the corresponding Profile.
- Define analytic prediction as ledger plus allocator allowance, not a prediction
  made by an old Profile. Neither comparison includes operator margin.
- Retain environment and device facts once through the corresponding Profile.
  Do not retain weights, full configurations or high-frequency sample sequences.
- Keep evidence publication separate from the runtime Profile schema. Retain one
  latest evidence file beside the selected Profile as `<profile-stem>.evidence.json`.
  For example, `memory.json` has a companion `memory.evidence.json`.
- Publish evidence only after all worlds, fitting and held-out validation
  succeed. Preparation, measurement, fit or held-out failure leaves the existing
  output files untouched. Do not create partial or failed calibration archives.

### Publish evidence before the runtime Profile

The CLI owns output publication. Write complete evidence atomically using a
temporary sibling and replacement, then invoke the existing atomic runtime
Profile writer. Print the actual output paths after success.

Evidence-write failure prevents Profile publication. Profile-write failure
reports failure, preserves the previous Profile and leaves the newly written
evidence available with its corresponding candidate Profile inside it. Report
both paths when publication is partial. The two files are individually atomic;
do not introduce a cross-file transaction or rollback mechanism.

Express publication with two adjacent exception boundaries: measurement and
evidence publication first, runtime Profile publication second. Each maps its
failure directly to `AgentError`; the second includes the published evidence
path and selected Profile path. Do not wrap a Profile-write error in an
intermediate `RuntimeError` only to translate it again. Keep independently
fallible sampler shutdown, coordinated native retirement and outer owner
cleanup under their existing exception boundaries.

Share the repeated temporary-sibling, atomic-replacement and failure-cleanup
operation through `xpool.utils.files.atomic_write_text`. Each writer retains its
configuration/path policy, serializer and business error context. Keep xkit's
fsync-backed result publication separate because its durability contract
differs; introduce no durability flag or dependency on development tooling.

### Resolve child configuration explicitly

Add optional `env: Mapping[str, str] | None` to runtime `init_global_config`.
`None` preserves the current use of `os.environ`; an explicit empty mapping means
no configuration environment overrides. Normal command precedence and global
installation checks remain unchanged.

Use `source.to_config_mapping()` for the experiment TOML, override the existing
experiment-owned fields and omit calibration input. Parent and child parse the
world with the same explicit configuration environment. Carry only env-only
configuration inputs that the snapshot cannot express, identified through the
existing registry rather than another environment-name list. Do not reapply ENV
settings over values already resolved into the snapshot.

This controls configuration resolution, not the actual process environment.
Device visibility, MPS and third-party runtime variables retain their existing
owners and continue to be passed to participants.

### Give the parent cancellation ownership

Temporarily block SIGINT/SIGTERM around process spawn and handle publication.
Children inherit that protection, ignore those signals at target entry and then
restore the pre-spawn signal mask. Parent handlers keep recording cancellation;
the existing stop/finalize commands coordinate retirement. Direct SIGTERM to an
internal child is not a separate exit interface. Cover interpreter/module import
before target entry, not only an already running target.

Use the standard-library resource-tracker signal pattern as a reference. Keep
`defer_signal_exceptions` semantics unchanged for its other callers. Any small
signal-mask mechanism belongs in the existing signal utility, not another process
framework.

Observe recorded cancellation before acquiring the next UID, MPS or participant
owner. If cancellation arrives during a protected spawn, first retain the
created child handle, then stop admitting participants and retire the owners
already acquired. Reuse the existing pre-join stop path rather than finishing
the participant launch loop after cancellation.

In the existing `receive` polling loop, recorded cancellation interrupts waits
for `uid` or `prepared` by raising `KeyboardInterrupt` into that cleanup path.
This covers cancellation during spawn and readiness waits without consuming the
remaining retirement budget waiting for preparation first. Cancellation does
not interrupt waits for `joined`, `execution` or `stopped`: membership, retirement
eligibility and process exit still require their existing confirmations.
Keep the resource-acquisition, phase-admission and final-verdict checks at their
own boundaries; add no cancellation helper, state type or timeout.

The existing unready UID-child cleanup must remain effective despite ignored
SIGTERM. Replace its exact-process `terminate()` with `kill()` only in that
Host-only branch: no valid UID has been received and no device participants have
been launched. Normal UID retirement continues to use `stop`; device participants
and process groups never acquire a corresponding force-kill path.

### Collect real-model resource evidence

The production FfnAgent records free memory for registration and admission, but
does not retain per-stage memory observations. Its existing lifecycle methods
and the native execution-installation entry already expose the required call
boundaries: checkpoint materialization, Fabric join, Graph capture, and execution
installation. The profiler already observes the capture/install handoff while
keeping `FfnExecutionRegistry.materialize` atomic.

- Collect real-model evidence with a throwaway qualification script in `/tmp`,
  outside version control. Wrap existing lifecycle and native call boundaries in
  self-owned qualification processes and forward to their original
  implementations. Run installed production startup, including activation and
  retirement, rather than manually reconstructing allocation or execution.
- Reuse the estimator's ledger, observation values, and passive sampling
  mechanics. Keep the qualification runner evidence-only; introduce no runtime
  configuration, control protocol, Hook Point, or permanent Observer solely for
  this run.
- Compare observations from the four startup phases with their corresponding
  analytic and calibrated predictions, recording the estimator mode being
  qualified separately from diagnostic comparisons. Require zero underprediction
  for that mode. Stop startup phase collection before activation launches
  persistent Lane Graphs. Successful pressure qualification still requires the
  complete production startup and confirmed retirement.
- Distinguish passive observation from calibration-specific synchronization and
  cache trimming. `PhaseRecorder.finish` synchronizes, and
  `record_registry_phases` also empties the allocator cache. Reusing those complete
  operations unchanged would alter production startup. Preserve calibration's
  existing synchronization and five observation points; do not add those actions
  to serving merely to obtain measurements.
- Keep cleaned `retained` evidence specific to the calibration measurement that
  actually performs cache trimming. Record production post-installation usage
  with its unflushed allocator state explicit, rather than comparing it with the
  cleaned retained estimate or forcing serving to release its cache. Such a
  diagnostic does not replace the four stage-specific accuracy checks.

### Position tight-pressure startup

Let `P` be the predicted startup peak and `F` the actual free bytes observed by
production admission. Successful tight-pressure startup requires
`P <= F <= P + 16 MiB`. Set the operator device-memory margin to zero and hold one
raw CUDA reserve across the complete ordered startup. Keep placement and
FfnAgent admission unchanged. Verify positioning from observed free bytes rather
than the requested reserve size.

The lower bound preserves admission; the upper bound establishes that the
experiment actually exercised startup near that boundary. The 16-MiB quantum
is experimental positioning resolution, not a CUDA requirement, an addition to
the estimate, or permission to underpredict. Baseline qualification still
requires zero underprediction for each measured phase and the estimator mode
being qualified. A below-peak rejection establishes an admission boundary but
does not substitute for a successful tight-pressure startup.

## Interface And Data-Structure Changes

### CUDA result checks and utility ownership

The shared utility accepts upstream status types directly:

```python
class CudaRuntimeError(RuntimeError):
    error_code: runtime.cudaError_t
    operation: str


class CudaDriverError(RuntimeError):
    error_code: driver.CUresult
    operation: str


def check_runtime(error_code: runtime.cudaError_t, operation: str) -> None: ...

def check_driver(error_code: driver.CUresult, operation: str) -> None: ...

def is_terminal_device_error(error: BaseException) -> bool: ...

def cuda_versions() -> tuple[int, int]: ...
```

The concrete exceptions retain their supplied enum and operation. Each check
returns on its domain's success status and otherwise raises its corresponding
exception. There is no generic tuple-unwrapping return type or compatibility
alias. Version queries check each Runtime result before issuing the next query;
the driver-version query remains a CUDA Runtime API call.

Migrate the profiler, runtime Profile compatibility checks, KV backing, Agent,
SGLang worker and native qualification harness to these owners. Publish the
supported utility interface through the normal explicit exports. Keep
`utils.device` responsible for inventory and visibility, and `utils.mps` for
MPS lifecycle and management commands. Synchronize Control Plane's terminal
error description and the affected failure-coverage documentation after the
implementation.

### Atomic text publication

Add one production file operation:

```python
def atomic_write_text(path: Path, text: str) -> None: ...
```

The helper writes UTF-8 text to a temporary sibling, closes it before
`os.replace`, and removes an unpublished temporary file on failure. The caller
supplies the existing parent directory and complete text, including its newline.
Filesystem errors propagate to the business writer's existing error mapping.
The operation preserves the current single-file atomicity without promising
fsync-backed crash durability or a two-file transaction.

Profile publication supplies `profile.model_dump_json(indent=2) + "\n"`.
Evidence publication supplies
`json.dumps(evidence.to_json_mapping(), indent=2, allow_nan=False) + "\n"`.
Keep their public signatures and publication order unchanged. Export the helper
from its owning module; retain no duplicate implementation or compatibility
facade.

### Private participant failure exchange

Keep the private pipe envelope `(tag, payload)` and existing control commands.
Participant failure payloads carry `message: str` and
`ready_to_finalize: bool`; the child retains the original exception locally.
Readiness means the joined child has ended its execution transaction and is
waiting to participate in parent-coordinated retirement. It is not evidence of
completed cleanup or authority to finalize any other participant.

The parent distinguishes execution outcome from retirement eligibility and
accepts a reported failed outcome without abandoning collection of the other
participants. Failure during join, a missing participant or required terminal
exit does not establish that readiness. After retirement begins, confirmation
and process-exit checks determine cleanup; never reinterpret a cleanup failure
as permission to send finalize again. Production registration and service-error
schemas remain unchanged by this private exchange.

### FFN compilation and execution boundary

Remove `architecture_name` from `FfnModelSpec`. Keep the source architecture on
`FfnSourceConfig` and compiler classes for discovery and source-format validation.
`architecture.load` continues to validate the returned Model ID and checkpoint
coverage; source-specific rules use its already selected source architecture.
Synthetic specifications require neither a real source architecture nor nullable
provenance. Remove the runtime `adapter_for` lookup after migrating its consumers.

Replace `RouterKind` with the closed `RouterAlgorithm` and `RouterPrecision`
enums in `xpool.ffn`. `MoeFfnSpec` has required `router_algorithm` and
`router_precision` fields; Dense specifications have neither field. Keep
correction-bias checkpoint identity and `renormalize` separate from both enums.
Preserve the existing three projection-precision paths while separating their
declarations; apply the selection-only correction contract below.

```python
class RouterAlgorithm(IntEnum):
    SOFTMAX = 1
    SIGMOID = 2


class RouterPrecision(IntEnum):
    PAYLOAD = 1
    FP32 = 2
    FP32_LOGITS = 3
```

`PAYLOAD` uses payload dtype for projection inputs, retained Router weights and
logits. `FP32` uses FP32 for all three, converting the Router input into its
caller-owned workspace. `FP32_LOGITS` keeps payload-precision inputs and weights
while producing FP32 logits. These policies describe Tensor dtypes rather than
changing TF32 or other backend arithmetic settings. Resolve concrete values with
`torch.dtype`; reuse `TorchDtype` wherever a concrete dtype crosses a JSON
boundary instead of introducing another scalar-type enum.

Qwen3 MoE declares `(SOFTMAX, PAYLOAD)`, DeepSeek-V2 declares
`(SOFTMAX, FP32_LOGITS)`, and GLM4 MoE Lite declares `(SIGMOID, FP32)`.
Signature construction and Graph capture consume these separate semantics while
preserving current sharing equivalence. Keep existing model declarations of bias
and normalization. Separating the enums does not itself admit new
algorithm/precision combinations. Remove the unused routing-callable protocol
and the model adapter's separate execution-method abstraction with their
consumers.

Resolve retained weight dtype through
`RouterPrecision.weight_dtype(*, payload_dtype: torch.dtype) -> torch.dtype`.
Keep all pure execution/resource geometry in
`xpool.runtime.ffnagent.execution`: Signature construction, Dense and Router
workspace sizes, MoE alignment-buffer shapes, workspace layout and Graph
storage derivations. These functions calculate dtypes, shapes, offsets and
byte extents; they neither allocate Tensors nor launch kernels.

Put Router projection, Softmax/Sigmoid TopK and final Routing Metadata
construction in `xpool.runtime.ffnagent.routing`. All TopK paths use the same
local `router_topk_kernel`, including Softmax without correction. The Host-side
entry `compute_routed_topk` owns validation, projection and direct kernel launch.
Move `finalize_moe_routing` and `finalize_moe_routing_kernel` to the same owner,
including the finalization launch constant. Preserve the compiled FP32 input
conversion, pre-capture warmup, Sigmoid selection-score rounding boundary,
existing tie behavior and zero-denominator guard. The selected weights follow
the original-score contract below. Module relocation may rebuild compilation
caches; it introduces no cache-reset or retry policy.

Keep Dense/Expert computation and its kernel configuration selection in
`operators.py`. Move `dense_workspace_bytes`,
`moe_alignment_workspace_shapes`, `OPERATOR_ALIGNMENT_BYTES` and
`QUALIFIED_MOE_BLOCK_SIZE_M_VALUES` from `operators.py` to `execution.py`;
move `router_workspace_bytes` there from `routing.py`. Operators and routing
reuse these formulas for their workspace checks. The block-size domain has
one declaration shared by sizing and kernel selection.

`execution.py` imports neither `routing` nor `operators`; the shared pure
geometry is below both computation modules:

```text
registry -> routing   -> execution
         -> operators -> execution
         -------------> execution

estimation / placement -> execution
```

Registry construction, including `MoeGraphCaptureWorkspace.allocate`, consumes
the same layout and size formulas. Preserve the fixed allocation covering all
admitted block sizes and the selected configuration's actual views, including
existing overlap and alignment. Preserve Router TopK, final Routing Metadata
construction and Expert computation order within capture.

Keep reusable low-level Dense/Expert imports in `operators.py` and local
Triton TopK execution in `routing.py`. Router TopK does not call SGLang's
low-level TopK operators; the existing Expert operations still require their
dependencies. Keep serving-runtime objects and retained state out of that
dependency seam. Update the FFN Execution design's import boundary with the
implementation. Add no workspace module.

Migrate all producers and consumers together: family compilers, synthetic corpus,
loader, estimator, `required_execution_signatures`, Registry, native
qualification and shared service fixtures. Keep Model ID, checkpoint coverage,
Plan/Spec identity and canonical registration agreement checked at their current
boundaries. Do not add metadata wrappers, compatibility aliases or a second
digest to support the migration.

The runtime memory Profile is a separate type. Its unchanged JSON contract does
not imply that `FfnModelSpec` or registration payloads are unchanged. Do not
require mixed-version participants to share a generation. Canonical Spec
serialization and the existing digest algorithm use the new field set; all
participants must use the same schema when forming a new generation. Production
TOML and source model configuration formats remain unchanged.

### Selection-only Router correction

For both algorithms, calculate activated FP32 scores from the projection logits,
select Experts using scores plus an optional per-Expert correction, and derive
weights from the original scores. Apply the model's existing normalization and
scaling afterward. `None` means no selection correction; keep bias optionality
independent of the algorithm. Existing compilers retain their checkpoint keys
and normalization declarations, including the admitted normalized Sigmoid path.

```python
scores = activation(logits)
selection_scores = scores if correction_bias is None else scores + correction_bias
selected_ids = topk(selection_scores)
selected_weights = gather(scores, selected_ids)
```

This is selection correction, not addition to projection logits. Do not pass a
non-null selection correction into the pinned `sgl_kernel.topk_softmax` bias
parameter; its documented-by-tests ordering is incompatible. See
[upstream evidence](router-correction-bias.md).

`compute_routed_topk` takes the separate Algorithm and Precision, constructs
logits using the selected projection-precision path, and directly launches
`router_topk_kernel` with `router_weights.correction_bias`. Keep validation,
grid and launch-argument construction at this Host entry; the device kernel
owns activation, selection and weight computation. Preserve caller-owned
contiguous `int32[C, K]` IDs and `float32[C, K]` weights. The implementation is:

- Softmax and Sigmoid, with or without correction, share `router_topk_kernel`,
  replacing `biased_sigmoid_topk_kernel`. Specialize algorithm, bias presence and
  normalization at compile time rather than duplicating the TopK selection and
  output logic. Softmax uses the stable FP32 max/exp/sum formula; Sigmoid uses
  FP32 sigmoid. Keep original scores and selection scores separate in kernel
  registers. `None` takes a static no-bias branch, not a zero Tensor allocation.
  Add no external TopK fast path, runtime backend selector or fallback.
- Preserve Sigmoid's FP32 selection-score write/read boundary in its existing
  logits scratch and its higher-Expert-ID tie rule. Read output weights from
  the original activated scores, not by subtracting bias from rounded selection
  scores. Softmax uses a lower-Expert-ID tie rule. Mask padded Experts
  out of selection and retain zero weights when normalization sums to zero.

Softmax changes its kernel implementation even without correction. The Sigmoid
weight calculation can differ from its current subtract-bias recovery in FP32
arithmetic. Both require independent model and Graph qualification rather than
an assumption of bitwise equivalence.

Retain the existing optional FP32 bias Tensor, checkpoint-key field, loader
copy, native binding projection and `correction_bias_present` Signature fact.
Replace the Signature's combined kind with separate algorithm and precision;
bias presence remains independent and determines the captured specialization.
Keep its existing resource accounting and Control Probe ownership. The new
selection contract needs no additional Tensor, workspace region, metadata
wrapper, configuration field or native protocol. Registers and opaque
compilation/Graph costs still require qualification rather than an assumption
that actual device memory is unchanged.

Update intrinsic validation to permit either algorithm's optional bias; retain
Expert cardinality, shared-Expert consistency, admitted precision and existing
normalization constraints. Capture fixes algorithm, precision, geometry and bias
presence; replay can consume changed Tensor contents at the established binding
addresses, not toggle optional ownership within an installed Signature.

The owning `MoeFfnSpec` validates admitted Algorithm/Precision combinations and
the Sigmoid normalization constraint. Execution consumes that established
relationship instead of repeating the same admission table. Keep concrete
Tensor shape, dtype, device and workspace validation at the execution boundary.

### Router kernel efficiency

Reduce the local kernel's avoidable cross-warp communication and repeated
reductions while preserving its selection-only correction contract. The initial
revision keeps one row per program and changes the direct launch to
`num_warps=1` for the admitted Router domain. Small 64- and 128-Expert rows do not
require the two-warp shared-memory reductions produced by the current launch.
Validate the existing Expert widths and dtypes rather than adding Model ID
dispatch, an automatic tuner, another configuration field or a backend selector.

Keep the explicit maximum and matching-ID reductions: Softmax's matching-ID
minimum preserves lower-ID ties, while Sigmoid's matching-ID maximum preserves
higher-ID ties. These execute within one warp under the new launch. An indexed
maximum is not required merely to reduce the source-level reduction count; its
value/ID comparison and compiler lowering must justify any replacement through
measurement. Both specializations remain in `router_topk_kernel` and share its
output and normalization logic; add no forwarding wrapper or second kernel.

When Softmax has no correction bias, the selected maximum already equals the
original activated weight; use it directly instead of reducing
`where(experts == expert, scores, 0)`. Keep gathering original scores when bias
is present and for Sigmoid. In particular, Sigmoid's stored-selection rounding
boundary does not authorize replacing its original activated weight with a
rounded selection value, even without bias.

Keep Sigmoid's selection-score store/read and its barrier, the normalization
store/read barrier, padding exclusion, zero-denominator protection, projection
precision and explicit workspace extents. A barrier is not removed merely
because the first launch revision uses one warp. Numerical reduction order and
opaque compilation/Graph costs may change and require affected qualification.

Measure the launch change and the reduction change separately against the same
inputs and former no-bias Softmax operator. Use CUDA Graph replay as well as the
ordinary-call diagnostic, keeping cache and timing methodology explicit. Stop
at these local changes before deciding whether the remaining measured gap
justifies a multi-row program layout. Retain a proposed optimization only when
the measurements establish a benefit without weakening the numerical contract;
a smaller reduction count alone is not that evidence. Investigating a multi-row
layout is conditional follow-up, not authorization to add another implementation
in this revision.

### Intrinsic construction and projection ownership

Put intrinsic operations on existing types rather than adding another helper
owner or base class:

- `RouterPrecision.weight_dtype` resolves the retained Router weight dtype from
  the policy and admitted BF16/FP16 payload dtype. It does not inspect the Router
  Algorithm or source architecture.
- `GatedFfnCheckpointKeys.tensor_keys()` and
  `MoeFfnCheckpointKeys.tensor_keys()` return their complete ordered key tuples.
  Callers use `layer.checkpoint.tensor_keys()` directly. Preserve gate/up/down
  order, Router and optional bias keys, routed Expert order and optional Shared
  Expert order. Source-specific checkpoint naming stays with its compiler.
- `MoeGraphCaptureWorkspace.allocate(*, signature, w13_config)` constructs the
  existing contiguous allocation and typed views. Preserve extents, alignment,
  overlapping regions, optional Router views and allocation order.
- `TensorReadDescriptor.from_key_view(...)` owns authoritative key lookup and
  descriptor construction with the current inputs and missing-key error.
  `descriptor.slot_view(slot)` interprets the staging prefix using its existing
  byte extent, dtype and shape. Neither operation performs checkpoint I/O.
- `DenseFfnWeights` and `MoeFfnWeights` expose `binding_projection()` with their
  precise native projection return types and `zeros_like()` returning their own
  type. Nested Router resources preserve optional bias and ownership. Keep
  address identity, shape, dtype, device and construction order unchanged.
  A zero-valued weight owner is independent of its Control Capture use.

Keep `prepare_layer_weight_request`, `validate_layer_weights_against_plan`,
Graph capture, checkpoint I/O, world execution and fitting at module scope:
they coordinate multiple owners or implement general algorithms. Remove the
stateless `PhaseRecorder.read_sample` forwarding method and use the existing
memory query directly, preserving sampling, synchronization and phase behavior.
Keep the shared pure Signature/workspace and Graph-storage derivations in
`execution.py` and allocation in the existing resource owner.

### Shared resource interfaces

The shared construction interfaces belong to existing owners:

- `xpool.fabric`: move the pure Layer Spec and ordered execution-group projection
  out of daemon placement; return the existing `FfnLayerPlan` variants. Report
  intrinsic invalid inputs with `ValueError`; the daemon maps service errors.
- `DenseFfnWeights` and `MoeFfnWeights`: add concrete allocation factories in
  `weights.py`, retaining precise return types. Inputs describe hidden size,
  local intermediate width, payload dtype and device. MoE allocation also uses
  the existing layer specification and selected Router dtype/ownership. Precision
  follows the declared FFN execution semantics rather than being discovered by
  the weight container or inferred from a source architecture.
- `FfnExecutionRegistry.materialize`: reuse the existing signature producer.
  Preserve capture, installation and resource rollback as one transaction.
- `MemoryProfileWorld`: preserve participant attribution rather than flattening
  observations and later inferring rank from positional groups.

Concrete allocation factories return their owning weight type:

```python
def place_ffn_layer(
    layer: ffn.FfnLayerSpec,
    execution_group: tuple[int, ...],
) -> FfnLayerPlan: ...

# Existing weight fields are omitted from these factory signatures.
class DenseFfnWeights:
    @classmethod
    def allocate(
        cls, *, hidden_size: int, local_intermediate_size: int,
        payload_dtype: torch.dtype, device: torch.device,
    ) -> DenseFfnWeights: ...

class MoeFfnWeights:
    @classmethod
    def allocate(
        cls, layer: ffn.MoeFfnSpec,
        *, hidden_size: int, local_intermediate_size: int,
        payload_dtype: torch.dtype, router_weight_dtype: torch.dtype | None,
        device: torch.device,
    ) -> MoeFfnWeights: ...
```

`router_weight_dtype=None` means this shard does not own Router tensors; do not
add another ownership flag or independently infer precision in the allocator.
Keep allocation empty and checkpoint-independent. Calibration performs its
existing Router/bias initialization after allocation. Preserve the production
allocation order, which calibration adopts.

### Calibration experiment values

Add `memory_profile/experiment.py` for the two concrete immutable value types.
Their intrinsic projections construct existing configuration and Plan types:

```python
@dataclass(frozen=True, slots=True)
class CalibrationModel:
    spec: ffn.FfnModelSpec
    ffn_tp_size: int
    topology: InstanceRankTopology
    group_sum_complete_admitted: bool

    def model_plan(self) -> FfnModelPlan: ...
    def instance_plan(self) -> FabricInstancePlan: ...
    def to_json_mapping(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class CalibrationExperiment:
    models: tuple[CalibrationModel, ...]

    @property
    def model_specs(self) -> tuple[ffn.FfnModelSpec, ...]: ...

    def config_snapshot(
        self, source: XpoolConfig, *, directory: Path,
    ) -> XpoolConfig: ...

    def fabric_plan(self, uid: str, *, config: XpoolConfig) -> FabricPlan: ...
    def to_json_mapping(self) -> dict[str, object]: ...
```

Every model's layers use the ordered primary FFN execution group
`tuple(range(ffn_tp_size))`. Model Plan projection reuses `place_ffn_layer`.
Instance Plan projection owns the existing BF16 payload, decode capacity 2048,
prefill capacity 4096, topology and output-admission declaration. Keep those
fixed settings with that projection; resource consumers use its effective
Profile. Existing Spec, topology and Plan validation retain their contracts.

The configuration snapshot preserves the existing experimental overrides and
complete-fleet attention configuration. The per-instance topology is projected
into the Instance Plan, not copied into the snapshot's `ModelConfig` attention
TP/DP fields. Configuration serialization, env-only inputs and file publication
remain in `runner.write_world_config`; its experiment argument supplies the
snapshot. Device allocation and native world lifecycle remain in the runner
and their existing resource owners.

`corpus.py` constructs the fixed sample specifications and composes the matrix
through one pure producer:

```python
def calibration_experiments(
    config: XpoolConfig,
) -> tuple[tuple[CalibrationExperiment, ...], CalibrationExperiment]: ...


fit_experiments, held_out_experiment = calibration_experiments(config)
```

Preserve fitting order, the conditional Dense TP2 experiment when at least two
FfnAgents are available, bounded TP widths, mixed held-out composition and
current repetition order. Reuse immutable sample specifications across those
definitions; every measurement still acquires fresh runtime resources. Pass
the selected experiment to snapshot construction, Fabric Plan construction and
`run_world` instead of reconstructing it from a selector in each consumer.
Synthetic Model IDs and checkpoint-key identities remain valid Spec fields;
their spelling does not dispatch execution.

The dependency direction is `corpus -> experiment -> core FFN/configuration/Plan
types`; `evidence` also consumes `experiment`, and the runner consumes both.
Keep experiment projection independent of evidence publication and process
execution. No additional package facade is needed.

An experiment's JSON definition contains its ordered `models`. Each model
records its Spec through the existing JSON serialization, `ffn_tp_size`, and
its effective Instance Plan through the existing Plan serialization. The latter
retains payload dtype, capacities, topology and output admission. This captures
the actual startup definition without a full deployment configuration, runtime
addresses, process state or weight contents.

### Measurement and evidence values

Keep the existing observation, software-environment and participant-evidence
values, moving their declarations from `fitting.py` to
`memory_profile/evidence.py`. That module owns the evidence JSON projection and
publication; `fitting.py` retains the numerical algorithm. Move declarations
rather than retaining aliases or adding parallel measurement types.

`MemoryProfileWorld` stores:

```text
experiment: CalibrationExperiment
repetition: int
participants: tuple[MemoryProfileParticipantEvidence, ...]
```

Repetitions are one-based. Participants follow ascending zero-based FfnAgent
index. Each participant retains the existing ordered five observations. Device,
environment and flattened observation views are derived from these participants,
not independently stored copies. Preserve the fitter's numerical input while
keeping attribution available to the evidence writer.

`run_world(experiment, source, *, repetition)` returns that world. Diagnostics
derive context from the experiment's concrete models and placement rather than
requiring an independently assigned experiment name.
`profile_ffn_memory()` returns one `MemoryProfileEvidence` containing
`profile: XpoolMemoryCalibrationProfile`,
`fit_worlds: tuple[MemoryProfileWorld, ...]` and
`held_out_worlds: tuple[MemoryProfileWorld, ...]`. These retain the candidate
Profile and its measurements together while preserving their fitting role.
Update its package export, CLI consumer and tests together. The CLI writes the
evidence, writes `evidence.profile` through the existing runtime writer and prints
both paths. Runtime Profile loading does not import the profiler package.

The evidence JSON retains the same grouping:

```text
profile: the complete candidate runtime Profile
fit_worlds:
  - experiment: the concrete experiment definition
    repetition: int
    participants:
      - ffnagent_index: int
        observations:
          - point: the existing DeviceMemoryPoint fields and features
            observed_device_bytes: int
            analytic_prediction_bytes: int
            calibrated_prediction_bytes: int
held_out_worlds: the same world structure for validation-only measurements
```

Within each collection, serialize experiments in matrix/repetition order, then
participant index and observation order. Calculate analytic and calibrated
predictions from the existing point and coefficient owners. Preserve individual
observations before fitting collapses equivalent rows. Environment and device
facts remain in the Profile. Existing retained evidence is left untouched;
no artifact version, experiment ID, digest, manifest or migration reader is
introduced.

The current isolated-child phase interception does not establish a need to split
Registry capture and installation into public operations. Keep that atomic owner;
do not add an observer framework as part of this task.

## Implementation

1. Express current FFN execution semantics independently of source architecture.
   Move Router execution policy to its shared owner and migrate model compilers,
   production consumers and synthetic inputs together. Remove avoidable Router
   cross-warp communication and duplicate Softmax reductions in separate measured
   steps without changing its mathematical or resource contract.
2. Consolidate TP projection, canonical allocation and signature construction with
   their production consumers. Construct the fixed matrix as concrete experiment
   values and migrate snapshot, Plan and runner consumers together. Keep
   production ordering, shapes, dtypes, Router ownership and storage lifetimes
   explicit.
3. Make experiment configuration and process/sampler ownership coherent across
   preparation, join, execution and retirement. Keep the existing control
   commands, extend private failure reports with retirement eligibility, and
   coordinate failed-world retirement separately from its execution verdict.
   Consolidate Runtime and Driver result checks and their consumers, preserve
   sampler failures regardless of thread liveness, and apply the terminal rule
   before KV diagnostics or compensation. Use explicit local KV allocation
   properties and their queried granularity. Stop qualification submissions at
   the existing failure boundary and complete terminal-exit diagnostics without
   changing parent-owned retirement.
4. Preserve attributed observations and their experiment definitions through
   fitting and publish world-grouped evidence under the accepted output policy.
   Flatten the CLI's publication error mapping and emit exception chains at
   existing owner-retention boundaries. Consolidate the two production writers'
   atomic text operation while retaining their serializers and error context.
   Keep runtime Profile loading independent and preserve the retirement protocol.
5. Migrate the shared reservation budget and placement expressions together.
   Recompute calibration from fresh worlds and qualify the resulting budget
   independently with real models; retain the
   fixed fitting method unless a further evidence-backed change is accepted.
6. Align the Control Plane memory-admission description, FFN Execution resource
   and Router ownership, and configuration instructions. Keep the supported
   low-level operator import boundary explicit when relocating code. Explain
   cleaned retained observations separately from instantaneous allocator-reserved
   memory after production startup. Align Qualification's Resource Evidence with
   the accepted tight-pressure success interval. Align Elastic Context State's storage
   description with the explicit local allocation properties. Update test
   documentation where coverage responsibilities change.

## Validation

- Add focused behavior regressions for inherited calibration output not becoming
  child input, sampler shutdown on setup failure, and cooperative pre-join failure
  cleanup with the original error preserved.
- Place actual spawn/terminal cancellation checks in Integration. Use self-owned
  Host processes to verify signal ownership and import/startup protection; do not
  exercise terminal signals against unrelated or real device processes.
  Extend the existing cancellation coverage to UID and final-participant readiness
  waits, verifying ordered retirement without requiring preparation to finish
  before stop is requested. Running child initialization remains cooperative.
- Verify an ordinary failure after confirmed world join participates in one
  parent-coordinated retirement, preserves the failed verdict and releases owners
  in order. Cover unconfirmed join and failed retirement without forced exit or
  repeated finalize. Use focused Host-only substitutes for these failure paths,
  not injected real device faults or an expanded serving matrix.
- Verify exceptional measurement exit skips success-only synchronization and
  sampling, preserves the original error, and applies the existing terminal
  predicate before initiating further cleanup calls.
- Cover retained sampler failures after the thread has stopped and cancellation
  between participant acquisitions using the existing Host-only lifecycle
  substitutes. Neither case may admit another phase or participant after the
  failure or cancellation has been observed.
- Verify Runtime and Driver checks preserve their original enum and operation,
  and terminal classification follows wrapped exceptions without another device
  query. Extend existing KV failure coverage for ordinary rollback, terminal
  failures before diagnostic or rollback calls, and a terminal diagnostic error
  retaining the allocation failure as its cause. Constructor failure coverage
  includes exception unwinding and proves no further device call follows the
  terminal result, rather than checking only that `close` was skipped. Use
  Host-side substitutes, not injected real device faults or a cross-product of
  all status codes.
- Verify the KV allocation properties are non-exportable and granularity and
  physical allocation use the same policy. Reuse existing real VMM expansion,
  shrink and Graph-replay coverage to check geometry and stable addresses;
  do not assert that a former probe helper is absent or that granularity has
  one fixed value.
- Extend existing Host-side qualification failure coverage to prove no later
  invocation starts after a retained failure is observed. Preserve the failed
  verdict and parent-owned retention rather than introducing another cleanup
  scenario or testing impossible cancellation of an in-flight native call.
- Prove held-out underprediction rejects a candidate. Verify evidence values
  reproduce the recorded before/after estimates and preserve
  experiment/repetition/rank/phase attribution and fitting role.
- Verify snapshot, Fabric Plan and recorded experiment definitions agree on
  their selected models and placement. Reuse fixed-geometry coverage for the
  fitting and held-out experiments, including the single-FfnAgent branch.
  Include the fixed samples' declared Router Algorithm and Precision alongside
  their geometry and normalization, preserving the conditional Dense TP2 case.
  Check that JSON retains the existing Spec and effective Instance Plan values,
  each participant's observations and the definition once per world. Keep tests
  at the typed construction and serialization interfaces, not selector spelling
  or assertions about absent implementation symbols.
- Exercise output-write failure under the accepted publication policy, including
  preservation of an existing runtime Profile. Reuse the CLI stage/path behavior
  tests rather than asserting exception nesting or the number of wrappers.
  Cover complete replacement, failure preservation and temporary-file cleanup
  once at the shared atomic-text operation; keep business tests focused on
  serialization, publication order and error context.
- Extend Host-only lifecycle coverage to verify the visible original-error and
  cleanup-error tracebacks at an owner-retention boundary. Reuse existing failure
  substitutes where applicable; do not assert logger call syntax, introduce
  production seams for tests or inject real device failures.
  Include the visible traceback for a later terminal error while preserving
  the original failure; avoid duplicate traceback output for a first terminal
  error. Reuse existing Agent and worker failure-path coverage.
- Reuse existing loader, Registry, topology and model coverage for canonical
  shapes, admitted dtypes and Router ownership. Replace duplicated sample-value
  inventories only where a real behavioral assertion owns the same contract.
- Verify source compilers select the declared execution semantics and production
  and synthetic consumers use the same resource rules. Preserve independent
  numerical references for the affected Router paths, their actual dtypes,
  workspace geometry and Signature sharing. Do not substitute source-text or
  removed-field assertions for those behaviors.
- Extend the existing routing coverage with definition-based activated-score
  references for both algorithms and optional correction. Include a nonzero
  correction that changes selection; use more than one selected Expert and
  unnormalized weights so one-element normalization cannot hide contaminated
  weights. Reuse representative cases for normalization, Expert padding and
  tie rules rather than multiplying every independent axis into another matrix.
  Preserve no-bias Softmax coverage and exercise Sigmoid without correction.
- Exercise corrected TopK through captured replay with changed inputs and bias
  contents at stable addresses, and reuse native model Graph qualification for
  binding/discovery. Reuse the shared size/accounting checks to prove unchanged
  explicit workspace extents and correct optional bias ownership. No test solely
  asserts a former kernel name or helper has disappeared.
- Record representative TopK timings against the former no-bias Softmax
  operator as report-only diagnostics, including the 64-Expert Top6 FP32 and
  128-Expert Top8 BF16 shapes, small and large row counts, and separate
  normalization comparisons. Distinguish ordinary calls with cache clearing from
  CUDA Graph replay; measure launch and reduction changes separately. Reuse
  existing numerical, padding, tie and captured-replay tests, extending them only
  for an uncovered behavior rather than adding a tuning-parameter test matrix.
  Do not add another benchmark framework, production fallback, source/assembly
  text assertion or automatic performance threshold for this comparison.
- Reuse checkpoint, loader and capture coverage for the owning-type methods:
  ordered key coverage, typed staging views, zero-valued shape-equivalent
  weights, exact binding addresses and contiguous workspace views. Exercise
  observable values and ownership rather than the absence of former helpers.
- Do not turn throwaway AST-extraction prototypes into permanent tests. Test
  supported behavior and resource ownership rather than implementation text.
- Provide the affected model qualifications with their declared real checkpoints
  and payload dtypes, including the FP16 `huggyllama/llama-7b` checkpoint. Acquire
  missing checkpoints through the existing `model-download` utility, preferring
  ModelScope and falling back to Hugging Face when the exact checkpoint is
  unavailable there. Keep downloaded weights machine-local and resolve them
  through existing model-path configuration. Another Dense model or synthetic
  corpus member does not replace a missing checkpoint.
- Cover allocation-class boundaries for the corrected reservation budget,
  positive request sizes, arithmetic limits, and conservative mixed-class
  fallback through the shared pure budget interface. Verify placement and the
  concrete estimator sum the same construction units. Reuse existing loader and
  Registry integration coverage; retain one direct canonical-allocation
  regression for the independently reproducible inactive-split discrepancy,
  rather than adding checkpoint or full-serving variants of that same verdict.
- Diagnose Graph coverage with paired observations under the temporary-script
  policy below. Preserve each phase's original peak as the accuracy verdict;
  endpoint allocator snapshots alone do not explain a previously sampled peak.
  Refit all accepted synthetic experiments after changing the analytic budget.
  Keep real qualification worlds independent from fitting and retain their
  failures when a candidate does not cover them.
- After implementation, follow
  [Qualification](../../designs/qualification.md#acceptance-and-invalidation):
  focused checks during iteration, complete strict acceptance on the final source
  and native identity, and real-model Resource Qualification because allocation
  and estimation consumers change. Include zero-underprediction baseline and
  zero-operator-margin pressure evidence. Complete the numerical and graph
  qualification invalidated by adapter or Router changes. Old fit/held-out
  observations remain evidence for the old execution, not measurements of
  relocated or changed allocation and capture paths. Reuse unaffected evidence
  as permitted by its owner; no full benchmark campaign is part of this refactor.

## Out Of Scope

- Changing the fitting algorithm, reducing the matrix or removing coefficients
  without independent measurement evidence.
- A universal Router/execution framework for unknown models or future backends,
  provider discovery, and provider or factory hierarchies, including corpus
  providers. Express only the currently supported execution behavior; do not add
  arbitrary combinations of routing algorithms and precision policies.
- New runtime configuration settings, another serving engine, or a new weight
  format.
- Running the full production daemon to coordinate calibration, adding a
  calibration daemon, importing xkit into production, or replacing shared process
  supervision.
- Automatic recovery, calibration resume, retries, generic TERM/KILL cleanup,
  teardown frameworks or device-driver fault diagnosis.
- A new benchmark/pin integration, database, artifact version, digest scheme or
  retention service for calibration evidence.

## Resource Budget Correction

### Bound physical allocator reservation

The allocator allowance bounds active-block excess, but physical device usage
also includes unused portions of reserved segments. Valid Dense weight geometry
can leave inactive splits in segments that still contain live tensors. Those
bytes remain reserved and can exceed the current allowance even when canonical
Tensor requests exactly match the Exact Resource Ledger. Replaying canonical
empty allocation is sufficient to distinguish this discrepancy from checkpoint
loading and distributed execution.

The relevant dependency is the pinned
[PyTorch native allocator](https://github.com/pytorch/pytorch/blob/cf30153c4c131c8164ee7798e5022d810682e2cb/c10/cuda/CUDACachingAllocator.cpp).
Its default request and segment sizes, also declared in the installed
`c10/core/AllocatorConfig.h`, give the following reservation rule. Round a
positive request to at least 512 bytes and then to a 512-byte multiple:

| Rounded request | Fresh segment |
| --- | --- |
| At most 1 MiB | 2 MiB |
| Greater than 1 MiB and less than 10 MiB | 20 MiB |
| At least 10 MiB | Request rounded up to 2 MiB |

Bound **new physical reservation during a known construction group** using the
following rules:

1. An empty group costs zero.
2. Consecutive requests on one stream and pool can share a 2-MiB charge when all
   are small and their rounded sum fits 2 MiB. They can share a 20-MiB charge
   when all are medium and their rounded sum fits 20 MiB.
3. Otherwise charge each request's complete fresh-segment size independently.
   Do not combine groups across layers, signatures, stream/pool changes, or
   intervening allocations.

For a shareable group, consider its first request that needs a new segment. Its
remaining requests fit that segment even if earlier requests used cached blocks.
Small-pool splitting preserves 512-byte-aligned capacity. In the medium pool,
an unsplit tail is at most 1 MiB and cannot consume space needed by another
medium request. This establishes a bound independent of the incoming cache
layout; it does not predict actual packing. The fallback follows directly from
charging every possible cache miss.

Apply the group rule only to construction sequences established by production
code: one canonical packed layer, one Control Probe, and one Dense signature's
input/partial/workspace sequence. The MoE retained-storage census is not such a
sequence: route-ID initialization occurs before workspace allocation. Charge
those known requests independently instead of inventing a chronological trace
from the census. Keep capture-body allocations on the capture stream/private
pool separate from explicit pre-capture construction.

The tradeoff is deliberate overprediction rather than exact fragmentation
simulation. For example, 24 Dense pairs of 8.3125 and 4.15625 MiB have 299.25 MiB
of logical storage and a 480-MiB grouped reservation bound; the observed
reservation can be 360 MiB. Twenty-eight pairs of 6 and 3 MiB have a 560-MiB
bound while observed reservation can be 280 MiB. These are budget examples, not
Model ID conditions. Cross-layer packing or packed backing-storage allocation
would need a separate design; do not silently add either to tighten this bound.

#### Shared interfaces and consumers

Replace the active-block-only helper in `device_memory.py` with:

```python
def allocator_reservation_bytes(storage_bytes: tuple[int, ...]) -> int: ...

def graph_capture_allocator_allowance_bytes(
    signature: execution.ExecutionSignature,
) -> int: ...
```

The first function returns the complete reservation bound for one consecutive
construction group under the pinned default allocator. Keep positive-size and
signed-64-bit validation. Its callers obtain allowance by subtracting the
group's logical bytes. The second function owns the per-signature composition
shared by placement and concrete estimation; it includes known construction-only
allocation requests as well as retained placeholders.

Keep `packed_weight_storage_bytes`, `control_capture_probe_storage_bytes`, and
`graph_capture_capacity_storage_bytes` as logical-storage producers. Preserve
their existing Tensor-byte/count feature meanings. Put the MoE initializer's
two short-lived `int32[effective_topk]` requests beside that geometry in
`execution.py`:

```python
def graph_capture_initialization_storage_bytes(
    signature: ExecutionSignature,
) -> tuple[int, ...]: ...
```

Return an empty tuple for Dense. Charge their complete possible reservation in
the Graph construction allowance, including their logical bytes, because the retained
ledger and features do not include those transient owners. Do not retain these
tensors longer or change their initialization to simplify measurement.

Sum independent costs per existing packed-layer/TP-rank variant, normalized
Control Probe, and signature. The concrete estimator must preserve those groups
before flattening logical storage for features. The placement solver must reuse
the same costs with its existing assignment counts and signature/probe presence
variables. Keep postsolve estimate agreement; introduce no ordering variables or
allocator-state model into CP-SAT.

Keep `DeviceMemoryPoint.allocator_allowance_bytes`, the five points, runtime
Profile fields, and fitted feature definitions unchanged. The allowance now
covers the known construction requests' possible reserved segments, including
cache left by short-lived initialization, rather than active-block excess alone.
Recompute coefficients from fresh calibration worlds after this change; old
Profiles must be regenerated even though their unchanged JSON schema remains
readable. Do not add compatibility aliases, configuration switches, a custom
allocator, or a storage-lifetime framework.

### Graph coverage investigation and conditional correction

Real Graph capture and installation can exceed the fitted estimate despite
successful synthetic fit and held-out validation. Allocator snapshots distinguish
active Tensor bytes, reserved bytes and inactive splits; these must be reconciled
with stage observations before attributing the entire discrepancy to Tensor
fragmentation. The weight-stage discrepancy alone does not explain every
Graph-stage byte.

First apply the analytic correction, then evaluate the unchanged
synthetic fit and independent real-model evidence. An observed total deficit
does not establish a need for another coefficient or a larger corpus. A derived
reservation bound may cover it without either change; record the resulting
headroom rather than describing it as precise allocation prediction.

Extend only the throwaway qualification script's evidence collection:

- Record baseline and same-boundary global free bytes, Torch allocated bytes,
  Torch reserved bytes, and inactive-split bytes. Bracket allocator-stat reads
  with free-memory observations so a changing sample is not treated as an
  atomic decomposition.
- Use the existing Registry capture function and native installation call as
  temporary wrapping seams, forwarding their original implementations. Retain
  paired observations around warmup/capture and installation, together with the
  original phase peak. Compare changes relative to the corresponding baseline;
  subtract only contemporaneous quantities, and do not subtract an endpoint
  snapshot from a different-time maximum. Read memory statistics outside active
  CUDA capture; do not insert observation calls into the captured launch body.
- Add no synchronization, cache trimming, activation-time observation, or
  permanent production telemetry. The pinned `torch.cuda.graph` context already
  synchronizes and empties cache before capture; preserve that dependency
  behavior. It is distinct from calibration's explicit post-install trim.
- Reconcile predictable CrossPool native allocations separately from Torch
  reservation. CUDA library/Graph internal allocations, and installation metadata
  whose size is discovered only after capture, remain candidates for the
  calibrated remainder. The native binding-site and update arrays are examples
  of the latter; their existence is not proof they cause the observed deficit.

Apply the first correction without changing the calibration method defined
above. Real serving uses capacities beyond the fixed corpus's 4K maximum, so
successful fit/held-out validation alone does not qualify those shapes. If a
newly fitted candidate still underpredicts, retain the failing paired evidence
and return a concrete proposal identifying the missing resource or
execution-shape coverage. A known stable allocation belongs in analytic
accounting; an opaque execution cost needs justified synthetic coverage.
Neither a Model ID case nor the failed real-model row becomes a fitting sample.
Any corpus, feature, or objective change still requires review before
implementation.
