# CrossPool Code Style

## Scope

This document is the canonical source for code-level conventions in CrossPool. It
applies to production code, tests, native bindings, and build definitions.
Current architecture belongs under `docs/designs/`, active target changes under
`docs/plans/`, configuration semantics in `docs/designs/control-plane.md`, test
organization in `tests/README.md`, and repository workflow routing in `AGENTS.md`.

Use English for documentation, comments, identifiers, tests, and commit
messages.

Name the system CrossPool in documentation, comments, docstrings, binding
documentation, and CLI help. Preserve executable identifiers such as `xpool`,
`XpoolConfig`, paths, URLs, configuration keys, and command examples. Runtime
logs and error messages use lowercase `xpool` for the system name.

## Documentation

Document supported APIs at their declaration site. Documentation must explain
the contract that a caller cannot infer from the signature: meaning, units,
ownership, lifecycle, side effects, and principal failures. Do not add
ceremonial descriptions that only repeat a name or type.

Use Google-style docstrings for Python and Doxygen comments for public C++ and
CUDA declarations. Native declarations under `src/cext-include/xpool` are the
documented project boundary. Internal helpers need comments only for
non-obvious ownership, synchronization, graph-capture, transport, or failure
semantics.

Tests do not need module docstrings when their path and names already describe
their behavioral scope. Add documentation when a test has a non-obvious
prerequisite, boundary, or acceptance strategy.

Use phase comments when they clarify meaningful protocol or lifecycle
boundaries. Name the phase and its invariant; omit labels that merely narrate
adjacent statements. Phase comments do not justify mixing unrelated
responsibilities.

Build and configuration comments explain only non-obvious toolchain,
dependency-discovery, ordering, package-layout, or side-effect constraints. Do
not translate declarations line by line, and do not hand-edit generated files
to add documentation.

Documentation gates cover supported declarations, not every implementation
parameter. They must not require ceremonial parameter, return-value, override,
or helper descriptions merely to satisfy a coverage rule.

Write current behavior in direct, positive sentences. Use negative wording when
the prohibited behavior is itself the contract; avoid stacking `not`, `no`,
`without`, `neither`, or `nor` in one explanation.

Runtime logs are low-frequency operational records, not a second protocol or
trace stream. Keep messages and field names lowercase, include process identity
relevant to the event owner explicitly, and write runtime logs to stderr while
keeping CLI data on stdout. Log state edges and completed slow phases at the
owning layer; keep retries, polling, and acknowledgements at debug level, and
report a failure only at the layer that terminates the operation. Prefer the
standard library logging package and one process-local formatter over
module-specific wrappers or context registries.

## Ownership And Abstraction

Put behavior at the narrowest layer that owns it. Model-specific behavior
belongs to its model adapter. Serving-integration imports, types, runtime
objects, hooks, compatibility rules, and devkit helpers belong under
`xpool.integrations.<engine>`. Translate them to CrossPool-owned values before they
cross into core configuration, runtime, protocol, or native interfaces. A
low-level operator implementation shipped in the same distribution is not
serving-integration code only when the relevant accepted design or active plan
explicitly accepts that narrow dependency and no serving-runtime type or
retained state crosses into core.

Do not introduce an abstraction for one implementation or a compatibility
layer for a private detail. A helper, wrapper, provider, manager, registry, or
base class must own a real boundary, repeated operation, invariant, or
lifecycle. Inline single-use indirection when that makes ownership clearer.

Do not add production seams solely so tests can replace dependencies. Tests
should patch the dependency at the module that owns the call. Production APIs
must describe runtime concepts rather than test mechanics.

Do not retain migration-only tests whose sole purpose is to name and reject a
removed, unsupported API, configuration field, CLI option, or environment
variable. Test the current public contract and generic invalid-input behavior
at the owning boundary instead. Keep a historical rejection case only when a
current accepted compatibility, security, or data-migration contract explicitly
requires that exact legacy input.

Bind intrinsic construction, validation, formatting, serialization, and
resource lifecycle to the type that owns the invariant. Keep cross-type
orchestration, discovery, plugin hooks, I/O, and general algorithms at module
or namespace scope. Do not use classes merely as namespaces.

Avoid duplicate sources of truth. Configuration, protocol fields, materialized
state, and derived values must each have one owner. Pass or derive values from
that owner instead of caching copies in adapters, registries, or globals.

Validate untrusted values once at the boundary that assumes ownership. Public
lifecycle operations continue to reject invalid state transitions, and external
runtime, device, IPC, and shared-memory results remain checked. Private helpers
rely on invariants already established by their owner; document non-obvious
preconditions instead of repeating the same runtime checks or replacing them
with release-disabled assertions.

Keep each exception boundary scoped to one operation, retry, or cleanup owner.
Nest exception handling only when compensation or reconciliation can fail
independently and both failures affect diagnostics or resource safety. Preserve
the original operation failure unless cleanup cannot prove resources safe to
release; suppress only explicitly ignorable idempotent cleanup outcomes.

Use repository terminology consistently. `AtnAgent` and `FfnAgent` are
indivisible role names in PascalCase identifiers. Use `ATNAGENT` and `FFNAGENT`
for Python wire-enum values, and `atnagent` and `ffnagent` in lowercase names.
Use `Agent` only for a shared lifecycle concept and `PE` only for NVSHMEM
behavior. `DevAgent` is not a current role or compatibility term.

## Python

Run Python tools through `uv run`. Format with `uv run ruff format`, lint with
`uv run ruff check`, and type-check with `uv run ty check`.

Prefer explicit concrete types. Use `Any` or `object` only for a genuinely
dynamic boundary and document the reason nearby. Import pinned third-party
types instead of inventing local look-alike protocols.

Do not use `TYPE_CHECKING` blocks or local imports to conceal ordinary
dependency cycles. Fix the ownership boundary.

Do not define CrossPool-owned names with a single leading underscore. Remove
unneeded bindings and keep framework-mandated parameters under their protocol
names. Python protocols and unavoidable private third-party names are exempt.
Do not preserve renamed private details through compatibility aliases.

Do not use import aliases. Import the owning module and qualify symbols where
names would collide.

Use explicit `__all__` declarations for supported package, facade, and stable
direct-module APIs. Omit `__all__` from internal implementation and test
harness modules. Do not use wildcard imports.

Migrate deprecated APIs instead of suppressing diagnostics. Keep the
`deprecated` type-check diagnostic enabled. Context-manager generators must use
the generator return type required by their decorator.

Prefer short local PEP 695 type-parameter names when scope makes their meaning
clear. Prefer `match` or a closed dispatch table when dispatching over a closed
set of states.

## C++ And CUDA

Format native files with `uv run clang-format` and the repository
`.clang-format`. Pass only explicit native source and header paths. Keep public
native documentation passing `doxygen Doxyfile`.

Order native includes in formatter-owned groups: the translation unit's
matching main header first, then C++ standard-library headers, third-party
headers, and CrossPool or local project headers. Let clang-format regroup and sort
these categories. Use the narrowest possible `clang-format off` region only
when preprocessing or another semantic dependency requires a different order,
and explain that dependency beside the exception.

Keep native production basenames free of underscores; directories express
subsystem boundaries. Flat core Devkit components use descriptive snake-case
basenames that match their component names. CTest files use the `*_test`
suffix.

Define short, stable member functions in the owning class definition. Keep
substantial algorithms, synchronization, resource lifecycle, failure recovery,
and implementation-only dependencies out of line. Do not introduce PIMPL
without a concrete ABI or dependency-isolation need.

Prefer `auto` for initialized local variables when the initializer fixes the
exact type. Keep public signatures, stored fields, protocol layouts, and
schema-bearing callbacks explicit.

Use an alias only for an intentional domain representation or genuinely
complex template expression. Do not alias primitives, familiar standard
containers, optionals, pointers, locks, or streams merely to shorten spelling.
Do not use `using namespace`.

Choose integers by domain:

- convert Torch operator integers once at the registration boundary;
- use `std::size_t` for sizes, offsets, strides, capacities, dimensions,
  counts, and indices;
- use fixed-width integers for wire values, enum representations, ABI
  versions, states, errors, sequences, and timestamps; and
- use an external API's native integer or handle type at that boundary.

Keep checked integer arithmetic type-preserving. Reject mixed integer types at
compile time rather than silently choosing a common type.

Represent trusted closed-set values with validated CrossPool-owned types. Keep
untrusted wire values in their explicit fixed-width representation and
validate them before constructing a trusted value.

Use `static_assert` for a real compile-time capability or external contract.
Do not freeze compiler-derived sizes, alignments, or member offsets when every
participant uses the declared type and `sizeof`. Express required alignment in
the declaration.

Do not prefix global storage with `g_`, and do not expose mutable storage as a
subsystem API. Ordinary callers use the owning accessor or lifecycle API.
Link-visible host/device storage is permitted only when required to implement
such an accessor; it remains an implementation detail.

Place native headers under `src/cext-include` and include them through the
`xpool/...` root. Directories under `src/cext` contain compilation units only.
Headers include the owning declarations they use instead of adding
namespace-scope forward declarations for project or third-party types.

Model nullable resource ownership in the owner itself. Do not wrap an already
nullable owner merely to express presence. Distributed resources require
explicit coordinated shutdown while their runtime is live; destruction is not
a substitute for that protocol.

Name private data members with a trailing underscore. Leave fields unsuffixed
only in intentional plain data aggregates such as wire records, arena layouts,
and device state.

Keep synchronization assumptions next to the code that depends on them.
Prefer Cooperative Groups for CUDA group behavior, then CCCL/libcu++, then an
existing typed CrossPool utility. Use raw intrinsics only when these cannot express
the required semantics and document the reason nearby.

Use the `XPOOL_*` annotations from `xpool/macros.hpp` for CUDA function
execution space, Device storage space, and justified forced inlining. Keep
compiler probes, Device intrinsics, and kernel launch syntax explicit. Guard
device-only declarations in ordinary `.hpp` files; `.cuh` files are CUDA-only.

Keep one global entry for a persistent CUDA role when its phases share resident
state and synchronization. Extract pure mechanisms or complete lifecycle
transactions, but keep barriers, ownership, publication, transport ordering,
shutdown, and fail-stop transitions visible. Split kernels or force inlining
only with same-toolchain evidence.

## Native Build Definitions

Manage native targets with CMake. Do not use `setup.py` as a parallel native
build system.

Apply project warning policy through the internal compiler-options target and
link it privately to project-owned targets. Keep normal C++ and CUDA warnings
enabled and treat them as errors. Warning escape hatches are for toolchain
diagnosis, not normal development.

Treat third-party include directories as CMake `SYSTEM` paths. Add a
compatibility header only for a real boundary that dependency versioning or
build metadata cannot express. Do not add wrappers that merely rename an
include or suppress project diagnostics.

Compiler controls that change generated CUDA code require workload-specific
profiling evidence; do not add them only to silence diagnostics.
