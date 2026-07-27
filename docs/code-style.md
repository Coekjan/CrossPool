# xpool Code Style

## Scope And Language

This document is the canonical source for code-level naming, typing,
ownership, abstraction, documentation, formatting, and native-language
conventions in xpool. It applies to production code, tests, native bindings,
and build definitions. Repository architecture, configuration, test layering,
developer environment, and workflow policy remain in `AGENTS.md`.

Documentation, comments, identifiers, tests, and commit messages must use
English.

Public APIs must be documented at the declaration site. For Python, public
modules, classes, functions, methods, dataclass fields, Pydantic fields, and
enum members need useful docstrings or field descriptions. Use Google-style
docstrings with `Args`, `Returns`, `Raises`, preconditions, postconditions, and
side effects where those sections apply.

Python API support is determined by an explicit package, facade, or stable
direct-module `__all__`, or by a wire/service boundary; the absence of a
leading underscore does not by itself make an internal implementation symbol a
supported API. FastAPI routes and their wire models document request, success,
and principal failure behavior. Every type, enum value, property, and function
exposed through `xpool.native` bindings carries useful runtime documentation at
its binding declaration. Internal production modules and reusable test-harness
modules still document module and type responsibilities plus nontrivial
lifecycle methods, but do not add ceremonial Google-style sections to simple
accessors. Ordinary `test_*.py` modules do not require a module docstring when
their path and test names already express the behavioral scope; add one only
when it records a non-obvious boundary, prerequisite, or acceptance strategy.
Repository `conftest.py` files document the session hooks and fixture effects
they install.

For C++ and CUDA, public namespaces, classes, structs, enum values, constants,
functions, and fields need Doxygen-style `///` or `/** ... */` documentation.
Public field documentation must explain meaning, units, ownership, lifecycle,
and why the field exists.

For native headers, public and protected declarations plus namespace-level
functions in `src/cext-include/xpool` form the documented project boundary.
Private storage and implementation-only helpers need comments only when their
ownership, units, synchronization, or lifecycle are not evident from the
surrounding documented contract.

Use inline comments only where they clarify ownership, device placement,
synchronization, CUDA graph capture, NVSHMEM ordering, IPC, or failure behavior
that is not obvious from the code.

Use phase comments only in functions with at least three genuine lifecycle or
protocol phases. Name the phase and explain the ownership, visibility, or
state-machine invariant established there. Do not add phase comments to short
helpers or use them to justify keeping unrelated responsibilities together.
Use the uniform `// Phase: Name - invariant` form in C++ and CUDA so phase
boundaries remain searchable across persistent kernels and device protocols.

## Ownership And Abstraction

Keep implementation structure deliberate. Model-specific code belongs under
the model adapter that owns it; global integration layers should expose only
generic registration, discovery, binding, and shim contracts.

Use repository terminology consistently. `AtnAgent` and `FfnAgent` are the two
xpool Agent runtime roles. Use the generic `Agent` term only for shared
lifecycle concepts; `DevAgent` and `devagent` are not current types, roles,
commands, or compatibility terms. Use `PE` only when directly describing
NVSHMEM APIs or behavior.

Treat `AtnAgent` and `FfnAgent` as indivisible role names in PascalCase C++ and
Python identifiers, including C++ enum values. Python wire-enum values use
`ATNAGENT` and `FFNAGENT`; modules, files, commands, process titles, config
values, and other lowercase identifiers use `atnagent` and `ffnagent`. Do not
write variants such as `Atnagent`, `Ffnagent`, `Atn_Agent`, or `ATN_AGENT`.

Avoid boilerplate helper functions, especially private helpers used only once
or twice, when inlining keeps the calling code clearer. Add an abstraction only
when it carries a real ownership boundary, repeated behavior, or a typed
contract that improves local reasoning.

Do not add one-line wrappers, renamed constants, or pass-through functions that
only obscure the real API. Examples include `call_native_*` wrappers,
single-use `*_to_outdir` helpers, or constants that merely rename one local
literal. Inline the call unless the wrapper owns validation, synchronization,
lifetime, or a stable typed boundary.

Do not add production callback or provider parameters solely to make tests
replace dependencies. Production signatures should expose runtime concepts;
tests should monkeypatch the dependency in the module that owns the call. A
short function is justified only when it implements a framework protocol, a
typed conversion, resource lifecycle, synchronization or failure semantics, or
a genuinely repeated domain operation. Before finalizing a change, inspect new
single-statement functions, renamed imports, and local helpers for needless
indirection.

Bind behavior to an xpool-owned type when it defines that type's intrinsic
construction, parsing, formatting, validation, serialization, or resource
lifecycle. Keep subsystem entry points, cross-type orchestration, discovery,
plugin hooks, I/O, and general algorithms at module or namespace scope. Do not
introduce manager, singleton, or namespace classes merely to group functions,
and do not retain pass-through wrappers around newly type-bound APIs.

## Python

Run Python project tools through `uv run`; do not invoke `.venv/bin/...`
commands directly. Format with `uv run ruff format`, lint with
`uv run ruff check`, and type-check with `uv run ty check`. Keep public Python
documentation passing Ruff's public docstring checks.

Prefer explicit concrete types. Avoid broad `Any` or `object` unless they are
required for a dynamic third-party surface and the reason is documented close
to the annotation. In SGLang integration code, import pinned SGLang concrete
types directly instead of inventing local `*Like` protocols. Do not use
`TYPE_CHECKING` blocks or local imports to hide ordinary dependency cycles; fix
the ownership boundary instead.

Do not define xpool-owned Python variables, parameters, constants, functions,
methods, attributes, classes, or type aliases with a single leading underscore.
Use descriptive names, remove bindings that are not needed, and keep
framework-mandated unused parameters under their protocol names instead of
prefixing or deleting them. Python double-underscore protocols and unavoidable
private names owned by standard-library or third-party APIs are exempt. Do not
add compatibility aliases for renamed private implementation details.

Do not use Python import aliases. Import the module that owns a symbol and use
its qualified name when local or third-party names would otherwise collide.
Python module filenames may use underscores when they express established
Python, configuration, integration, or model terminology.

Use `__all__` to document supported package, facade, and stable direct-module
APIs. Enumerate those exports explicitly. Omit `__all__` from internal
implementation and test-harness modules, and do not use wildcard imports.

Deprecated APIs must be migrated rather than suppressed. Keep ty's `deprecated`
diagnostic enabled as an error for PEP 702-decorated APIs. This gate is not
exhaustive for typeshed overload deprecations, so migrate those directly when
another supported tool or review identifies them. A function decorated with
`contextmanager` returns `Generator[Yield, None, None]`; one decorated with
`asynccontextmanager` returns `AsyncGenerator[Yield, None]`. Undecorated pytest
yield fixtures may continue to return `Iterator[Yield]`.

For PEP 695 generic functions, prefer short local type parameter names such as
`R` and `W` when the scope is obvious. Avoid legacy-style verbose names such as
`ReturnT` or `WeightT` for local generic function parameters.

Prefer `match` statements when dispatching over a closed set of enum-like
states; avoid long `if`/`elif` ladders when a closed dispatch table or `match`
would make exhaustiveness clearer.

## C++ And CUDA

Format C++, CUDA, and headers with `uv run clang-format`. The repository-root
`.clang-format` is the only native formatting policy and sets the same
120-column limit as Ruff. Pass only explicit `.c`, `.cc`, `.cpp`, `.cu`, `.h`,
`.hh`, `.hpp`, or `.cuh` paths; never run clang-format over the repository root,
a mixed-language file list, Markdown, Python, TOML, YAML, or extensionless
files. The root `.clang-format-ignore` is a final safeguard against accidental
mixed-language invocation, not a substitute for selecting native inputs
correctly. When adding a native extension to the CMake source or header globs,
update the ignore allowlist in the same change.

Keep native C++/CUDA file basenames free of underscores; express subsystem and
role boundaries through directories instead of compound filenames.
Native CTest source files use the explicit `*_test.{c,cc,cpp,cu}` discovery
suffix as the sole basename exception; their directory still carries subsystem
ownership.

Define short, stable class and struct member functions directly in the class
definition in the owning `.hpp` or `.cuh`. In-class definitions are already
implicitly inline; do not add a redundant `inline` specifier. Simple accessors,
singleton accessors, thin overload delegation, move operations, and short
resource delegation are typical candidates.

Keep member functions out of line when they require an incomplete private type,
an implementation-only dependency such as a CUDA device symbol, NVSHMEM, or a
JSON parser, or when they implement substantial algorithms, synchronization
protocols, resource lifecycles, or failure recovery. Do not introduce PIMPL
solely to hide short member functions or reduce header content; require a
concrete ABI firewall or dependency-isolation reason.

Keep public C++/CUDA documentation passing `doxygen Doxyfile`; Doxygen is a
repository development prerequisite.

Prefer `auto` for initialized C++ and CUDA local variables when the initializer
determines the exact type. Preserve semantic integer width or handle type with
an explicit typed initializer on the right-hand side. Keep public signatures,
stored fields, protocol layouts, and schema-bearing callback parameters
explicit.

Prefer `using` over `typedef`, but introduce a type alias only when it names an
intentional domain representation or hides genuinely complex template
machinery. A public alias is part of the owning header's API and must document
whether callers may rely on exact substitutability with its underlying type. Do
not alias primitive integers, familiar standard-library containers, optionals,
smart pointers, locks, streams, or template instantiations merely to shorten
spelling; use an xpool-owned wrapper or struct when values with the same
representation must not be mixed. Keep implementation-convenience aliases local
or private, and do not use `using namespace` directives.

Choose native integer types by semantic domain. Accept Torch operator integers
as `std::int64_t` only at the registration boundary and convert them once to the
owning domain type. Use `std::size_t` for host/device arena sizes, byte offsets,
strides, capacities, tensor dimensions, and container, model, layer, lane,
slot, and rank counts or indices. The native ABI targets 64-bit Linux/CUDA and
may enforce 64-bit `std::size_t` as a compile-time platform contract. Use
fixed-width integers for raw enum values, ABI versions, states, errors,
monotonic sequences, and timestamps; use external API-native types such as
`c10::DeviceIndex` and NVSHMEM's `int`. Remove redundant casts after values
enter their domain type. Use `std::uintptr_t` only when address bits are
intentionally represented as an integer, not for ordinary pointer or
arena-offset arithmetic.

Keep checked integer arithmetic type-preserving. `checked::sum` and
`checked::prod` accept same-type non-boolean integral operands and return the
exact operand type; `checked::align_up` additionally requires an unsigned
operand type. Reject mixed integer types at compile time instead of silently
selecting a common type.

Represent trusted closed-set native values with xpool-owned validated value
types, while keeping untrusted Transport and Fabric wire enum, state, and error
fields in their explicit fixed-width integer representation. Generic enum-value
constructors may accept the type's own enum or non-boolean integral inputs, but
ordinary integers must require explicit construction and invalid construction
must fail-stop. Recoverable operator and wire boundaries must validate before
constructing the trusted value.

Use `static_assert` to enforce a real compile-time capability or external
contract, such as trivial copying across CUDA/NVSHMEM boundaries, standard
layout where required, or a generic template's type requirements. Do not freeze
compiler-derived object sizes, alignments, or member offsets when all producers,
consumers, arena planners, and transfers use the declared type and `sizeof(T)`.
Express required alignment with `alignas` at the declaration.

Do not prefix C++ or CUDA global storage with `g_`, and do not expose mutable
global storage as the subsystem API. Access process-lifetime owners through a
type-bound static `singleton()` accessor backed by a function-local static
object. CUDA and NVSHMEM resources require explicit lifecycle cleanup before
process teardown; static destruction only terminates an already empty or
finalized host owner and is not a recovery path for live device resources. Keep
process-lifetime resource state directly in its owning type unless an accepted
PIMPL boundary has a concrete ABI or dependency-isolation purpose. Host and
device debug code must read their target-specific storage through
`debug::options()` rather than accessing the `_h` or `_d` storage directly.

Model nullable native resource ownership directly in the owning type: default
construction is empty, explicit boolean conversion reports ownership, typed
acquisition creates or attaches the resource, normal cleanup reports errors,
and destruction is only a best-effort fallback for resources that one process
can release independently. Collective or distributed resources must use their
explicit coordinated shutdown while the required runtime is live; their
destructors verify the owner is empty and fail-stop instead of initiating an
implicit collective. Do not wrap an already nullable owner in `std::optional`
or `std::unique_ptr` solely to represent presence.

Name private C++ and CUDA data members with a trailing underscore. Keep public
fields unsuffixed only for intentional structs such as wire records, arena
layouts, device state, and plain data aggregates. Types that own a resource or
enforce parsing, formatting, validation, or lifecycle invariants must keep their
storage private and expose behavior through member functions.

Keep CUDA/NVSHMEM synchronization assumptions close to the code that depends on
them.

Prefer Cooperative Groups for CUDA group behavior, then CCCL/libcu++ for
standard algorithms and utilities, then an existing typed xpool utility. Use a
raw block/warp synchronization, shuffle, atomic, or cooperative-launch intrinsic
only when the preferred libraries cannot express the required semantics and the
reason is accepted and documented close to the implementation. Raw CUDA
allocation, copy, and set APIs remain appropriate at IPC ownership and one-shot
host-initialization boundaries; NVSHMEM remote operations and device pointers
remain required at cross-address-space boundaries.

Annotate native function execution space through `XPOOL_HOST_FN`,
`XPOOL_DEVICE_FN`, and `XPOOL_HOST_DEVICE_FN` from `xpool/macros.hpp`. These
macros express execution space only: keep `inline`, `__forceinline__`,
`__global__`, CUDA storage qualifiers, and kernel launch syntax explicit.
Ordinary `.hpp` files must retain `__CUDACC__` guards around device-only
declarations; `.cuh` files are CUDA-only and need no redundant guard.

Keep one global entry per persistent CUDA role when its phases share resident
state and synchronization. Extract only pure mechanisms, typed local contexts,
or complete lifecycle transactions, while leaving barriers, warp/block
ownership, queue transfer, publication, NVSHMEM ordering, shutdown, and
fail-stop transitions visible in the kernel body. Do not split a resident
protocol into additional kernels or add forced inlining without same-toolchain
resource and profiling evidence.

## Native Build Definitions

Manage native targets with CMake; do not reintroduce `setup.py` as the primary
native build system.

Apply native warning policy through the internal `xpool_compiler_options`
target, linked privately by project-owned extension and test targets. Keep
`-Wall -Wextra` enabled for C++ and the CUDA host compiler, keep the selected
CUDA frontend diagnostics enabled, and treat warnings as errors by default.
`XPOOL_WARNINGS_AS_ERRORS=OFF` is a toolchain-diagnosis escape hatch, not a
normal development mode. Do not place project warning policy on public or
dependency interface targets such as `xpool_abi`.

Treat third-party include directories as CMake `SYSTEM` paths. Add a documented
header under `xpool/third-party/` only when it implements a real compatibility
boundary that cannot be expressed by dependency versioning or build-system
metadata. Do not add wrappers that merely rename an include or suppress
diagnostics, and do not add translation-unit-wide suppressions that also hide
xpool diagnostics.

Do not enable `--Wmissing-launch-bounds` or add `__launch_bounds__` merely to
satisfy a warning gate. Launch bounds affect CUDA code generation and require
workload-specific profiling and performance evidence.
