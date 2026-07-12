# Test Suite

xpool separates test behavior from execution requirements. Directories define
the testing layer; pytest markers describe required resources.

## Layers

### Unit

`tests/unit/` mirrors `src/xpool/`. Unit tests do not exercise native behavior,
launch subprocesses, or load model weights. Every pytest session still performs
the mandatory native-op preflight. Unit tests may use pinned SGLang concrete
types when validating local translation logic.

| Area | Purpose |
| --- | --- |
| `cli/` | Current command discovery, config output, readiness, and command dispatch. |
| `config/` | Schema validation, source precedence, environment allowlisting, and model lookup. |
| `devkit/` | Role-aware observer discovery, graph events, and transport snapshots. |
| `ops/` | Typed wrapper delegation and transport-handle validation. |
| `runtime/agent/test_heartbeat.py` | Shared heartbeat retry, registration-loss, and fatal-error behavior. |
| `runtime/atnagent/test_lifecycle.py` | AtnAgent construction, residency, and daemon health behavior. |
| `runtime/atnagent/test_transport_lifecycle.py` | Arena creation, publication, recovery, drain, and destruction. |
| `runtime/instance/` | Instance lifecycle, heartbeat recovery, transport attachment, and cleanup. |
| `service/` | HTTP client retry, readiness, protocol, and arena-acquisition behavior. |
| `utils/` | Background-thread and signal-handler lifecycle. |

### Integration

`tests/integration/` covers contracts that cross an xpool module boundary or a
pinned external/native interface without launching a serving engine.

| Area | Purpose | Requirements |
| --- | --- | --- |
| `native/test_runtime_ops.py` | Native role initialization and ABI behavior. | Built extension and CUDA |
| `native/test_arena_ops.py` | Arena attachment, detachment, and geometry behavior. | Built extension and CUDA |
| `native/test_shim_ops.py` | Shim meta dispatch and graph compilation behavior. | Built extension and CUDA |
| `native/test_instance_loopback.py` | Instance-local native shim dtype, validation, and CUDA graph behavior in isolated processes. | Built extension and CUDA |
| `native/test_atnagent_loopback.py` | AtnAgent persistent transport lifecycle, metadata, contention, and slot reuse. | Built extension and CUDA |
| `sglang/models/deepseek_v2/` | DeepSeek shim construction, translation, hooks, and model invariants. | SGLang |
| `sglang/plugin/` | Plugin lifecycle, transport sizing, and fail-closed behavior. | SGLang |
| `sglang/test_registry.py` | Adapter discovery and strict discovery failures. | SGLang |
| `sglang/test_server_args.py` | Server-argument gates and pinned forward-mode compatibility drift. | SGLang |
| `sglang/test_topology.py` | Attention and FFN topology derivation. | SGLang |
| `service/daemon/` | ASGI daemon registration, readiness, arena ownership, drain, and process identity. | POSIX subprocesses |
| `utils/test_procs.py` | Process identity rejects unreaped zombies. | Linux `/proc` and subprocesses |

### End To End

`tests/e2e/` contains real CUDA process and serving-engine workflows. Every file
uses a `test_e2e_*.py` name. Shim and transport files own independent setup and
eager baselines.

| Area | Evidence | Requirements |
| --- | --- | --- |
| `sglang/test_e2e_instance_loopback.py` | Instance-loopback token parity and observer events across eager, full graph, and piecewise graph paths. | Native extension, CUDA, SGLang, model weights |
| `sglang/test_e2e_atnagent_loopback.py` | The same graph evidence through daemon, AtnAgent, arena, and persistent transport. | Native extension, CUDA, SGLang, model weights |

Shared process, probe, fake, and native utilities live under `tests/harness/`
and are not collected as tests.

### Native CTest

| Source | Purpose |
| --- | --- |
| `cext/abi_test.cpp` | Stable wire values, sizes, offsets, type traits, and debug options. |
| `cext/utils/ring_queue_test.cu` | Host queue images and sequential/concurrent device queues. |
| `cext/transport/shutdown_test.cu` | Normal drain and abandoned-slot shutdown reclamation. |

## Markers

| Marker | Meaning |
| --- | --- |
| `requires_cuda(min_devices=1, bf16=False)` | Requires visible CUDA capacity and optional BF16 support. |
| `requires_config` | Requires an explicit config path in `XPOOL_CONFIG`. |
| `requires_model_weights(model_id)` | Resolves one model through that config and requires `config.json`. |

Resource markers are executable. Unavailable resources skip by default and fail
under `--strict-requirements`; malformed explicit configuration always fails.
The native extension has no marker because every pytest session preflights it
before collection. Pytest has no local-file or dotenv fallback; canonical
commands conditionally ask uv to load the repository-root `.env` before pytest
starts.

## Commands

```bash
# Load optional local pytest/E2E configuration through uv
if [ -f .env ]; then export UV_ENV_FILE="$PWD/.env"; fi

# Default unit, integration, and resource-eligible E2E suite
uv run pytest

# Individual layers and resources
uv run pytest tests/unit
uv run pytest tests/integration
uv run pytest --strict-requirements tests/integration/native -s

# Targeted E2E suites; strict mode is not required to execute them
XPOOL_CONFIG=/absolute/xpool.toml uv run pytest tests/e2e/sglang -s

# Requirement policy checks
env -u XPOOL_CONFIG uv run pytest tests/e2e/sglang
env -u XPOOL_CONFIG uv run pytest --strict-requirements -x tests/e2e/sglang

# Unit coverage report; no percentage threshold is enforced
uv run pytest tests/unit \
  --cov=xpool \
  --cov-branch \
  --cov-report=term-missing \
  --cov-report=xml:build/coverage/unit.xml
```

Use `pytest --collect-only` when an exact case count is needed. Build the native
extension through uv/scikit-build before running CTest or native E2E tests.
