# CrossPool Design Map

The documents in this directory describe the implemented and accepted CrossPool
system. They are current-state references, not task plans or design history.

- [System Overview](overview.md) defines the supported boundary, process roles,
  public interface ownership, and cross-module invariants.
- [Control Plane](control-plane.md) defines configuration, generation planning,
  placement, memory admission, and lifecycle coordination.
- [Transport](transport.md) defines the Instance-to-AtnAgent mailbox.
- [Fabric](fabric.md) defines distributed invocation, coordination, delivery,
  scheduling, and failure.
- [FFN Execution](ffn-execution.md) defines model materialization, operators,
  GraphTemplates, and Executor Lanes.
- [Elastic KV-cache Pooling](elastic-kv-cache.md) defines stable attention-side
  KV storage, logical admission, physical capacity coordination, and
  reclamation.
- [Devkit](devkit.md) defines Hook Points and observer evidence.
- [Qualification](qualification.md) defines readiness and acceptance evidence.

The root [CONTEXT.md](../../CONTEXT.md) owns domain terminology. Active target
changes live under `docs/plans/<task>/README.md`; completed plans remain until
user-confirmed cleanup through `write-design`. A relevant active plan is a scoped
delta over these current documents; source declarations and generated native stubs remain
authoritative for exact interfaces.

The [Roadmap](../plans/README.md) maps long-term candidate workstreams and their
relationships without treating them as accepted architecture or active plans.
