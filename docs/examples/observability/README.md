# Synthetic observation traces

`bundle/` is a small native protobuf example generated from deterministic
observation fixtures. It contains Graph, Transport, Fabric and Routing events,
an offset-calibrated Device domain, independently displayed uncalibrated
domains, and a Routing attachment recorded as not collected.

Sources named `synthetic/producer-*.xpool` are illustrative provenance references,
not bundled raw recordings. The fixture clock samples are synthetic, not serving
or hardware accuracy measurements. The bundle's display origins and file hashes
are recorded in `bundle.json`; calibration and quality evidence remain available
both as reports and as trace arguments.

```bash
uv run xpool trace setup
uv run xpool trace query --input-dir docs/examples/observability/bundle \
  --output-dir example-query-results
uv run xpool trace query --input-dir docs/examples/observability/bundle \
  --output-dir example-identities --sql docs/examples/observability/identities.sql
```

Open an individual `.pftrace` member with Perfetto UI. Unified and raw members
are independent views; user SQL executes separately on each. Built-in queries
deduplicate equivalent evidence by identity. See the
[trace tutorial](../../tutorials/observation-traces.md) for timing interpretation,
raw input manifests, error behavior and tool qualification.
