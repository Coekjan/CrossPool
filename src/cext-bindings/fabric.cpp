#include "bindings.hpp"

#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <xpool/abi.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace {

constexpr auto trace_kind(xpool::fabric::AtnAgentTraceEvent) {
  return xpool::fabric::FabricTraceKind::AtnAgent;
}

constexpr auto trace_kind(xpool::fabric::CoordinatorTraceEvent) {
  return xpool::fabric::FabricTraceKind::Coordinator;
}

constexpr auto trace_kind(xpool::fabric::ExecutionTraceEvent) {
  return xpool::fabric::FabricTraceKind::Execution;
}

template <typename Event>
void require_trace_event_family(const xpool::fabric::FabricTraceRecord &record, Event event) {
  if (record.kind() != trace_kind(event)) {
    throw py::value_error("Fabric trace event family does not match record kind");
  }
}

template <typename Event>
bool trace_recorded(const xpool::fabric::FabricTraceRecord &record, Event event) {
  require_trace_event_family(record, event);
  return record.recorded(event);
}

template <typename Event>
std::uint64_t trace_timestamp(const xpool::fabric::FabricTraceRecord &record, Event event) {
  require_trace_event_family(record, event);
  return record.timestamp(event);
}

} // namespace

namespace xpool::bindings {

void bind_fabric(py::module_ &module) {
  py::class_<xpool::fabric::FabricLayerMetadata>(
      module, "FabricLayerMetadata", "One ordered FFN layer supplied to native Fabric join.")
      .def(py::init([](std::size_t layer_id, std::int64_t kind) {
             TORCH_CHECK(xpool::fabric::FfnLayerKind::is_valid(kind),
                         "xpool FabricLayerMetadata received an invalid kind");
             return xpool::fabric::FabricLayerMetadata{
                 .layer_id = layer_id,
                 .kind = xpool::fabric::FfnLayerKind{kind},
             };
           }),
           py::arg("layer_id"), py::arg("kind"))
      .def_readonly("layer_id", &xpool::fabric::FabricLayerMetadata::layer_id,
                    "Concrete decoder layer identifier.")
      .def_property_readonly("kind", [](const xpool::fabric::FabricLayerMetadata &metadata) {
        return static_cast<std::uint32_t>(metadata.kind.value());
      }, "Stable FFN layer-kind value.");

  py::class_<xpool::fabric::FabricModelMetadata>(
      module, "FabricModelMetadata", "One model's capacities, topology, and layers supplied to native Fabric join.")
      .def(py::init([](std::size_t max_decode_rows, std::size_t max_prefill_rows, std::int64_t dtype,
                       std::size_t hidden_size, std::size_t atn_tp_size, std::size_t atn_dp_size,
                       std::vector<xpool::fabric::FabricLayerMetadata> layers) {
             TORCH_CHECK(xpool::abi::TensorDType::is_valid(dtype),
                         "xpool FabricModelMetadata received an invalid dtype");
             return xpool::fabric::FabricModelMetadata{
                 .max_decode_rows = max_decode_rows,
                 .max_prefill_rows = max_prefill_rows,
                 .dtype = xpool::abi::TensorDType{dtype},
                 .hidden_size = hidden_size,
                 .atn_tp_size = atn_tp_size,
                 .atn_dp_size = atn_dp_size,
                 .layers = std::move(layers),
             };
           }),
           py::arg("max_decode_rows"), py::arg("max_prefill_rows"), py::arg("dtype"), py::arg("hidden_size"),
           py::arg("atn_tp_size"), py::arg("atn_dp_size"), py::arg("layers"))
      .def_readonly("max_decode_rows", &xpool::fabric::FabricModelMetadata::max_decode_rows,
                    "Maximum physical Decode rows per invocation.")
      .def_readonly("max_prefill_rows", &xpool::fabric::FabricModelMetadata::max_prefill_rows,
                    "Maximum physical Prefill rows per invocation.")
      .def_property_readonly("dtype", [](const xpool::fabric::FabricModelMetadata &metadata) {
        return static_cast<std::uint32_t>(metadata.dtype.value());
      }, "Stable hidden-state tensor dtype value.")
      .def_readonly("hidden_size", &xpool::fabric::FabricModelMetadata::hidden_size,
                    "Model hidden width in elements.")
      .def_readonly("atn_tp_size", &xpool::fabric::FabricModelMetadata::atn_tp_size,
                    "Attention tensor-parallel participant count.")
      .def_readonly("atn_dp_size", &xpool::fabric::FabricModelMetadata::atn_dp_size,
                    "Attention data-parallel participant count.")
      .def_readonly("layers", &xpool::fabric::FabricModelMetadata::layers,
                    "Ordered FFN layer metadata.");

      py::class_<xpool::fabric::FfnSchedulerPolicy>(module, "FfnSchedulerPolicy",
                                                "Immutable native Fabric scheduling policy.")
      .def_static("fifo", &xpool::fabric::FfnSchedulerPolicy::fifo,
                  "Create the deterministic FIFO scheduling policy.")
      .def_static("random", &xpool::fabric::FfnSchedulerPolicy::random, py::arg("seed"),
                  "Create the random scheduling policy from a nonzero seed.");

  py::class_<xpool::fabric::FabricJoinMetadata>(
      module, "FabricJoinMetadata", "Complete semantic input for one native Fabric participant.")
      .def(py::init([](const std::string &uid, int pe, std::size_t atnagent_count,
                       std::size_t ffnagent_count, std::size_t executor_count,
                       xpool::fabric::FfnSchedulerPolicy scheduler_policy,
                       std::vector<xpool::fabric::FabricModelMetadata> models) {
             return xpool::fabric::FabricJoinMetadata{
                 .uid = xpool::fabric::FabricUid::decode(uid),
                 .pe = pe,
                 .atnagent_count = atnagent_count,
                 .ffnagent_count = ffnagent_count,
                 .executor_count = executor_count,
                 .scheduler_policy = std::move(scheduler_policy),
                 .models = std::move(models),
             };
           }),
           py::arg("uid"), py::arg("pe"), py::arg("atnagent_count"), py::arg("ffnagent_count"),
           py::arg("executor_count"), py::arg("scheduler_policy"), py::arg("models"))
      .def_property_readonly("uid", [](const xpool::fabric::FabricJoinMetadata &metadata) {
        return metadata.uid.encode();
      }, "Opaque NVSHMEM bootstrap identity.")
      .def_readonly("pe", &xpool::fabric::FabricJoinMetadata::pe,
                    "This participant's canonical PE index.")
      .def_readonly("atnagent_count", &xpool::fabric::FabricJoinMetadata::atnagent_count,
                    "Number of AtnAgent participants.")
      .def_readonly("ffnagent_count", &xpool::fabric::FabricJoinMetadata::ffnagent_count,
                    "Number of FfnAgent participants.")
      .def_readonly("executor_count", &xpool::fabric::FabricJoinMetadata::executor_count,
                    "Number of distributed FFN Executors.")
      .def_readonly("scheduler_policy", &xpool::fabric::FabricJoinMetadata::scheduler_policy,
                    "Coordinator scheduling policy.")
      .def_readonly("models", &xpool::fabric::FabricJoinMetadata::models,
                    "Config-order model metadata.");

  py::class_<xpool::fabric::FfnInvocationKey>(module, "FfnInvocationKey", "Identity of one FFN invocation.")
      .def_readonly("model_index", &xpool::fabric::FfnInvocationKey::model_index,
                    "Config-order model index.")
      .def_readonly("invocation_sequence", &xpool::fabric::FfnInvocationKey::invocation_sequence,
                    "Model-local invocation sequence.");

  py::class_<xpool::fabric::FabricFailurePayload>(module, "FabricFailurePayload",
                                                  "Immutable canonical Fabric failure payload.")
      .def_readonly("result_code", &xpool::fabric::FabricFailurePayload::result_code,
                    "Stable FFN result code.")
      .def_readonly("origin_pe", &xpool::fabric::FabricFailurePayload::origin_pe,
                    "PE that first claimed the failure.")
      .def_readonly("key", &xpool::fabric::FabricFailurePayload::key,
                    "Failed invocation identity.")
      .def_readonly("layer_ordinal", &xpool::fabric::FabricFailurePayload::layer_ordinal,
                    "Failed config-order FFN layer ordinal.");

  py::class_<xpool::fabric::FabricFailure>(module, "FabricFailure", "Published canonical Fabric failure.")
      .def_readonly("claim", &xpool::fabric::FabricFailure::claim,
                    "Canonical failure claim word.")
      .def_readonly("publication", &xpool::fabric::FabricFailure::publication,
                    "Canonical failure publication word.")
      .def_readonly("payload", &xpool::fabric::FabricFailure::payload,
                    "Published immutable failure payload.");

  py::enum_<xpool::fabric::FabricTraceKind>(module, "FabricTraceKind",
                                           "State-machine kind represented by one Fabric trace record.")
      .value("ATNAGENT", xpool::fabric::FabricTraceKind::AtnAgent,
             "Trace emitted by one AtnAgent request path.")
      .value("COORDINATOR", xpool::fabric::FabricTraceKind::Coordinator,
             "Trace emitted by the Fabric Coordinator.")
      .value("EXECUTION", xpool::fabric::FabricTraceKind::Execution,
             "Trace emitted by one FfnAgent Execution path.");

  py::enum_<xpool::fabric::AtnAgentTraceEvent>(module, "AtnAgentTraceEvent",
                                               "Ordered semantic AtnAgent trace events.")
      .value("SUBMISSION_PREPARED", xpool::fabric::AtnAgentTraceEvent::SubmissionPrepared,
             "Submission record and invocation identity were prepared.")
      .value("DECODE_INPUT_STAGED", xpool::fabric::AtnAgentTraceEvent::DecodeInputStaged,
             "Decode input was staged in model payload storage.")
      .value("SUBMISSION_PUBLISHED", xpool::fabric::AtnAgentTraceEvent::SubmissionPublished,
             "Local Submission publication completed; remote observation is not implied.")
      .value("ADMISSION_OBSERVED", xpool::fabric::AtnAgentTraceEvent::AdmissionObserved,
             "Matching Executor admission was observed.")
      .value("PREFILL_INPUT_STAGED", xpool::fabric::AtnAgentTraceEvent::PrefillInputStaged,
             "Prefill input was staged in leased Executor storage.")
      .value("PREFILL_INPUT_PUBLISHED", xpool::fabric::AtnAgentTraceEvent::PrefillInputPublished,
             "Local Prefill input publication calls completed for every FfnAgent.")
      .value("RESULT_OBSERVED", xpool::fabric::AtnAgentTraceEvent::ResultObserved,
             "Matching distributed result was observed.")
      .value("OUTPUT_PREPARED", xpool::fabric::AtnAgentTraceEvent::OutputPrepared,
             "Rank-local Transport output was materialized.")
      .value("TRANSPORT_EVALUATED_PUBLISHED",
             xpool::fabric::AtnAgentTraceEvent::TransportEvaluatedPublished,
             "Successful Transport result was published.")
      .value("ACKNOWLEDGEMENT_PUBLISHED", xpool::fabric::AtnAgentTraceEvent::AcknowledgementPublished,
             "Local result-acknowledgement publication completed.");

  py::enum_<xpool::fabric::CoordinatorTraceEvent>(module, "CoordinatorTraceEvent",
                                                   "Ordered semantic Coordinator trace events.")
      .value("ENQUEUED", xpool::fabric::CoordinatorTraceEvent::Enqueued,
             "Complete invocation entered the configured Scheduler.")
      .value("SCHEDULED", xpool::fabric::CoordinatorTraceEvent::Scheduled,
             "An Executor lease was selected.")
      .value("ADMISSIONS_PUBLISHED", xpool::fabric::CoordinatorTraceEvent::AdmissionsPublished,
             "Local Admission publication calls completed for every AtnAgent.")
      .value("INVOCATIONS_PUBLISHED", xpool::fabric::CoordinatorTraceEvent::InvocationsPublished,
             "Local Invocation publication calls completed for every FfnAgent.")
      .value("COMPLETIONS_OBSERVED", xpool::fabric::CoordinatorTraceEvent::CompletionsObserved,
             "Every matching FfnAgent completion was observed.")
      .value("RESULTS_PUBLISHED", xpool::fabric::CoordinatorTraceEvent::ResultsPublished,
             "Local Result publication calls completed for every AtnAgent.")
      .value("ACKNOWLEDGEMENTS_OBSERVED", xpool::fabric::CoordinatorTraceEvent::AcknowledgementsObserved,
             "Every matching AtnAgent acknowledgement was observed.")
      .value("SCHEDULER_RELEASED", xpool::fabric::CoordinatorTraceEvent::SchedulerReleased,
             "Scheduler entry and Executor lease were released.");

  py::enum_<xpool::fabric::ExecutionTraceEvent>(module, "ExecutionTraceEvent",
                                                 "Ordered semantic FfnAgent Execution trace events.")
      .value("INVOCATION_OBSERVED", xpool::fabric::ExecutionTraceEvent::InvocationObserved,
             "Matching invocation metadata was observed.")
      .value("DECODE_INPUT_PULL_STARTED", xpool::fabric::ExecutionTraceEvent::DecodeInputPullStarted,
             "Decode input pull from the Input Publisher began.")
      .value("DECODE_INPUT_PULL_COMPLETED", xpool::fabric::ExecutionTraceEvent::DecodeInputPullCompleted,
             "Decode input pull completed.")
      .value("PREFILL_INPUT_READY_OBSERVED", xpool::fabric::ExecutionTraceEvent::PrefillInputReadyObserved,
             "Prefill input-ready publication was observed.")
      .value("EXECUTION_STARTED", xpool::fabric::ExecutionTraceEvent::ExecutionStarted,
             "FFN execution or configured loopback began.")
      .value("EXECUTION_COMPLETED", xpool::fabric::ExecutionTraceEvent::ExecutionCompleted,
             "FFN execution or configured loopback completed.")
      .value("COMPLETION_PUBLISHED", xpool::fabric::ExecutionTraceEvent::CompletionPublished,
             "Local Completion publication to the Coordinator completed.");

  py::class_<xpool::fabric::FabricTraceRecord>(module, "FabricTraceRecord", "One PE-local Fabric trace record.")
      .def_property_readonly("local_trace_id", &xpool::fabric::FabricTraceRecord::local_trace_id,
                             "PE-local monotonic trace identity.")
      .def_property_readonly("kind", &xpool::fabric::FabricTraceRecord::kind,
                             "Site-specific trace state-machine kind.")
      .def_property_readonly("key", &xpool::fabric::FabricTraceRecord::key,
                             "Traced invocation identity.")
      .def_property_readonly("layer_ordinal", &xpool::fabric::FabricTraceRecord::layer_ordinal,
                             "Config-order FFN layer ordinal.")
      .def_property_readonly("layer_id", &xpool::fabric::FabricTraceRecord::layer_id,
                             "Concrete decoder layer identifier.")
      .def_property_readonly("result_handoff", &xpool::fabric::FabricTraceRecord::result_handoff,
                             "Stable result-handoff value.")
      .def_property_readonly("dp_padding_mode", &xpool::fabric::FabricTraceRecord::dp_padding_mode,
                             "Stable DP-padding value.")
      .def_property_readonly("submission_payload_rows",
                             &xpool::fabric::FabricTraceRecord::submission_payload_rows,
                             "AtnAgent submission rows, when applicable.")
      .def_property_readonly("invocation_payload_rows",
                             &xpool::fabric::FabricTraceRecord::invocation_payload_rows,
                             "Coordinator or Execution invocation rows, when applicable.")
      .def_property_readonly("local_token_count", &xpool::fabric::FabricTraceRecord::local_token_count,
                             "AtnAgent rank-local live-token count, when applicable.")
      .def_property_readonly("forward_mode", &xpool::fabric::FabricTraceRecord::forward_mode,
                             "AtnAgent forward mode, when applicable.")
      .def_property_readonly("input_pe", &xpool::fabric::FabricTraceRecord::input_pe,
                             "Derived Input Publisher PE, when applicable.")
      .def_property_readonly("executor_index", &xpool::fabric::FabricTraceRecord::executor_index,
                             "Admitted or scheduled Executor index, when established.")
      .def_property_readonly("execution_mode", &xpool::fabric::FabricTraceRecord::execution_mode,
                             "Derived execution mode, when established.")
      .def_property_readonly("result_contribution", &xpool::fabric::FabricTraceRecord::result_contribution,
                             "AtnAgent result contribution, when observed.")
      .def_property_readonly("ready_ticket", &xpool::fabric::FabricTraceRecord::ready_ticket,
                             "FIFO ready ticket, when applicable.")
      .def("recorded", &trace_recorded<xpool::fabric::AtnAgentTraceEvent>,
           py::arg("event"), "Return whether an AtnAgent event was recorded.")
      .def("recorded", &trace_recorded<xpool::fabric::CoordinatorTraceEvent>,
           py::arg("event"), "Return whether a Coordinator event was recorded.")
      .def("recorded", &trace_recorded<xpool::fabric::ExecutionTraceEvent>,
           py::arg("event"), "Return whether an Execution event was recorded.")
      .def("timestamp", &trace_timestamp<xpool::fabric::AtnAgentTraceEvent>,
           py::arg("event"), "Return an AtnAgent event's raw global-timer timestamp.")
      .def("timestamp", &trace_timestamp<xpool::fabric::CoordinatorTraceEvent>,
           py::arg("event"), "Return a Coordinator event's raw global-timer timestamp.")
      .def("timestamp", &trace_timestamp<xpool::fabric::ExecutionTraceEvent>,
           py::arg("event"), "Return an Execution event's raw global-timer timestamp.");

  py::class_<xpool::fabric::FabricTraceModelTopology>(module, "FabricTraceModelTopology",
                                                       "Static model topology retained in a Fabric trace snapshot.")
      .def_readonly("atn_tp_size", &xpool::fabric::FabricTraceModelTopology::atn_tp_size,
                    "Attention tensor-parallel participant count.")
      .def_readonly("atn_dp_size", &xpool::fabric::FabricTraceModelTopology::atn_dp_size,
                    "Attention data-parallel participant count.");

  py::class_<xpool::fabric::FabricTraceSnapshot>(module, "FabricTraceSnapshot",
                                                  "Host-owned Fabric trace snapshot.")
      .def_readonly("pe", &xpool::fabric::FabricTraceSnapshot::pe,
                    "PE that produced the snapshot.")
      .def_readonly("atnagent_count", &xpool::fabric::FabricTraceSnapshot::atnagent_count,
                    "Number of AtnAgent participants.")
      .def_readonly("ffnagent_count", &xpool::fabric::FabricTraceSnapshot::ffnagent_count,
                    "Number of FfnAgent participants.")
      .def_readonly("model_topologies", &xpool::fabric::FabricTraceSnapshot::model_topologies,
                    "Config-order model topology projections.")
      .def_readonly("sequence", &xpool::fabric::FabricTraceSnapshot::sequence,
                    "Next PE-local trace sequence.")
      .def_readonly("dropped", &xpool::fabric::FabricTraceSnapshot::dropped,
                    "Number of traces dropped after capacity exhaustion.")
      .def_readonly("records", &xpool::fabric::FabricTraceSnapshot::records,
                    "Retained PE-local trace records.");

  auto fabric = module.def_submodule("fabric", "Native NVSHMEM Fabric control and lifecycle functions.");
  fabric.def(
      "create_uid",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Daemon,
                                                      "xpool.native.fabric.create_uid");
        return xpool::fabric::create_uid().encode();
      },
      "Create an opaque daemon-owned NVSHMEM bootstrap identity.");
  fabric.def(
      "join",
      [](const xpool::fabric::FabricJoinMetadata &metadata) {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.join");
        xpool::fabric::FabricRuntime::singleton().join(
            xpool::RuntimeState::singleton().cuda_device("xpool.native.fabric.join"), metadata);
      },
      py::arg("metadata"), "Join the exact Fabric generation described by metadata.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "activate",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent,
                                                      "xpool.native.fabric.activate");
        xpool::fabric::FabricRuntime::singleton().activate();
      },
      "Launch the FfnAgent coordinator and Executor resident kernels.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "check_health",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.check_health");
        xpool::fabric::FabricRuntime::singleton().check_health();
      },
      "Raise when the active Fabric runtime has failed.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "drain_async",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.drain_async");
        xpool::fabric::FabricRuntime::singleton().drain_async();
      },
      "Begin asynchronous cooperative Fabric drain.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "drain_pending",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.drain_pending");
        return xpool::fabric::FabricRuntime::singleton().drain_pending();
      },
      "Return whether asynchronous Fabric drain remains pending.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "failure",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.failure");
        return xpool::fabric::FabricRuntime::singleton().failure();
      },
      "Return the published canonical Fabric failure, if any.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "read_trace",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.read_trace");
        return xpool::fabric::FabricRuntime::singleton().read_trace();
      },
      "Copy the local Fabric trace buffer into a host-owned snapshot, if enabled.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "shutdown",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent}, "xpool.native.fabric.shutdown");
        xpool::fabric::FabricRuntime::singleton().shutdown();
      },
      "Finalize NVSHMEM and release all local Fabric resources.",
      py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
