#include "bindings.hpp"

#include <pybind11/stl.h>
#include <torch/csrc/utils/pybind.h>

#include <cstddef>
#include <cstdint>
#include <map>

#include <xpool/devkit/fabric_observer.hpp>
#include <xpool/devkit/ffn_routing_observer.hpp>
#include <xpool/devkit/graph_observer.hpp>
#include <xpool/devkit/transport_observer.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace {

using FabricAtnAgentEvent = xpool::hooks::FabricAtnAgentProtocolEvent::Kind;
using FabricCoordinatorEvent = xpool::hooks::FabricCoordinatorProtocolEvent::Kind;
using FabricFfnAgentEvent = xpool::hooks::FabricFfnAgentProtocolEvent::Kind;
using PublicationRecord = xpool::devkit::fabric_observer::Record;
using FabricRecordKind = xpool::devkit::fabric_observer::RecordKind;
using TransportEvent = xpool::hooks::TransportProtocolEventKind;
using TransportRecord = xpool::devkit::transport_observer::Record;

constexpr auto trace_kind(FabricAtnAgentEvent) { return FabricRecordKind::AtnAgent; }
constexpr auto trace_kind(FabricCoordinatorEvent) { return FabricRecordKind::Coordinator; }
constexpr auto trace_kind(FabricFfnAgentEvent) { return FabricRecordKind::FfnAgent; }

template <typename Event> void require_trace_event_family(const PublicationRecord &record, Event event) {
  if (record.kind() != trace_kind(event)) {
    throw py::value_error("Fabric trace event family does not match record kind");
  }
}

template <typename Event> bool trace_recorded(const PublicationRecord &record, Event event) {
  require_trace_event_family(record, event);
  return record.recorded(event);
}

template <typename Event> std::uint64_t trace_timestamp(const PublicationRecord &record, Event event) {
  require_trace_event_family(record, event);
  return record.timestamp(event);
}

void bind_fabric_observer_types(py::module_ &module) {
  py::enum_<FabricRecordKind>(module, "RecordKind", "State-machine kind represented by one Fabric trace record.")
      .value("ATNAGENT", FabricRecordKind::AtnAgent, "Trace emitted by one AtnAgent request path.")
      .value("COORDINATOR", FabricRecordKind::Coordinator, "Trace emitted by the Fabric Coordinator.")
      .value("FFNAGENT", FabricRecordKind::FfnAgent, "Trace emitted by one FfnAgent execution path.");

  py::enum_<FabricAtnAgentEvent>(module, "AtnAgentEvent", "Ordered semantic AtnAgent trace events.")
      .value("SUBMISSION_PREPARED", FabricAtnAgentEvent::SubmissionPrepared,
             "Submission record and invocation identity were prepared.")
      .value("SUBMISSION_PUBLISHED", FabricAtnAgentEvent::SubmissionPublished,
             "Local Submission publication completed; remote observation is not implied.")
      .value("ADMISSION_OBSERVED", FabricAtnAgentEvent::AdmissionObserved, "Matching Executor admission was observed.")
      .value("INPUT_READY_PUBLISHED", FabricAtnAgentEvent::InputReadyPublished,
             "Input payload publication completed for every selected FfnAgent.")
      .value("OUTPUT_COMMIT_OBSERVED", FabricAtnAgentEvent::OutputCommitObserved,
             "Matching output commit was observed.")
      .value("OUTPUT_ACKNOWLEDGEMENT_PUBLISHED", FabricAtnAgentEvent::OutputAcknowledgementPublished,
             "Local output acknowledgement publication completed.");

  py::enum_<FabricCoordinatorEvent>(module, "CoordinatorEvent", "Ordered semantic Coordinator trace events.")
      .value("ENQUEUED", FabricCoordinatorEvent::Enqueued, "Complete invocation entered the configured Scheduler.")
      .value("SCHEDULED", FabricCoordinatorEvent::Scheduled, "An Executor lease was selected.")
      .value("ADMISSION_PUBLISHED", FabricCoordinatorEvent::AdmissionPublished,
             "Admission publication completed for every AtnAgent.")
      .value("LANE_EXECUTION_PUBLISHED", FabricCoordinatorEvent::LaneExecutionPublished,
             "Lane execution publication completed for every selected FfnAgent.")
      .value("FFNAGENT_COMPLETIONS_OBSERVED", FabricCoordinatorEvent::FfnAgentCompletionsObserved,
             "Every matching FfnAgent completion was observed.")
      .value("OUTPUT_COMMIT_PUBLISHED", FabricCoordinatorEvent::OutputCommitPublished,
             "Output commit publication completed for every AtnAgent.")
      .value("OUTPUT_ACKNOWLEDGEMENTS_OBSERVED", FabricCoordinatorEvent::OutputAcknowledgementsObserved,
             "Every matching AtnAgent output acknowledgement was observed.")
      .value("LANE_RELEASED", FabricCoordinatorEvent::LaneReleased,
             "Scheduler ownership and the Executor Lane were released.");

  py::enum_<FabricFfnAgentEvent>(module, "FfnAgentEvent", "Ordered semantic FfnAgent trace events.")
      .value("LANE_EXECUTION_OBSERVED", FabricFfnAgentEvent::LaneExecutionObserved)
      .value("INPUT_READY_OBSERVED", FabricFfnAgentEvent::InputReadyObserved)
      .value("ROUTING_METADATA_PUBLISHED", FabricFfnAgentEvent::RoutingMetadataPublished)
      .value("ROUTING_METADATA_OBSERVED", FabricFfnAgentEvent::RoutingMetadataObserved)
      .value("COMPUTE_STARTED", FabricFfnAgentEvent::ComputeStarted)
      .value("COMPUTE_COMPLETED", FabricFfnAgentEvent::ComputeCompleted)
      .value("PARTIAL_READY_PUBLISHED", FabricFfnAgentEvent::PartialReadyPublished)
      .value("PEER_PARTIALS_READY_OBSERVED", FabricFfnAgentEvent::PeerPartialsReadyObserved)
      .value("COMPLETION_PUBLISHED", FabricFfnAgentEvent::CompletionPublished);

  py::class_<PublicationRecord>(module, "Record", "One PE-local Fabric trace record.")
      .def_property_readonly("local_trace_id", &PublicationRecord::local_trace_id, "PE-local monotonic trace identity.")
      .def_property_readonly("kind", &PublicationRecord::kind, "Site-specific trace state-machine kind.")
      .def_property_readonly("key", &PublicationRecord::key, "Traced invocation identity.")
      .def_property_readonly("layer_ordinal", &PublicationRecord::layer_ordinal, "Config-order FFN layer ordinal.")
      .def_property_readonly("payload_rows", &PublicationRecord::payload_rows, "Physical invocation payload row count.")
      .def_property_readonly("output_requirement", &PublicationRecord::output_requirement,
                             "Stable output-requirement value.")
      .def_property_readonly("dp_row_layout", &PublicationRecord::dp_row_layout,
                             "AtnAgent DP-row layout, when applicable.")
      .def_property_readonly("dp_rank_payload_rows", &PublicationRecord::dp_rank_payload_rows,
                             "AtnAgent rank-local physical row count, when applicable.")
      .def_property_readonly("forward_mode", &PublicationRecord::forward_mode,
                             "AtnAgent forward mode, when applicable.")
      .def_property_readonly("executor_lane_index", &PublicationRecord::executor_lane_index,
                             "Admitted or scheduled Executor index, when established.")
      .def_property_readonly("executor_lease_sequence", &PublicationRecord::executor_lease_sequence,
                             "Executor Lane lease identity, when established.")
      .def_property_readonly("ready_ticket", &PublicationRecord::ready_ticket, "FIFO ready ticket, when applicable.")
      .def_property_readonly("payload_row_capacity", &PublicationRecord::payload_row_capacity,
                             "FfnAgent-executed row capacity, when applicable.")
      .def_property_readonly("delivery", &PublicationRecord::delivery,
                             "FfnAgent-executed delivery branch, when applicable.")
      .def("recorded", &trace_recorded<FabricAtnAgentEvent>, py::arg("event"),
           "Return whether an AtnAgent event was recorded.")
      .def("recorded", &trace_recorded<FabricCoordinatorEvent>, py::arg("event"),
           "Return whether a Coordinator event was recorded.")
      .def("recorded", &trace_recorded<FabricFfnAgentEvent>, py::arg("event"),
           "Return whether an FfnAgent event was recorded.")
      .def("timestamp", &trace_timestamp<FabricAtnAgentEvent>, py::arg("event"),
           "Return an AtnAgent event's raw global-timer timestamp.")
      .def("timestamp", &trace_timestamp<FabricCoordinatorEvent>, py::arg("event"),
           "Return a Coordinator event's raw global-timer timestamp.")
      .def("timestamp", &trace_timestamp<FabricFfnAgentEvent>, py::arg("event"),
           "Return an FfnAgent event's raw global-timer timestamp.");

  py::class_<xpool::devkit::fabric_observer::ModelTopology>(
      module, "ModelTopology", "Static model topology retained in a Fabric trace snapshot.")
      .def_readonly("atn_tp_size", &xpool::devkit::fabric_observer::ModelTopology::atn_tp_size,
                    "Attention tensor-parallel participant count.")
      .def_readonly("atn_dp_size", &xpool::devkit::fabric_observer::ModelTopology::atn_dp_size,
                    "Attention data-parallel participant count.");

  py::class_<xpool::devkit::fabric_observer::Snapshot>(module, "Snapshot", "Host-owned Fabric trace snapshot.")
      .def_readonly("pe", &xpool::devkit::fabric_observer::Snapshot::pe, "PE that produced the snapshot.")
      .def_readonly("atnagent_count", &xpool::devkit::fabric_observer::Snapshot::atnagent_count,
                    "Number of AtnAgent participants.")
      .def_readonly("ffnagent_count", &xpool::devkit::fabric_observer::Snapshot::ffnagent_count,
                    "Number of FfnAgent participants.")
      .def_readonly("model_topologies", &xpool::devkit::fabric_observer::Snapshot::model_topologies,
                    "Config-order model topology projections.")
      .def_readonly("sequence", &xpool::devkit::fabric_observer::Snapshot::sequence,
                    "Next PE-local trace sequence.")
      .def_readonly("dropped", &xpool::devkit::fabric_observer::Snapshot::dropped,
                    "Number of traces dropped after capacity exhaustion.")
      .def_readonly("records", &xpool::devkit::fabric_observer::Snapshot::records,
                    "Retained PE-local trace records.");
}

void bind_transport_observer_types(py::module_ &module) {
  py::class_<TransportRecord>(module, "Record", "One native Transport trace record.")
      .def_property_readonly("trace_id", &TransportRecord::trace_id, "Process-local monotonic trace identity.")
      .def_property_readonly("payload_rows", &TransportRecord::payload_rows,
                             "Physical hidden-state rows carried by the request.")
      .def_property_readonly("layer_ordinal", &TransportRecord::layer_ordinal, "Config-order FFN layer ordinal.")
      .def_property_readonly("forward_mode", &TransportRecord::forward_mode, "Stable forward-mode value.")
      .def_property_readonly("output_requirement", &TransportRecord::output_requirement,
                             "Stable output-requirement value.")
      .def_property_readonly("dp_row_layout", &TransportRecord::dp_row_layout, "Stable physical DP-row layout value.")
      .def_property_readonly("result_code", &TransportRecord::result_code, "Stable FFN result code.")
      .def_property_readonly(
          "request_staging_started",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::RequestStagingStarted); },
          "Raw timestamp when Instance payload staging started.")
      .def_property_readonly(
          "request_staging_completed",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::RequestStagingCompleted); },
          "Raw timestamp when Instance payload staging completed.")
      .def_property_readonly(
          "request_published",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::RequestPublished); },
          "Raw timestamp when Instance request publication completed.")
      .def_property_readonly(
          "request_observed",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::RequestObserved); },
          "Raw timestamp when AtnAgent observed the published request.")
      .def_property_readonly(
          "execution_started",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::ExecutionStarted); },
          "Raw timestamp when AtnAgent request execution started.")
      .def_property_readonly(
          "execution_completed",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::ExecutionCompleted); },
          "Raw timestamp when the selected execution path completed.")
      .def_property_readonly(
          "result_published",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::ResultPublished); },
          "Raw timestamp when AtnAgent published the request result.")
      .def_property_readonly(
          "result_observed",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::ResultObserved); },
          "Raw timestamp when Instance observed the effective request result.")
      .def_property_readonly(
          "output_copied", [](const TransportRecord &record) { return record.timestamp(TransportEvent::OutputCopied); },
          "Raw timestamp when Instance finished copying the output payload.")
      .def_property_readonly(
          "result_acknowledged",
          [](const TransportRecord &record) { return record.timestamp(TransportEvent::ResultAcknowledged); },
          "Raw timestamp when Instance acknowledged successful completion.")
      .def_property_readonly(
          "closed", [](const TransportRecord &record) { return record.timestamp(TransportEvent::Closed); },
          "Raw timestamp when the Transport request entered terminal closure.");

  py::class_<xpool::devkit::transport_observer::EndpointSnapshot>(
      module, "EndpointSnapshot", "One endpoint's process-local Transport trace snapshot.")
      .def_readonly("instance_index", &xpool::devkit::transport_observer::EndpointSnapshot::instance_index,
                    "Stable instance index associated with the endpoint.")
      .def_readonly("instance_rank", &xpool::devkit::transport_observer::EndpointSnapshot::instance_rank,
                    "Instance rank associated with the endpoint.")
      .def_readonly("sequence", &xpool::devkit::transport_observer::EndpointSnapshot::sequence,
                    "Next process-local trace sequence.")
      .def_readonly("dropped", &xpool::devkit::transport_observer::EndpointSnapshot::dropped,
                    "Number of traces dropped after capacity exhaustion.")
      .def_readonly("records", &xpool::devkit::transport_observer::EndpointSnapshot::records,
                    "Retained Transport trace records.");

  py::class_<xpool::devkit::transport_observer::Snapshot>(module, "Snapshot",
                                                          "Process-local Transport observation.")
      .def_readonly("endpoints", &xpool::devkit::transport_observer::Snapshot::endpoints,
                    "Every open Transport endpoint in this process.");
}

py::dict python_node_counts(const std::map<cudaGraphNodeType, std::size_t> &counts) {
  auto result = py::dict{};
  for (const auto &[kind, count] : counts) {
    result[py::int_(static_cast<int>(kind))] = count;
  }
  return result;
}

} // namespace

namespace xpool::bindings {

void bind_devkit(py::module_ &module) {
  auto devkit = module.def_submodule("devkit", "Development-only native observation interfaces.");
  auto fabric = devkit.def_submodule("fabric_observer", "Per-PE Fabric protocol observation.");
  auto graph = devkit.def_submodule("graph_observer", "FfnAgent CUDA Graph observation.");
  auto routing = devkit.def_submodule("ffn_routing_observer", "FfnAgent routing observation.");
  auto transport = devkit.def_submodule("transport_observer", "Per-process Transport protocol observation.");

  bind_fabric_observer_types(fabric);
  bind_transport_observer_types(transport);

  py::class_<xpool::devkit::graph_observer::PrimaryGraphSnapshot>(graph, "PrimaryGraphSnapshot",
                                                                  "Actual Primary Graph structure and binding sites.")
      .def_property_readonly(
          "node_counts",
          [](const xpool::devkit::graph_observer::PrimaryGraphSnapshot &value) {
            return python_node_counts(value.node_counts);
          },
          "Observed Primary Graph node counts keyed by CUDA node type.")
      .def_readonly("binding_site_count", &xpool::devkit::graph_observer::PrimaryGraphSnapshot::binding_site_count,
                    "Observed device-pointer binding site count.");

  py::class_<xpool::devkit::graph_observer::LaneGraphSnapshot>(graph, "LaneGraphSnapshot",
                                                               "Actual recursive Lane Graph structure.")
      .def_property_readonly(
          "node_counts",
          [](const xpool::devkit::graph_observer::LaneGraphSnapshot &value) {
            return python_node_counts(value.node_counts);
          },
          "Observed recursive Lane Graph node counts keyed by CUDA node type.")
      .def_readonly("compute_branch_count", &xpool::devkit::graph_observer::LaneGraphSnapshot::compute_branch_count,
                    "Compute branches reachable from this Lane Graph.")
      .def_readonly("delivery_branch_count", &xpool::devkit::graph_observer::LaneGraphSnapshot::delivery_branch_count,
                    "Delivery branches reachable from this Lane Graph.");

  py::class_<xpool::devkit::graph_observer::Snapshot>(graph, "Snapshot", "Immutable native Graph Observer snapshot.")
      .def_readonly("primary_graphs", &xpool::devkit::graph_observer::Snapshot::primary_graphs,
                    "Actual structure for every installed Primary Graph.")
      .def_readonly("lane_graphs", &xpool::devkit::graph_observer::Snapshot::lane_graphs,
                    "Actual structure for every installed Executor Lane Graph.");

  py::class_<xpool::devkit::ffn_routing_observer::Record>(routing, "Record",
                                                          "One compact Host-owned semantic routing record.")
      .def_readonly("key", &xpool::devkit::ffn_routing_observer::Record::key,
                    "Invocation whose Router result was consumed.")
      .def_readonly("layer_ordinal", &xpool::devkit::ffn_routing_observer::Record::layer_ordinal,
                    "Config-order FFN layer ordinal.")
      .def_readonly("topk_ids", &xpool::devkit::ffn_routing_observer::Record::topk_ids, "Compact CPU int32 Expert ids.")
      .def_readonly("topk_weights", &xpool::devkit::ffn_routing_observer::Record::topk_weights,
                    "Compact CPU float32 Expert weights.");

  py::class_<xpool::devkit::ffn_routing_observer::Snapshot>(routing, "Snapshot",
                                                            "Repeatable Host-owned routing readout.")
      .def_readonly("sequence", &xpool::devkit::ffn_routing_observer::Snapshot::sequence, "Reservation-attempt count.")
      .def_readonly("dropped", &xpool::devkit::ffn_routing_observer::Snapshot::dropped, "Dropped record count.")
      .def_property_readonly(
          "records",
          [](const xpool::devkit::ffn_routing_observer::Snapshot &snapshot) {
            auto records = py::tuple(snapshot.records.size());
            for (auto index = std::size_t{0}; index < snapshot.records.size(); ++index) {
              records[index] = py::cast(snapshot.records[index]);
            }
            return records;
          },
          "Retained routing records in reservation order.");

  fabric.def(
      "read",
      []() {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.devkit.fabric_observer.read");
        return xpool::devkit::fabric_observer::read();
      },
      "Return the current process-local Fabric snapshot, if enabled.", py::call_guard<py::gil_scoped_release>());
  fabric.def("allocation_bytes", &xpool::devkit::fabric_observer::allocation_bytes, py::arg("instance_count"),
             py::arg("executor_lane_count"), "Return the optional process-local Fabric Observer allocation size.");

  graph.def(
      "read",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent,
                                                      "xpool.native.devkit.graph_observer.read");
        return xpool::devkit::graph_observer::read();
      },
      "Return the installed process-local Graph snapshot, if enabled.");

  routing.def(
      "read",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent,
                                                      "xpool.native.devkit.ffn_routing_observer.read");
        return xpool::devkit::ffn_routing_observer::read();
      },
      "Return the current process-local routing snapshot, if enabled.", py::call_guard<py::gil_scoped_release>());
  routing.def("allocation_bytes", &xpool::devkit::ffn_routing_observer::allocation_bytes, py::arg("element_capacity"),
              "Return the optional process-local Routing Observer allocation size.");

  transport.def(
      "read",
      []() {
        xpool::RuntimeState::singleton().require_role(
            {xpool::RuntimeRole::Instance, xpool::RuntimeRole::AtnAgent},
            "xpool.native.devkit.transport_observer.read");
        return xpool::devkit::transport_observer::read();
      },
      "Return the process-local Transport snapshot, if enabled.", py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
