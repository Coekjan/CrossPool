#include "bindings.hpp"

#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <xpool/runtime.hpp>
#include <xpool/transport/atnagent.hpp>
#include <xpool/transport/instance.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_transport(py::module_ &module) {
  py::class_<xpool::transport::TransportTraceRecord>(module, "TransportTraceRecord",
                                                     "One native transport trace record.")
      .def_property_readonly("trace_id", &xpool::transport::TransportTraceRecord::trace_id,
                             "Arena-local monotonic trace identity.")
      .def_property_readonly("payload_rows", &xpool::transport::TransportTraceRecord::payload_rows,
                             "Physical hidden-state rows carried by the request.")
      .def_property_readonly("layer_ordinal", &xpool::transport::TransportTraceRecord::layer_ordinal,
                             "Config-order FFN layer ordinal.")
      .def_property_readonly("forward_mode", &xpool::transport::TransportTraceRecord::forward_mode,
                             "Stable forward-mode value.")
      .def_property_readonly("result_handoff", &xpool::transport::TransportTraceRecord::result_handoff,
                             "Stable result-handoff value.")
      .def_property_readonly("dp_padding_mode", &xpool::transport::TransportTraceRecord::dp_padding_mode,
                             "Stable DP-padding value.")
      .def_property_readonly("result_code", &xpool::transport::TransportTraceRecord::result_code,
                             "Stable FFN result code.")
      .def_property_readonly("staging_started",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::StagingStarted);
                             },
                             "Raw timestamp when Instance input staging began.")
      .def_property_readonly("staging_completed",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::StagingCompleted);
                             },
                             "Raw timestamp when Instance input staging completed.")
      .def_property_readonly("published",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::Published);
                             },
                             "Raw timestamp immediately before Instance request release-store.")
      .def_property_readonly("published_observed",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::PublishedObserved);
                             },
                             "Raw timestamp when AtnAgent observed publication.")
      .def_property_readonly("execution_started",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::ExecutionStarted);
                             },
                             "Raw timestamp when AtnAgent execution began.")
      .def_property_readonly("execution_admitted",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::ExecutionAdmitted);
                             },
                             "Raw timestamp when Fabric admitted execution.")
      .def_property_readonly("execution_completed",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::ExecutionCompleted);
                             },
                             "Raw timestamp when AtnAgent execution completed.")
      .def_property_readonly("evaluated",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::Evaluated);
                             },
                             "Raw timestamp immediately before AtnAgent result release-store.")
      .def_property_readonly("evaluated_observed",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::EvaluatedObserved);
                             },
                             "Raw timestamp when Instance observed the result.")
      .def_property_readonly("output_copied",
                             [](const xpool::transport::TransportTraceRecord &record) {
                               return record.timestamp(xpool::transport::TransportTraceEvent::OutputCopied);
                             },
                             "Raw timestamp when Instance output copy completed.")
      .def_property_readonly("acknowledged", [](const xpool::transport::TransportTraceRecord &record) {
        return record.timestamp(xpool::transport::TransportTraceEvent::Acknowledged);
      }, "Raw timestamp immediately before Instance acknowledgement release-store.")
      .def_property_readonly("closed", [](const xpool::transport::TransportTraceRecord &record) {
        return record.timestamp(xpool::transport::TransportTraceEvent::Closed);
      }, "Raw timestamp when the request entered the terminal Closed state.");

  py::class_<xpool::transport::TransportTraceSnapshot>(module, "TransportTraceSnapshot",
                                                       "Host-owned transport trace snapshot.")
      .def_readonly("sequence", &xpool::transport::TransportTraceSnapshot::sequence,
                    "Next arena-local trace sequence.")
      .def_readonly("dropped", &xpool::transport::TransportTraceSnapshot::dropped,
                    "Number of traces dropped after capacity exhaustion.")
      .def_readonly("records", &xpool::transport::TransportTraceSnapshot::records,
                    "Retained transport trace records.");

  auto transport = module.def_submodule("transport", "Native CUDA IPC transport control and lifecycle functions.");
  transport.def(
      "create_arena",
      [](std::size_t instance_index, std::size_t instance_rank, std::size_t max_tokens, std::size_t hidden_size,
         std::int64_t dtype, std::size_t atn_tp_rank, std::size_t atn_tp_size, std::size_t atn_dp_rank,
         std::size_t atn_dp_size) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.create_arena");
        TORCH_CHECK(xpool::abi::TensorDType::is_valid(dtype),
                    "xpool native transport received an invalid dtype");
        return xpool::transport::AtnAgentTransportRuntime::singleton()
            .create_arena(xpool::RuntimeState::singleton().cuda_device("xpool.native.transport.create_arena"),
                          instance_index, instance_rank, max_tokens, hidden_size, xpool::abi::TensorDType{dtype},
                          atn_tp_rank, atn_tp_size, atn_dp_rank, atn_dp_size)
            .encode();
      },
      py::arg("instance_index"), py::arg("instance_rank"), py::arg("max_tokens"), py::arg("hidden_size"),
      py::arg("dtype"), py::arg("atn_tp_rank"), py::arg("atn_tp_size"), py::arg("atn_dp_rank"),
      py::arg("atn_dp_size"), "Create one AtnAgent-owned CUDA IPC Transport arena.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "activate",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.activate");
        xpool::transport::AtnAgentTransportRuntime::singleton().activate();
      },
      "Launch resident Transport kernels for every created arena.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "check_health",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.check_health");
        xpool::transport::AtnAgentTransportRuntime::singleton().check_health();
      },
      "Raise when an active Transport resident has failed.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_async",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_async");
        xpool::transport::AtnAgentTransportRuntime::singleton().drain_async();
      },
      "Begin asynchronous drain for every active Transport resident.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "drain_pending",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.drain_pending");
        return xpool::transport::AtnAgentTransportRuntime::singleton().drain_pending();
      },
      "Return whether any Transport resident drain remains pending.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "read_trace",
      [](const std::string &handle) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.read_trace");
        return xpool::transport::AtnAgentTransportRuntime::singleton().read_trace(
            xpool::transport::TransportArenaHandle::decode(handle));
      },
      py::arg("handle"), "Copy one arena trace buffer into a host-owned snapshot, if enabled.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "destroy_arenas",
      [](const std::vector<std::string> &handles) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::AtnAgent,
                                                      "xpool.native.transport.destroy_arenas");
        auto decoded = std::vector<xpool::transport::TransportArenaHandle>{};
        decoded.reserve(handles.size());
        for (const auto &handle : handles) {
          decoded.push_back(xpool::transport::TransportArenaHandle::decode(handle));
        }
        xpool::transport::AtnAgentTransportRuntime::singleton().destroy_arenas(decoded);
      },
      py::arg("handles"), "Destroy the specified drained AtnAgent-owned arenas.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "attach_arena",
      [](std::size_t instance_index, std::size_t rank, const std::string &handle) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.attach_arena");
        xpool::transport::InstanceTransportRuntime::singleton().attach_arena(
            instance_index, rank, xpool::transport::TransportArenaHandle::decode(handle));
      },
      py::arg("instance_index"), py::arg("rank"), py::arg("handle"),
      "Attach one Instance process to its rank-local CUDA IPC arena.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "detach_arena",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.detach_arena");
        xpool::transport::InstanceTransportRuntime::singleton().detach_arena();
      },
      "Detach the Instance process from its current CUDA IPC arena.",
      py::call_guard<py::gil_scoped_release>());
  transport.def(
      "read_generation_failure",
      []() -> std::uint32_t {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Instance,
                                                      "xpool.native.transport.read_generation_failure");
        return static_cast<std::uint32_t>(xpool::transport::InstanceTransportRuntime::singleton()
                                              .read_generation_failure()
                                              .value());
      },
      "Return the attached arena's generation-wide failure code.",
      py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
