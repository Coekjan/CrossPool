#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/native_enum.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <xpool/bindings.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffn.hpp>
#include <xpool/ffnagent/runtime.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace xpool::bindings {

void bind_fabric(py::module_ &module) {
  auto fabric = module.def_submodule("fabric", "Native NVSHMEM Fabric control and lifecycle functions.");
  fabric.attr("UID_HEX_LENGTH") = xpool::fabric::Uid::encoded_size;

  py::class_<xpool::fabric::InstanceLayerProjection>(fabric, "InstanceLayerProjection",
                                                     "One ordered FFN layer supplied to native Fabric join.")
      .def(py::init([](std::size_t layer_id, xpool::ffn::LayerKind kind, std::size_t effective_topk,
                       std::vector<std::size_t> ffnagent_indices) {
             TORCH_CHECK(xpool::ffn::is_valid(kind), "xpool InstanceLayerProjection received an invalid kind");
             return xpool::fabric::InstanceLayerProjection{
                 .layer_id = layer_id,
                 .kind = kind,
                 .effective_topk = effective_topk,
                 .ffnagent_indices = std::move(ffnagent_indices),
             };
           }),
           py::arg("layer_id"), py::arg("kind"), py::arg("effective_topk"), py::arg("ffnagent_indices"));

  py::class_<xpool::fabric::InstanceProjection>(fabric, "InstanceProjection",
                                                "One Instance supplied to native Fabric join.")
      .def(
          py::init([](const py::object &payload_dtype, std::size_t hidden_size, std::size_t decode_payload_row_capacity,
                      std::size_t prefill_payload_row_capacity, bool group_sum_complete_admitted,
                      std::size_t atn_tp_size, std::size_t atn_dp_size, std::vector<std::size_t> atnagent_indices,
                      std::vector<xpool::fabric::InstanceLayerProjection> layers) {
            return xpool::fabric::InstanceProjection{
                .decode_payload_row_capacity = decode_payload_row_capacity,
                .prefill_payload_row_capacity = prefill_payload_row_capacity,
                .payload_dtype = require_payload_dtype(payload_dtype, "FFN"),
                .hidden_size = hidden_size,
                .group_sum_complete_admitted = group_sum_complete_admitted,
                .atn_tp_size = atn_tp_size,
                .atn_dp_size = atn_dp_size,
                .atnagent_indices = std::move(atnagent_indices),
                .layers = std::move(layers),
            };
          }),
          py::arg("payload_dtype"), py::arg("hidden_size"), py::arg("decode_payload_row_capacity"),
          py::arg("prefill_payload_row_capacity"), py::arg("group_sum_complete_admitted"), py::arg("atn_tp_size"),
          py::arg("atn_dp_size"), py::arg("atnagent_indices"), py::arg("layers"));

  py::class_<xpool::fabric::SchedulerPolicy>(fabric, "SchedulerPolicy", "Immutable native Fabric scheduling policy.")
      .def_static("fifo", &xpool::fabric::SchedulerPolicy::fifo, "Create the deterministic FIFO scheduling policy.")
      .def_static("random", &xpool::fabric::SchedulerPolicy::random, py::arg("seed"),
                  "Create the random scheduling policy from a nonzero seed.");

  py::class_<xpool::fabric::ArenaProjection>(fabric, "ArenaProjection",
                                             "Complete common semantic input for native Fabric join.")
      .def(py::init([](std::uint64_t generation_high, std::uint64_t generation_low, const std::string &uid,
                       std::size_t atnagent_count, std::size_t ffnagent_count, std::size_t executor_lane_count,
                       xpool::fabric::SchedulerPolicy scheduler,
                       std::vector<xpool::fabric::InstanceProjection> instances) {
             auto projection = xpool::fabric::ArenaProjection{
                 .generation_high = generation_high,
                 .generation_low = generation_low,
                 .uid = xpool::fabric::Uid::decode(uid),
                 .atnagent_count = atnagent_count,
                 .ffnagent_count = ffnagent_count,
                 .executor_lane_count = executor_lane_count,
                 .scheduler = std::move(scheduler),
                 .instances = std::move(instances),
             };
             projection.validate();
             return projection;
           }),
           py::arg("generation_high"), py::arg("generation_low"), py::arg("uid"), py::arg("atnagent_count"),
           py::arg("ffnagent_count"), py::arg("executor_lane_count"), py::arg("scheduler"), py::arg("instances"));

  py::class_<xpool::fabric::InvocationKey>(fabric, "InvocationKey", "Identity of one FFN invocation.")
      .def_readonly("instance_index", &xpool::fabric::InvocationKey::instance_index, "Config-order Instance index.")
      .def_readonly("invocation_sequence", &xpool::fabric::InvocationKey::invocation_sequence,
                    "Model-local invocation sequence.");

  py::class_<xpool::fabric::FailurePayload>(fabric, "FailurePayload", "Immutable canonical Fabric failure payload.")
      .def_readonly("result_code", &xpool::fabric::FailurePayload::result_code, "Stable FFN result code.")
      .def_readonly("origin_pe", &xpool::fabric::FailurePayload::origin_pe, "PE that first claimed the failure.")
      .def_readonly("key", &xpool::fabric::FailurePayload::key, "Failed invocation identity.")
      .def_readonly("layer_ordinal", &xpool::fabric::FailurePayload::layer_ordinal,
                    "Failed config-order FFN layer ordinal.");

  py::native_enum<xpool::fabric::DeliveryVariant>(fabric, "DeliveryVariant", "enum.IntEnum",
                                                  "Observed Fabric output-delivery branch.")
      .value("DIRECT_PARTIAL", xpool::fabric::DeliveryVariant::DirectPartial)
      .value("SINGLE_COMPLETE", xpool::fabric::DeliveryVariant::SingleComplete)
      .value("REPLICATED_COMPLETE", xpool::fabric::DeliveryVariant::ReplicatedComplete)
      .finalize();

  fabric.def("arena_allocation_bytes", &xpool::fabric::arena_allocation_bytes, py::arg("atnagent_count"),
             py::arg("ffnagent_count"), py::arg("instance_count"), py::arg("executor_lane_count"),
             py::arg("layer_entry_count"), py::arg("atnagent_pe_entry_count"), py::arg("ffnagent_pe_entry_count"),
             py::arg("maximum_lane_payload_bytes"), py::arg("maximum_routing_metadata_elements"),
             "Return exact native Fabric arena allocation bytes.");
  fabric.def("ffnagent_control_allocation_bytes", &xpool::fabric::ffnagent_control_allocation_bytes,
             py::arg("is_coordinator"), py::arg("instance_count"),
             "Return exact native FfnAgent control allocation bytes.");
  fabric.def(
      "create_uid",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::Daemon, "xpool.native.fabric.create_uid");
        return xpool::fabric::create_uid().encode();
      },
      "Create an opaque daemon-owned NVSHMEM bootstrap identity.");
  fabric.def(
      "join",
      [](const xpool::fabric::ArenaProjection &projection, int pe) {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.fabric.join");
        xpool::fabric::Runtime::singleton().join(xpool::RuntimeState::singleton().device("xpool.native.fabric.join"),
                                                 projection, pe);
      },
      py::arg("projection"), py::arg("pe"), "Join the exact Fabric generation described by the Projection.",
      py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "drain_async",
      []() {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.fabric.drain_async");
        xpool::fabric::Runtime::singleton().drain_async();
      },
      "Begin asynchronous cooperative Fabric drain.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "drain_pending",
      []() {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.fabric.drain_pending");
        return xpool::fabric::Runtime::singleton().drain_pending();
      },
      "Return whether asynchronous Fabric drain remains pending.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "failure",
      []() {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.fabric.failure");
        return xpool::fabric::Runtime::singleton().failure();
      },
      "Return the published canonical Fabric failure, if any.", py::call_guard<py::gil_scoped_release>());
  fabric.def(
      "finalize",
      []() {
        xpool::RuntimeState::singleton().require_role({xpool::RuntimeRole::AtnAgent, xpool::RuntimeRole::FfnAgent},
                                                      "xpool.native.fabric.finalize");
        xpool::fabric::Runtime::singleton().finalize();
      },
      "Finalize NVSHMEM and release all local Fabric resources.", py::call_guard<py::gil_scoped_release>());
}

} // namespace xpool::bindings
