#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/native_enum.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include "bindings.hpp"
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffn.hpp>
#include <xpool/ffnagent/runtime.hpp>
#include <xpool/runtime.hpp>

namespace py = pybind11;

namespace {

c10::ScalarType require_payload_dtype(const py::object &dtype) {
  const auto scalar_type = torch::python::detail::py_object_to_dtype(dtype);
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(scalar_type),
              "xpool FFN payload dtype must be torch.float16 or torch.bfloat16");
  return scalar_type;
}

std::uintptr_t require_cuda_address(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), "xpool ", name, " must be a CUDA Tensor");
  TORCH_CHECK(tensor.numel() != 0 && tensor.is_contiguous(), "xpool ", name,
              " must be a nonempty contiguous CUDA Tensor");
  return reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
}

} // namespace

namespace xpool::bindings {

void bind_fabric(py::module_ &module) {
  auto fabric = module.def_submodule("fabric", "Native NVSHMEM Fabric control and lifecycle functions.");
  auto ffnagent = module.def_submodule("ffnagent", "Native FfnAgent execution lifecycle functions.");
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
                .payload_dtype = require_payload_dtype(payload_dtype),
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

  py::class_<xpool::ffnagent::DenseBindingResourceProjection>(ffnagent, "DenseBindingResourceProjection",
                                                              "Non-owning Dense weight addresses.")
      .def(py::init([](const at::Tensor &gate_up_weight, const at::Tensor &down_weight) {
             return xpool::ffnagent::DenseBindingResourceProjection{
                 .gate_up_weight_address = require_cuda_address(gate_up_weight, "Dense gate/up weight"),
                 .down_weight_address = require_cuda_address(down_weight, "Dense down weight"),
             };
           }),
           py::arg("gate_up_weight"), py::arg("down_weight"));

  py::class_<xpool::ffnagent::MoeRouterBindingResourceProjection>(ffnagent, "MoeRouterBindingResourceProjection",
                                                                  "Non-owning Router weight addresses.")
      .def(py::init([](const at::Tensor &weight, const std::optional<at::Tensor> &correction_bias) {
             return xpool::ffnagent::MoeRouterBindingResourceProjection{
                 .weight_address = require_cuda_address(weight, "MoE Router weight"),
                 .correction_bias_address =
                     correction_bias.has_value()
                         ? std::optional{require_cuda_address(*correction_bias, "MoE Router correction bias")}
                         : std::nullopt,
             };
           }),
           py::arg("weight"), py::arg("correction_bias"));

  py::class_<xpool::ffnagent::MoeBindingResourceProjection>(ffnagent, "MoeBindingResourceProjection",
                                                            "Non-owning MoE weight addresses.")
      .def(py::init([](const at::Tensor &expert_gate_up_weight, const at::Tensor &expert_down_weight,
                       std::optional<xpool::ffnagent::MoeRouterBindingResourceProjection> router) {
             return xpool::ffnagent::MoeBindingResourceProjection{
                 .expert_gate_up_weight_address =
                     require_cuda_address(expert_gate_up_weight, "MoE Expert gate/up weights"),
                 .expert_down_weight_address = require_cuda_address(expert_down_weight, "MoE Expert down weights"),
                 .router = std::move(router),
             };
           }),
           py::arg("expert_gate_up_weight"), py::arg("expert_down_weight"), py::arg("router"));

  py::class_<xpool::ffnagent::DenseExecutionSignatureProjection>(ffnagent, "DenseExecutionSignatureProjection",
                                                                 "Captured Dense execution signature.")
      .def(py::init([](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                       std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                       std::uintptr_t control_graph_address, const at::Tensor &capture_input,
                       const at::Tensor &capture_partial, const at::Tensor &capture_workspace,
                       std::size_t compute_workspace_bytes,
                       xpool::ffnagent::DenseBindingResourceProjection primary_capture_resources,
                       xpool::ffnagent::DenseBindingResourceProjection control_capture_resources) {
             return xpool::ffnagent::DenseExecutionSignatureProjection{
                 .payload_dtype = require_payload_dtype(payload_dtype),
                 .payload_row_capacity = payload_row_capacity,
                 .hidden_size = hidden_size,
                 .local_intermediate_size = local_intermediate_size,
                 .primary_graph_address = primary_graph_address,
                 .control_graph_address = control_graph_address,
                 .capture_input_address = require_cuda_address(capture_input, "Dense capture input"),
                 .capture_partial_address = require_cuda_address(capture_partial, "Dense capture partial"),
                 .capture_workspace_address = require_cuda_address(capture_workspace, "Dense capture workspace"),
                 .compute_workspace_bytes = compute_workspace_bytes,
                 .primary_capture_resources = primary_capture_resources,
                 .control_capture_resources = control_capture_resources,
             };
           }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input"), py::arg("capture_partial"), py::arg("capture_workspace"),
           py::arg("compute_workspace_bytes"), py::arg("primary_capture_resources"),
           py::arg("control_capture_resources"));

  py::class_<xpool::ffnagent::MoeExecutionSignatureProjection>(ffnagent, "MoeExecutionSignatureProjection",
                                                               "Captured MoE execution signature.")
      .def(py::init([](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                       std::size_t local_intermediate_size, std::size_t expert_count, std::size_t effective_topk,
                       std::optional<std::size_t> routed_expert_count, std::uintptr_t primary_graph_address,
                       std::uintptr_t control_graph_address, const at::Tensor &capture_input,
                       const at::Tensor &capture_partial, const at::Tensor &capture_workspace,
                       std::size_t compute_workspace_bytes, const at::Tensor &capture_routing_metadata,
                       const std::optional<at::Tensor> &capture_payload_rows,
                       xpool::ffnagent::MoeBindingResourceProjection primary_capture_resources,
                       xpool::ffnagent::MoeBindingResourceProjection control_capture_resources) {
             return xpool::ffnagent::MoeExecutionSignatureProjection{
                 .payload_dtype = require_payload_dtype(payload_dtype),
                 .payload_row_capacity = payload_row_capacity,
                 .hidden_size = hidden_size,
                 .local_intermediate_size = local_intermediate_size,
                 .expert_count = expert_count,
                 .effective_topk = effective_topk,
                 .routed_expert_count = routed_expert_count,
                 .primary_graph_address = primary_graph_address,
                 .control_graph_address = control_graph_address,
                 .capture_input_address = require_cuda_address(capture_input, "MoE capture input"),
                 .capture_partial_address = require_cuda_address(capture_partial, "MoE capture partial"),
                 .capture_workspace_address = require_cuda_address(capture_workspace, "MoE capture workspace"),
                 .compute_workspace_bytes = compute_workspace_bytes,
                 .capture_routing_metadata_address =
                     require_cuda_address(capture_routing_metadata, "MoE capture routing metadata"),
                 .capture_payload_rows_address =
                     capture_payload_rows.has_value()
                         ? std::optional{require_cuda_address(*capture_payload_rows, "MoE capture payload rows")}
                         : std::nullopt,
                 .primary_capture_resources = std::move(primary_capture_resources),
                 .control_capture_resources = std::move(control_capture_resources),
             };
           }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("expert_count"), py::arg("effective_topk"),
           py::arg("routed_expert_count"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input"), py::arg("capture_partial"), py::arg("capture_workspace"),
           py::arg("compute_workspace_bytes"), py::arg("capture_routing_metadata"), py::arg("capture_payload_rows"),
           py::arg("primary_capture_resources"), py::arg("control_capture_resources"));

  py::class_<xpool::ffnagent::LayerExecutionProjection>(ffnagent, "LayerExecutionProjection",
                                                        "One Plan-addressed local FFN layer.")
      .def(py::init([](std::size_t instance_index, std::size_t layer_ordinal,
                       std::vector<std::size_t> execution_signature_indices,
                       xpool::ffnagent::BindingResourceProjection layer_resource_targets) {
             return xpool::ffnagent::LayerExecutionProjection{
                 .instance_index = instance_index,
                 .layer_ordinal = layer_ordinal,
                 .execution_signature_indices = std::move(execution_signature_indices),
                 .layer_resource_targets = std::move(layer_resource_targets),
             };
           }),
           py::arg("instance_index"), py::arg("layer_ordinal"), py::arg("execution_signature_indices"),
           py::arg("layer_resource_targets"));

  py::class_<xpool::ffnagent::ExecutionProjection>(ffnagent, "ExecutionProjection",
                                                   "Complete one-time native FFN installation input.")
      .def(py::init([](std::vector<xpool::ffnagent::ExecutionSignatureProjection> signatures,
                       std::vector<xpool::ffnagent::LayerExecutionProjection> layers) {
             auto projection = xpool::ffnagent::ExecutionProjection{
                 .signatures = std::move(signatures),
                 .layers = std::move(layers),
             };
             projection.validate();
             return projection;
           }),
           py::arg("signatures"), py::arg("layers"));

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
        xpool::fabric::Runtime::singleton().join(
            xpool::RuntimeState::singleton().cuda_device("xpool.native.fabric.join"), projection, pe);
      },
      py::arg("projection"), py::arg("pe"), "Join the exact Fabric generation described by the Projection.",
      py::call_guard<py::gil_scoped_release>());
  ffnagent.def("execution_state_allocation_bytes", &xpool::ffnagent::execution_state_allocation_bytes,
               py::arg("local_layer_count"), py::arg("local_capacity_count"), py::arg("local_signature_count"),
               py::arg("executor_lane_count"), "Return exact retained FfnAgent execution-state allocation bytes.");
  ffnagent.def(
      "install_execution",
      [](const xpool::ffnagent::ExecutionProjection &projection) {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent,
                                                      "xpool.native.ffnagent.install_execution");
        xpool::fabric::Runtime::singleton().install_ffnagent_execution(projection);
      },
      py::arg("projection"), "Atomically install one production FfnAgent execution.",
      py::call_guard<py::gil_scoped_release>());
  ffnagent.def(
      "activate",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent, "xpool.native.ffnagent.activate");
        xpool::fabric::Runtime::singleton().activate_ffnagent();
      },
      "Launch the Fabric Coordinator and installed Executor Lane Graphs.", py::call_guard<py::gil_scoped_release>());
  ffnagent.def(
      "check_health",
      []() {
        xpool::RuntimeState::singleton().require_role(xpool::RuntimeRole::FfnAgent,
                                                      "xpool.native.ffnagent.check_health");
        xpool::fabric::Runtime::singleton().check_ffnagent_health();
      },
      "Raise when the active FfnAgent runtime has failed.", py::call_guard<py::gil_scoped_release>());
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
