#include "bindings.hpp"

#include <pybind11/native_enum.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <xpool/ffn.hpp>
#include <xpool/fabric/runtime.hpp>
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

py::object python_dtype(c10::ScalarType dtype) {
  return py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(torch::getTHPDtype(dtype)));
}

} // namespace

namespace xpool::bindings {

void bind_fabric(py::module_ &module) {
  auto fabric = module.def_submodule("fabric", "Native NVSHMEM Fabric control and lifecycle functions.");
  auto ffnagent = module.def_submodule("ffnagent", "Native FfnAgent execution lifecycle functions.");

  py::class_<xpool::fabric::InstanceLayerProjection>(fabric, "InstanceLayerProjection",
                                                           "One ordered FFN layer supplied to native Fabric join.")
      .def(py::init([](std::size_t layer_id, xpool::ffn::LayerKind kind, std::size_t effective_topk,
                       std::vector<std::size_t> ffnagent_indices) {
             TORCH_CHECK(xpool::ffn::is_valid(kind),
                         "xpool InstanceLayerProjection received an invalid kind");
             return xpool::fabric::InstanceLayerProjection{
                 .layer_id = layer_id,
                 .kind = kind,
                 .effective_topk = effective_topk,
                 .ffnagent_indices = std::move(ffnagent_indices),
             };
           }),
           py::arg("layer_id"), py::arg("kind"), py::arg("effective_topk"), py::arg("ffnagent_indices"))
      .def_readonly("layer_id", &xpool::fabric::InstanceLayerProjection::layer_id,
                    "Concrete decoder layer identifier.")
      .def_readonly("kind", &xpool::fabric::InstanceLayerProjection::kind, "FFN layer kind.")
      .def_readonly("effective_topk", &xpool::fabric::InstanceLayerProjection::effective_topk,
                    "Final routing width, or zero for Dense layers.")
      .def_readonly("ffnagent_indices", &xpool::fabric::InstanceLayerProjection::ffnagent_indices,
                    "Ordered local FfnAgent indices by FFN TP rank.");

  py::class_<xpool::fabric::InstanceProjection>(fabric, "InstanceProjection",
                                                      "One Instance supplied to native Fabric join.")
      .def(py::init([](const py::object &payload_dtype, std::size_t hidden_size, std::size_t decode_payload_row_capacity,
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
           py::arg("atn_dp_size"), py::arg("atnagent_indices"), py::arg("layers"))
      .def_readonly("decode_payload_row_capacity",
                    &xpool::fabric::InstanceProjection::decode_payload_row_capacity,
                    "Maximum physical Decode rows per invocation.")
      .def_readonly("prefill_payload_row_capacity",
                    &xpool::fabric::InstanceProjection::prefill_payload_row_capacity,
                    "Maximum physical Prefill rows per invocation.")
      .def_property_readonly(
          "payload_dtype",
          [](const xpool::fabric::InstanceProjection &projection) {
            return python_dtype(projection.payload_dtype);
          },
          "Stable hidden-state tensor dtype value.")
      .def_readonly("hidden_size", &xpool::fabric::InstanceProjection::hidden_size,
                    "Model hidden width in elements.")
      .def_readonly("group_sum_complete_admitted",
                    &xpool::fabric::InstanceProjection::group_sum_complete_admitted,
                    "Whether Group-Sum Complete output is admitted.")
      .def_readonly("atn_tp_size", &xpool::fabric::InstanceProjection::atn_tp_size,
                    "Attention tensor-parallel participant count.")
      .def_readonly("atn_dp_size", &xpool::fabric::InstanceProjection::atn_dp_size,
                    "Attention data-parallel participant count.")
      .def_readonly("atnagent_indices", &xpool::fabric::InstanceProjection::atnagent_indices,
                    "Ordered AtnAgent indices in TP-fastest rank order.")
      .def_readonly("layers", &xpool::fabric::InstanceProjection::layers, "Ordered FFN layer metadata.");

  py::class_<xpool::fabric::SchedulerPolicy>(fabric, "SchedulerPolicy",
                                                "Immutable native Fabric scheduling policy.")
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
           py::arg("ffnagent_count"), py::arg("executor_lane_count"), py::arg("scheduler"), py::arg("instances"))
      .def_readonly("generation_high", &xpool::fabric::ArenaProjection::generation_high,
                    "High 64 bits of the Fabric generation identity.")
      .def_readonly("generation_low", &xpool::fabric::ArenaProjection::generation_low,
                    "Low 64 bits of the Fabric generation identity.")
      .def_property_readonly(
          "uid", [](const xpool::fabric::ArenaProjection &projection) { return projection.uid.encode(); },
          "Opaque NVSHMEM bootstrap identity.")
      .def_readonly("atnagent_count", &xpool::fabric::ArenaProjection::atnagent_count,
                    "Number of AtnAgent participants.")
      .def_readonly("ffnagent_count", &xpool::fabric::ArenaProjection::ffnagent_count,
                    "Number of FfnAgent participants.")
      .def_readonly("executor_lane_count", &xpool::fabric::ArenaProjection::executor_lane_count,
                    "Number of distributed FFN Executors.")
      .def_readonly("scheduler", &xpool::fabric::ArenaProjection::scheduler, "Coordinator scheduling policy.")
      .def_readonly("instances", &xpool::fabric::ArenaProjection::instances,
                    "Config-order Instance Projections.");

  py::class_<xpool::ffnagent::DenseBindingResourceProjection>(ffnagent, "DenseBindingResourceProjection",
                                                                 "Non-owning Dense weight addresses.")
      .def(py::init<std::uintptr_t, std::uintptr_t>(), py::arg("gate_up_weight_address"),
           py::arg("down_weight_address"))
      .def_readonly("gate_up_weight_address",
                    &xpool::ffnagent::DenseBindingResourceProjection::gate_up_weight_address,
                    "Device address of the packed gate-up weight tensor.")
      .def_readonly("down_weight_address", &xpool::ffnagent::DenseBindingResourceProjection::down_weight_address,
                    "Device address of the down-projection weight tensor.");

  py::class_<xpool::ffnagent::MoeRouterBindingResourceProjection>(ffnagent, "MoeRouterBindingResourceProjection",
                                                                  "Non-owning Router weight addresses.")
      .def(py::init<std::uintptr_t, std::optional<std::uintptr_t>>(), py::arg("weight_address"),
           py::arg("correction_bias_address"))
      .def_readonly("weight_address", &xpool::ffnagent::MoeRouterBindingResourceProjection::weight_address,
                    "Device address of the Router weight tensor.")
      .def_readonly("correction_bias_address",
                    &xpool::ffnagent::MoeRouterBindingResourceProjection::correction_bias_address,
                    "Optional device address of the Router correction-bias tensor.");

  py::class_<xpool::ffnagent::MoeBindingResourceProjection>(ffnagent, "MoeBindingResourceProjection",
                                                               "Non-owning MoE weight addresses.")
      .def(py::init<std::uintptr_t, std::uintptr_t,
                    std::optional<xpool::ffnagent::MoeRouterBindingResourceProjection>>(),
           py::arg("expert_gate_up_weight_address"), py::arg("expert_down_weight_address"), py::arg("router"))
      .def_readonly("expert_gate_up_weight_address",
                    &xpool::ffnagent::MoeBindingResourceProjection::expert_gate_up_weight_address,
                    "Device address of the packed Expert gate-up weights.")
      .def_readonly("expert_down_weight_address",
                    &xpool::ffnagent::MoeBindingResourceProjection::expert_down_weight_address,
                    "Device address of the packed Expert down weights.")
      .def_readonly("router", &xpool::ffnagent::MoeBindingResourceProjection::router,
                    "Router resources on the owning TP rank, or no resources otherwise.");

  py::class_<xpool::ffnagent::DenseExecutionSignatureProjection>(ffnagent, "DenseExecutionSignatureProjection",
                                                                    "Captured Dense execution signature.")
      .def(py::init([](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                       std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                       std::uintptr_t control_graph_address, std::uintptr_t capture_input_address,
                       std::uintptr_t capture_partial_address, std::uintptr_t capture_workspace_address,
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
                 .capture_input_address = capture_input_address,
                 .capture_partial_address = capture_partial_address,
                 .capture_workspace_address = capture_workspace_address,
                 .compute_workspace_bytes = compute_workspace_bytes,
                 .primary_capture_resources = primary_capture_resources,
                 .control_capture_resources = control_capture_resources,
             };
           }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input_address"), py::arg("capture_partial_address"), py::arg("capture_workspace_address"),
           py::arg("compute_workspace_bytes"), py::arg("primary_capture_resources"),
           py::arg("control_capture_resources"))
      .def_property_readonly(
          "payload_dtype",
          [](const xpool::ffnagent::DenseExecutionSignatureProjection &value) {
            return python_dtype(value.payload_dtype);
          },
          "Stable hidden-state payload dtype value.")
      .def_readonly("payload_row_capacity",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::payload_row_capacity,
                    "Physical row capacity captured by this Primary Graph.")
      .def_readonly("hidden_size", &xpool::ffnagent::DenseExecutionSignatureProjection::hidden_size,
                    "Hidden-state width in elements.")
      .def_readonly("local_intermediate_size",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::local_intermediate_size,
                    "Intermediate width owned by this TP rank.")
      .def_readonly("primary_graph_address",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::primary_graph_address,
                    "Opaque CUDA Graph address for the primary representative layer.")
      .def_readonly("control_graph_address",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::control_graph_address,
                    "Opaque CUDA Graph address for binding-schema control.")
      .def_readonly("capture_input_address",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::capture_input_address,
                    "Captured source address of the input tensor.")
      .def_readonly("capture_partial_address",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::capture_partial_address,
                    "Captured source address of the rank-local partial output tensor.")
      .def_readonly("capture_workspace_address",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::capture_workspace_address,
                    "Captured source address of the caller-owned compute workspace.")
      .def_readonly("compute_workspace_bytes",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::compute_workspace_bytes,
                    "Exact logical source workspace byte extent.")
      .def_readonly("primary_capture_resources",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::primary_capture_resources,
                    "Primary representative layer's captured weight addresses.")
      .def_readonly("control_capture_resources",
                    &xpool::ffnagent::DenseExecutionSignatureProjection::control_capture_resources,
                    "Control representative layer's captured weight addresses.");

  py::class_<xpool::ffnagent::MoeExecutionSignatureProjection>(ffnagent, "MoeExecutionSignatureProjection",
                                                                  "Captured MoE execution signature.")
      .def(py::init([](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                       std::size_t local_intermediate_size, std::size_t expert_count, std::size_t effective_topk,
                       std::optional<std::size_t> routed_expert_count, std::uintptr_t primary_graph_address,
                       std::uintptr_t control_graph_address, std::uintptr_t capture_input_address,
                       std::uintptr_t capture_partial_address, std::uintptr_t capture_workspace_address,
                       std::size_t compute_workspace_bytes, std::uintptr_t capture_routing_metadata_address,
                       std::optional<std::uintptr_t> capture_payload_rows_address,
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
                 .capture_input_address = capture_input_address,
                 .capture_partial_address = capture_partial_address,
                 .capture_workspace_address = capture_workspace_address,
                 .compute_workspace_bytes = compute_workspace_bytes,
                 .capture_routing_metadata_address = capture_routing_metadata_address,
                 .capture_payload_rows_address = capture_payload_rows_address,
                 .primary_capture_resources = std::move(primary_capture_resources),
                 .control_capture_resources = std::move(control_capture_resources),
             };
           }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("expert_count"), py::arg("effective_topk"),
           py::arg("routed_expert_count"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input_address"), py::arg("capture_partial_address"), py::arg("capture_workspace_address"),
           py::arg("compute_workspace_bytes"), py::arg("capture_routing_metadata_address"),
           py::arg("capture_payload_rows_address"), py::arg("primary_capture_resources"),
           py::arg("control_capture_resources"))
      .def_property_readonly(
          "payload_dtype",
          [](const xpool::ffnagent::MoeExecutionSignatureProjection &value) { return python_dtype(value.payload_dtype); },
          "Stable hidden-state payload dtype value.")
      .def_readonly("payload_row_capacity", &xpool::ffnagent::MoeExecutionSignatureProjection::payload_row_capacity,
                    "Physical row capacity captured by this Primary Graph.")
      .def_readonly("hidden_size", &xpool::ffnagent::MoeExecutionSignatureProjection::hidden_size,
                    "Hidden-state width in elements.")
      .def_readonly("local_intermediate_size",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::local_intermediate_size,
                    "Intermediate width owned by this TP rank and Expert.")
      .def_readonly("expert_count", &xpool::ffnagent::MoeExecutionSignatureProjection::expert_count,
                    "Total routed and shared Expert slots.")
      .def_readonly("effective_topk", &xpool::ffnagent::MoeExecutionSignatureProjection::effective_topk,
                    "Final routed and always-selected Expert slots per row.")
      .def_readonly("routed_expert_count", &xpool::ffnagent::MoeExecutionSignatureProjection::routed_expert_count,
                    "Routed Expert count on the Router-owning TP rank, or None.")
      .def_readonly("primary_graph_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::primary_graph_address,
                    "Opaque CUDA Graph address for the primary representative layer.")
      .def_readonly("control_graph_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::control_graph_address,
                    "Opaque CUDA Graph address for binding-schema control.")
      .def_readonly("capture_input_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::capture_input_address,
                    "Captured source address of the input tensor.")
      .def_readonly("capture_partial_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::capture_partial_address,
                    "Captured source address of the rank-local partial output tensor.")
      .def_readonly("capture_workspace_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::capture_workspace_address,
                    "Captured source address of the caller-owned compute workspace.")
      .def_readonly("compute_workspace_bytes",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::compute_workspace_bytes,
                    "Exact logical source workspace byte extent.")
      .def_readonly("capture_routing_metadata_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::capture_routing_metadata_address,
                    "Captured source address of packed Expert ids and weights.")
      .def_readonly("capture_payload_rows_address",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::capture_payload_rows_address,
                    "Router-owner captured source address of the device-visible live-row count, or None.")
      .def_readonly("primary_capture_resources",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::primary_capture_resources,
                    "Primary representative layer's captured weight addresses.")
      .def_readonly("control_capture_resources",
                    &xpool::ffnagent::MoeExecutionSignatureProjection::control_capture_resources,
                    "Control representative layer's captured weight addresses.");

  py::class_<xpool::ffnagent::LayerExecutionProjection>(ffnagent, "LayerExecutionProjection",
                                                           "One Plan-addressed local FFN layer.")
      .def(py::init([](std::size_t instance_index, std::size_t layer_ordinal,
                       std::vector<std::size_t> execution_signature_indices, const py::object &target) {
             if (py::isinstance<xpool::ffnagent::DenseBindingResourceProjection>(target)) {
               return xpool::ffnagent::LayerExecutionProjection{
                   .instance_index = instance_index,
                   .layer_ordinal = layer_ordinal,
                   .execution_signature_indices = std::move(execution_signature_indices),
                   .layer_resource_targets = target.cast<xpool::ffnagent::DenseBindingResourceProjection>(),
               };
             }
             if (py::isinstance<xpool::ffnagent::MoeBindingResourceProjection>(target)) {
               return xpool::ffnagent::LayerExecutionProjection{
                   .instance_index = instance_index,
                   .layer_ordinal = layer_ordinal,
                   .execution_signature_indices = std::move(execution_signature_indices),
                   .layer_resource_targets = target.cast<xpool::ffnagent::MoeBindingResourceProjection>(),
               };
             }
             throw py::type_error("layer_resource_targets must be a Dense or MoE FFN resource projection");
           }),
           py::arg("instance_index"), py::arg("layer_ordinal"), py::arg("execution_signature_indices"),
           py::arg("layer_resource_targets"))
      .def_readonly("instance_index", &xpool::ffnagent::LayerExecutionProjection::instance_index,
                    "Config-order Instance Plan index.")
      .def_readonly("layer_ordinal", &xpool::ffnagent::LayerExecutionProjection::layer_ordinal,
                    "Config-order FFN layer ordinal.")
      .def_readonly("execution_signature_indices",
                    &xpool::ffnagent::LayerExecutionProjection::execution_signature_indices,
                    "Execution Signatures eligible for this layer in ascending capacity order.")
      .def_property_readonly(
          "layer_resource_targets",
          [](const xpool::ffnagent::LayerExecutionProjection &value) {
            return std::visit([](const auto &target) { return py::cast(target); }, value.layer_resource_targets);
          },
          "Installed layer weight addresses used to bind eligible Graph Templates.");

  py::class_<xpool::ffnagent::ExecutionProjection>(ffnagent, "ExecutionProjection",
                                                      "Complete one-time native FFN installation input.")
      .def(py::init([](const py::sequence &signature_values,
                       std::vector<xpool::ffnagent::LayerExecutionProjection> layers) {
             auto signatures = std::vector<xpool::ffnagent::ExecutionSignatureProjection>{};
             signatures.reserve(signature_values.size());
             for (const auto item : signature_values) {
               const auto value = py::reinterpret_borrow<py::object>(item);
               if (py::isinstance<xpool::ffnagent::DenseExecutionSignatureProjection>(value)) {
                 signatures.emplace_back(value.cast<xpool::ffnagent::DenseExecutionSignatureProjection>());
               } else if (py::isinstance<xpool::ffnagent::MoeExecutionSignatureProjection>(value)) {
                 signatures.emplace_back(value.cast<xpool::ffnagent::MoeExecutionSignatureProjection>());
               } else {
                 throw py::type_error("signatures must contain only Dense or MoE FFN signature projections");
               }
             }
             auto projection = xpool::ffnagent::ExecutionProjection{
                 .signatures = std::move(signatures),
                 .layers = std::move(layers),
             };
             projection.validate();
             return projection;
           }),
           py::arg("signatures"), py::arg("layers"))
      .def_property_readonly(
          "signatures",
          [](const xpool::ffnagent::ExecutionProjection &projection) {
            auto values = py::tuple(projection.signatures.size());
            for (auto index = std::size_t{0}; index < projection.signatures.size(); ++index) {
              values[index] =
                  std::visit([](const auto &value) { return py::cast(value); }, projection.signatures[index]);
            }
            return values;
          },
          "Captured Dense and MoE execution signatures.")
      .def_readonly("layers", &xpool::ffnagent::ExecutionProjection::layers,
                    "Plan-addressed local FFN layer bindings.");

  py::class_<xpool::fabric::InvocationKey>(fabric, "InvocationKey", "Identity of one FFN invocation.")
      .def_readonly("instance_index", &xpool::fabric::InvocationKey::instance_index, "Config-order Instance index.")
      .def_readonly("invocation_sequence", &xpool::fabric::InvocationKey::invocation_sequence,
                    "Model-local invocation sequence.");

  py::class_<xpool::fabric::FailurePayload>(fabric, "FailurePayload",
                                                  "Immutable canonical Fabric failure payload.")
      .def_readonly("result_code", &xpool::fabric::FailurePayload::result_code, "Stable FFN result code.")
      .def_readonly("origin_pe", &xpool::fabric::FailurePayload::origin_pe, "PE that first claimed the failure.")
      .def_readonly("key", &xpool::fabric::FailurePayload::key, "Failed invocation identity.")
      .def_readonly("layer_ordinal", &xpool::fabric::FailurePayload::layer_ordinal,
                    "Failed config-order FFN layer ordinal.");

  py::class_<xpool::fabric::Failure>(fabric, "Failure", "Published canonical Fabric failure.")
      .def_readonly("claim", &xpool::fabric::Failure::claim, "Canonical failure claim word.")
      .def_readonly("publication", &xpool::fabric::Failure::publication, "Canonical failure publication word.")
      .def_readonly("payload", &xpool::fabric::Failure::payload, "Published immutable failure payload.");


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
