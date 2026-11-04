#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <c10/util/Exception.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <xpool/bindings.hpp>
#include <xpool/fabric/runtime.hpp>
#include <xpool/ffnagent/execution.hpp>
#include <xpool/ffnagent/runtime.hpp>
#include <xpool/runtime.hpp>
#include <xpool/utils/checked.hpp>

namespace py = pybind11;

namespace {

std::uintptr_t require_cuda_address(const at::Tensor &tensor, const char *name) {
  TORCH_CHECK(tensor.is_cuda(), "xpool ", name, " must be a CUDA Tensor");
  TORCH_CHECK(tensor.numel() != 0 && tensor.is_contiguous(), "xpool ", name,
              " must be a nonempty contiguous CUDA Tensor");
  return reinterpret_cast<std::uintptr_t>(tensor.data_ptr());
}

std::uintptr_t require_cuda_storage(const at::Tensor &tensor, const char *name, std::size_t minimum_bytes) {
  const auto address = require_cuda_address(tensor, name);
  TORCH_CHECK(tensor.nbytes() >= minimum_bytes, "xpool ", name, " requires at least ", minimum_bytes, " bytes, found ",
              tensor.nbytes());
  return address;
}

} // namespace

namespace xpool::bindings {

void bind_ffnagent(py::module_ &module) {
  auto ffnagent = module.def_submodule("ffnagent", "Native FfnAgent execution lifecycle functions.");

  py::class_<xpool::ffnagent::CapturedResourceRange>(ffnagent, "CapturedResourceRange",
                                                     "Logical storage borrowed from a retained pool or explicit owner.")
      .def(py::init<std::string, std::uintptr_t, std::uintptr_t, std::size_t, bool>(), py::arg("key"),
           py::arg("primary_address"), py::arg("control_address"), py::arg("bytes"),
           py::arg("shared_read_only") = false)
      .def_readonly("key", &xpool::ffnagent::CapturedResourceRange::key, "Source identity within one capture.")
      .def_readonly("primary_address", &xpool::ffnagent::CapturedResourceRange::primary_address,
                    "Primary address retained for service.")
      .def_readonly("control_address", &xpool::ffnagent::CapturedResourceRange::control_address,
                    "Control address retained through installation.")
      .def_readonly("bytes", &xpool::ffnagent::CapturedResourceRange::bytes, "Logical extent of each source range.")
      .def_readonly("shared_read_only", &xpool::ffnagent::CapturedResourceRange::shared_read_only,
                    "Source-declared immutable storage permitted across Lanes.");

  py::class_<xpool::ffnagent::ProjectionBindingView>(ffnagent, "ProjectionBindingView",
                                                     "Non-owning projection addresses and Tensor byte extents.")
      .def(py::init([](const at::Tensor &values, const std::optional<at::Tensor> &scales,
                       const std::optional<at::Tensor> &bias) {
             return xpool::ffnagent::ProjectionBindingView{
                 .values = require_cuda_address(values, "projection values"),
                 .values_bytes = values.nbytes(),
                 .scales = scales.has_value() ? std::optional{require_cuda_address(*scales, "projection scales")}
                                              : std::nullopt,
                 .scales_bytes = scales.has_value() ? scales->nbytes() : 0,
                 .bias =
                     bias.has_value() ? std::optional{require_cuda_address(*bias, "projection bias")} : std::nullopt,
                 .bias_bytes = bias.has_value() ? bias->nbytes() : 0,
             };
           }),
           py::arg("values"), py::arg("scales"), py::arg("bias"));

  py::class_<xpool::ffnagent::DenseBindingView>(ffnagent, "DenseBindingView", "Non-owning Dense resources.")
      .def(py::init([](xpool::ffnagent::ProjectionBindingView gate_up, xpool::ffnagent::ProjectionBindingView down) {
             return xpool::ffnagent::DenseBindingView{.gate_up = gate_up, .down = down};
           }),
           py::arg("gate_up"), py::arg("down"));

  py::class_<xpool::ffnagent::MoeRouterBindingView>(ffnagent, "MoeRouterBindingView", "Non-owning Router resources.")
      .def(py::init(
               [](xpool::ffnagent::ProjectionBindingView projection, const std::optional<at::Tensor> &correction_bias) {
                 return xpool::ffnagent::MoeRouterBindingView{
                     .projection = projection,
                     .correction_bias_address =
                         correction_bias.has_value()
                             ? std::optional{require_cuda_address(*correction_bias, "MoE Router correction bias")}
                             : std::nullopt,
                     .correction_bias_bytes = correction_bias.has_value() ? correction_bias->nbytes() : 0,
                 };
               }),
           py::arg("projection"), py::arg("correction_bias"));

  py::class_<xpool::ffnagent::MoeBindingView>(ffnagent, "MoeBindingView", "Non-owning MoE resources.")
      .def(py::init([](xpool::ffnagent::ProjectionBindingView gate_up, xpool::ffnagent::ProjectionBindingView down,
                       std::optional<xpool::ffnagent::MoeRouterBindingView> router) {
             return xpool::ffnagent::MoeBindingView{.gate_up = gate_up, .down = down, .router = std::move(router)};
           }),
           py::arg("gate_up"), py::arg("down"), py::arg("router"));

  py::class_<xpool::ffnagent::DenseCaptureDescriptor>(ffnagent, "DenseCaptureDescriptor",
                                                      "Captured Dense execution signature.")
      .def(py::init([](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                       std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                       std::uintptr_t control_graph_address, const at::Tensor &capture_input,
                       const at::Tensor &capture_partial,
                       std::vector<xpool::ffnagent::CapturedResourceRange> capture_ranges,
                       xpool::ffnagent::DenseBindingView primary_weight_bindings,
                       xpool::ffnagent::DenseBindingView control_weight_bindings) {
             const auto dtype = require_payload_dtype(payload_dtype, "FFN");
             const auto io_bytes = xpool::utils::checked::prod(payload_row_capacity, hidden_size,
                                                               static_cast<std::size_t>(c10::elementSize(dtype)));
             TORCH_CHECK(capture_input.scalar_type() == dtype && capture_partial.scalar_type() == dtype,
                         "xpool capture input and partial must match payload dtype");
             return xpool::ffnagent::DenseCaptureDescriptor{
                 .payload_dtype = dtype,
                 .payload_row_capacity = payload_row_capacity,
                 .hidden_size = hidden_size,
                 .local_intermediate_size = local_intermediate_size,
                 .primary_graph_address = primary_graph_address,
                 .control_graph_address = control_graph_address,
                 .capture_input_address = require_cuda_storage(capture_input, "Dense capture input", io_bytes),
                 .capture_partial_address = require_cuda_storage(capture_partial, "Dense capture partial", io_bytes),
                 .capture_ranges = std::move(capture_ranges),
                 .primary_weight_bindings = primary_weight_bindings,
                 .control_weight_bindings = control_weight_bindings,
             };
           }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input"), py::arg("capture_partial"), py::arg("capture_ranges"),
           py::arg("primary_weight_bindings"), py::arg("control_weight_bindings"));

  py::class_<xpool::ffnagent::MoeCaptureDescriptor>(ffnagent, "MoeCaptureDescriptor",
                                                    "Captured MoE execution signature.")
      .def(py::init(
               [](const py::object &payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                  std::size_t local_intermediate_size, std::size_t expert_count, std::size_t effective_topk,
                  std::optional<std::size_t> routed_expert_count, std::uintptr_t primary_graph_address,
                  std::uintptr_t control_graph_address, const at::Tensor &capture_input,
                  const at::Tensor &capture_partial, std::vector<xpool::ffnagent::CapturedResourceRange> capture_ranges,
                  const at::Tensor &capture_routing_metadata, const std::optional<at::Tensor> &capture_payload_rows,
                  xpool::ffnagent::MoeBindingView primary_weight_bindings,
                  xpool::ffnagent::MoeBindingView control_weight_bindings) {
                 const auto dtype = require_payload_dtype(payload_dtype, "FFN");
                 const auto io_bytes = xpool::utils::checked::prod(payload_row_capacity, hidden_size,
                                                                   static_cast<std::size_t>(c10::elementSize(dtype)));
                 const auto route_bytes = xpool::utils::checked::prod(payload_row_capacity, effective_topk,
                                                                      sizeof(std::int32_t) + sizeof(float));
                 TORCH_CHECK(capture_input.scalar_type() == dtype && capture_partial.scalar_type() == dtype,
                             "xpool capture input and partial must match payload dtype");
                 TORCH_CHECK(!capture_payload_rows.has_value() ||
                                 capture_payload_rows->scalar_type() == c10::ScalarType::Long,
                             "xpool MoE capture payload rows must have int64 dtype");
                 return xpool::ffnagent::MoeCaptureDescriptor{
                     .payload_dtype = dtype,
                     .payload_row_capacity = payload_row_capacity,
                     .hidden_size = hidden_size,
                     .local_intermediate_size = local_intermediate_size,
                     .expert_count = expert_count,
                     .effective_topk = effective_topk,
                     .routed_expert_count = routed_expert_count,
                     .primary_graph_address = primary_graph_address,
                     .control_graph_address = control_graph_address,
                     .capture_input_address = require_cuda_storage(capture_input, "MoE capture input", io_bytes),
                     .capture_partial_address = require_cuda_storage(capture_partial, "MoE capture partial", io_bytes),
                     .capture_ranges = std::move(capture_ranges),
                     .capture_routing_metadata_address =
                         require_cuda_storage(capture_routing_metadata, "MoE capture routing metadata", route_bytes),
                     .capture_payload_rows_address =
                         capture_payload_rows.has_value()
                             ? std::optional{require_cuda_storage(*capture_payload_rows, "MoE capture payload rows",
                                                                  sizeof(std::int64_t))}
                             : std::nullopt,
                     .primary_weight_bindings = std::move(primary_weight_bindings),
                     .control_weight_bindings = std::move(control_weight_bindings),
                 };
               }),
           py::arg("payload_dtype"), py::arg("payload_row_capacity"), py::arg("hidden_size"),
           py::arg("local_intermediate_size"), py::arg("expert_count"), py::arg("effective_topk"),
           py::arg("routed_expert_count"), py::arg("primary_graph_address"), py::arg("control_graph_address"),
           py::arg("capture_input"), py::arg("capture_partial"), py::arg("capture_ranges"),
           py::arg("capture_routing_metadata"), py::arg("capture_payload_rows"), py::arg("primary_weight_bindings"),
           py::arg("control_weight_bindings"));

  py::class_<xpool::ffnagent::LayerExecutionDescriptor>(ffnagent, "LayerExecutionDescriptor",
                                                        "One Plan-addressed local FFN layer.")
      .def(py::init([](std::size_t instance_index, std::size_t layer_ordinal,
                       std::vector<std::size_t> execution_signature_indices,
                       xpool::ffnagent::BindingView layer_resource_targets) {
             return xpool::ffnagent::LayerExecutionDescriptor{
                 .instance_index = instance_index,
                 .layer_ordinal = layer_ordinal,
                 .execution_signature_indices = std::move(execution_signature_indices),
                 .layer_resource_targets = std::move(layer_resource_targets),
             };
           }),
           py::arg("instance_index"), py::arg("layer_ordinal"), py::arg("execution_signature_indices"),
           py::arg("layer_resource_targets"));

  py::class_<xpool::ffnagent::LaneCaptureDescriptor>(ffnagent, "LaneCaptureDescriptor",
                                                     "Independently owned captures in deterministic Signature order.")
      .def(py::init<std::vector<xpool::ffnagent::CaptureDescriptor>>(), py::arg("signatures"));

  py::class_<xpool::ffnagent::ExecutionInstallDescriptor>(ffnagent, "ExecutionInstallDescriptor",
                                                          "Complete one-time native FFN installation input.")
      .def(py::init([](std::vector<xpool::ffnagent::LaneCaptureDescriptor> lanes,
                       std::vector<xpool::ffnagent::LayerExecutionDescriptor> layers) {
             auto projection = xpool::ffnagent::ExecutionInstallDescriptor{
                 .lanes = std::move(lanes),
                 .layers = std::move(layers),
             };
             projection.validate();
             return projection;
           }),
           py::arg("lanes"), py::arg("layers"));

  ffnagent.def("execution_state_allocation_bytes", &xpool::ffnagent::execution_state_allocation_bytes,
               py::arg("local_layer_count"), py::arg("local_capacity_count"), py::arg("local_signature_count"),
               py::arg("executor_lane_count"),
               "Return logical fixed-table and Lane-state request bytes, excluding variable binding/update arrays.");
  ffnagent.def(
      "install_execution",
      [](const xpool::ffnagent::ExecutionInstallDescriptor &projection) {
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
}

} // namespace xpool::bindings
