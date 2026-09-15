#include <xpool/ffnagent/projection.hpp>

#include <cstddef>
#include <cstdint>
#include <variant>

#include <c10/util/Exception.h>

#include <xpool/ffn.hpp>

namespace xpool::ffnagent {

namespace {

void validate_common(c10::ScalarType payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                     std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                     std::uintptr_t control_graph_address, std::size_t compute_workspace_bytes) {
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
              "xpool FFN Execution Projection requires BF16 or FP16 execution");
  TORCH_CHECK(payload_row_capacity != 0 && hidden_size != 0 && local_intermediate_size != 0,
              "xpool FFN Execution Projection contains zero execution geometry");
  TORCH_CHECK(primary_graph_address != control_graph_address,
              "xpool FFN Execution Projection Primary and Control Graphs must be distinct");
  TORCH_CHECK(compute_workspace_bytes != 0,
              "xpool FFN Execution Projection contains an invalid captured compute-workspace size");
}

void validate_signature(const DenseExecutionSignatureProjection &signature) {
  validate_common(signature.payload_dtype, signature.payload_row_capacity, signature.hidden_size,
                  signature.local_intermediate_size, signature.primary_graph_address, signature.control_graph_address,
                  signature.compute_workspace_bytes);
  const auto &primary = signature.primary_capture_resources;
  const auto &control = signature.control_capture_resources;
  TORCH_CHECK(primary.gate_up_weight_address != control.gate_up_weight_address &&
                  primary.down_weight_address != control.down_weight_address,
              "xpool FFN Execution Projection Primary and Control Dense Capture resources must use distinct "
              "corresponding addresses");
}

void validate_signature(const MoeExecutionSignatureProjection &signature) {
  validate_common(signature.payload_dtype, signature.payload_row_capacity, signature.hidden_size,
                  signature.local_intermediate_size, signature.primary_graph_address, signature.control_graph_address,
                  signature.compute_workspace_bytes);
  TORCH_CHECK(signature.expert_count != 0 && signature.effective_topk != 0 &&
                  signature.effective_topk <= signature.expert_count,
              "xpool FFN Execution Projection contains invalid MoE Expert geometry");
  const auto router_owner = signature.routed_expert_count.has_value();
  const auto &primary = signature.primary_capture_resources;
  const auto &control = signature.control_capture_resources;
  TORCH_CHECK(signature.capture_payload_rows_address.has_value() == router_owner &&
                  primary.router.has_value() == router_owner && control.router.has_value() == router_owner,
              "xpool FFN Execution Projection MoE Router ownership and Capture resources disagree");
  TORCH_CHECK(primary.expert_gate_up_weight_address != control.expert_gate_up_weight_address &&
                  primary.expert_down_weight_address != control.expert_down_weight_address,
              "xpool FFN Execution Projection Primary and Control MoE Expert resources must use distinct "
              "corresponding addresses");
  if (!router_owner) {
    return;
  }
  TORCH_CHECK(*signature.routed_expert_count != 0 && *signature.routed_expert_count <= signature.expert_count,
              "xpool FFN Execution Projection contains invalid routed Expert geometry");
  const auto &primary_router = *primary.router;
  const auto &control_router = *control.router;
  TORCH_CHECK(primary_router.correction_bias_address.has_value() == control_router.correction_bias_address.has_value(),
              "xpool FFN Execution Projection Primary and Control Router schemas disagree");
  TORCH_CHECK(primary_router.weight_address != control_router.weight_address &&
                  (!primary_router.correction_bias_address.has_value() ||
                   primary_router.correction_bias_address != control_router.correction_bias_address),
              "xpool FFN Execution Projection Primary and Control Router resources must use distinct corresponding "
              "addresses");
}

bool binding_schema_matches(const ExecutionSignatureProjection &signature, const BindingResourceProjection &target) {
  if (std::holds_alternative<DenseExecutionSignatureProjection>(signature)) {
    return std::holds_alternative<DenseBindingResourceProjection>(target);
  }
  const auto *resources = std::get_if<MoeBindingResourceProjection>(&target);
  if (resources == nullptr) {
    return false;
  }
  const auto &captured = std::get<MoeExecutionSignatureProjection>(signature).primary_capture_resources;
  return captured.router.has_value() == resources->router.has_value() &&
         (!captured.router.has_value() || captured.router->correction_bias_address.has_value() ==
                                              resources->router->correction_bias_address.has_value());
}

bool weight_geometry_matches(const ExecutionSignatureProjection &left, const ExecutionSignatureProjection &right) {
  if (const auto *left_dense = std::get_if<DenseExecutionSignatureProjection>(&left)) {
    const auto *right_dense = std::get_if<DenseExecutionSignatureProjection>(&right);
    return right_dense != nullptr && left_dense->hidden_size == right_dense->hidden_size &&
           left_dense->local_intermediate_size == right_dense->local_intermediate_size;
  }
  const auto *left_moe = std::get_if<MoeExecutionSignatureProjection>(&left);
  const auto *right_moe = std::get_if<MoeExecutionSignatureProjection>(&right);
  if (right_moe == nullptr) {
    return false;
  }
  const auto left_bias = left_moe->primary_capture_resources.router.has_value() &&
                         left_moe->primary_capture_resources.router->correction_bias_address.has_value();
  const auto right_bias = right_moe->primary_capture_resources.router.has_value() &&
                          right_moe->primary_capture_resources.router->correction_bias_address.has_value();
  return left_moe->hidden_size == right_moe->hidden_size &&
         left_moe->local_intermediate_size == right_moe->local_intermediate_size &&
         left_moe->expert_count == right_moe->expert_count &&
         left_moe->routed_expert_count == right_moe->routed_expert_count && left_bias == right_bias;
}

} // namespace

void ExecutionProjection::validate() const {
  for (const auto &signature : signatures) {
    std::visit([](const auto &value) { validate_signature(value); }, signature);
  }
  TORCH_CHECK(signatures.empty() == layers.empty(),
              "xpool FFN Execution Projection must contain both local signatures and layers, or neither");

  auto referenced = std::vector<bool>(signatures.size(), false);
  auto next_first_use = std::size_t{0};
  auto previous_instance = std::size_t{0};
  auto previous_layer = std::size_t{0};
  auto first_layer = true;
  for (const auto &layer : layers) {
    TORCH_CHECK(first_layer || layer.instance_index > previous_instance ||
                    (layer.instance_index == previous_instance && layer.layer_ordinal > previous_layer),
                "xpool FFN Execution Projection layers are not in strict instance/layer order");
    first_layer = false;
    previous_instance = layer.instance_index;
    previous_layer = layer.layer_ordinal;
    TORCH_CHECK(!layer.execution_signature_indices.empty(),
                "xpool FFN Execution Projection contains a layer with no Execution Signature");
    std::optional<std::size_t> previous_capacity;
    const ExecutionSignatureProjection *representative = nullptr;
    for (const auto signature_index : layer.execution_signature_indices) {
      TORCH_CHECK(signature_index < signatures.size(),
                  "xpool FFN Execution Projection contains an out-of-range Execution Signature index");
      const auto &signature = signatures[signature_index];
      const auto current_capacity = std::visit([](const auto &value) { return value.payload_row_capacity; }, signature);
      TORCH_CHECK(!previous_capacity.has_value() || current_capacity > *previous_capacity,
                  "xpool FFN Execution Projection layer Execution Signatures are not in increasing Capacity order");
      previous_capacity = current_capacity;
      TORCH_CHECK(binding_schema_matches(signature, layer.layer_resource_targets),
                  "xpool FFN Execution Projection layer target resources disagree with its Execution Signature");
      if (representative == nullptr) {
        representative = &signature;
      } else {
        TORCH_CHECK(weight_geometry_matches(*representative, signature),
                    "xpool FFN Execution Projection layer Execution Signatures disagree on weight geometry");
      }
      if (!referenced[signature_index]) {
        TORCH_CHECK(signature_index == next_first_use,
                    "xpool FFN Execution Projection Execution Signatures do not follow first-use Plan order");
        referenced[signature_index] = true;
        ++next_first_use;
      }
    }
  }
  TORCH_CHECK(next_first_use == signatures.size(),
              "xpool FFN Execution Projection contains an unreferenced Execution Signature");
}

} // namespace xpool::ffnagent
