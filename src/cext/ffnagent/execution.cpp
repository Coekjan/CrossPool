#include <xpool/ffnagent/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <variant>

#include <c10/util/Exception.h>

#include <xpool/ffn.hpp>

namespace xpool::ffnagent {

void ProjectionBindingView::validate() const {
  TORCH_CHECK(values != 0 && values_bytes != 0,
              "xpool projection values require a nonnull address and positive extent");
  TORCH_CHECK(scales.has_value() ? *scales != 0 && scales_bytes != 0 : scales_bytes == 0,
              "xpool projection scale presence and byte extent disagree");
  TORCH_CHECK(bias.has_value() ? *bias != 0 && bias_bytes != 0 : bias_bytes == 0,
              "xpool projection bias presence and byte extent disagree");
}

bool ProjectionBindingView::schema_matches(const ProjectionBindingView &other) const {
  return values_bytes == other.values_bytes && scales.has_value() == other.scales.has_value() &&
         scales_bytes == other.scales_bytes && bias.has_value() == other.bias.has_value() &&
         bias_bytes == other.bias_bytes;
}

void MoeRouterBindingView::validate() const {
  projection.validate();
  TORCH_CHECK(correction_bias_address.has_value() ? *correction_bias_address != 0 && correction_bias_bytes != 0
                                                  : correction_bias_bytes == 0,
              "xpool Router correction-bias presence and byte extent disagree");
}

bool MoeRouterBindingView::schema_matches(const MoeRouterBindingView &other) const {
  return projection.schema_matches(other.projection) &&
         correction_bias_address.has_value() == other.correction_bias_address.has_value() &&
         correction_bias_bytes == other.correction_bias_bytes;
}

namespace {

void validate_capture_pair(const ProjectionBindingView &primary, const ProjectionBindingView &control) {
  primary.validate();
  control.validate();
  TORCH_CHECK(primary.schema_matches(control), "xpool Primary and Control projection schemas disagree");
  TORCH_CHECK(primary.values != control.values && (!primary.scales.has_value() || primary.scales != control.scales) &&
                  (!primary.bias.has_value() || primary.bias != control.bias),
              "xpool Primary and Control projection resources must use distinct corresponding addresses");
}

void validate_common(c10::ScalarType payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                     std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                     std::uintptr_t control_graph_address, std::size_t compute_workspace_bytes) {
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
              "xpool ffn capture descriptor requires BF16 or FP16 execution");
  TORCH_CHECK(payload_row_capacity != 0 && hidden_size != 0 && local_intermediate_size != 0,
              "xpool ffn capture descriptor contains zero execution geometry");
  TORCH_CHECK(primary_graph_address != control_graph_address,
              "xpool ffn capture descriptor primary and control graphs must be distinct");
  TORCH_CHECK(compute_workspace_bytes != 0,
              "xpool ffn capture descriptor contains an invalid captured compute-workspace size");
}

void validate_signature(const DenseCaptureDescriptor &signature) {
  validate_common(signature.payload_dtype, signature.payload_row_capacity, signature.hidden_size,
                  signature.local_intermediate_size, signature.primary_graph_address, signature.control_graph_address,
                  signature.compute_workspace_bytes);
  const auto &primary = signature.primary_capture_resources;
  const auto &control = signature.control_capture_resources;
  validate_capture_pair(primary.gate_up, control.gate_up);
  validate_capture_pair(primary.down, control.down);
}

void validate_signature(const MoeCaptureDescriptor &signature) {
  validate_common(signature.payload_dtype, signature.payload_row_capacity, signature.hidden_size,
                  signature.local_intermediate_size, signature.primary_graph_address, signature.control_graph_address,
                  signature.compute_workspace_bytes);
  TORCH_CHECK(signature.expert_count != 0 && signature.effective_topk != 0 &&
                  signature.effective_topk <= signature.expert_count,
              "xpool moe capture descriptor contains invalid expert geometry");
  const auto router_owner = signature.routed_expert_count.has_value();
  const auto &primary = signature.primary_capture_resources;
  const auto &control = signature.control_capture_resources;
  TORCH_CHECK(signature.capture_payload_rows_address.has_value() == router_owner &&
                  primary.router.has_value() == router_owner && control.router.has_value() == router_owner,
              "xpool moe capture descriptor router ownership and capture resources disagree");
  validate_capture_pair(primary.gate_up, control.gate_up);
  validate_capture_pair(primary.down, control.down);
  if (!router_owner) {
    return;
  }
  TORCH_CHECK(*signature.routed_expert_count != 0 && *signature.routed_expert_count <= signature.expert_count,
              "xpool moe capture descriptor contains invalid routed expert geometry");
  const auto &primary_router = *primary.router;
  const auto &control_router = *control.router;
  primary_router.validate();
  control_router.validate();
  TORCH_CHECK(primary_router.schema_matches(control_router),
              "xpool moe capture descriptor primary and control router schemas disagree");
  validate_capture_pair(primary_router.projection, control_router.projection);
  TORCH_CHECK(!primary_router.correction_bias_address.has_value() ||
                  primary_router.correction_bias_address != control_router.correction_bias_address,
              "xpool moe capture descriptor primary and control router resources must use distinct corresponding "
              "addresses");
}

bool binding_schema_matches(const CaptureDescriptor &signature, const BindingView &target) {
  std::visit(
      [](const auto &value) {
        value.gate_up.validate();
        value.down.validate();
      },
      target);
  if (const auto *dense = std::get_if<DenseCaptureDescriptor>(&signature)) {
    const auto *resources = std::get_if<DenseBindingView>(&target);
    return resources != nullptr && dense->primary_capture_resources.gate_up.schema_matches(resources->gate_up) &&
           dense->primary_capture_resources.down.schema_matches(resources->down);
  }
  const auto *resources = std::get_if<MoeBindingView>(&target);
  if (resources == nullptr) {
    return false;
  }
  if (resources->router.has_value()) {
    resources->router->validate();
  }
  const auto &captured = std::get<MoeCaptureDescriptor>(signature).primary_capture_resources;
  return captured.gate_up.schema_matches(resources->gate_up) && captured.down.schema_matches(resources->down) &&
         captured.router.has_value() == resources->router.has_value() &&
         (!captured.router.has_value() || captured.router->schema_matches(*resources->router));
}

bool weight_geometry_matches(const CaptureDescriptor &left, const CaptureDescriptor &right) {
  if (const auto *left_dense = std::get_if<DenseCaptureDescriptor>(&left)) {
    const auto *right_dense = std::get_if<DenseCaptureDescriptor>(&right);
    return right_dense != nullptr && left_dense->payload_dtype == right_dense->payload_dtype &&
           left_dense->hidden_size == right_dense->hidden_size &&
           left_dense->local_intermediate_size == right_dense->local_intermediate_size &&
           left_dense->primary_capture_resources.gate_up.schema_matches(
               right_dense->primary_capture_resources.gate_up) &&
           left_dense->primary_capture_resources.down.schema_matches(right_dense->primary_capture_resources.down);
  }
  const auto *left_moe = std::get_if<MoeCaptureDescriptor>(&left);
  const auto *right_moe = std::get_if<MoeCaptureDescriptor>(&right);
  if (right_moe == nullptr) {
    return false;
  }
  const auto &left_resources = left_moe->primary_capture_resources;
  const auto &right_resources = right_moe->primary_capture_resources;
  return left_moe->payload_dtype == right_moe->payload_dtype && left_moe->hidden_size == right_moe->hidden_size &&
         left_moe->local_intermediate_size == right_moe->local_intermediate_size &&
         left_moe->expert_count == right_moe->expert_count &&
         left_moe->routed_expert_count == right_moe->routed_expert_count &&
         left_resources.gate_up.schema_matches(right_resources.gate_up) &&
         left_resources.down.schema_matches(right_resources.down) &&
         left_resources.router.has_value() == right_resources.router.has_value() &&
         (!left_resources.router.has_value() || left_resources.router->schema_matches(*right_resources.router));
}

} // namespace

void ExecutionInstallDescriptor::validate() const {
  for (const auto &signature : signatures) {
    std::visit([](const auto &value) { validate_signature(value); }, signature);
  }
  TORCH_CHECK(signatures.empty() == layers.empty(),
              "xpool ffn installation descriptor must contain both local signatures and layers, or neither");

  auto referenced = std::vector<bool>(signatures.size(), false);
  auto next_first_use = std::size_t{0};
  auto previous_instance = std::size_t{0};
  auto previous_layer = std::size_t{0};
  auto first_layer = true;
  for (const auto &layer : layers) {
    TORCH_CHECK(first_layer || layer.instance_index > previous_instance ||
                    (layer.instance_index == previous_instance && layer.layer_ordinal > previous_layer),
                "xpool ffn installation descriptor layers are not in strict instance/layer order");
    first_layer = false;
    previous_instance = layer.instance_index;
    previous_layer = layer.layer_ordinal;
    TORCH_CHECK(!layer.execution_signature_indices.empty(),
                "xpool ffn installation descriptor contains a layer with no execution signature");
    std::optional<std::size_t> previous_capacity;
    const CaptureDescriptor *representative = nullptr;
    for (const auto signature_index : layer.execution_signature_indices) {
      TORCH_CHECK(signature_index < signatures.size(),
                  "xpool ffn installation descriptor contains an out-of-range execution signature index");
      const auto &signature = signatures[signature_index];
      const auto current_capacity = std::visit([](const auto &value) { return value.payload_row_capacity; }, signature);
      TORCH_CHECK(!previous_capacity.has_value() || current_capacity > *previous_capacity,
                  "xpool ffn installation descriptor layer execution signatures are not in increasing capacity order");
      previous_capacity = current_capacity;
      TORCH_CHECK(binding_schema_matches(signature, layer.layer_resource_targets),
                  "xpool ffn installation descriptor layer target resources disagree with its execution signature");
      if (representative == nullptr) {
        representative = &signature;
      } else {
        TORCH_CHECK(weight_geometry_matches(*representative, signature),
                    "xpool ffn installation descriptor layer execution signatures disagree on weight geometry");
      }
      if (!referenced[signature_index]) {
        TORCH_CHECK(signature_index == next_first_use,
                    "xpool ffn installation descriptor execution signatures do not follow first-use plan order");
        referenced[signature_index] = true;
        ++next_first_use;
      }
    }
  }
  TORCH_CHECK(next_first_use == signatures.size(),
              "xpool ffn installation descriptor contains an unreferenced execution signature");
}

} // namespace xpool::ffnagent
