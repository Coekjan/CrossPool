#include <xpool/ffnagent/execution.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>

#include <c10/util/Exception.h>

#include <xpool/ffn.hpp>
#include <xpool/utils/checked.hpp>

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

void append_projection_intervals(std::vector<std::pair<std::uintptr_t, std::size_t>> &intervals,
                                 const ProjectionBindingView &projection) {
  intervals.emplace_back(projection.values, xpool::utils::checked::sum(projection.values, projection.values_bytes));
  if (projection.scales.has_value()) {
    intervals.emplace_back(*projection.scales, xpool::utils::checked::sum(*projection.scales, projection.scales_bytes));
  }
  if (projection.bias.has_value()) {
    intervals.emplace_back(*projection.bias, xpool::utils::checked::sum(*projection.bias, projection.bias_bytes));
  }
}

template <typename Signature> void validate_resources(const Signature &signature) {
  const auto io_bytes =
      xpool::utils::checked::prod(signature.payload_row_capacity, signature.hidden_size,
                                  static_cast<std::size_t>(c10::elementSize(signature.payload_dtype)));
  auto shared = std::vector<std::pair<std::uintptr_t, std::size_t>>{
      {signature.capture_input_address, xpool::utils::checked::sum(signature.capture_input_address, io_bytes)},
      {signature.capture_partial_address, xpool::utils::checked::sum(signature.capture_partial_address, io_bytes)},
  };
  if constexpr (std::is_same_v<Signature, MoeCaptureDescriptor>) {
    const auto route_bytes = xpool::utils::checked::prod(signature.payload_row_capacity, signature.effective_topk,
                                                         sizeof(std::int32_t) + sizeof(float));
    shared.emplace_back(signature.capture_routing_metadata_address,
                        xpool::utils::checked::sum(signature.capture_routing_metadata_address, route_bytes));
    if (signature.capture_payload_rows_address.has_value()) {
      shared.emplace_back(*signature.capture_payload_rows_address,
                          xpool::utils::checked::sum(*signature.capture_payload_rows_address, sizeof(std::int64_t)));
    }
  }
  for (const auto &[address, end] : shared) {
    TORCH_CHECK(address != 0, "xpool capture bindings require nonnull addresses");
  }
  auto primary = shared;
  auto control = shared;
  const auto append_resources = [](auto &intervals, const auto &resources) {
    append_projection_intervals(intervals, resources.gate_up);
    append_projection_intervals(intervals, resources.down);
    if constexpr (std::is_same_v<std::remove_cvref_t<decltype(resources)>, MoeBindingView>) {
      if (resources.router.has_value()) {
        append_projection_intervals(intervals, resources.router->projection);
        if (resources.router->correction_bias_address.has_value()) {
          intervals.emplace_back(*resources.router->correction_bias_address,
                                 xpool::utils::checked::sum(*resources.router->correction_bias_address,
                                                            resources.router->correction_bias_bytes));
        }
      }
    }
  };
  append_resources(primary, signature.primary_weight_bindings);
  append_resources(control, signature.control_weight_bindings);
  auto keys = std::unordered_set<std::string>{};
  for (const auto &region : signature.capture_ranges) {
    TORCH_CHECK(!region.key.empty() && keys.insert(region.key).second,
                "xpool captured resource identities must be nonempty and unique");
    TORCH_CHECK(region.bytes != 0, "xpool captured resource requires a positive logical extent");
    const auto append_region = [&](auto &intervals, std::uintptr_t address) {
      TORCH_CHECK(address != 0, "xpool captured resource requires a nonnull address");
      const auto end = xpool::utils::checked::sum(address, static_cast<std::uintptr_t>(region.bytes));
      for (const auto &[base, other_end] : intervals) {
        TORCH_CHECK(address >= other_end || base >= end, "xpool captured resource overlaps another declared resource");
      }
      intervals.emplace_back(address, end);
    };
    append_region(primary, region.primary_address);
    append_region(control, region.control_address);
  }
}

void validate_distinct_capture_addresses(const ProjectionBindingView &primary, const ProjectionBindingView &control) {
  TORCH_CHECK(primary.values != control.values && (!primary.scales.has_value() || primary.scales != control.scales) &&
                  (!primary.bias.has_value() || primary.bias != control.bias),
              "xpool Primary and Control projection resources must use distinct corresponding addresses");
}

void validate_capture_pair(const ProjectionBindingView &primary, const ProjectionBindingView &control) {
  primary.validate();
  control.validate();
  TORCH_CHECK(primary.schema_matches(control), "xpool Primary and Control projection schemas disagree");
  validate_distinct_capture_addresses(primary, control);
}

void validate_common(c10::ScalarType payload_dtype, std::size_t payload_row_capacity, std::size_t hidden_size,
                     std::size_t local_intermediate_size, std::uintptr_t primary_graph_address,
                     std::uintptr_t control_graph_address) {
  TORCH_CHECK(xpool::ffn::is_supported_payload_dtype(payload_dtype),
              "xpool ffn capture descriptor requires BF16 or FP16 execution");
  TORCH_CHECK(payload_row_capacity != 0 && hidden_size != 0 && local_intermediate_size != 0,
              "xpool ffn capture descriptor contains zero execution geometry");
  TORCH_CHECK(primary_graph_address != 0 && control_graph_address != 0 &&
                  primary_graph_address != control_graph_address,
              "xpool ffn capture descriptor primary and control graphs must be nonnull and distinct");
}

bool binding_schema_matches(const CaptureDescriptor &signature, const BindingView &target) {
  if (const auto *dense = std::get_if<DenseCaptureDescriptor>(&signature)) {
    const auto *resources = std::get_if<DenseBindingView>(&target);
    return resources != nullptr && dense->primary_weight_bindings.gate_up.schema_matches(resources->gate_up) &&
           dense->primary_weight_bindings.down.schema_matches(resources->down);
  }
  const auto *resources = std::get_if<MoeBindingView>(&target);
  if (resources == nullptr) {
    return false;
  }
  const auto &captured = std::get<MoeCaptureDescriptor>(signature).primary_weight_bindings;
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
           left_dense->primary_weight_bindings.gate_up.schema_matches(right_dense->primary_weight_bindings.gate_up) &&
           left_dense->primary_weight_bindings.down.schema_matches(right_dense->primary_weight_bindings.down);
  }
  const auto *left_moe = std::get_if<MoeCaptureDescriptor>(&left);
  const auto *right_moe = std::get_if<MoeCaptureDescriptor>(&right);
  if (right_moe == nullptr) {
    return false;
  }
  const auto &left_resources = left_moe->primary_weight_bindings;
  const auto &right_resources = right_moe->primary_weight_bindings;
  return left_moe->payload_dtype == right_moe->payload_dtype && left_moe->hidden_size == right_moe->hidden_size &&
         left_moe->local_intermediate_size == right_moe->local_intermediate_size &&
         left_moe->expert_count == right_moe->expert_count &&
         left_moe->routed_expert_count == right_moe->routed_expert_count &&
         left_resources.gate_up.schema_matches(right_resources.gate_up) &&
         left_resources.down.schema_matches(right_resources.down) &&
         left_resources.router.has_value() == right_resources.router.has_value() &&
         (!left_resources.router.has_value() || left_resources.router->schema_matches(*right_resources.router));
}

void validate_layers(const std::vector<CaptureDescriptor> &signatures,
                     const std::vector<LayerExecutionDescriptor> &layers) {
  TORCH_CHECK(signatures.empty() == layers.empty(),
              "xpool ffn installation descriptor must contain both local signatures and layers, or neither");

  auto next_first_use = std::size_t{0};
  auto previous_instance = std::size_t{0};
  auto previous_layer = std::size_t{0};
  auto first_layer = true;
  for (const auto &layer : layers) {
    std::visit(
        [](const auto &value) {
          value.gate_up.validate();
          value.down.validate();
          if constexpr (std::is_same_v<std::remove_cvref_t<decltype(value)>, MoeBindingView>) {
            if (value.router.has_value()) {
              value.router->validate();
            }
          }
        },
        layer.layer_resource_targets);
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
      TORCH_CHECK(signature_index <= next_first_use,
                  "xpool ffn installation descriptor execution signatures do not follow first-use plan order");
      if (signature_index == next_first_use) {
        ++next_first_use;
      }
    }
  }
  TORCH_CHECK(next_first_use == signatures.size(),
              "xpool ffn installation descriptor contains an unreferenced execution signature");
}

} // namespace

void DenseCaptureDescriptor::validate() const {
  validate_common(payload_dtype, payload_row_capacity, hidden_size, local_intermediate_size, primary_graph_address,
                  control_graph_address);
  const auto &primary = primary_weight_bindings;
  const auto &control = control_weight_bindings;
  validate_capture_pair(primary.gate_up, control.gate_up);
  validate_capture_pair(primary.down, control.down);
  validate_resources(*this);
}

void MoeCaptureDescriptor::validate() const {
  validate_common(payload_dtype, payload_row_capacity, hidden_size, local_intermediate_size, primary_graph_address,
                  control_graph_address);
  TORCH_CHECK(expert_count != 0 && effective_topk != 0 && effective_topk <= expert_count,
              "xpool moe capture descriptor contains invalid expert geometry");
  const auto router_owner = routed_expert_count.has_value();
  const auto &primary = primary_weight_bindings;
  const auto &control = control_weight_bindings;
  TORCH_CHECK(capture_payload_rows_address.has_value() == router_owner && primary.router.has_value() == router_owner &&
                  control.router.has_value() == router_owner,
              "xpool moe capture descriptor router ownership and capture resources disagree");
  validate_capture_pair(primary.gate_up, control.gate_up);
  validate_capture_pair(primary.down, control.down);
  validate_resources(*this);
  if (!router_owner) {
    return;
  }
  TORCH_CHECK(*routed_expert_count != 0 && *routed_expert_count <= expert_count,
              "xpool moe capture descriptor contains invalid routed expert geometry");
  const auto &primary_router = *primary.router;
  const auto &control_router = *control.router;
  primary_router.validate();
  control_router.validate();
  TORCH_CHECK(primary_router.schema_matches(control_router),
              "xpool moe capture descriptor primary and control router schemas disagree");
  validate_distinct_capture_addresses(primary_router.projection, control_router.projection);
  TORCH_CHECK(!primary_router.correction_bias_address.has_value() ||
                  primary_router.correction_bias_address != control_router.correction_bias_address,
              "xpool moe capture descriptor primary and control router resources must use distinct corresponding "
              "addresses");
}

void ExecutionInstallDescriptor::validate() const {
  TORCH_CHECK(!lanes.empty(), "xpool execution installation requires its Fabric Lane inventory");
  const auto &reference = lanes.front().signatures;
  for (const auto &lane : lanes) {
    for (const auto &signature : lane.signatures) {
      std::visit([](const auto &value) { value.validate(); }, signature);
    }
  }
  validate_layers(reference, layers);
  for (auto lane_index = std::size_t{1}; lane_index < lanes.size(); ++lane_index) {
    const auto &signatures = lanes[lane_index].signatures;
    TORCH_CHECK(signatures.size() == reference.size(), "xpool captured Lanes disagree on Signature count");
    for (auto index = std::size_t{0}; index < signatures.size(); ++index) {
      const auto &left = reference[index];
      const auto &right = signatures[index];
      TORCH_CHECK(weight_geometry_matches(left, right), "xpool captured Lanes disagree on weight geometry");
      std::visit(
          [&](const auto &capture) {
            using Signature = std::remove_cvref_t<decltype(capture)>;
            const auto &other = std::get<Signature>(right);
            TORCH_CHECK(capture.payload_row_capacity == other.payload_row_capacity,
                        "xpool captured Lanes disagree on Capacity order");
            if constexpr (std::is_same_v<Signature, MoeCaptureDescriptor>) {
              TORCH_CHECK(capture.effective_topk == other.effective_topk,
                          "xpool captured Lanes disagree on routing geometry");
            }
            TORCH_CHECK(capture.capture_ranges.size() == other.capture_ranges.size(),
                        "xpool captured Lanes disagree on resource schema");
            for (auto resource = std::size_t{0}; resource < capture.capture_ranges.size(); ++resource) {
              const auto &expected = capture.capture_ranges[resource];
              const auto &actual = other.capture_ranges[resource];
              TORCH_CHECK(expected.key == actual.key && expected.bytes == actual.bytes &&
                              expected.shared_read_only == actual.shared_read_only,
                          "xpool captured Lanes disagree on resource schema");
            }
          },
          left);
    }
    for (auto previous = std::size_t{0}; previous < lane_index; ++previous) {
      for (const auto &current : signatures) {
        for (const auto &earlier : lanes[previous].signatures) {
          std::visit(
              [](const auto &left, const auto &right) {
                TORCH_CHECK(left.primary_graph_address != right.primary_graph_address &&
                                left.primary_graph_address != right.control_graph_address &&
                                left.control_graph_address != right.primary_graph_address &&
                                left.control_graph_address != right.control_graph_address,
                            "xpool Lanes must own distinct capture graphs");
                for (const auto &a : left.capture_ranges) {
                  for (const auto &b : right.capture_ranges) {
                    for (const auto a_address : {a.primary_address, a.control_address}) {
                      for (const auto b_address : {b.primary_address, b.control_address}) {
                        const auto overlap = a_address < b_address + b.bytes && b_address < a_address + a.bytes;
                        TORCH_CHECK(!overlap || (a.shared_read_only && b.shared_read_only && a.key == b.key &&
                                                 a_address == b_address && a.bytes == b.bytes),
                                    "xpool writable capture resources overlap across Lanes");
                      }
                    }
                  }
                }
              },
              current, earlier);
        }
      }
    }
  }
}

} // namespace xpool::ffnagent
