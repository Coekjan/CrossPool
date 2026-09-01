#include <xpool/ffnagent/parameterization.hpp>
#include <xpool/utils/graph.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/util/Exception.h>

namespace xpool::ffnagent {

namespace {

void parameterize_graph_recursive(
    cudaGraph_t primary_graph, cudaGraph_t control_graph, cuda::std::span<const ResourceReplacement> resources,
    cuda::std::span<const LaneAddressReplacement> lane_replacements, cuda::std::span<std::size_t> resource_matches,
    cuda::std::span<std::size_t> lane_matches, std::uintptr_t capture_routing_metadata_address,
    std::uintptr_t capture_routing_weights_address, std::uintptr_t capture_payload_rows_address,
    GraphParameterization &parameterization) {
  // Phase: Verify Topology - Primary and Control captures must have identical
  // direct topology before their nodes can be compared recursively.
  const auto primary_nodes = xpool::utils::graph::nodes(primary_graph);
  const auto control_nodes = xpool::utils::graph::nodes(control_graph);
  TORCH_CHECK(xpool::utils::graph::same_direct_topology(primary_graph, control_graph),
              "xpool Primary and Control Graph topologies differ");

  for (auto node_index = std::size_t{0}; node_index < primary_nodes.size(); ++node_index) {
    const auto primary_type = xpool::utils::graph::node_type(primary_nodes[node_index]);
    if (primary_type == cudaGraphNodeTypeGraph) {
      const auto primary_child = xpool::utils::graph::child_graph(primary_nodes[node_index]);
      const auto control_child = xpool::utils::graph::child_graph(control_nodes[node_index]);
      parameterize_graph_recursive(primary_child, control_child, resources, lane_replacements, resource_matches,
                                   lane_matches, capture_routing_metadata_address, capture_routing_weights_address,
                                   capture_payload_rows_address, parameterization);
      continue;
    }
    if (primary_type == cudaGraphNodeTypeEmpty) {
      continue;
    }
    TORCH_CHECK(primary_type == cudaGraphNodeTypeKernel, "xpool Primary Graph contains an unsupported node kind");

    auto primary = xpool::utils::graph::KernelNodeParameters::read(primary_nodes[node_index]);
    const auto control = xpool::utils::graph::KernelNodeParameters::read(control_nodes[node_index]);
    TORCH_CHECK(primary.same_schema(control), "xpool Primary and Control Graph kernel schemas differ");
    const auto routing_candidate = capture_payload_rows_address != 0 &&
                                   primary.contains_address(capture_payload_rows_address) &&
                                   primary.contains_address(capture_routing_metadata_address) &&
                                   primary.contains_address(capture_routing_weights_address);
    if (routing_candidate) {
      TORCH_CHECK(!parameterization.routing_finalization.has_value(),
                  "xpool Primary Graph has multiple Routing Finalization candidates");
      parameterization.routing_finalization =
          PrimaryGraphLocation{.graph = primary_graph, .node = primary_nodes[node_index]};
    }

    // Phase: Discover Bindings - The captures may differ only at declared
    // layer-resource addresses; each match becomes one Device-updatable site.
    auto changed = false;
    auto node_matches = std::vector<std::pair<std::size_t, std::size_t>>{};
    for (auto parameter_index = std::size_t{0}; parameter_index < primary.arguments().size(); ++parameter_index) {
      auto &primary_value = primary.arguments()[parameter_index];
      const auto &control_value = control.arguments()[parameter_index];
      auto covered = std::vector<bool>(primary_value.bytes.size(), false);
      for (auto byte_offset = std::size_t{0}; byte_offset + sizeof(std::uintptr_t) <= primary_value.bytes.size();
           ++byte_offset) {
        const auto primary_address = primary_value.read_address(byte_offset);
        const auto control_address = control_value.read_address(byte_offset);
        auto matched_resource = resources.size();
        for (auto resource_index = std::size_t{0}; resource_index < resources.size(); ++resource_index) {
          const auto &resource = resources[resource_index];
          if (primary_address == resource.primary_address && control_address == resource.control_address) {
            TORCH_CHECK(matched_resource == resources.size(),
                        "xpool Primary Graph resource address match is ambiguous");
            matched_resource = resource_index;
          }
        }
        if (matched_resource == resources.size()) {
          continue;
        }
        TORCH_CHECK(std::none_of(covered.begin() + static_cast<std::ptrdiff_t>(byte_offset),
                                 covered.begin() + static_cast<std::ptrdiff_t>(byte_offset + sizeof(std::uintptr_t)),
                                 [](bool value) { return value; }),
                    "xpool Primary Graph resource sites overlap");
        std::fill_n(covered.begin() + static_cast<std::ptrdiff_t>(byte_offset), sizeof(std::uintptr_t), true);
        node_matches.emplace_back(primary_value.offset_bytes + byte_offset, matched_resource);
        ++resource_matches[matched_resource];
        primary_value.write_address(byte_offset, resources[matched_resource].target_address);
        changed = true;
      }
      for (auto byte_offset = std::size_t{0}; byte_offset < primary_value.bytes.size(); ++byte_offset) {
        // Any uncovered byte delta changes behavior rather than merely naming
        // a resource that the runtime intends to rebind.
        TORCH_CHECK(primary_value.bytes[byte_offset] == control_value.bytes[byte_offset] || covered[byte_offset],
                    "xpool Primary Graph contains an undeclared parameter-byte delta");
      }

      // Phase: Relocate Lane Addresses - Captured input, output, routing, and
      // workspace addresses become stable storage owned by this Lane.
      for (auto byte_offset = std::size_t{0}; byte_offset + sizeof(std::uintptr_t) <= primary_value.bytes.size();
           ++byte_offset) {
        const auto address = primary_value.read_address(byte_offset);
        auto matched_replacement = lane_replacements.size();
        auto target = std::uintptr_t{0};
        for (auto replacement_index = std::size_t{0}; replacement_index < lane_replacements.size();
             ++replacement_index) {
          const auto &replacement = lane_replacements[replacement_index];
          if (address < replacement.capture_address || address - replacement.capture_address >= replacement.bytes) {
            continue;
          }
          TORCH_CHECK(matched_replacement == lane_replacements.size(),
                      "xpool Primary Graph lane-resource address match is ambiguous");
          matched_replacement = replacement_index;
          target = replacement.target_address + (address - replacement.capture_address);
        }
        if (matched_replacement == lane_replacements.size()) {
          continue;
        }
        primary_value.write_address(byte_offset, target);
        ++lane_matches[matched_replacement];
        changed = true;
      }
    }

    // Phase: Apply Node Changes - Record runtime binding offsets before
    // committing the rewritten argument storage to the embedded graph.
    if (!node_matches.empty()) {
      const auto device_node = xpool::utils::graph::make_device_updatable(primary_nodes[node_index]);
      for (const auto &[parameter_offset, resource_index] : node_matches) {
        parameterization.binding_sites.push_back(BindingSite{
            .node = device_node,
            .parameter_offset_bytes = parameter_offset,
            .value_offset_bytes = resources[resource_index].value_offset_bytes,
        });
      }
    }
    if (changed) {
      primary.apply(primary_nodes[node_index]);
    }
  }
}

} // namespace

GraphParameterization parameterize_graph(cudaGraph_t primary_graph, cudaGraph_t control_graph,
                                         cuda::std::span<const ResourceReplacement> resources,
                                         cuda::std::span<const LaneAddressReplacement> lane_replacements,
                                         std::uintptr_t capture_routing_metadata_address,
                                         std::uintptr_t capture_routing_weights_address,
                                         std::uintptr_t capture_payload_rows_address) {
  auto parameterization = GraphParameterization{};
  auto resource_matches = std::vector<std::size_t>(resources.size(), 0);
  auto lane_matches = std::vector<std::size_t>(lane_replacements.size(), 0);
  parameterize_graph_recursive(primary_graph, control_graph, resources, lane_replacements, resource_matches,
                               lane_matches, capture_routing_metadata_address, capture_routing_weights_address,
                               capture_payload_rows_address, parameterization);
  TORCH_CHECK(std::ranges::all_of(resource_matches, [](std::size_t count) { return count != 0; }),
              "xpool Primary Graph omits a required layer-resource binding");
  TORCH_CHECK(std::ranges::all_of(lane_matches, [](std::size_t count) { return count != 0; }),
              "xpool Primary Graph omits a required lane-resource binding");
  return parameterization;
}

} // namespace xpool::ffnagent
