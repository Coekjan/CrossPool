#include <xpool/ffnagent/parameterization.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <xpool/utils/checked.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace xpool::ffnagent {

namespace {

std::uintptr_t bind_memory_address(std::uintptr_t primary, std::uintptr_t control, std::size_t bytes, bool writing,
                                   cuda::std::span<const LaneAddressReplacement> replacements,
                                   cuda::std::span<std::size_t> matches) {
  auto found = false;
  auto target = primary;
  for (auto index = std::size_t{0}; index < replacements.size(); ++index) {
    const auto &range = replacements[index];
    const auto in_primary = primary >= range.primary_address && primary - range.primary_address < range.bytes;
    const auto in_control = control >= range.control_address && control - range.control_address < range.bytes;
    if (!in_primary && !in_control) {
      continue;
    }
    TORCH_CHECK(in_primary && in_control && primary - range.primary_address == control - range.control_address,
                "xpool typed graph access changes its resource or relative offset");
    TORCH_CHECK(!found, "xpool typed graph access has ambiguous resources");
    const auto offset = primary - range.primary_address;
    TORCH_CHECK(bytes <= range.bytes - offset, "xpool typed graph access exceeds a declared resource");
    TORCH_CHECK(!writing || !range.read_only, "xpool typed graph writes a shared read-only resource");
    target = xpool::utils::checked::sum(range.target_address, offset);
    ++matches[index];
    found = true;
  }
  TORCH_CHECK(found, "xpool typed graph access has no declared resource owner");
  return target;
}

void parameterize_graph_recursive(
    cudaGraph_t primary_graph, cudaGraph_t control_graph, cuda::std::span<const ResourceReplacement> resources,
    cuda::std::span<const LaneAddressReplacement> lane_replacements, cuda::std::span<std::size_t> resource_matches,
    cuda::std::span<std::size_t> lane_matches, std::uintptr_t capture_routing_metadata_address,
    std::uintptr_t capture_routing_weights_address, std::uintptr_t capture_payload_rows_address,
    GraphParameterization &parameterization) {
  // Phase: Verify Topology - Primary and Control captures must have identical
  // direct topology before their nodes can be compared recursively.
  const auto primary_nodes = xpool::utils::cuda::graph::nodes(primary_graph);
  const auto control_nodes = xpool::utils::cuda::graph::nodes(control_graph);
  TORCH_CHECK(xpool::utils::cuda::graph::same_direct_topology(primary_graph, control_graph),
              "xpool Primary and Control Graph topologies differ");

  for (auto node_index = std::size_t{0}; node_index < primary_nodes.size(); ++node_index) {
    const auto primary_type = xpool::utils::cuda::graph::node_type(primary_nodes[node_index]);
    if (primary_type == cudaGraphNodeTypeGraph) {
      const auto primary_child = xpool::utils::cuda::graph::child_graph(primary_nodes[node_index]);
      const auto control_child = xpool::utils::cuda::graph::child_graph(control_nodes[node_index]);
      parameterize_graph_recursive(primary_child, control_child, resources, lane_replacements, resource_matches,
                                   lane_matches, capture_routing_metadata_address, capture_routing_weights_address,
                                   capture_payload_rows_address, parameterization);
      continue;
    }
    if (primary_type == cudaGraphNodeTypeMemset) {
      auto primary = xpool::utils::cuda::graph::memset_parameters(primary_nodes[node_index]);
      const auto control = xpool::utils::cuda::graph::memset_parameters(control_nodes[node_index]);
      TORCH_CHECK(primary.value == control.value && primary.elementSize == control.elementSize &&
                      primary.width == control.width && primary.height == control.height &&
                      primary.pitch == control.pitch,
                  "xpool Primary and Control Memset metadata differ");
      const auto bytes = xpool::utils::cuda::graph::memset_access_bytes(primary);
      primary.dst = reinterpret_cast<void *>(bind_memory_address(reinterpret_cast<std::uintptr_t>(primary.dst),
                                                                 reinterpret_cast<std::uintptr_t>(control.dst), bytes,
                                                                 true, lane_replacements, lane_matches));
      xpool::utils::cuda::graph::set_memset_parameters(primary_nodes[node_index], primary);
      continue;
    }
    if (primary_type == cudaGraphNodeTypeMemcpy) {
      auto primary = xpool::utils::cuda::graph::memcpy_parameters(primary_nodes[node_index]);
      const auto control = xpool::utils::cuda::graph::memcpy_parameters(control_nodes[node_index]);
      const auto bytes = xpool::utils::cuda::graph::memcpy_access_bytes(primary);
      TORCH_CHECK(bytes == xpool::utils::cuda::graph::memcpy_access_bytes(control) && primary.kind == control.kind,
                  "xpool Primary and Control Memcpy metadata differ");
      primary.srcPtr.ptr = reinterpret_cast<void *>(bind_memory_address(
          reinterpret_cast<std::uintptr_t>(primary.srcPtr.ptr), reinterpret_cast<std::uintptr_t>(control.srcPtr.ptr),
          bytes, false, lane_replacements, lane_matches));
      primary.dstPtr.ptr = reinterpret_cast<void *>(bind_memory_address(
          reinterpret_cast<std::uintptr_t>(primary.dstPtr.ptr), reinterpret_cast<std::uintptr_t>(control.dstPtr.ptr),
          bytes, true, lane_replacements, lane_matches));
      xpool::utils::cuda::graph::set_memcpy_parameters(primary_nodes[node_index], primary);
      continue;
    }
    if (primary_type == cudaGraphNodeTypeEmpty) {
      continue;
    }
    TORCH_CHECK(primary_type == cudaGraphNodeTypeKernel, "xpool Primary Graph contains an unsupported node kind");

    auto primary = xpool::utils::cuda::graph::KernelNodeParameters::read(primary_nodes[node_index]);
    const auto control = xpool::utils::cuda::graph::KernelNodeParameters::read(control_nodes[node_index]);
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

    // Phase: Discover Bindings - Declared addresses may differ; weight matches
    // become Device-updatable sites and storage ranges receive static bindings.
    auto changed = false;
    auto node_matches = std::vector<std::pair<std::size_t, std::size_t>>{};
    for (auto parameter_index = std::size_t{0}; parameter_index < primary.arguments().size(); ++parameter_index) {
      auto &primary_value = primary.arguments()[parameter_index];
      const auto &control_value = control.arguments()[parameter_index];
      const auto captured_value = primary_value;
      auto covered = std::vector<bool>(primary_value.bytes.size(), false);
      for (auto byte_offset = std::size_t{0}; byte_offset + sizeof(std::uintptr_t) <= captured_value.bytes.size();
           ++byte_offset) {
        const auto p = captured_value.read_address(byte_offset);
        const auto c = control_value.read_address(byte_offset);
        auto resource_index = resources.size();
        for (auto index = std::size_t{0}; index < resources.size(); ++index) {
          if (p == resources[index].primary_address && c == resources[index].control_address) {
            TORCH_CHECK(resource_index == resources.size(), "xpool ambiguous weight binding");
            resource_index = index;
          }
        }
        auto lane_index = lane_replacements.size();
        auto relative_offset = std::size_t{0};
        for (auto index = std::size_t{0}; index < lane_replacements.size(); ++index) {
          const auto &region = lane_replacements[index];
          const auto in_primary = p >= region.primary_address && p - region.primary_address < region.bytes;
          const auto in_control = c >= region.control_address && c - region.control_address < region.bytes;
          if (!in_primary && !in_control) {
            continue;
          }
          TORCH_CHECK(in_primary && in_control && p - region.primary_address == c - region.control_address,
                      "xpool resource pair changes range or relative offset");
          TORCH_CHECK(lane_index == lane_replacements.size(), "xpool ambiguous resource-range binding");
          lane_index = index;
          relative_offset = p - region.primary_address;
        }
        const auto weight = resource_index != resources.size();
        const auto storage_range = lane_index != lane_replacements.size();
        TORCH_CHECK(!(weight && storage_range), "xpool weight and resource-range bindings overlap");
        if (!weight && !storage_range) {
          continue;
        }
        TORCH_CHECK(std::none_of(covered.begin() + static_cast<std::ptrdiff_t>(byte_offset),
                                 covered.begin() + static_cast<std::ptrdiff_t>(byte_offset + sizeof(std::uintptr_t)),
                                 [](bool value) { return value; }),
                    "xpool parameter binding sites overlap");
        std::fill_n(covered.begin() + static_cast<std::ptrdiff_t>(byte_offset), sizeof(std::uintptr_t), true);
        if (weight) {
          primary_value.write_address(byte_offset, resources[resource_index].target_address);
          node_matches.emplace_back(primary_value.offset_bytes + byte_offset, resource_index);
          ++resource_matches[resource_index];
        } else {
          primary_value.write_address(byte_offset,
                                      xpool::utils::checked::sum(lane_replacements[lane_index].target_address,
                                                                 static_cast<std::uintptr_t>(relative_offset)));
          ++lane_matches[lane_index];
        }
        changed = true;
      }
      for (auto byte_offset = std::size_t{0}; byte_offset < captured_value.bytes.size(); ++byte_offset) {
        TORCH_CHECK(captured_value.bytes[byte_offset] == control_value.bytes[byte_offset] || covered[byte_offset],
                    "xpool Primary Graph contains an undeclared parameter-byte delta at node ", node_index,
                    ", argument ", parameter_index, ", byte ", byte_offset);
      }
    }

    // Phase: Apply Node Changes - Record runtime binding offsets before
    // committing the rewritten argument storage to the embedded graph.
    if (!node_matches.empty()) {
      const auto device_node = xpool::utils::cuda::graph::make_device_updatable(primary_nodes[node_index]);
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
  for (auto index = std::size_t{0}; index < lane_replacements.size(); ++index) {
    TORCH_CHECK(!lane_replacements[index].required || lane_matches[index] != 0,
                "xpool Primary Graph omits a required lane-resource binding");
  }
  return parameterization;
}

} // namespace xpool::ffnagent
