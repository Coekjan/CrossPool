#include <xpool/devkit/graph_observer.hpp>

#include <c10/util/Exception.h>

#include <algorithm>
#include <mutex>
#include <optional>
#include <ranges>

#include <xpool/debug/options.hpp>
#include <xpool/devkit/adapters.hpp>
#include <xpool/utils/graph.hpp>

namespace xpool::devkit::graph_observer {

namespace {

std::mutex state_mutex;
std::optional<Snapshot> snapshot;

} // namespace

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionInstallPreEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  snapshot.emplace();
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnPrimaryGraphParameterizePostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  TORCH_CHECK(snapshot.has_value(), "xpool Graph Observer received a Primary Graph before installation");
  if (snapshot->primary_graphs.size() <= context.execution_signature_index) {
    snapshot->primary_graphs.resize(context.execution_signature_index + 1);
  }
  auto &primary_graph = snapshot->primary_graphs[context.execution_signature_index];
  xpool::utils::graph::for_each_reachable_node(
      context.graph,
      [&primary_graph](cudaGraphNode_t, cudaGraphNodeType type) { ++primary_graph.node_counts[type]; });
  primary_graph.binding_site_count = context.binding_site_count;
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnLaneGraphBuildPostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  auto lane_graph = LaneGraphSnapshot{};
  xpool::utils::graph::for_each_reachable_node(
      context.graph, [&lane_graph](cudaGraphNode_t, cudaGraphNodeType type) { ++lane_graph.node_counts[type]; });
  const auto compute_branches = xpool::utils::graph::conditional_bodies(context.compute_switch);
  lane_graph.compute_branch_count = static_cast<std::size_t>(std::ranges::distance(
      compute_branches | std::views::filter([](cudaGraph_t branch) {
        return std::ranges::any_of(xpool::utils::graph::nodes(branch), [](cudaGraphNode_t node) {
          return xpool::utils::graph::node_type(node) == cudaGraphNodeTypeGraph;
        });
      })));
  const auto delivery_branches = xpool::utils::graph::conditional_bodies(context.delivery_switch);
  lane_graph.delivery_branch_count = static_cast<std::size_t>(std::ranges::distance(
      delivery_branches | std::views::filter([](cudaGraph_t branch) {
        return std::ranges::any_of(xpool::utils::graph::nodes(branch), [](cudaGraphNode_t node) {
          return xpool::utils::graph::node_type(node) == cudaGraphNodeTypeKernel;
        });
      })));

  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  TORCH_CHECK(snapshot.has_value(), "xpool Graph Observer received a Lane Graph before installation");
  if (snapshot->lane_graphs.size() <= context.executor_lane_index) {
    snapshot->lane_graphs.resize(context.executor_lane_index + 1);
  }
  snapshot->lane_graphs[context.executor_lane_index] = std::move(lane_graph);
}

XPOOL_HOST_HOOK_FN(xpool::hooks::FfnExecutionFinalizePostEvent, HostAdapter::observe, context) {
  if (!xpool::debug::options().graph_observer.enable) {
    return;
  }
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  snapshot.reset();
}

std::optional<Snapshot> read() {
  const auto lock = std::lock_guard<std::mutex>{state_mutex};
  return snapshot;
}

} // namespace xpool::devkit::graph_observer
