#pragma once

/// \file xpool/devkit/graph_observer.hpp
/// \brief Development-only CUDA Graph structure snapshots.

#include <cstddef>
#include <map>
#include <optional>
#include <vector>

#include <cuda_runtime.h>

namespace xpool::devkit::graph_observer {

/// Actual node inventory and binding-site count for one Primary Graph.
struct PrimaryGraphSnapshot {
  /// Recursive counts indexed by CUDA node kind.
  std::map<cudaGraphNodeType, std::size_t> node_counts;
  /// Discovered device-updatable weight-address sites.
  std::size_t binding_site_count;
};

/// Actual recursive node and branch inventory for one Lane Graph.
struct LaneGraphSnapshot {
  /// Recursive counts indexed by CUDA node kind.
  std::map<cudaGraphNodeType, std::size_t> node_counts;
  /// Installed conditional compute bodies.
  std::size_t compute_branch_count;
  /// Installed conditional delivery bodies.
  std::size_t delivery_branch_count;
};

/// Immutable Graph Observer snapshot retained after execution installation.
struct Snapshot {
  /// Observations for each installed Primary Graph signature.
  std::vector<PrimaryGraphSnapshot> primary_graphs;
  /// Observations for each lane-owned Graph.
  std::vector<LaneGraphSnapshot> lane_graphs;
};

/// Return the process-local Graph Observer snapshot when enabled and installed.
/// \return A repeatable Host-owned snapshot, or no value outside the installed lifecycle.
std::optional<Snapshot> read();

} // namespace xpool::devkit::graph_observer
