#pragma once

/// \file xpool/utils/graph.hpp
/// \brief Host-side CUDA Graph inspection and mutation helpers.
///
/// Borrowed graph and node handles remain owned by CUDA. CUDA query, creation,
/// and mutation failures surface as c10::Error through the owning implementation.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace xpool::utils::graph {

/// One owned CUDA Kernel Node argument and its ABI offset.
struct KernelNodeArgument {
  /// CUDA-reported offset of this argument in the packed parameter buffer.
  std::size_t offset_bytes;
  /// Owned argument bytes retained independently of CUDA's transient pointers.
  std::vector<std::byte> bytes;

  /// Read one pointer-sized address from the argument.
  /// \throws c10::Error when the pointer-sized window exceeds the retained bytes.
  [[nodiscard]] std::uintptr_t read_address(std::size_t byte_offset) const;

  /// Write one pointer-sized address into the argument.
  /// \throws c10::Error when the pointer-sized window exceeds the retained bytes.
  void write_address(std::size_t byte_offset, std::uintptr_t address);
};

/// Owned Host snapshot of one CUDA Graph Kernel Node's launch schema and arguments.
class KernelNodeParameters {
public:
  /// Read one Kernel Node's complete launch schema and argument bytes.
  [[nodiscard]] static KernelNodeParameters read(cudaGraphNode_t node);

  /// Compare launch identity and argument geometry without comparing values.
  [[nodiscard]] bool same_schema(const KernelNodeParameters &other) const;

  /// Return whether any pointer-sized byte window contains an address.
  [[nodiscard]] bool contains_address(std::uintptr_t address) const;

  /// Return mutable owned arguments.
  [[nodiscard]] std::span<KernelNodeArgument> arguments() { return arguments_; }

  /// Return immutable owned arguments.
  [[nodiscard]] std::span<const KernelNodeArgument> arguments() const { return arguments_; }

  /// Apply the retained launch schema and current argument values to a Kernel Node.
  void apply(cudaGraphNode_t node) const;

private:
  CUfunction function_ = nullptr;
  CUkernel kernel_ = nullptr;
  CUcontext context_ = nullptr;
  dim3 grid_{};
  dim3 block_{};
  unsigned int shared_memory_bytes_ = 0;
  std::vector<KernelNodeArgument> arguments_;
};

/// Return one CUDA Graph node's runtime type.
[[nodiscard]] cudaGraphNodeType node_type(cudaGraphNode_t node);

/// Return the direct nodes of one CUDA Graph in CUDA's enumeration order.
[[nodiscard]] std::vector<cudaGraphNode_t> nodes(cudaGraph_t graph);

/// Return the borrowed body graphs owned by one conditional node.
/// The returned handles remain owned by the conditional node.
[[nodiscard]] std::vector<cudaGraph_t> conditional_bodies(cudaGraphNode_t node);

/// Return the borrowed graph embedded by one child-graph node.
[[nodiscard]] cudaGraph_t child_graph(cudaGraphNode_t node);

/// Create one conditional handle owned by a graph.
[[nodiscard]] cudaGraphConditionalHandle
create_conditional_handle(cudaGraph_t owner, unsigned int default_launch_value = 0, unsigned int flags = 0);

/// Add one WHILE conditional node.
[[nodiscard]] cudaGraphNode_t add_while_node(cudaGraph_t graph, cudaGraphConditionalHandle handle);

/// Add one SWITCH conditional node with a positive body count.
[[nodiscard]] cudaGraphNode_t add_switch_node(cudaGraph_t graph, cudaGraphConditionalHandle handle,
                                              std::size_t body_count);

/// Embed a source graph as one child node and return CUDA's embedded clone.
/// The returned handle is borrowed from the child node.
[[nodiscard]] cudaGraph_t embed_child_graph(cudaGraph_t graph, cudaGraph_t source);

/// Enable one kernel node for device-side parameter updates.
/// The returned device-node handle remains owned by the graph node.
[[nodiscard]] cudaGraphDeviceNode_t make_device_updatable(cudaGraphNode_t node);

/// Visit every node reachable through child graphs and conditional bodies.
/// The visitor is invoked synchronously and is not retained.
void for_each_reachable_node(cudaGraph_t graph, const std::function<void(cudaGraphNode_t, cudaGraphNodeType)> &visitor);

/// Add one kernel node and return its graph-owned handle.
/// CUDA copies the supplied kernel argument values during this call.
[[nodiscard]] cudaGraphNode_t add_kernel_node(cudaGraph_t graph, const void *function, dim3 grid, dim3 block,
                                              std::size_t shared_memory_bytes, void **arguments,
                                              std::span<const cudaGraphNode_t> predecessors = {});

/// Add one default dependency edge between existing nodes in the same graph.
void add_dependency(cudaGraph_t graph, cudaGraphNode_t predecessor, cudaGraphNode_t successor);

/// Compare direct node kinds and indexed dependency edges of two CUDA Graphs.
/// Child and conditional body graphs are not traversed.
[[nodiscard]] bool same_direct_topology(cudaGraph_t left, cudaGraph_t right);

/// Splice one existing unconnected node after a predecessor.
/// Every direct default-edge successor is rewired through the inserted node.
void insert_node_after(cudaGraph_t graph, cudaGraphNode_t predecessor, cudaGraphNode_t inserted);

} // namespace xpool::utils::graph
