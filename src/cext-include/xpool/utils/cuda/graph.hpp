#pragma once

/// \file xpool/utils/cuda/graph.hpp
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

#include <xpool/utils/cuda.hpp>

namespace xpool::utils::cuda::graph {

/// Locate references to declared borrowed intervals in supported graph nodes.
/// Typed Memset/Memcpy nodes validate full access spans; Kernel arguments identify
/// numeric references, not arbitrary pointer types, access lengths or ownership.
/// \throws c10::Error for invalid intervals or unsupported graph nodes.
std::vector<std::size_t> used_resource_regions(cudaGraph_t graph, const std::vector<CaptureAllocation> &allocations);

/// Create a graph directly in the caller's retained output slot.
/// \throws c10::AcceleratorError if CUDA rejects creation.
void create(cudaGraph_t &graph);

/// Instantiate into the caller's retained executable slot, including partial output on failure.
/// \throws c10::AcceleratorError if CUDA rejects instantiation.
void instantiate(cudaGraphExec_t &executable, cudaGraph_t graph);

/// Enqueue graph upload without waiting for stream completion.
/// \throws c10::AcceleratorError if CUDA rejects upload.
void upload(cudaGraphExec_t executable, cudaStream_t stream);

/// Enqueue graph execution without waiting for stream completion.
/// \throws c10::AcceleratorError if CUDA rejects launch.
void launch(cudaGraphExec_t executable, cudaStream_t stream);

/// Destroy a graph; the caller clears its retained slot after successful return.
/// \throws c10::AcceleratorError if CUDA rejects destruction.
void destroy(cudaGraph_t graph);

/// Destroy an executable; the caller clears its retained slot after successful return.
/// \throws c10::AcceleratorError if CUDA rejects destruction.
void destroy(cudaGraphExec_t executable);

/// Read typed Memset parameters from a borrowed node.
/// \throws c10::AcceleratorError if CUDA rejects the query.
[[nodiscard]] cudaMemsetParams memset_parameters(cudaGraphNode_t node);

/// Set typed Memset parameters; CUDA copies the supplied values during the call.
/// \throws c10::AcceleratorError if CUDA rejects the change.
void set_memset_parameters(cudaGraphNode_t node, const cudaMemsetParams &parameters);

/// Read typed Memcpy parameters from a borrowed node.
/// \throws c10::AcceleratorError if CUDA rejects the query.
[[nodiscard]] cudaMemcpy3DParms memcpy_parameters(cudaGraphNode_t node);

/// Set typed Memcpy parameters; CUDA copies the supplied values during the call.
/// \throws c10::AcceleratorError if CUDA rejects the change.
void set_memcpy_parameters(cudaGraphNode_t node, const cudaMemcpy3DParms &parameters);

/// Return the number of direct dependent nodes.
/// \throws c10::AcceleratorError if CUDA rejects the query.
[[nodiscard]] std::size_t dependent_count(cudaGraphNode_t node);

/// Add one graph-owned empty node.
/// \throws c10::AcceleratorError if CUDA rejects the addition.
[[nodiscard]] cudaGraphNode_t add_empty_node(cudaGraph_t graph);

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
  /// Read one Kernel Node's complete launch schema from exactly one CUDA
  /// parameter transport and normalize it into owned argument bytes.
  /// \throws c10::Error when the transport, packed options, or packed argument
  /// layout is invalid.
  [[nodiscard]] static KernelNodeParameters read(cudaGraphNode_t node);

  /// Compare launch identity and argument geometry without comparing values.
  [[nodiscard]] bool same_schema(const KernelNodeParameters &other) const;

  /// Return whether any pointer-sized byte window contains an address.
  [[nodiscard]] bool contains_address(std::uintptr_t address) const;

  /// Return mutable owned arguments.
  [[nodiscard]] std::span<KernelNodeArgument> arguments() { return arguments_; }

  /// Return immutable owned arguments.
  [[nodiscard]] std::span<const KernelNodeArgument> arguments() const { return arguments_; }

  /// Apply the retained launch schema and current values through CUDA's
  /// pointer-array parameter transport.
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

/// Return the covering span from the first through last Memset write, including pitch gaps.
/// \throws c10::Error for empty geometry, invalid element size/pitch or overflow.
[[nodiscard]] std::size_t memset_access_bytes(const cudaMemsetParams &parameters);

/// Return the byte span of a contiguous one-dimensional device-to-device copy.
/// \throws c10::Error for unsupported geometry, direction or storage.
[[nodiscard]] std::size_t memcpy_access_bytes(const cudaMemcpy3DParms &parameters);

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

} // namespace xpool::utils::cuda::graph
