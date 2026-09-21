#include <xpool/utils/graph.hpp>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/driver_api.h>
#include <c10/util/Exception.h>
#include <c10/util/TypeCast.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace xpool::utils::graph {

namespace {

struct GraphEdge {
  std::size_t from;
  std::size_t to;
  cudaGraphEdgeData data;
};

std::vector<GraphEdge> graph_edges(cudaGraph_t graph, const std::vector<cudaGraphNode_t> &graph_nodes) {
  auto count = std::size_t{0};
  C10_CUDA_CHECK(cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &count));
  auto sources = std::vector<cudaGraphNode_t>(count);
  auto destinations = std::vector<cudaGraphNode_t>(count);
  auto data = std::vector<cudaGraphEdgeData>(count);
  if (count != 0) {
    C10_CUDA_CHECK(cudaGraphGetEdges(graph, sources.data(), destinations.data(), data.data(), &count));
  }
  auto indices = std::unordered_map<cudaGraphNode_t, std::size_t>{};
  indices.reserve(graph_nodes.size());
  for (auto index = std::size_t{0}; index < graph_nodes.size(); ++index) {
    indices.emplace(graph_nodes[index], index);
  }
  // CUDA node handles differ across cloned Graphs; enumeration positions make
  // otherwise identical dependency topology comparable.
  auto edges = std::vector<GraphEdge>{};
  edges.reserve(count);
  for (auto index = std::size_t{0}; index < count; ++index) {
    const auto from = indices.find(sources[index]);
    const auto to = indices.find(destinations[index]);
    TORCH_CHECK(from != indices.end() && to != indices.end(),
                "xpool CUDA Graph edge references a node outside its direct graph");
    edges.push_back(GraphEdge{.from = from->second, .to = to->second, .data = data[index]});
  }
  std::sort(edges.begin(), edges.end(), [](const GraphEdge &left, const GraphEdge &right) {
    return std::tie(left.from, left.to, left.data.from_port, left.data.to_port, left.data.type) <
           std::tie(right.from, right.to, right.data.from_port, right.data.to_port, right.data.type);
  });
  return edges;
}

bool same_edges(const std::vector<GraphEdge> &left, const std::vector<GraphEdge> &right) {
  if (left.size() != right.size()) {
    return false;
  }
  for (auto index = std::size_t{0}; index < left.size(); ++index) {
    if (left[index].from != right[index].from || left[index].to != right[index].to ||
        left[index].data.from_port != right[index].data.from_port ||
        left[index].data.to_port != right[index].data.to_port || left[index].data.type != right[index].data.type) {
      return false;
    }
  }
  return true;
}

cudaGraphNode_t add_conditional_node(cudaGraph_t graph, cudaGraphConditionalHandle handle,
                                     cudaGraphConditionalNodeType type, std::size_t body_count) {
  TORCH_CHECK(body_count != 0, "xpool CUDA Graph conditional node requires at least one body");
  auto parameters = cudaGraphNodeParams{};
  parameters.type = cudaGraphNodeTypeConditional;
  parameters.conditional.handle = handle;
  parameters.conditional.type = type;
  parameters.conditional.size = c10::checked_convert<unsigned int>(body_count, "CUDA Graph conditional body count");
  auto node = cudaGraphNode_t{};
  C10_CUDA_CHECK(cudaGraphAddNode(&node, graph, nullptr, nullptr, 0, &parameters));
  return node;
}

std::span<const std::byte> packed_kernel_parameters(void **extra) {
  auto *buffer = static_cast<const std::byte *>(nullptr);
  auto size = std::size_t{0};
  auto has_size = false;
  for (auto **option = extra; *option != CU_LAUNCH_PARAM_END; option += 2) {
    if (*option == CU_LAUNCH_PARAM_BUFFER_POINTER) {
      buffer = static_cast<const std::byte *>(option[1]);
    } else if (*option == CU_LAUNCH_PARAM_BUFFER_SIZE) {
      TORCH_CHECK(option[1] != nullptr, "xpool CUDA Graph packed parameter buffer has no size value");
      size = *static_cast<const std::size_t *>(option[1]);
      has_size = true;
    } else {
      TORCH_CHECK(false, "xpool CUDA Graph Kernel Node uses an unsupported extra option");
    }
  }
  TORCH_CHECK(buffer != nullptr && has_size, "xpool CUDA Graph Kernel Node has incomplete packed parameters");
  return {buffer, size};
}

} // namespace

std::uintptr_t KernelNodeArgument::read_address(std::size_t byte_offset) const {
  TORCH_CHECK(byte_offset <= bytes.size() && sizeof(std::uintptr_t) <= bytes.size() - byte_offset,
              "xpool CUDA Graph address read exceeds its Kernel argument");
  auto address = std::uintptr_t{0};
  std::memcpy(&address, bytes.data() + byte_offset, sizeof(address));
  return address;
}

void KernelNodeArgument::write_address(std::size_t byte_offset, std::uintptr_t address) {
  TORCH_CHECK(byte_offset <= bytes.size() && sizeof(address) <= bytes.size() - byte_offset,
              "xpool CUDA Graph address write exceeds its Kernel argument");
  std::memcpy(bytes.data() + byte_offset, &address, sizeof(address));
}

KernelNodeParameters KernelNodeParameters::read(cudaGraphNode_t node) {
  auto parameters = CUDA_KERNEL_NODE_PARAMS{};
  C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeGetParams(node, &parameters));
  const auto pointer_array = parameters.kernelParams != nullptr;
  const auto packed_buffer = parameters.extra != nullptr;
  TORCH_CHECK(pointer_array != packed_buffer, "xpool CUDA Graph Kernel Node uses invalid parameter transport");
  const auto packed = packed_buffer ? packed_kernel_parameters(parameters.extra) : std::span<const std::byte>{};
  auto count = std::size_t{0};
  if (parameters.func != nullptr) {
    C10_CUDA_DRIVER_CHECK(cuFuncGetParamCount(parameters.func, &count));
  } else {
    TORCH_CHECK(parameters.kern != nullptr, "xpool CUDA Graph Kernel Node has no function identity");
    C10_CUDA_DRIVER_CHECK(cuKernelGetParamCount(parameters.kern, &count));
  }
  auto result = KernelNodeParameters{};
  result.function_ = parameters.func;
  result.kernel_ = parameters.kern;
  result.context_ = parameters.ctx;
  result.grid_ = dim3{parameters.gridDimX, parameters.gridDimY, parameters.gridDimZ};
  result.block_ = dim3{parameters.blockDimX, parameters.blockDimY, parameters.blockDimZ};
  result.shared_memory_bytes_ = parameters.sharedMemBytes;
  result.arguments_.reserve(count);
  for (auto index = std::size_t{0}; index < count; ++index) {
    auto offset = std::size_t{0};
    auto size = std::size_t{0};
    if (parameters.func != nullptr) {
      C10_CUDA_DRIVER_CHECK(cuFuncGetParamInfo(parameters.func, index, &offset, &size));
    } else {
      C10_CUDA_DRIVER_CHECK(cuKernelGetParamInfo(parameters.kern, index, &offset, &size));
    }
    auto bytes = std::vector<std::byte>(size);
    if (pointer_array) {
      std::memcpy(bytes.data(), parameters.kernelParams[index], size);
    } else {
      TORCH_CHECK(offset <= packed.size() && size <= packed.size() - offset,
                  "xpool CUDA Graph packed parameter buffer is smaller than its Kernel argument layout");
      std::memcpy(bytes.data(), packed.data() + offset, size);
    }
    result.arguments_.push_back(KernelNodeArgument{.offset_bytes = offset, .bytes = std::move(bytes)});
  }
  return result;
}

bool KernelNodeParameters::same_schema(const KernelNodeParameters &other) const {
  if (function_ != other.function_ || kernel_ != other.kernel_ || context_ != other.context_ ||
      grid_.x != other.grid_.x || grid_.y != other.grid_.y || grid_.z != other.grid_.z || block_.x != other.block_.x ||
      block_.y != other.block_.y || block_.z != other.block_.z || shared_memory_bytes_ != other.shared_memory_bytes_ ||
      arguments_.size() != other.arguments_.size()) {
    return false;
  }
  for (auto index = std::size_t{0}; index < arguments_.size(); ++index) {
    if (arguments_[index].offset_bytes != other.arguments_[index].offset_bytes ||
        arguments_[index].bytes.size() != other.arguments_[index].bytes.size()) {
      return false;
    }
  }
  return true;
}

bool KernelNodeParameters::contains_address(std::uintptr_t address) const {
  for (const auto &argument : arguments_) {
    for (auto offset = std::size_t{0}; offset + sizeof(address) <= argument.bytes.size(); ++offset) {
      if (argument.read_address(offset) == address) {
        return true;
      }
    }
  }
  return false;
}

void KernelNodeParameters::apply(cudaGraphNode_t node) const {
  auto pointers = std::vector<void *>{};
  pointers.reserve(arguments_.size());
  for (const auto &argument : arguments_) {
    pointers.push_back(const_cast<std::byte *>(argument.bytes.data()));
  }
  auto parameters = CUDA_KERNEL_NODE_PARAMS{
      .func = function_,
      .gridDimX = grid_.x,
      .gridDimY = grid_.y,
      .gridDimZ = grid_.z,
      .blockDimX = block_.x,
      .blockDimY = block_.y,
      .blockDimZ = block_.z,
      .sharedMemBytes = shared_memory_bytes_,
      .kernelParams = pointers.data(),
      .extra = nullptr,
      .kern = kernel_,
      .ctx = context_,
  };
  C10_CUDA_DRIVER_CHECK(cuGraphKernelNodeSetParams(node, &parameters));
}

cudaGraphNodeType node_type(cudaGraphNode_t node) {
  auto type = cudaGraphNodeTypeCount;
  C10_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
  TORCH_CHECK(type != cudaGraphNodeTypeCount, "xpool CUDA Graph returned the node-kind sentinel");
  return type;
}

std::vector<cudaGraphNode_t> nodes(cudaGraph_t graph) {
  auto count = std::size_t{0};
  C10_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &count));
  auto result = std::vector<cudaGraphNode_t>(count);
  if (count != 0) {
    C10_CUDA_CHECK(cudaGraphGetNodes(graph, result.data(), &count));
    result.resize(count);
  }
  return result;
}

std::vector<cudaGraph_t> conditional_bodies(cudaGraphNode_t node) {
  auto parameters = CUgraphNodeParams{};
  C10_CUDA_DRIVER_CHECK(cuGraphNodeGetParams(reinterpret_cast<CUgraphNode>(node), &parameters));
  TORCH_CHECK(parameters.type == CU_GRAPH_NODE_TYPE_CONDITIONAL, "xpool CUDA Graph node is not conditional");
  auto result = std::vector<cudaGraph_t>{};
  result.reserve(parameters.conditional.size);
  for (auto index = 0U; index < parameters.conditional.size; ++index) {
    result.push_back(reinterpret_cast<cudaGraph_t>(parameters.conditional.phGraph_out[index]));
  }
  return result;
}

cudaGraph_t child_graph(cudaGraphNode_t node) {
  auto result = cudaGraph_t{};
  C10_CUDA_CHECK(cudaGraphChildGraphNodeGetGraph(node, &result));
  return result;
}

cudaGraphConditionalHandle create_conditional_handle(cudaGraph_t owner, unsigned int default_launch_value,
                                                     unsigned int flags) {
  auto handle = cudaGraphConditionalHandle{};
  C10_CUDA_CHECK(cudaGraphConditionalHandleCreate(&handle, owner, default_launch_value, flags));
  return handle;
}

cudaGraphNode_t add_while_node(cudaGraph_t graph, cudaGraphConditionalHandle handle) {
  return add_conditional_node(graph, handle, cudaGraphCondTypeWhile, 1);
}

cudaGraphNode_t add_switch_node(cudaGraph_t graph, cudaGraphConditionalHandle handle, std::size_t body_count) {
  return add_conditional_node(graph, handle, cudaGraphCondTypeSwitch, body_count);
}

cudaGraph_t embed_child_graph(cudaGraph_t graph, cudaGraph_t source) {
  auto node = cudaGraphNode_t{};
  C10_CUDA_CHECK(cudaGraphAddChildGraphNode(&node, graph, nullptr, 0, source));
  return child_graph(node);
}

cudaGraphDeviceNode_t make_device_updatable(cudaGraphNode_t node) {
  auto set_value = cudaKernelNodeAttrValue{};
  set_value.deviceUpdatableKernelNode.deviceUpdatable = 1;
  C10_CUDA_CHECK(cudaGraphKernelNodeSetAttribute(node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &set_value));
  auto get_value = cudaKernelNodeAttrValue{};
  C10_CUDA_CHECK(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &get_value));
  TORCH_CHECK(get_value.deviceUpdatableKernelNode.devNode != nullptr,
              "xpool failed to obtain a device-updatable CUDA Graph node");
  return get_value.deviceUpdatableKernelNode.devNode;
}

void for_each_reachable_node(cudaGraph_t graph,
                             const std::function<void(cudaGraphNode_t, cudaGraphNodeType)> &visitor) {
  for (const auto node : nodes(graph)) {
    const auto type = node_type(node);
    visitor(node, type);
    if (type == cudaGraphNodeTypeGraph) {
      for_each_reachable_node(child_graph(node), visitor);
    } else if (type == cudaGraphNodeTypeConditional) {
      for (const auto body : conditional_bodies(node)) {
        for_each_reachable_node(body, visitor);
      }
    }
  }
}

cudaGraphNode_t add_kernel_node(cudaGraph_t graph, const void *function, dim3 grid, dim3 block,
                                std::size_t shared_memory_bytes, void **arguments,
                                std::span<const cudaGraphNode_t> predecessors) {
  auto parameters = cudaKernelNodeParams{};
  parameters.func = const_cast<void *>(function);
  parameters.gridDim = grid;
  parameters.blockDim = block;
  parameters.sharedMemBytes =
      c10::checked_convert<unsigned int>(shared_memory_bytes, "CUDA Graph kernel shared-memory bytes");
  parameters.kernelParams = arguments;
  auto node = cudaGraphNode_t{};
  C10_CUDA_CHECK(cudaGraphAddKernelNode(&node, graph, predecessors.data(), predecessors.size(), &parameters));
  return node;
}

void add_dependency(cudaGraph_t graph, cudaGraphNode_t predecessor, cudaGraphNode_t successor) {
  C10_CUDA_CHECK(cudaGraphAddDependencies(graph, &predecessor, &successor, nullptr, 1));
}

bool same_direct_topology(cudaGraph_t left, cudaGraph_t right) {
  const auto left_nodes = nodes(left);
  const auto right_nodes = nodes(right);
  if (left_nodes.size() != right_nodes.size()) {
    return false;
  }
  for (auto index = std::size_t{0}; index < left_nodes.size(); ++index) {
    if (node_type(left_nodes[index]) != node_type(right_nodes[index])) {
      return false;
    }
  }
  return same_edges(graph_edges(left, left_nodes), graph_edges(right, right_nodes));
}

void insert_node_after(cudaGraph_t graph, cudaGraphNode_t predecessor, cudaGraphNode_t inserted) {
  auto successor_count = std::size_t{0};
  C10_CUDA_CHECK(cudaGraphNodeGetDependentNodes(predecessor, nullptr, nullptr, &successor_count));
  auto successors = std::vector<cudaGraphNode_t>(successor_count);
  auto edge_data = std::vector<cudaGraphEdgeData>(successor_count);
  if (successor_count != 0) {
    C10_CUDA_CHECK(cudaGraphNodeGetDependentNodes(predecessor, successors.data(), edge_data.data(), &successor_count));
  }
  for (auto index = std::size_t{0}; index < successor_count; ++index) {
    TORCH_CHECK(edge_data[index].from_port == 0 && edge_data[index].to_port == 0 &&
                    edge_data[index].type == cudaGraphDependencyTypeDefault,
                "xpool CUDA Graph dependency splice requires direct default edges");
    C10_CUDA_CHECK(cudaGraphRemoveDependencies(graph, &predecessor, &successors[index], &edge_data[index], 1));
  }
  add_dependency(graph, predecessor, inserted);
  for (const auto successor : successors) {
    add_dependency(graph, inserted, successor);
  }
}

} // namespace xpool::utils::graph
