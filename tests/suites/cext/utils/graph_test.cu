#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <xpool/macros.hpp>
#include <xpool/utils/graph.hpp>

namespace {

XPOOL_KERNEL_FN void empty_kernel() {}

XPOOL_KERNEL_FN void write_value_kernel(std::int32_t *output, std::int32_t value) { *output = value; }

XPOOL_KERNEL_FN void update_write_value_kernel(cudaGraphDeviceNode_t node, std::int32_t *output, std::int32_t value,
                                               std::size_t output_offset, std::size_t value_offset,
                                               cudaError_t *status) {
  const cudaGraphKernelNodeUpdate updates[] = {
      {.node = node,
       .field = cudaGraphKernelNodeFieldParam,
       .updateData = {.param = {.pValue = &output, .offset = output_offset, .size = sizeof(output)}}},
      {.node = node,
       .field = cudaGraphKernelNodeFieldParam,
       .updateData = {.param = {.pValue = &value, .offset = value_offset, .size = sizeof(value)}}},
  };
  *status = cudaGraphKernelNodeUpdatesApply(updates, std::size(updates));
}

TEST(GraphTest, ConstructsAndVisitsRetainedNodeKinds) {
  auto device_count = 0;
  const auto availability = cudaGetDeviceCount(&device_count);
  if (availability != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(availability);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto source = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&source, 0), cudaSuccess);
  auto source_node = cudaGraphNode_t{};
  ASSERT_EQ(cudaGraphAddEmptyNode(&source_node, source, nullptr, 0), cudaSuccess);

  auto graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  const auto while_handle = xpool::utils::graph::create_conditional_handle(graph, 1, cudaGraphCondAssignDefault);
  const auto while_node = xpool::utils::graph::add_while_node(graph, while_handle);
  const auto while_bodies = xpool::utils::graph::conditional_bodies(while_node);
  ASSERT_EQ(while_bodies.size(), 1U);

  const auto switch_handle = xpool::utils::graph::create_conditional_handle(while_bodies.front());
  const auto switch_node = xpool::utils::graph::add_switch_node(while_bodies.front(), switch_handle, 2);
  const auto switch_bodies = xpool::utils::graph::conditional_bodies(switch_node);
  ASSERT_EQ(switch_bodies.size(), 2U);

  const auto embedded = xpool::utils::graph::embed_child_graph(switch_bodies.front(), source);
  const auto child_nodes = xpool::utils::graph::nodes(switch_bodies.front());
  ASSERT_EQ(child_nodes.size(), 1U);
  EXPECT_EQ(xpool::utils::graph::node_type(child_nodes.front()), cudaGraphNodeTypeGraph);
  EXPECT_EQ(xpool::utils::graph::child_graph(child_nodes.front()), embedded);
  const auto embedded_nodes = xpool::utils::graph::nodes(embedded);
  ASSERT_EQ(embedded_nodes.size(), 1U);
  EXPECT_EQ(xpool::utils::graph::node_type(embedded_nodes.front()), cudaGraphNodeTypeEmpty);

  auto node_counts = std::map<cudaGraphNodeType, std::size_t>{};
  xpool::utils::graph::for_each_reachable_node(graph,
                                               [&](cudaGraphNode_t, cudaGraphNodeType type) { ++node_counts[type]; });
  EXPECT_EQ(node_counts[cudaGraphNodeTypeConditional], 2U);
  EXPECT_EQ(node_counts[cudaGraphNodeTypeGraph], 1U);
  EXPECT_EQ(node_counts[cudaGraphNodeTypeEmpty], 1U);

  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(source), cudaSuccess);
}

TEST(GraphTest, ReturnsDeviceUpdatableKernelNode) {
  auto device_count = 0;
  const auto availability = cudaGetDeviceCount(&device_count);
  if (availability != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(availability);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  const auto node = xpool::utils::graph::add_kernel_node(graph, reinterpret_cast<const void *>(empty_kernel), dim3{1},
                                                         dim3{1}, 0, nullptr);
  const auto device_node = xpool::utils::graph::make_device_updatable(node);

  auto attribute = cudaKernelNodeAttrValue{};
  ASSERT_EQ(cudaGraphKernelNodeGetAttribute(node, cudaKernelNodeAttributeDeviceUpdatableKernelNode, &attribute),
            cudaSuccess);
  EXPECT_EQ(attribute.deviceUpdatableKernelNode.deviceUpdatable, 1U);
  EXPECT_EQ(attribute.deviceUpdatableKernelNode.devNode, device_node);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
}

TEST(GraphTest, NormalizesPackedKernelParametersForDeviceUpdates) {
  auto device_count = 0;
  const auto availability = cudaGetDeviceCount(&device_count);
  if (availability != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(availability);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto first_output = static_cast<std::int32_t *>(nullptr);
  auto second_output = static_cast<std::int32_t *>(nullptr);
  auto update_status = static_cast<cudaError_t *>(nullptr);
  ASSERT_EQ(cudaMalloc(&first_output, sizeof(*first_output)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&second_output, sizeof(*second_output)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&update_status, sizeof(*update_status)), cudaSuccess);
  ASSERT_EQ(cudaMemset(first_output, 0, sizeof(*first_output)), cudaSuccess);
  ASSERT_EQ(cudaMemset(second_output, 0, sizeof(*second_output)), cudaSuccess);

  auto function = cudaFunction_t{};
  ASSERT_EQ(cudaGetFuncBySymbol(&function, reinterpret_cast<const void *>(write_value_kernel)), cudaSuccess);
  const auto driver_function = reinterpret_cast<CUfunction>(function);
  auto offsets = std::array<std::size_t, 2>{};
  auto sizes = std::array<std::size_t, 2>{};
  auto packed_size = std::size_t{0};
  for (auto index = std::size_t{0}; index < offsets.size(); ++index) {
    ASSERT_EQ(cuFuncGetParamInfo(driver_function, index, &offsets[index], &sizes[index]), CUDA_SUCCESS);
    packed_size = std::max(packed_size, offsets[index] + sizes[index]);
  }
  ASSERT_EQ(sizes[0], sizeof(first_output));
  ASSERT_EQ(sizes[1], sizeof(std::int32_t));

  auto value = std::int32_t{17};
  auto pointer_graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&pointer_graph, 0), cudaSuccess);
  auto pointer_arguments = std::array<void *, 2>{&first_output, &value};
  const auto pointer_node = xpool::utils::graph::add_kernel_node(
      pointer_graph, reinterpret_cast<const void *>(write_value_kernel), dim3{1}, dim3{1}, 0, pointer_arguments.data());
  const auto pointer_parameters = xpool::utils::graph::KernelNodeParameters::read(pointer_node);

  auto packed = std::vector<std::byte>(packed_size);
  std::memcpy(packed.data() + offsets[0], &first_output, sizes[0]);
  std::memcpy(packed.data() + offsets[1], &value, sizes[1]);
  auto extra = std::array<void *, 5>{CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(), CU_LAUNCH_PARAM_BUFFER_SIZE,
                                     &packed_size, CU_LAUNCH_PARAM_END};
  auto raw_parameters = CUDA_KERNEL_NODE_PARAMS{
      .func = driver_function,
      .gridDimX = 1,
      .gridDimY = 1,
      .gridDimZ = 1,
      .blockDimX = 1,
      .blockDimY = 1,
      .blockDimZ = 1,
      .sharedMemBytes = 0,
      .kernelParams = nullptr,
      .extra = extra.data(),
      .kern = nullptr,
      .ctx = nullptr,
  };
  auto packed_graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&packed_graph, 0), cudaSuccess);
  auto packed_node = cudaGraphNode_t{};
  ASSERT_EQ(cuGraphAddKernelNode(reinterpret_cast<CUgraphNode *>(&packed_node), reinterpret_cast<CUgraph>(packed_graph),
                                 nullptr, 0, &raw_parameters),
            CUDA_SUCCESS);
  auto observed_packed = CUDA_KERNEL_NODE_PARAMS{};
  ASSERT_EQ(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(packed_node), &observed_packed), CUDA_SUCCESS);
  ASSERT_EQ(observed_packed.kernelParams, nullptr);
  ASSERT_NE(observed_packed.extra, nullptr);
  auto packed_parameters = xpool::utils::graph::KernelNodeParameters::read(packed_node);
  ASSERT_TRUE(pointer_parameters.same_schema(packed_parameters));
  ASSERT_EQ(pointer_parameters.arguments().size(), packed_parameters.arguments().size());
  for (auto index = std::size_t{0}; index < pointer_parameters.arguments().size(); ++index) {
    EXPECT_EQ(pointer_parameters.arguments()[index].offset_bytes, packed_parameters.arguments()[index].offset_bytes);
    EXPECT_EQ(pointer_parameters.arguments()[index].bytes, packed_parameters.arguments()[index].bytes);
  }
  EXPECT_TRUE(packed_parameters.contains_address(reinterpret_cast<std::uintptr_t>(first_output)));
  packed_parameters.arguments()[0].write_address(0, reinterpret_cast<std::uintptr_t>(second_output));
  EXPECT_TRUE(packed_parameters.contains_address(reinterpret_cast<std::uintptr_t>(second_output)));

  auto cloned_graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphClone(&cloned_graph, packed_graph), cudaSuccess);
  const auto cloned_nodes = xpool::utils::graph::nodes(cloned_graph);
  ASSERT_EQ(cloned_nodes.size(), 1U);
  EXPECT_TRUE(pointer_parameters.same_schema(xpool::utils::graph::KernelNodeParameters::read(cloned_nodes.front())));

  auto parent_graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&parent_graph, 0), cudaSuccess);
  const auto embedded_graph = xpool::utils::graph::embed_child_graph(parent_graph, packed_graph);
  const auto child_nodes = xpool::utils::graph::nodes(parent_graph);
  ASSERT_EQ(child_nodes.size(), 1U);
  const auto embedded_nodes = xpool::utils::graph::nodes(embedded_graph);
  ASSERT_EQ(embedded_nodes.size(), 1U);
  const auto embedded_node = embedded_nodes.front();
  auto observed_embedded = CUDA_KERNEL_NODE_PARAMS{};
  ASSERT_EQ(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(embedded_node), &observed_embedded), CUDA_SUCCESS);
  ASSERT_EQ(observed_embedded.kernelParams, nullptr);
  ASSERT_NE(observed_embedded.extra, nullptr);
  auto device_node = xpool::utils::graph::make_device_updatable(embedded_node);
  const auto embedded_parameters = xpool::utils::graph::KernelNodeParameters::read(embedded_node);
  ASSERT_TRUE(pointer_parameters.same_schema(embedded_parameters));
  packed_parameters.apply(embedded_node);

  auto normalized = CUDA_KERNEL_NODE_PARAMS{};
  ASSERT_EQ(cuGraphKernelNodeGetParams(reinterpret_cast<CUgraphNode>(embedded_node), &normalized), CUDA_SUCCESS);
  EXPECT_NE(normalized.kernelParams, nullptr);
  EXPECT_EQ(normalized.extra, nullptr);

  auto rebound_value = std::int32_t{31};
  auto update_arguments =
      std::array<void *, 6>{&device_node, &first_output, &rebound_value, &offsets[0], &offsets[1], &update_status};
  const auto update_node =
      xpool::utils::graph::add_kernel_node(parent_graph, reinterpret_cast<const void *>(update_write_value_kernel),
                                           dim3{1}, dim3{1}, 0, update_arguments.data());
  xpool::utils::graph::add_dependency(parent_graph, update_node, child_nodes.front());

  auto executable = cudaGraphExec_t{};
  ASSERT_EQ(cudaGraphInstantiate(&executable, parent_graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(executable, nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto first_value = std::int32_t{0};
  auto second_value = std::int32_t{0};
  auto observed_status = cudaErrorUnknown;
  ASSERT_EQ(cudaMemcpy(&first_value, first_output, sizeof(first_value), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&second_value, second_output, sizeof(second_value), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&observed_status, update_status, sizeof(observed_status), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(first_value, rebound_value);
  EXPECT_EQ(second_value, 0);
  EXPECT_EQ(observed_status, cudaSuccess);

  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(parent_graph), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(cloned_graph), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(packed_graph), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(pointer_graph), cudaSuccess);
  EXPECT_EQ(cudaFree(update_status), cudaSuccess);
  EXPECT_EQ(cudaFree(second_output), cudaSuccess);
  EXPECT_EQ(cudaFree(first_output), cudaSuccess);
}

TEST(GraphTest, UpdatesKernelParametersAndSplicesDependencies) {
  auto device_count = 0;
  const auto availability = cudaGetDeviceCount(&device_count);
  if (availability != cudaSuccess || device_count == 0) {
    GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(availability);
  }
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto first_output = static_cast<std::int32_t *>(nullptr);
  auto second_output = static_cast<std::int32_t *>(nullptr);
  ASSERT_EQ(cudaMalloc(&first_output, sizeof(*first_output)), cudaSuccess);
  ASSERT_EQ(cudaMalloc(&second_output, sizeof(*second_output)), cudaSuccess);
  ASSERT_EQ(cudaMemset(first_output, 0, sizeof(*first_output)), cudaSuccess);
  ASSERT_EQ(cudaMemset(second_output, 0, sizeof(*second_output)), cudaSuccess);

  auto graph = cudaGraph_t{};
  ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  auto predecessor = cudaGraphNode_t{};
  ASSERT_EQ(cudaGraphAddEmptyNode(&predecessor, graph, nullptr, 0), cudaSuccess);
  auto value = std::int32_t{17};
  auto arguments = std::array<void *, 2>{&first_output, &value};
  const auto kernel =
      xpool::utils::graph::add_kernel_node(graph, reinterpret_cast<const void *>(write_value_kernel), dim3{1}, dim3{1},
                                           0, arguments.data(), std::span<const cudaGraphNode_t>{&predecessor, 1});
  auto inserted = cudaGraphNode_t{};
  ASSERT_EQ(cudaGraphAddEmptyNode(&inserted, graph, nullptr, 0), cudaSuccess);

  auto parameters = xpool::utils::graph::KernelNodeParameters::read(kernel);
  ASSERT_EQ(parameters.arguments().size(), 2U);
  EXPECT_TRUE(parameters.contains_address(reinterpret_cast<std::uintptr_t>(first_output)));
  parameters.arguments()[0].write_address(0, reinterpret_cast<std::uintptr_t>(second_output));
  parameters.apply(kernel);
  xpool::utils::graph::insert_node_after(graph, predecessor, inserted);

  auto predecessor_successors = std::size_t{0};
  ASSERT_EQ(cudaGraphNodeGetDependentNodes(predecessor, nullptr, nullptr, &predecessor_successors), cudaSuccess);
  EXPECT_EQ(predecessor_successors, 1U);
  auto inserted_successors = std::size_t{0};
  ASSERT_EQ(cudaGraphNodeGetDependentNodes(inserted, nullptr, nullptr, &inserted_successors), cudaSuccess);
  EXPECT_EQ(inserted_successors, 1U);

  auto executable = cudaGraphExec_t{};
  ASSERT_EQ(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(executable, nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto first_value = std::int32_t{0};
  auto second_value = std::int32_t{0};
  ASSERT_EQ(cudaMemcpy(&first_value, first_output, sizeof(first_value), cudaMemcpyDeviceToHost), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&second_value, second_output, sizeof(second_value), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(first_value, 0);
  EXPECT_EQ(second_value, value);

  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  EXPECT_EQ(cudaFree(second_output), cudaSuccess);
  EXPECT_EQ(cudaFree(first_output), cudaSuccess);
}

} // namespace
