#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>

#include <gtest/gtest.h>

#include <xpool/macros.hpp>
#include <xpool/utils/graph.hpp>

namespace {

XPOOL_KERNEL_FN void empty_kernel() {}

XPOOL_KERNEL_FN void write_value_kernel(std::int32_t *output, std::int32_t value) { *output = value; }

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
  const auto kernel = xpool::utils::graph::add_kernel_node(
      graph, reinterpret_cast<const void *>(write_value_kernel), dim3{1}, dim3{1}, 0, arguments.data(),
      std::span<const cudaGraphNode_t>{&predecessor, 1});
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
