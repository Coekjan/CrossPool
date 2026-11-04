#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/ffnagent/parameterization.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace {

constexpr auto kScratchBytes = std::size_t{32};

XPOOL_KERNEL_FN void use_scratch(const std::int32_t *weight, std::int32_t *scratch, std::int32_t *output,
                                 std::int32_t tag) {
  *scratch = *weight;
  *output = *scratch + tag;
}

cudaGraph_t scratch_graph(std::int32_t *weight, std::int32_t *scratch, std::int32_t *output, std::size_t offset = 1,
                          std::int32_t tag = 1) {
  auto graph = cudaGraph_t{};
  C10_CUDA_CHECK(cudaGraphCreate(&graph, 0));
  auto *interior = scratch + offset;
  auto arguments = std::array<void *, 4>{&weight, &interior, &output, &tag};
  static_cast<void>(xpool::utils::cuda::graph::add_kernel_node(graph, reinterpret_cast<const void *>(use_scratch),
                                                               dim3{1}, dim3{1}, 0, arguments.data()));
  return graph;
}

} // namespace

TEST(FfnParameterizationTest, RetiresCaptureGraphsBeforeIndependentLaneReplay) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto sources = std::array<std::int32_t *, 5>{};
  for (auto &pointer : sources) {
    ASSERT_EQ(cudaMalloc(&pointer, kScratchBytes), cudaSuccess);
  }
  const auto primary = scratch_graph(sources[0], sources[2], sources[4]);
  const auto control = scratch_graph(sources[1], sources[3], sources[4]);
  auto graphs = std::array<cudaGraph_t, 2>{};
  auto executables = std::array<cudaGraphExec_t, 2>{};
  auto streams = std::array<cudaStream_t, 2>{};
  auto workspaces = std::array<std::int32_t *, 2>{};
  auto weights = std::array<std::int32_t *, 2>{};
  auto outputs = std::array<std::int32_t *, 2>{};
  for (auto lane = std::size_t{0}; lane < graphs.size(); ++lane) {
    ASSERT_EQ(cudaMalloc(&workspaces[lane], kScratchBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(workspaces[lane], 0, kScratchBytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&weights[lane], sizeof(std::int32_t)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&outputs[lane], sizeof(std::int32_t)), cudaSuccess);
    const auto value = static_cast<std::int32_t>(18 + lane);
    ASSERT_EQ(cudaMemcpy(weights[lane], &value, sizeof(value), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaGraphCreate(&graphs[lane], 0), cudaSuccess);
    const auto embedded = xpool::utils::cuda::graph::embed_child_graph(graphs[lane], primary);
    const auto resources = std::vector<xpool::ffnagent::ResourceReplacement>{
        {reinterpret_cast<std::uintptr_t>(sources[0]), reinterpret_cast<std::uintptr_t>(sources[1]),
         reinterpret_cast<std::uintptr_t>(weights[lane]), 0},
    };
    const auto replacements = std::vector<xpool::ffnagent::LaneAddressReplacement>{
        {reinterpret_cast<std::uintptr_t>(sources[2]), reinterpret_cast<std::uintptr_t>(sources[3]), kScratchBytes,
         reinterpret_cast<std::uintptr_t>(workspaces[lane])},
        {reinterpret_cast<std::uintptr_t>(sources[4]), reinterpret_cast<std::uintptr_t>(sources[4]),
         sizeof(std::int32_t), reinterpret_cast<std::uintptr_t>(outputs[lane])},
    };
    const auto bindings = xpool::ffnagent::parameterize_graph(embedded, control, resources, replacements, 0, 0, 0);
    ASSERT_EQ(bindings.binding_sites.size(), 1);
    const auto parameters =
        xpool::utils::cuda::graph::KernelNodeParameters::read(xpool::utils::cuda::graph::nodes(embedded).front());
    EXPECT_TRUE(parameters.contains_address(reinterpret_cast<std::uintptr_t>(workspaces[lane] + 1)));
    EXPECT_FALSE(parameters.contains_address(reinterpret_cast<std::uintptr_t>(sources[2] + 1)));
    ASSERT_EQ(cudaGraphInstantiate(&executables[lane], graphs[lane], 0), cudaSuccess);
    ASSERT_EQ(cudaStreamCreateWithFlags(&streams[lane], cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaGraphUpload(executables[lane], streams[lane]), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(streams[lane]), cudaSuccess);
  }
  ASSERT_EQ(cudaGraphDestroy(primary), cudaSuccess);
  ASSERT_EQ(cudaGraphDestroy(control), cudaSuccess);
  for (auto pointer : sources) {
    ASSERT_EQ(cudaFree(pointer), cudaSuccess);
  }
  auto poison = std::array<std::int32_t *, 5>{};
  for (auto &pointer : poison) {
    ASSERT_EQ(cudaMalloc(&pointer, kScratchBytes), cudaSuccess);
    ASSERT_EQ(cudaMemset(pointer, 0xa5, kScratchBytes), cudaSuccess);
  }
  for (auto iteration = 0; iteration < 100; ++iteration) {
    for (auto lane = std::size_t{0}; lane < graphs.size(); ++lane) {
      ASSERT_EQ(cudaGraphLaunch(executables[lane], streams[lane]), cudaSuccess);
    }
  }
  for (auto lane = std::size_t{0}; lane < graphs.size(); ++lane) {
    ASSERT_EQ(cudaStreamSynchronize(streams[lane]), cudaSuccess);
    auto output = std::int32_t{};
    auto scratch = std::array<std::int32_t, kScratchBytes / sizeof(std::int32_t)>{};
    ASSERT_EQ(cudaMemcpy(&output, outputs[lane], sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(scratch.data(), workspaces[lane], kScratchBytes, cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(output, static_cast<std::int32_t>(19 + lane));
    EXPECT_EQ(scratch[1], static_cast<std::int32_t>(18 + lane));
    EXPECT_EQ(scratch[0], 0);
    EXPECT_EQ(scratch[2], 0);
    EXPECT_EQ(cudaGraphExecDestroy(executables[lane]), cudaSuccess);
    EXPECT_EQ(cudaGraphDestroy(graphs[lane]), cudaSuccess);
    EXPECT_EQ(cudaFree(workspaces[lane]), cudaSuccess);
    EXPECT_EQ(cudaFree(weights[lane]), cudaSuccess);
    EXPECT_EQ(cudaFree(outputs[lane]), cudaSuccess);
    EXPECT_EQ(cudaStreamDestroy(streams[lane]), cudaSuccess);
  }
  for (auto pointer : poison) {
    auto bytes = std::array<std::uint8_t, kScratchBytes>{};
    ASSERT_EQ(cudaMemcpy(bytes.data(), pointer, bytes.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_TRUE(std::ranges::all_of(bytes, [](std::uint8_t value) { return value == 0xa5; }));
    EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
}

TEST(FfnParameterizationTest, RejectsMismatchedOffsetsAndUndeclaredScalarChanges) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto storage = std::array<std::int32_t *, 5>{};
  for (auto &pointer : storage) {
    ASSERT_EQ(cudaMalloc(&pointer, kScratchBytes), cudaSuccess);
  }
  const auto primary = scratch_graph(storage[0], storage[2], storage[4]);
  const auto wrong_offset = scratch_graph(storage[1], storage[3], storage[4], 2);
  const auto wrong_scalar = scratch_graph(storage[1], storage[3], storage[4], 1, 2);
  const auto resources = std::vector<xpool::ffnagent::ResourceReplacement>{
      {reinterpret_cast<std::uintptr_t>(storage[0]), reinterpret_cast<std::uintptr_t>(storage[1]),
       reinterpret_cast<std::uintptr_t>(storage[0]), 0},
  };
  const auto replacements = std::vector<xpool::ffnagent::LaneAddressReplacement>{
      {reinterpret_cast<std::uintptr_t>(storage[2]), reinterpret_cast<std::uintptr_t>(storage[3]), kScratchBytes,
       reinterpret_cast<std::uintptr_t>(storage[2])},
  };
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(primary, wrong_offset, resources, replacements, 0, 0, 0),
               c10::Error);
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(primary, wrong_scalar, resources, replacements, 0, 0, 0),
               c10::Error);
  EXPECT_EQ(cudaGraphDestroy(primary), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(wrong_offset), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(wrong_scalar), cudaSuccess);
  for (auto pointer : storage) {
    EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
}

TEST(FfnParameterizationTest, AllowsUnusedOptionalRangeAndRejectsUnusedRequiredRange) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto graphs = std::array<cudaGraph_t, 2>{};
  for (auto &graph : graphs) {
    ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
  }
  auto range = xpool::ffnagent::LaneAddressReplacement{0x1000, 0x2000, 64, 0x3000, false};
  EXPECT_NO_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {&range, 1}, 0, 0, 0));
  range.required = true;
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {&range, 1}, 0, 0, 0), c10::Error);
  for (auto graph : graphs) {
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
}

TEST(FfnParameterizationTest, RejectsTwoDimensionalDeviceMemcpy) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto storage = std::array<void *, 2>{};
  for (auto &pointer : storage) {
    ASSERT_EQ(cudaMalloc(&pointer, 64), cudaSuccess);
  }
  auto graphs = std::array<cudaGraph_t, 2>{};
  auto parameters = cudaMemcpy3DParms{};
  parameters.srcPtr = make_cudaPitchedPtr(storage[0], 16, 16, 4);
  parameters.dstPtr = make_cudaPitchedPtr(storage[1], 16, 16, 4);
  parameters.extent = make_cudaExtent(8, 2, 1);
  parameters.kind = cudaMemcpyDeviceToDevice;
  for (auto &graph : graphs) {
    ASSERT_EQ(cudaGraphCreate(&graph, 0), cudaSuccess);
    auto node = cudaGraphNode_t{};
    ASSERT_EQ(cudaGraphAddMemcpyNode(&node, graph, nullptr, 0, &parameters), cudaSuccess);
  }
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {}, 0, 0, 0), c10::Error);
  for (auto graph : graphs) {
    EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
  }
  for (auto pointer : storage) {
    EXPECT_EQ(cudaFree(pointer), cudaSuccess);
  }
}

TEST(FfnParameterizationTest, BindsMemsetWithPitchAndRejectsIncompleteWriteSpans) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto storage = std::array<std::uint8_t *, 3>{};
  for (auto &pointer : storage) {
    ASSERT_EQ(cudaMalloc(&pointer, 64), cudaSuccess);
  }
  auto graphs = std::array<cudaGraph_t, 2>{};
  auto nodes = std::array<cudaGraphNode_t, 2>{};
  for (auto index = std::size_t{0}; index < graphs.size(); ++index) {
    ASSERT_EQ(cudaGraphCreate(&graphs[index], 0), cudaSuccess);
    const auto parameters = cudaMemsetParams{storage[index] + 4, 16, 7, 1, 12, 2};
    ASSERT_EQ(cudaGraphAddMemsetNode(&nodes[index], graphs[index], nullptr, 0, &parameters), cudaSuccess);
  }
  auto replacement = xpool::ffnagent::LaneAddressReplacement{reinterpret_cast<std::uintptr_t>(storage[0]),
                                                             reinterpret_cast<std::uintptr_t>(storage[1]), 31,
                                                             reinterpret_cast<std::uintptr_t>(storage[2])};
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {&replacement, 1}, 0, 0, 0), c10::Error);
  replacement.bytes = 64;
  replacement.read_only = true;
  EXPECT_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {&replacement, 1}, 0, 0, 0), c10::Error);
  replacement.read_only = false;
  ASSERT_NO_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, {&replacement, 1}, 0, 0, 0));
  ASSERT_EQ(cudaGraphDestroy(graphs[1]), cudaSuccess);
  ASSERT_EQ(cudaFree(storage[0]), cudaSuccess);
  ASSERT_EQ(cudaFree(storage[1]), cudaSuccess);
  ASSERT_EQ(cudaMemset(storage[2], 0, 64), cudaSuccess);
  auto executable = cudaGraphExec_t{};
  ASSERT_EQ(cudaGraphInstantiate(&executable, graphs[0], 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(executable, nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto result = std::array<std::uint8_t, 64>{};
  ASSERT_EQ(cudaMemcpy(result.data(), storage[2], result.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (auto index = std::size_t{0}; index < result.size(); ++index) {
    EXPECT_EQ(result[index], (index >= 4 && index < 16) || (index >= 20 && index < 32) ? 7 : 0);
  }
  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graphs[0]), cudaSuccess);
  EXPECT_EQ(cudaFree(storage[2]), cudaSuccess);
}

TEST(FfnParameterizationTest, BindsDeviceMemcpyEndpointsBeforeSourceRetirement) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  auto storage = std::array<std::uint8_t *, 6>{};
  for (auto &pointer : storage) {
    ASSERT_EQ(cudaMalloc(&pointer, 64), cudaSuccess);
  }
  auto graphs = std::array<cudaGraph_t, 2>{};
  for (auto index = std::size_t{0}; index < graphs.size(); ++index) {
    ASSERT_EQ(cudaGraphCreate(&graphs[index], 0), cudaSuccess);
    auto node = cudaGraphNode_t{};
    ASSERT_EQ(cudaGraphAddMemcpyNode1D(&node, graphs[index], nullptr, 0, storage[index + 2] + 4, storage[index] + 8, 16,
                                       cudaMemcpyDeviceToDevice),
              cudaSuccess);
  }
  const auto replacements = std::array<xpool::ffnagent::LaneAddressReplacement, 2>{
      xpool::ffnagent::LaneAddressReplacement{reinterpret_cast<std::uintptr_t>(storage[0]),
                                              reinterpret_cast<std::uintptr_t>(storage[1]), 64,
                                              reinterpret_cast<std::uintptr_t>(storage[4]), true, true},
      xpool::ffnagent::LaneAddressReplacement{reinterpret_cast<std::uintptr_t>(storage[2]),
                                              reinterpret_cast<std::uintptr_t>(storage[3]), 64,
                                              reinterpret_cast<std::uintptr_t>(storage[5])},
  };
  ASSERT_NO_THROW(xpool::ffnagent::parameterize_graph(graphs[0], graphs[1], {}, replacements, 0, 0, 0));
  ASSERT_EQ(cudaGraphDestroy(graphs[1]), cudaSuccess);
  for (auto index = std::size_t{0}; index < 4; ++index) {
    ASSERT_EQ(cudaFree(storage[index]), cudaSuccess);
  }
  ASSERT_EQ(cudaMemset(storage[4], 13, 64), cudaSuccess);
  ASSERT_EQ(cudaMemset(storage[5], 0, 64), cudaSuccess);
  auto executable = cudaGraphExec_t{};
  ASSERT_EQ(cudaGraphInstantiate(&executable, graphs[0], 0), cudaSuccess);
  ASSERT_EQ(cudaGraphLaunch(executable, nullptr), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  auto result = std::array<std::uint8_t, 64>{};
  ASSERT_EQ(cudaMemcpy(result.data(), storage[5], result.size(), cudaMemcpyDeviceToHost), cudaSuccess);
  for (auto index = std::size_t{0}; index < result.size(); ++index) {
    EXPECT_EQ(result[index], index >= 4 && index < 20 ? 13 : 0);
  }
  EXPECT_EQ(cudaGraphExecDestroy(executable), cudaSuccess);
  EXPECT_EQ(cudaGraphDestroy(graphs[0]), cudaSuccess);
  EXPECT_EQ(cudaFree(storage[4]), cudaSuccess);
  EXPECT_EQ(cudaFree(storage[5]), cudaSuccess);
}
