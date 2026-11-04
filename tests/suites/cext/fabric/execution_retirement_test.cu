#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/fabric/arena.hpp>
#include <xpool/fabric/layout.hpp>
#include <xpool/fabric/projection.hpp>
#include <xpool/ffnagent/runtime.hpp>
#include <xpool/macros.hpp>
#include <xpool/utils/cuda/graph.hpp>

namespace {

struct CudaFaults {
  bool enabled = false;
  int instantiations = 0;
  int fail_instantiation = 0;
  bool intercept_launches = false;
  int launches = 0;
  int fail_launch = 0;
  int queries = 0;
  int fail_query = 0;
  bool fail_exec_destroy = false;
  bool fail_free = false;
  int synchronization_failures = 0;
  cudaStream_t unconfirmed_stream = nullptr;
  bool destroyed_before_confirmation = false;
  std::unordered_set<void *> allocations;
  std::unordered_set<cudaGraphExec_t> executables;
  std::unordered_set<void *> borrowed_allocations;
  bool borrowed_free_attempted = false;
};

CudaFaults faults;
std::atomic<bool> producer_release{true};

void CUDART_CB wait_for_producer_release(void *) {
  while (!producer_release.load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

} // namespace

extern "C" cudaError_t __real_cudaMalloc(void **pointer, std::size_t bytes);
extern "C" cudaError_t __real_cudaFree(void *pointer);
extern "C" cudaError_t __real_cudaGraphInstantiateWithParams(cudaGraphExec_t *executable, cudaGraph_t graph,
                                                             cudaGraphInstantiateParams *parameters);
extern "C" cudaError_t __real_cudaGraphExecDestroy(cudaGraphExec_t executable);
extern "C" cudaError_t __real_cudaStreamSynchronize(cudaStream_t stream);
extern "C" cudaError_t __real_cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream);
extern "C" cudaError_t __real_cudaStreamQuery(cudaStream_t stream);

extern "C" cudaError_t __wrap_cudaGraphLaunch(cudaGraphExec_t executable, cudaStream_t stream) {
  if (faults.enabled && faults.intercept_launches) {
    if (++faults.launches == faults.fail_launch) {
      return cudaErrorInvalidValue;
    }
    // Real queued work models a resident producer without starting NVSHMEM kernels.
    return cudaLaunchHostFunc(stream, wait_for_producer_release, nullptr);
  }
  return __real_cudaGraphLaunch(executable, stream);
}

extern "C" cudaError_t __wrap_cudaStreamQuery(cudaStream_t stream) {
  if (faults.enabled && ++faults.queries == faults.fail_query) {
    return cudaErrorIllegalAddress;
  }
  return __real_cudaStreamQuery(stream);
}

extern "C" cudaError_t __wrap_cudaStreamSynchronize(cudaStream_t stream) {
  if (faults.enabled && faults.synchronization_failures > 0) {
    --faults.synchronization_failures;
    faults.unconfirmed_stream = stream;
    return cudaErrorInvalidValue;
  }
  const auto result = __real_cudaStreamSynchronize(stream);
  if (result == cudaSuccess && stream == faults.unconfirmed_stream) {
    faults.unconfirmed_stream = nullptr;
  }
  return result;
}

extern "C" cudaError_t __wrap_cudaMalloc(void **pointer, std::size_t bytes) {
  const auto result = __real_cudaMalloc(pointer, bytes);
  if (faults.enabled && result == cudaSuccess) {
    faults.allocations.insert(*pointer);
  }
  return result;
}

extern "C" cudaError_t __wrap_cudaFree(void *pointer) {
  if (faults.enabled && faults.borrowed_allocations.contains(pointer)) {
    faults.borrowed_free_attempted = true;
    return cudaErrorInvalidValue;
  }
  faults.destroyed_before_confirmation |= faults.unconfirmed_stream != nullptr;
  if (faults.enabled && faults.fail_free) {
    faults.fail_free = false;
    return cudaErrorInvalidValue;
  }
  const auto result = __real_cudaFree(pointer);
  if (result == cudaSuccess) {
    faults.allocations.erase(pointer);
  }
  return result;
}

extern "C" cudaError_t __wrap_cudaGraphInstantiateWithParams(cudaGraphExec_t *executable, cudaGraph_t graph,
                                                             cudaGraphInstantiateParams *parameters) {
  if (faults.enabled && ++faults.instantiations == faults.fail_instantiation) {
    return cudaErrorMemoryAllocation;
  }
  const auto result = __real_cudaGraphInstantiateWithParams(executable, graph, parameters);
  if (faults.enabled && result == cudaSuccess) {
    faults.executables.insert(*executable);
  }
  return result;
}

extern "C" cudaError_t __wrap_cudaGraphExecDestroy(cudaGraphExec_t executable) {
  faults.destroyed_before_confirmation |= faults.unconfirmed_stream != nullptr;
  if (faults.enabled && faults.fail_exec_destroy) {
    faults.fail_exec_destroy = false;
    return cudaErrorInvalidValue;
  }
  const auto result = __real_cudaGraphExecDestroy(executable);
  if (result == cudaSuccess) {
    faults.executables.erase(executable);
  }
  return result;
}

namespace {

class ExecutionRetirementTest : public ::testing::Test {
protected:
  void SetUp() override {
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    projection_.validate();
    descriptor_.lanes.resize(layout_.executor_lane_count);
    descriptor_.validate();
    ASSERT_EQ(cudaMalloc(&arena_, layout_.header.total_bytes), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&activation_count_, sizeof(std::uint32_t)), cudaSuccess);
    faults = {};
    faults.enabled = true;
  }

  void TearDown() override {
    producer_release.store(true, std::memory_order_release);
    faults.fail_query = 0;
    if (faults.intercept_launches) {
      ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    faults.fail_instantiation = 0;
    faults.fail_exec_destroy = false;
    faults.fail_free = false;
    faults.synchronization_failures = 0;
    ASSERT_NO_THROW(runtime_.finalize());
    EXPECT_TRUE(faults.allocations.empty());
    EXPECT_TRUE(faults.executables.empty());
    EXPECT_FALSE(faults.destroyed_before_confirmation);
    faults.enabled = false;
    EXPECT_EQ(cudaFree(activation_count_), cudaSuccess);
    EXPECT_EQ(cudaFree(arena_), cudaSuccess);
  }

  void install() {
    // Upload real resident graphs without activating them.
    runtime_.install(descriptor_, xpool::fabric::ArenaView{arena_}, layout_, projection_, 1, activation_count_);
  }

  xpool::ffnagent::ExecutionRuntime runtime_;
  xpool::ffnagent::ExecutionInstallDescriptor descriptor_;
  xpool::fabric::ArenaProjection projection_{
      .generation_high = 1,
      .generation_low = 2,
      .uid = xpool::fabric::Uid::decode(std::string(sizeof(nvshmemx_uniqueid_t) * 2, 'a')),
      .atnagent_count = 1,
      .ffnagent_count = 2,
      .executor_lane_count = 2,
      .scheduler = xpool::fabric::SchedulerPolicy::fifo(),
      .instances = {{.decode_payload_row_capacity = 1,
                     .prefill_payload_row_capacity = 1,
                     .payload_dtype = c10::ScalarType::BFloat16,
                     .hidden_size = 16,
                     .group_sum_complete_admitted = false,
                     .atn_tp_size = 1,
                     .atn_dp_size = 1,
                     .atnagent_indices = {0},
                     .layers = {{.layer_id = 0,
                                 .kind = xpool::ffn::LayerKind::Dense,
                                 .effective_topk = 0,
                                 .ffnagent_indices = {0}}}}},
  };
  xpool::fabric::ArenaLayout layout_ = xpool::fabric::ArenaLayout::create(1, 2, 1, 2, 1, 1, 1, 32, 0);
  std::uint8_t *arena_ = nullptr;
  std::uint32_t *activation_count_ = nullptr;
};

TEST_F(ExecutionRetirementTest, CompleteInstallationFinalizesOnce) {
  ASSERT_NO_THROW(install());
  EXPECT_TRUE(runtime_.installed());
  EXPECT_EQ(faults.instantiations, 2);
  EXPECT_EQ(faults.executables.size(), 2);
  ASSERT_NO_THROW(runtime_.finalize());
  EXPECT_FALSE(runtime_.installed());
  EXPECT_TRUE(faults.allocations.empty());
  EXPECT_TRUE(faults.executables.empty());
  EXPECT_NO_THROW(runtime_.finalize());
  EXPECT_THROW(install(), c10::Error);
}

TEST_F(ExecutionRetirementTest, PartialActivationRetainsPendingProducerUntilCompletion) {
  ASSERT_NO_THROW(install());
  faults.intercept_launches = true;
  faults.fail_launch = 2;
  producer_release.store(false, std::memory_order_release);
  EXPECT_THROW(runtime_.activate(), c10::Error);
  EXPECT_TRUE(runtime_.drain_pending());
  producer_release.store(true, std::memory_order_release);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  EXPECT_FALSE(runtime_.drain_pending());
}

TEST_F(ExecutionRetirementTest, PartialInstallationPreservesOperationErrorAfterSuccessfulCleanup) {
  faults.fail_instantiation = 2;
  try {
    install();
    FAIL() << "injected installation failure was ignored";
  } catch (const c10::Error &error) {
    EXPECT_NE(std::string(error.what()).find("out of memory"), std::string::npos);
  }
  EXPECT_FALSE(runtime_.installed());
  EXPECT_TRUE(faults.allocations.empty());
  EXPECT_TRUE(faults.executables.empty());
  EXPECT_THROW(install(), c10::Error);
}

XPOOL_KERNEL_FN void borrowed_compute(const std::uint32_t *input, const std::uint32_t *gate, const std::uint32_t *down,
                                      std::uint32_t *scratch, std::uint32_t *output) {
  *scratch = *input + *gate + *down;
  *output = *scratch;
}

class BorrowedExecutionRetirementTest : public ExecutionRetirementTest {
protected:
  void SetUp() override {
    ExecutionRetirementTest::SetUp();
    if (HasFatalFailure()) {
      return;
    }
    faults.enabled = false;
    ASSERT_EQ(cudaMalloc(&weights_, std::size_t{256}), cudaSuccess);
    ASSERT_EQ(cudaMemset(weights_, 0x5a, std::size_t{256}), cudaSuccess);
    faults.borrowed_allocations.insert(weights_);
    const auto projection = [this](std::size_t offset, std::size_t bytes) {
      return xpool::ffnagent::ProjectionBindingView{.values = reinterpret_cast<std::uintptr_t>(weights_ + offset),
                                                    .values_bytes = bytes,
                                                    .scales = std::nullopt,
                                                    .scales_bytes = 0,
                                                    .bias = std::nullopt,
                                                    .bias_bytes = 0};
    };
    const auto primary = xpool::ffnagent::DenseBindingView{.gate_up = projection(0, 64), .down = projection(64, 32)};
    const auto control = xpool::ffnagent::DenseBindingView{.gate_up = projection(128, 64), .down = projection(192, 32)};
    projection_.instances[0].layers[0].ffnagent_indices = {1};
    projection_.validate();
    descriptor_.layers = {{.instance_index = 0,
                           .layer_ordinal = 0,
                           .execution_signature_indices = {0},
                           .layer_resource_targets = primary}};
    for (auto lane = std::size_t{0}; lane < backing_.size(); ++lane) {
      ASSERT_EQ(cudaMalloc(&backing_[lane], std::size_t{96}), cudaSuccess);
      ASSERT_EQ(cudaMemset(backing_[lane], 0x5a, std::size_t{96}), cudaSuccess);
      faults.borrowed_allocations.insert(backing_[lane]);
      auto *input = reinterpret_cast<std::uint32_t *>(backing_[lane]);
      auto *output = reinterpret_cast<std::uint32_t *>(backing_[lane] + 32);
      auto *scratch = reinterpret_cast<std::uint32_t *>(backing_[lane] + 64);
      const auto capture = [&](const xpool::ffnagent::DenseBindingView &view) {
        auto graph = cudaGraph_t{};
        C10_CUDA_CHECK(cudaGraphCreate(&graph, 0));
        source_graphs_.push_back(graph);
        auto *gate = reinterpret_cast<const std::uint32_t *>(view.gate_up.values);
        auto *down = reinterpret_cast<const std::uint32_t *>(view.down.values);
        auto args = std::array<void *, 5>{&input, &gate, &down, &scratch, &output};
        static_cast<void>(xpool::utils::cuda::graph::add_kernel_node(
            graph, reinterpret_cast<const void *>(borrowed_compute), dim3{1}, dim3{1}, 0, args.data()));
        return reinterpret_cast<std::uintptr_t>(graph);
      };
      descriptor_.lanes[lane].signatures.push_back(xpool::ffnagent::DenseCaptureDescriptor{
          .payload_dtype = c10::ScalarType::BFloat16,
          .payload_row_capacity = 1,
          .hidden_size = 16,
          .local_intermediate_size = 1,
          .primary_graph_address = capture(primary),
          .control_graph_address = capture(control),
          .capture_input_address = reinterpret_cast<std::uintptr_t>(input),
          .capture_partial_address = reinterpret_cast<std::uintptr_t>(output),
          .capture_ranges = {{"compute", reinterpret_cast<std::uintptr_t>(scratch),
                              reinterpret_cast<std::uintptr_t>(scratch), 32}},
          .primary_weight_bindings = primary,
          .control_weight_bindings = control,
      });
    }
    descriptor_.validate();
    faults.enabled = true;
  }

  void TearDown() override {
    ExecutionRetirementTest::TearDown();
    EXPECT_FALSE(faults.borrowed_free_attempted);
    for (const auto graph : source_graphs_) {
      EXPECT_EQ(cudaGraphDestroy(graph), cudaSuccess);
    }
    for (auto *pointer : backing_) {
      EXPECT_EQ(cudaFree(pointer), cudaSuccess);
    }
    EXPECT_EQ(cudaFree(weights_), cudaSuccess);
  }

  void expect_borrowed_backing_alive() {
    ASSERT_FALSE(faults.borrowed_free_attempted);
    for (auto *pointer : backing_) {
      auto value = std::uint32_t{};
      ASSERT_EQ(cudaMemcpy(&value, pointer + 64, sizeof(value), cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(value, std::uint32_t{0x5a5a5a5a});
    }
  }

  std::uint8_t *weights_ = nullptr;
  std::array<std::uint8_t *, 2> backing_{};
  std::vector<cudaGraph_t> source_graphs_;
};

TEST_F(BorrowedExecutionRetirementTest, PartialInstallationRetiresBeforeBorrowedOwnerRelease) {
  faults.fail_instantiation = 2;
  EXPECT_THROW(install(), c10::Error);
  EXPECT_TRUE(faults.executables.empty());
  EXPECT_TRUE(faults.allocations.empty());
  expect_borrowed_backing_alive();
}

} // namespace
