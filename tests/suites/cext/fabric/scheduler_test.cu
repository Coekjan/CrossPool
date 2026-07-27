/// \file tests/suites/cext/fabric/scheduler_test.cu
/// \brief Device behavior tests for FIFO and Random FFN scheduling.

#include <c10/cuda/CUDAException.h>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <xpool/abi.hpp>
#include <xpool/fabric/arena.cuh>
#include <xpool/fabric/scheduler.cuh>

namespace {

constexpr auto kModelCount = std::size_t{3};
constexpr auto kPayloadCapacity = xpool::arena::kPayloadAlignment;

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class SchedulerArena {
public:
  SchedulerArena(const xpool::fabric::FfnSchedulerPolicy &policy, std::size_t executor_count)
      : layout_(xpool::fabric::FabricArenaLayout::create(
            1, 1, executor_count, kModelCount, kModelCount, kModelCount * kPayloadCapacity,
            kPayloadCapacity)) {
    auto *allocation = static_cast<void *>(nullptr);
    C10_CUDA_CHECK(cudaMallocManaged(&allocation, layout_.header.total_bytes));
    base_ = static_cast<std::uint8_t *>(allocation);
    C10_CUDA_CHECK(cudaMemset(base_, 0, layout_.header.total_bytes));

    std::memcpy(base_, &layout_, sizeof(layout_));
    auto models = std::array<xpool::fabric::FabricModelLayout, kModelCount>{};
    auto layers = std::array<xpool::fabric::FabricLayerLayout, kModelCount>{};
    for (auto model_index = std::size_t{0}; model_index < kModelCount; ++model_index) {
      models[model_index] = xpool::fabric::FabricModelLayout{
          .dtype = xpool::abi::TensorDType::Fp32,
          .hidden_size = 2,
          .atn_tp_size = 1,
          .atn_dp_size = 1,
          .layer_begin = model_index,
          .layer_count = 1,
          .decode_payload_offset = model_index * kPayloadCapacity,
          .decode_payload_capacity_bytes = kPayloadCapacity,
          .prefill_payload_capacity_bytes = kPayloadCapacity,
      };
      layers[model_index] = xpool::fabric::FabricLayerLayout{
          .layer_id = model_index,
          .kind = xpool::fabric::FfnLayerKind::Dense,
      };
    }
    std::memcpy(
        base_ + layout_.model_layouts_offset, models.data(), sizeof(models));
    std::memcpy(
        base_ + layout_.layer_layouts_offset, layers.data(), sizeof(layers));
    const auto scheduler = xpool::fabric::FfnScheduler::from(policy);
    std::memcpy(base_ + layout_.scheduler_offset, &scheduler, sizeof(scheduler));
  }

  ~SchedulerArena() {
    if (base_ != nullptr) {
      C10_CUDA_IGNORE_ERROR(cudaFree(base_));
    }
  }

  SchedulerArena(const SchedulerArena &) = delete;
  SchedulerArena &operator=(const SchedulerArena &) = delete;

  xpool::fabric::FabricArenaView view() const {
    return xpool::fabric::FabricArenaView{base_};
  }

private:
  xpool::fabric::FabricArenaLayout layout_;
  std::uint8_t *base_ = nullptr;
};

XPOOL_HOST_DEVICE_FN xpool::fabric::FfnInvocation invocation(std::size_t model_index) {
  return xpool::fabric::FfnInvocation{
      .key = {.model_index = model_index, .invocation_sequence = 1},
      .layer_ordinal = 0,
      .payload_rows = 1,
      .input_pe = 0,
      .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
}

__global__ void exercise_fifo(
    xpool::fabric::FabricArenaView arena, std::size_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &scheduler = arena.scheduler();
  scheduler.enqueue(arena, invocation(2));
  scheduler.enqueue(arena, invocation(0));
  scheduler.enqueue(arena, invocation(1));
  for (auto index = std::size_t{0}; index < kModelCount; ++index) {
    const auto decision = scheduler.schedule(arena);
    observed[index] = decision.invocation().key.model_index;
    observed[kModelCount + index] = decision.executor_index();
    scheduler.release(arena, decision.invocation().key, decision.executor_index());
  }
  observed[6] = scheduler.schedule(arena) ? 1 : 0;
  observed[7] = arena.scheduler_entry(2).next_sequence();
}

__global__ void exercise_concurrent_executors(
    xpool::fabric::FabricArenaView arena, std::size_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &scheduler = arena.scheduler();
  for (auto model_index = std::size_t{0}; model_index < kModelCount; ++model_index) {
    scheduler.enqueue(arena, invocation(model_index));
  }
  const auto first = scheduler.schedule(arena);
  const auto second = scheduler.schedule(arena);
  observed[0] = first.invocation().key.model_index;
  observed[1] = first.executor_index();
  observed[2] = second.invocation().key.model_index;
  observed[3] = second.executor_index();
  observed[4] = scheduler.schedule(arena) ? 1 : 0;
  scheduler.release(arena, first.invocation().key, first.executor_index());
  const auto third = scheduler.schedule(arena);
  observed[5] = third.invocation().key.model_index;
  observed[6] = third.executor_index();
}

__global__ void exercise_single_executor_queue(
    xpool::fabric::FabricArenaView arena, std::size_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &scheduler = arena.scheduler();
  scheduler.enqueue(arena, invocation(0));
  scheduler.enqueue(arena, invocation(1));
  const auto first = scheduler.schedule(arena);
  observed[0] = first.invocation().key.model_index;
  observed[1] = first.executor_index();
  observed[2] = scheduler.schedule(arena) ? 1 : 0;
  scheduler.release(arena, first.invocation().key, first.executor_index());
  const auto second = scheduler.schedule(arena);
  observed[3] = second.invocation().key.model_index;
  observed[4] = second.executor_index();
}

__global__ void exercise_random(
    xpool::fabric::FabricArenaView arena, std::size_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &scheduler = arena.scheduler();
  for (auto attempt = 0; attempt < 3; ++attempt) {
    observed[attempt] = scheduler.schedule(arena) ? 1 : 0;
  }
  for (auto model_index = std::size_t{0}; model_index < kModelCount; ++model_index) {
    scheduler.enqueue(arena, invocation(model_index));
  }
  for (auto index = std::size_t{0}; index < kModelCount; ++index) {
    const auto decision = scheduler.schedule(arena);
    observed[3 + index] = decision.invocation().key.model_index;
    scheduler.release(arena, decision.invocation().key, decision.executor_index());
  }
}

class FfnSchedulerTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto device_count = int{0};
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(error);
    }
    ASSERT_TRUE(cuda_succeeded(cudaSetDevice(0)));
  }
};

} // namespace

TEST_F(FfnSchedulerTest, FifoPreservesEnqueueOrderAndReleasesHistory) {
  auto arena = SchedulerArena{xpool::fabric::FfnSchedulerPolicy::fifo(), 1};
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 8 * sizeof(*observed))));

  exercise_fifo<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[1], observed[2]}), (std::array<std::size_t, 3>{2, 0, 1}));
  EXPECT_EQ((std::array{observed[3], observed[4], observed[5]}), (std::array<std::size_t, 3>{0, 0, 0}));
  EXPECT_EQ(observed[6], 0U);
  EXPECT_EQ(observed[7], 2U);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
}

TEST_F(FfnSchedulerTest, LeasesDistinctExecutorsUntilOneIsReleased) {
  auto arena = SchedulerArena{xpool::fabric::FfnSchedulerPolicy::fifo(), 2};
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 7 * sizeof(*observed))));

  exercise_concurrent_executors<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[2], observed[5]}), (std::array<std::size_t, 3>{0, 1, 2}));
  EXPECT_EQ((std::array{observed[1], observed[3], observed[6]}), (std::array<std::size_t, 3>{0, 1, 0}));
  EXPECT_EQ(observed[4], 0U);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
}

TEST_F(FfnSchedulerTest, QueuesUntilTheOnlyExecutorIsReleased) {
  auto arena = SchedulerArena{xpool::fabric::FfnSchedulerPolicy::fifo(), 1};
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 5 * sizeof(*observed))));

  exercise_single_executor_queue<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[3]}), (std::array<std::size_t, 2>{0, 1}));
  EXPECT_EQ((std::array{observed[1], observed[4]}), (std::array<std::size_t, 2>{0, 0}));
  EXPECT_EQ(observed[2], 0U);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
}

TEST_F(FfnSchedulerTest, RandomIsDeterministicAndIgnoresEmptyPolls) {
  auto arena = SchedulerArena{xpool::fabric::FfnSchedulerPolicy::random(7), 1};
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 6 * sizeof(*observed))));

  exercise_random<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[1], observed[2]}), (std::array<std::size_t, 3>{0, 0, 0}));
  EXPECT_EQ((std::array{observed[3], observed[4], observed[5]}), (std::array<std::size_t, 3>{1, 2, 0}));
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
}
