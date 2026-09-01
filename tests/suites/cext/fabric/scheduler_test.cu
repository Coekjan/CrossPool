#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <xpool/ffn.hpp>
#include <xpool/fabric/scheduler.cuh>
#include <xpool/macros.hpp>

namespace {

constexpr auto kInstanceCount = std::size_t{3};

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

XPOOL_HOST_DEVICE_FN xpool::fabric::Invocation invocation(std::size_t instance_index) {
  return {
      .key = {.instance_index = instance_index, .invocation_sequence = 1},
      .layer_ordinal = 0,
      .payload_rows = 1,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
  };
}

XPOOL_KERNEL_FN void exercise_fifo(xpool::fabric::Scheduler scheduler, std::size_t *observed) {
  observed[0] = scheduler.enqueue(invocation(2));
  observed[1] = scheduler.enqueue(invocation(0));
  observed[2] = scheduler.enqueue(invocation(1));
  for (auto index = std::size_t{0}; index < kInstanceCount; ++index) {
    const auto decision = scheduler.try_schedule();
    observed[3 + index] = decision->invocation().key.instance_index;
    observed[6 + index] = decision->executor_lane_index();
    scheduler.release(decision->invocation().key, decision->executor_lane_index());
  }
  observed[9] = scheduler.try_schedule().has_value() ? 1 : 0;
  observed[10] = scheduler.has_unresolved_invocation(2) ? 1 : 0;
}

XPOOL_KERNEL_FN void exercise_two_lanes(xpool::fabric::Scheduler scheduler, std::size_t *observed) {
  for (auto instance_index = std::size_t{0}; instance_index < kInstanceCount; ++instance_index) {
    scheduler.enqueue(invocation(instance_index));
  }
  const auto first = scheduler.try_schedule();
  const auto second = scheduler.try_schedule();
  observed[0] = first->invocation().key.instance_index;
  observed[1] = first->executor_lane_index();
  observed[2] = second->invocation().key.instance_index;
  observed[3] = second->executor_lane_index();
  observed[4] = scheduler.try_schedule().has_value() ? 1 : 0;
  const auto active = scheduler.active_decision(first->invocation().key.instance_index);
  observed[5] = active->executor_lane_index();
  scheduler.release(first->invocation().key, first->executor_lane_index());
  const auto third = scheduler.try_schedule();
  observed[6] = third->invocation().key.instance_index;
  observed[7] = third->executor_lane_index();
}

XPOOL_KERNEL_FN void exercise_random(xpool::fabric::Scheduler scheduler, std::size_t *observed) {
  observed[0] = scheduler.try_schedule().has_value() ? 1 : 0;
  observed[1] = scheduler.try_schedule().has_value() ? 1 : 0;
  for (auto instance_index = std::size_t{0}; instance_index < kInstanceCount; ++instance_index) {
    scheduler.enqueue(invocation(instance_index));
  }
  for (auto index = std::size_t{0}; index < kInstanceCount; ++index) {
    const auto decision = scheduler.try_schedule();
    observed[2 + index] = decision->invocation().key.instance_index;
    scheduler.release(decision->invocation().key, decision->executor_lane_index());
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

TEST_F(FfnSchedulerTest, FifoPreservesReadyOrderAndReleasesState) {
  auto *entries = static_cast<xpool::fabric::SchedulerEntry *>(nullptr);
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&entries, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(entries, 0, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 11 * sizeof(*observed))));
  const auto scheduler =
      xpool::fabric::Scheduler::from(xpool::fabric::SchedulerPolicy::fifo(), entries, kInstanceCount, 1);

  exercise_fifo<<<1, 1, 0, nullptr>>>(scheduler, observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[1], observed[2]}), (std::array<std::size_t, 3>{1, 2, 3}));
  EXPECT_EQ((std::array{observed[3], observed[4], observed[5]}), (std::array<std::size_t, 3>{2, 0, 1}));
  EXPECT_EQ((std::array{observed[6], observed[7], observed[8]}), (std::array<std::size_t, 3>{0, 0, 0}));
  EXPECT_EQ(observed[9], 0U);
  EXPECT_EQ(observed[10], 0U);
  EXPECT_EQ(entries[2].completed_sequence, 1U);

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(entries)));
}

TEST_F(FfnSchedulerTest, HoldsQueuedWorkUntilOneOfTwoLanesIsReleased) {
  auto *entries = static_cast<xpool::fabric::SchedulerEntry *>(nullptr);
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&entries, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(entries, 0, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 8 * sizeof(*observed))));
  const auto scheduler =
      xpool::fabric::Scheduler::from(xpool::fabric::SchedulerPolicy::fifo(), entries, kInstanceCount, 2);

  exercise_two_lanes<<<1, 1, 0, nullptr>>>(scheduler, observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[2], observed[6]}), (std::array<std::size_t, 3>{0, 1, 2}));
  EXPECT_EQ((std::array{observed[1], observed[3], observed[5], observed[7]}), (std::array<std::size_t, 4>{0, 1, 0, 0}));
  EXPECT_EQ(observed[4], 0U);

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(entries)));
}

TEST_F(FfnSchedulerTest, RandomSelectionIsDeterministicAndEmptyPollsDoNotAdvanceIt) {
  auto *entries = static_cast<xpool::fabric::SchedulerEntry *>(nullptr);
  auto *observed = static_cast<std::size_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&entries, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(entries, 0, kInstanceCount * sizeof(*entries))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 5 * sizeof(*observed))));
  const auto scheduler =
      xpool::fabric::Scheduler::from(xpool::fabric::SchedulerPolicy::random(7), entries, kInstanceCount, 1);

  exercise_random<<<1, 1, 0, nullptr>>>(scheduler, observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ((std::array{observed[0], observed[1]}), (std::array<std::size_t, 2>{0, 0}));
  EXPECT_EQ((std::array{observed[2], observed[3], observed[4]}), (std::array<std::size_t, 3>{1, 2, 0}));

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(entries)));
}
