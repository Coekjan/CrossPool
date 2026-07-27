/// \file tests/suites/cext/trace_test.cu
/// \brief Common trace timeline and device-buffer behavior tests.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <xpool/trace.cuh>
#include <xpool/utils/wait.cuh>

namespace {

enum class TestEvent : std::uint32_t {
  Begin,
  End,
  Count,
};

struct TestRecord {
  std::uint64_t sequence;
};

struct TestArena {
  xpool::trace::BufferState state;
  TestRecord records[2];
};

static_assert(xpool::utils::wait::Poll<decltype([] { return true; })>);
static_assert(!xpool::utils::wait::Poll<decltype([] { return 1; })>);
static_assert(std::is_trivially_copyable_v<xpool::trace::Timeline<TestEvent>>);

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

__global__ void reserve_trace_records(TestArena *arena, std::uint64_t *reservations) {
  if (threadIdx.x != 0) {
    return;
  }
  const auto layout = xpool::trace::BufferLayout{
      .records_offset = offsetof(TestArena, records),
      .capacity = 2,
  };
  const auto buffer = xpool::trace::Buffer<TestRecord>{reinterpret_cast<std::uint8_t *>(arena), layout, arena->state};
  for (auto index = std::size_t{0}; index < 3; ++index) {
    const auto entry = buffer.reserve();
    reservations[index] = entry ? entry.sequence() : 0;
    if (entry) {
      entry.record().sequence = entry.sequence();
    }
  }
}

class TraceBufferTest : public ::testing::Test {
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

TEST(TraceTimelineTest, RecordsDependencyGraph) {
  auto timeline = xpool::trace::Timeline<TestEvent>{};

  timeline.record<TestEvent::Begin>(10);
  timeline.record<TestEvent::End, TestEvent::Begin>(20);

  EXPECT_TRUE(timeline.recorded(TestEvent::Begin));
  EXPECT_TRUE(timeline.recorded(TestEvent::End));
  EXPECT_EQ(timeline.timestamp(TestEvent::Begin), 10U);
  EXPECT_EQ(timeline.timestamp(TestEvent::End), 20U);
}

TEST(TraceTimelineTest, FailStopsOnMissingDependency) {
  EXPECT_DEATH(([] {
                 auto timeline = xpool::trace::Timeline<TestEvent>{};
                 timeline.record<TestEvent::End, TestEvent::Begin>(20);
               }()),
               "");
}

TEST(TraceTimelineTest, FailStopsOnDuplicateEvent) {
  EXPECT_DEATH(([] {
                 auto timeline = xpool::trace::Timeline<TestEvent>{};
                 timeline.record<TestEvent::Begin>(10);
                 timeline.record<TestEvent::Begin>(20);
               }()),
               "");
}

TEST_F(TraceBufferTest, DropsNewRecordsAfterCapacity) {
  auto *arena = static_cast<TestArena *>(nullptr);
  auto *reservations = static_cast<std::uint64_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&arena, sizeof(TestArena))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&reservations, 3 * sizeof(std::uint64_t))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(arena, 0, sizeof(TestArena))));

  reserve_trace_records<<<1, 1, 0, nullptr>>>(arena, reservations);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(arena->state.sequence, 3U);
  EXPECT_EQ(arena->state.dropped, 1U);
  EXPECT_EQ(arena->records[0].sequence, 1U);
  EXPECT_EQ(arena->records[1].sequence, 2U);
  EXPECT_EQ(reservations[0], 1U);
  EXPECT_EQ(reservations[1], 2U);
  EXPECT_EQ(reservations[2], 0U);

  EXPECT_TRUE(cuda_succeeded(cudaFree(reservations)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(arena)));
}
