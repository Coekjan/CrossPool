/// \file tests/cext/utils/ring_queue_test.cu
/// \brief GoogleTest coverage for xpool::utils::queue::RingQueue.

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include <xpool/utils/queue.cuh>

namespace {

using xpool::utils::queue::RingQueue;
using xpool::utils::queue::RingQueueCell;
using xpool::utils::queue::RingQueueHostImage;
using xpool::utils::queue::RingQueueState;

constexpr int kConcurrentItemCount = 64;

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class RingQueueCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    const cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: "
                   << cudaGetErrorString(error);
    }
    ASSERT_TRUE(cuda_succeeded(cudaSetDevice(0)));
  }
};

__global__ void
sequential_ring_queue_kernel(std::uint32_t *observed, RingQueueState *state,
                             RingQueueCell<std::uint32_t> *cells) {
  if (threadIdx.x != 0) {
    return;
  }

  state->capacity = 2ULL;
  state->head = 0ULL;
  state->tail = 0ULL;
  cells[0] = RingQueueCell<std::uint32_t>{0ULL, 0U};
  cells[1] = RingQueueCell<std::uint32_t>{1ULL, 0U};

  RingQueue<std::uint32_t> queue{state, cells};
  std::uint32_t value = 0U;
  observed[0] = queue.try_pop(value) ? 1U : 0U;
  observed[1] = queue.try_push(11U) ? 1U : 0U;
  observed[2] = queue.try_push(22U) ? 1U : 0U;
  observed[3] = queue.try_push(33U) ? 1U : 0U;
  observed[4] = queue.try_pop(value) ? value : 0U;
  observed[5] = queue.try_push(33U) ? 1U : 0U;
  observed[6] = queue.try_pop(value) ? value : 0U;
  observed[7] = queue.try_pop(value) ? value : 0U;
  observed[8] = queue.try_pop(value) ? 1U : 0U;
  observed[9] = queue.try_pop_with_timeout(value, 1ULL, 1U) ? 1U : 0U;
  (void)queue.try_push(44U);
  (void)queue.try_push(55U);
  observed[10] = queue.try_push_with_timeout(66U, 1ULL, 1U) ? 1U : 0U;
  observed[11] = queue.full() ? 1U : 0U;
  cells[0].sequence = 4ULL;
  observed[12] = queue.full() ? 1U : 0U;
  cells[0].sequence = 5ULL;
  observed[13] = queue.full() ? 1U : 0U;
}

__global__ void
concurrent_ring_queue_kernel(std::uint32_t *push_results,
                             std::uint32_t *pop_results, std::uint32_t *seen,
                             RingQueueState *state,
                             RingQueueCell<std::uint32_t> *cells) {
  const auto item = static_cast<std::uint32_t>(threadIdx.x);
  if (threadIdx.x >= kConcurrentItemCount) {
    return;
  }

  if (threadIdx.x == 0) {
    state->capacity = static_cast<std::uint64_t>(kConcurrentItemCount);
    state->head = 0ULL;
    state->tail = 0ULL;
  }
  cells[threadIdx.x] =
      RingQueueCell<std::uint32_t>{static_cast<std::uint64_t>(threadIdx.x), 0U};
  __syncthreads();

  RingQueue<std::uint32_t> queue{state, cells};
  push_results[threadIdx.x] = queue.try_push(item) ? 1U : 0U;
  __syncthreads();

  std::uint32_t value = 0U;
  if (queue.try_pop(value)) {
    pop_results[threadIdx.x] = 1U;
    if (value < static_cast<std::uint32_t>(kConcurrentItemCount)) {
      atomicAdd(&seen[value], 1U);
    }
  } else {
    pop_results[threadIdx.x] = 0U;
  }
}

} // namespace

TEST(RingQueueHostImageTest, FullImageInitializesReadyToPopCells) {
  const std::array<std::uint32_t, 3> values{7U, 11U, 13U};

  const auto image = RingQueueHostImage<std::uint32_t>::full(
      std::span<const std::uint32_t>{values.data(), values.size()});

  EXPECT_EQ(image.state.capacity, values.size());
  EXPECT_EQ(image.state.head, 0ULL);
  EXPECT_EQ(image.state.tail, values.size());
  ASSERT_EQ(image.cells.size(), values.size());
  for (std::size_t index = 0; index < values.size(); ++index) {
    EXPECT_EQ(image.cells[index].sequence, index + 1);
    EXPECT_EQ(image.cells[index].value, values[index]);
  }
}

TEST(RingQueueHostImageTest, EmptyImageInitializesWrappedEmptyCells) {
  const auto image = RingQueueHostImage<std::uint32_t>::empty(3);

  EXPECT_EQ(image.state.capacity, 3U);
  EXPECT_EQ(image.state.head, 0ULL);
  EXPECT_EQ(image.state.tail, 0ULL);
  ASSERT_EQ(image.cells.size(), 3U);
  for (std::size_t index = 0; index < image.cells.size(); ++index) {
    EXPECT_EQ(image.cells[index].sequence, index);
    EXPECT_EQ(image.cells[index].value, 0U);
  }
}

TEST_F(RingQueueCudaTest, SequentialPushPopWrapAndTimeoutPathsWork) {
  std::uint32_t *observed = nullptr;
  RingQueueState *state = nullptr;
  RingQueueCell<std::uint32_t> *cells = nullptr;
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(
      reinterpret_cast<void **>(&observed), 14 * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&state), sizeof(*state))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&cells),
                        2 * sizeof(RingQueueCell<std::uint32_t>))));

  sequential_ring_queue_kernel<<<1, 1>>>(observed, state, cells);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  const std::array<std::uint32_t, 14> expected{0U,  1U, 1U, 0U, 11U, 1U, 22U,
                                               33U, 0U, 0U, 0U, 1U,  0U, 1U};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    EXPECT_EQ(observed[index], expected[index]) << "observed[" << index << "]";
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(state)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(cells)));
}

TEST_F(RingQueueCudaTest, SingleBlockConcurrentPushPopHasNoDuplicatesOrDrops) {
  std::uint32_t *push_results = nullptr;
  std::uint32_t *pop_results = nullptr;
  std::uint32_t *seen = nullptr;
  RingQueueState *state = nullptr;
  RingQueueCell<std::uint32_t> *cells = nullptr;
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&push_results),
                        kConcurrentItemCount * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&pop_results),
                        kConcurrentItemCount * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&seen),
                        kConcurrentItemCount * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocManaged(reinterpret_cast<void **>(&state), sizeof(*state))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(
      reinterpret_cast<void **>(&cells),
      kConcurrentItemCount * sizeof(RingQueueCell<std::uint32_t>))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(
      push_results, 0, kConcurrentItemCount * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(
      pop_results, 0, kConcurrentItemCount * sizeof(std::uint32_t))));
  ASSERT_TRUE(cuda_succeeded(
      cudaMemset(seen, 0, kConcurrentItemCount * sizeof(std::uint32_t))));

  concurrent_ring_queue_kernel<<<1, kConcurrentItemCount>>>(
      push_results, pop_results, seen, state, cells);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  for (int index = 0; index < kConcurrentItemCount; ++index) {
    EXPECT_EQ(push_results[index], 1U) << "push thread " << index;
    EXPECT_EQ(pop_results[index], 1U) << "pop thread " << index;
    EXPECT_EQ(seen[index], 1U) << "value " << index;
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(push_results)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(pop_results)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(seen)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(state)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(cells)));
}
