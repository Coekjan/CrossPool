#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <thread>

#include <xpool/abi.hpp>
#include <xpool/atomic.cuh>
#include <xpool/transport.hpp>
#include <xpool/transport/protocol.cuh>

namespace {

constexpr std::int64_t kSlotCount = 3;

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class TransportShutdownCudaTest : public ::testing::Test {
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

bool wait_stream(cudaStream_t stream) {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < deadline) {
    const cudaError_t error = cudaStreamQuery(stream);
    if (error == cudaSuccess) {
      return true;
    }
    if (error != cudaErrorNotReady) {
      return false;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  return false;
}

__global__ void publish_all_slots(xpool::transport::TransportArena arena) {
  if (threadIdx.x >= kSlotCount) {
    return;
  }
  std::uint32_t slot = 0U;
  if (!arena.free_queue().try_pop(slot)) {
    xpool::utils::device::trap();
  }
  auto &request = arena.request(slot);
  request.header.abi_version = xpool::abi::kAbiVersion;
  request.state.slot_id = slot;
  xpool::atomic::store_release(request.state.status,
                               xpool::abi::DescriptorStatus::kPublished);
  if (!arena.used_queue().try_push(slot)) {
    xpool::utils::device::trap();
  }
}

__global__ void abandon_one_slot(xpool::transport::TransportArena arena) {
  std::uint32_t slot = 0U;
  if (threadIdx.x == 0 && !arena.free_queue().try_pop(slot)) {
    xpool::utils::device::trap();
  }
}

} // namespace

TEST_F(TransportShutdownCudaTest,
       DrainsEveryUsedSlotBeforeResidentKernelExits) {
  const xpool::transport::TransportArenaLayout layout{kSlotCount, 1, 2, 2, 1};
  auto arena = xpool::transport::TransportArena::create(0, layout);
  cudaStream_t producer_stream = nullptr;
  cudaStream_t resident_stream = nullptr;
  xpool::abi::FfnResultDescriptor *results = nullptr;
  ASSERT_TRUE(cuda_succeeded(
      cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking)));
  ASSERT_TRUE(cuda_succeeded(
      cudaStreamCreateWithFlags(&resident_stream, cudaStreamNonBlocking)));
  ASSERT_TRUE(cuda_succeeded(
      cudaMallocHost(reinterpret_cast<void **>(&results),
                     kSlotCount * sizeof(xpool::abi::FfnResultDescriptor))));

  publish_all_slots<<<1, kSlotCount, 0, producer_stream>>>(arena);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(producer_stream)));
  arena.request_shutdown(producer_stream);
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(producer_stream)));

  xpool::transport::launch_devagent_transport_kernel(arena, resident_stream);
  bool results_ready = false;
  const auto result_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds{5};
  while (std::chrono::steady_clock::now() < result_deadline) {
    ASSERT_TRUE(cuda_succeeded(
        cudaMemcpyAsync(results, arena.base + layout.result_base_offset,
                        kSlotCount * sizeof(xpool::abi::FfnResultDescriptor),
                        cudaMemcpyDeviceToHost, producer_stream)));
    ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(producer_stream)));
    results_ready = true;
    for (std::int64_t slot = 0; slot < kSlotCount; ++slot) {
      results_ready &=
          results[slot].state.status == xpool::abi::DescriptorStatus::kFailed;
    }
    if (results_ready) {
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{1});
  }
  ASSERT_TRUE(results_ready)
      << "transport resident kernel did not drain every used slot";

  for (std::int64_t slot = 0; slot < kSlotCount; ++slot) {
    EXPECT_EQ(results[slot].error_code,
              xpool::abi::FfnResultErrorCode::kShutdown);
  }

  const bool resident_finished = wait_stream(resident_stream);
  ASSERT_TRUE(resident_finished)
      << "transport resident kernel did not exit after draining used slots";

  EXPECT_TRUE(cuda_succeeded(cudaFreeHost(results)));
  EXPECT_TRUE(cuda_succeeded(cudaStreamDestroy(producer_stream)));
  EXPECT_TRUE(cuda_succeeded(cudaStreamDestroy(resident_stream)));
  arena.destroy();
}

TEST_F(TransportShutdownCudaTest, ExitsWhenProducerAbandonsClaimedSlot) {
  const xpool::transport::TransportArenaLayout layout{1, 1, 2, 2, 1};
  auto arena = xpool::transport::TransportArena::create(0, layout);
  cudaStream_t producer_stream = nullptr;
  cudaStream_t resident_stream = nullptr;
  ASSERT_TRUE(cuda_succeeded(
      cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking)));
  ASSERT_TRUE(cuda_succeeded(
      cudaStreamCreateWithFlags(&resident_stream, cudaStreamNonBlocking)));

  abandon_one_slot<<<1, 1, 0, producer_stream>>>(arena);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(producer_stream)));
  arena.request_shutdown(producer_stream);
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(producer_stream)));

  xpool::transport::launch_devagent_transport_kernel(arena, resident_stream);
  ASSERT_TRUE(wait_stream(resident_stream))
      << "transport resident kernel waited for an abandoned producer slot";

  EXPECT_TRUE(cuda_succeeded(cudaStreamDestroy(producer_stream)));
  EXPECT_TRUE(cuda_succeeded(cudaStreamDestroy(resident_stream)));
  arena.destroy();
}
