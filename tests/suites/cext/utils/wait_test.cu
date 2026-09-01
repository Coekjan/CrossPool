#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include <xpool/macros.hpp>
#include <xpool/utils/wait.cuh>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class DeviceWaitTest : public ::testing::Test {
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

struct WaitObservations {
  xpool::utils::wait::Status until[4];
  xpool::utils::wait::Status poll[5];
};

XPOOL_KERNEL_FN void observe_wait_precedence(WaitObservations *observations) {
  const auto expired = xpool::utils::wait::Deadline::after(0);
  observations->until[0] = xpool::utils::wait::until(
      expired, [] { return true; }, [] { return true; });
  observations->until[1] = xpool::utils::wait::until(
      expired, [] { return false; }, [] { return true; });
  observations->until[2] = xpool::utils::wait::until(
      expired, [] { return false; }, [] { return false; });
  observations->until[3] = xpool::utils::wait::until(
      xpool::utils::wait::Deadline::never(), [] { return false; }, [] { return true; });
  observations->poll[0] = xpool::utils::wait::poll_once(
      expired, [] { return true; }, [] { return true; });
  observations->poll[1] = xpool::utils::wait::poll_once(
      expired, [] { return false; }, [] { return true; });
  observations->poll[2] = xpool::utils::wait::poll_once(
      expired, [] { return false; }, [] { return false; });
  observations->poll[3] = xpool::utils::wait::poll_once(
      xpool::utils::wait::Deadline::never(), [] { return false; }, [] { return false; });
  observations->poll[4] = xpool::utils::wait::poll_once(
      xpool::utils::wait::Deadline::from_start(xpool::utils::time::now(), 0), [] { return false; },
      [] { return false; });
}

} // namespace

TEST_F(DeviceWaitTest, ReadinessPrecedesCancellationAndTimeout) {
  auto observations = static_cast<WaitObservations *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observations, sizeof(WaitObservations))));

  observe_wait_precedence<<<1, 1, 0, nullptr>>>(observations);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  const std::array expected_until{
      xpool::utils::wait::Status::Ready,
      xpool::utils::wait::Status::Cancelled,
      xpool::utils::wait::Status::TimedOut,
      xpool::utils::wait::Status::Cancelled,
  };
  for (auto index = std::size_t{0}; index < expected_until.size(); ++index) {
    EXPECT_EQ(observations->until[index], expected_until[index]);
  }
  const std::array expected_poll{
      xpool::utils::wait::Status::Ready,   xpool::utils::wait::Status::Cancelled, xpool::utils::wait::Status::TimedOut,
      xpool::utils::wait::Status::Pending, xpool::utils::wait::Status::TimedOut,
  };
  for (auto index = std::size_t{0}; index < expected_poll.size(); ++index) {
    EXPECT_EQ(observations->poll[index], expected_poll[index]);
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(observations)));
}
