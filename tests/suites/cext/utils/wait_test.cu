/// \file tests/suites/cext/utils/wait_test.cu
/// \brief Device behavior tests for bounded polling precedence.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstdint>

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

__global__ void observe_wait_precedence(xpool::utils::wait::Result *results) {
  if (threadIdx.x != 0) {
    return;
  }
  const auto expired = xpool::utils::wait::Deadline::after(0);
  results[0] = xpool::utils::wait::until(expired, [] { return true; }, [] { return true; });
  results[1] = xpool::utils::wait::until(expired, [] { return false; }, [] { return true; });
  results[2] = xpool::utils::wait::until(expired, [] { return false; }, [] { return false; });
  results[3] =
      xpool::utils::wait::until(xpool::utils::wait::Deadline::never(), [] { return false; }, [] { return true; });
}

} // namespace

TEST_F(DeviceWaitTest, ReadinessPrecedesCancellationAndTimeout) {
  auto results = static_cast<xpool::utils::wait::Result *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&results, 4 * sizeof(xpool::utils::wait::Result))));

  observe_wait_precedence<<<1, 1, 0, nullptr>>>(results);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  const std::array expected{
      xpool::utils::wait::Result::Ready,
      xpool::utils::wait::Result::Cancelled,
      xpool::utils::wait::Result::TimedOut,
      xpool::utils::wait::Result::Cancelled,
  };
  for (auto index = std::size_t{0}; index < expected.size(); ++index) {
    EXPECT_EQ(results[index], expected[index]);
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(results)));
}
