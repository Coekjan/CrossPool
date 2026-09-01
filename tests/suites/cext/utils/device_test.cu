#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <atomic>
#include <thread>
#include <utility>

#include <xpool/utils/device.hpp>

namespace {

void CUDART_CB wait_for_release(void *state) {
  const auto release = static_cast<std::atomic<bool> *>(state);
  while (!release->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }
}

class OwnedCudaStreamTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto device_count = int{0};
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(error);
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
  }
};

TEST_F(OwnedCudaStreamTest, DefaultOwnerIsEmpty) {
  xpool::utils::device::OwnedCudaStream stream;

  EXPECT_FALSE(static_cast<bool>(stream));
  EXPECT_EQ(stream.get(), nullptr);
  EXPECT_TRUE(stream.query());
  EXPECT_NO_THROW(stream.destroy());
}

TEST_F(OwnedCudaStreamTest, QueryDistinguishesPendingAndCompletedWork) {
  auto stream = xpool::utils::device::OwnedCudaStream::create();
  auto release = std::atomic<bool>{false};

  EXPECT_TRUE(static_cast<bool>(stream));
  ASSERT_EQ(cudaLaunchHostFunc(stream.get(), wait_for_release, &release), cudaSuccess);
  EXPECT_FALSE(stream.query());

  release.store(true, std::memory_order_release);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  EXPECT_TRUE(stream.query());
  EXPECT_NO_THROW(stream.destroy());
  EXPECT_FALSE(static_cast<bool>(stream));
  EXPECT_TRUE(stream.query());
  EXPECT_NO_THROW(stream.destroy());
}

TEST_F(OwnedCudaStreamTest, MoveLeavesSourceEmptyAndTransfersOwnership) {
  auto source = xpool::utils::device::OwnedCudaStream::create();
  xpool::utils::device::OwnedCudaStream destination{std::move(source)};

  EXPECT_FALSE(static_cast<bool>(source));
  EXPECT_TRUE(static_cast<bool>(destination));
  EXPECT_TRUE(source.query());
  EXPECT_TRUE(destination.query());
  EXPECT_NO_THROW(destination.destroy());
}

TEST_F(OwnedCudaStreamTest, WritesDeviceValueInStreamOrder) {
  auto stream = xpool::utils::device::OwnedCudaStream::create();
  auto *value = static_cast<std::uint32_t *>(nullptr);
  ASSERT_EQ(cudaMalloc(&value, sizeof(*value)), cudaSuccess);

  stream.write_value(value, 17);
  ASSERT_EQ(cudaStreamSynchronize(stream.get()), cudaSuccess);
  auto observed = std::uint32_t{0};
  ASSERT_EQ(cudaMemcpy(&observed, value, sizeof(observed), cudaMemcpyDeviceToHost), cudaSuccess);
  EXPECT_EQ(observed, 17U);

  ASSERT_EQ(cudaFree(value), cudaSuccess);
  stream.destroy();
}

TEST_F(OwnedCudaStreamTest, CleanupWriteRejectsEmptyResourcesWithoutThrowing) {
  auto stream = xpool::utils::device::OwnedCudaStream{};
  EXPECT_FALSE(stream.try_write_value(nullptr, 1));
}

} // namespace
