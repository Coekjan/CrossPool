#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>

#include <xpool/macros.hpp>
#include <xpool/utils/cooperative.cuh>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class CooperativeCopyTest : public ::testing::TestWithParam<int> {
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

XPOOL_KERNEL_FN void copy_kernel(void *destination, const void *source, std::size_t count) {
  const auto group = cooperative_groups::this_thread_block();
  xpool::utils::cooperative::copy(group, cuda::std::span{static_cast<std::uint8_t *>(destination), count},
                                  cuda::std::span{static_cast<const std::uint8_t *>(source), count});
}

XPOOL_KERNEL_FN void transform_kernel(std::uint32_t *destination, const std::int64_t *source, std::size_t count) {
  const auto group = cooperative_groups::this_thread_block();
  xpool::utils::cooperative::transform(
      group, cuda::std::span{destination, count}, cuda::std::span{source, count},
      [] XPOOL_DEVICE_FN(std::int64_t value) { return static_cast<std::uint32_t>(value * 2); });
}

void expect_copy(int thread_count, std::size_t destination_offset, std::size_t source_offset, std::size_t byte_count) {
  constexpr auto allocation_size = std::size_t{256};
  constexpr auto destination_sentinel = std::uint8_t{0xa5};
  auto *source = static_cast<std::uint8_t *>(nullptr);
  auto *destination = static_cast<std::uint8_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&source, allocation_size)));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&destination, allocation_size)));

  for (auto index = std::size_t{0}; index < allocation_size; ++index) {
    source[index] = static_cast<std::uint8_t>((index * 17 + 3) % 251);
    destination[index] = destination_sentinel;
  }

  copy_kernel<<<1, thread_count, 0, nullptr>>>(destination + destination_offset, source + source_offset, byte_count);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  for (auto index = std::size_t{0}; index < allocation_size; ++index) {
    const auto expected_source = static_cast<std::uint8_t>((index * 17 + 3) % 251);
    EXPECT_EQ(source[index], expected_source) << "source byte " << index;
    const auto copied = index >= destination_offset && index < destination_offset + byte_count;
    const auto expected_destination =
        copied ? source[source_offset + index - destination_offset] : destination_sentinel;
    EXPECT_EQ(destination[index], expected_destination) << "destination byte " << index;
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(destination)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(source)));
}

} // namespace

TEST_P(CooperativeCopyTest, CopiesAlignedRange) { expect_copy(GetParam(), 32, 16, 128); }

TEST_P(CooperativeCopyTest, CopiesUnalignedRange) { expect_copy(GetParam(), 7, 3, 128); }

TEST_P(CooperativeCopyTest, CopiesNonAlignedTail) { expect_copy(GetParam(), 32, 16, 127); }

TEST_P(CooperativeCopyTest, AcceptsZeroLengthWithNullPointers) {
  copy_kernel<<<1, GetParam(), 0, nullptr>>>(nullptr, nullptr, 0);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));
}

TEST_P(CooperativeCopyTest, TransformsEqualSizeRanges) {
  constexpr auto count = std::size_t{7};
  auto *source = static_cast<std::int64_t *>(nullptr);
  auto *destination = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&source, count * sizeof(*source))));
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&destination, count * sizeof(*destination))));
  for (auto index = std::size_t{0}; index < count; ++index) {
    source[index] = static_cast<std::int64_t>(index + 1);
    destination[index] = 0;
  }

  transform_kernel<<<1, GetParam(), 0, nullptr>>>(destination, source, count);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));
  for (auto index = std::size_t{0}; index < count; ++index) {
    EXPECT_EQ(destination[index], 2 * (index + 1));
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(destination)));
  EXPECT_TRUE(cuda_succeeded(cudaFree(source)));
}

INSTANTIATE_TEST_SUITE_P(ThreadGroups, CooperativeCopyTest, ::testing::Values(32, 256));
