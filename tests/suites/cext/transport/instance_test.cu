/// \file tests/suites/cext/transport/instance_test.cu
/// \brief Instance-side Transport request-kernel tests.

#include <ATen/ATen.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <xpool/abi.hpp>
#include <xpool/atomic.cuh>
#include <xpool/debug/options.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/instance.hpp>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class InstanceTransportCudaTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto device_count = int{0};
    const auto error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device is not available: " << cudaGetErrorString(error);
    }
    ASSERT_TRUE(cuda_succeeded(cudaSetDevice(0)));

    auto debug_options = xpool::debug::DebugOptions{};
    debug_options.loopback = {true, xpool::debug::LoopbackSite::AtnAgent};
    debug_options.transport_observer = {true, 2};
    xpool::debug::configure(debug_options, 0);

    const auto layout = xpool::transport::TransportArenaLayout::create(
        0, 0, 0, 1, 0, 1, 2, 2,
        xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
    arena_ = xpool::transport::TransportArena::create(0, layout);
    ASSERT_TRUE(cuda_succeeded(
        cudaStreamCreateWithFlags(&request_stream_, cudaStreamNonBlocking)));

    const auto options = at::TensorOptions{}.device(at::kCUDA).dtype(at::kFloat);
    hidden_states_ =
        at::tensor({1.0F, 3.0F, 2.0F, 4.0F}, options).reshape({2, 2});
    output_ = at::empty_like(hidden_states_);
  }

  void TearDown() override {
    if (request_stream_ != nullptr) {
      EXPECT_TRUE(cuda_succeeded(cudaStreamDestroy(request_stream_)));
    }
    if (arena_) {
      EXPECT_NO_THROW(arena_.destroy());
    }
  }

  void enqueue_request() {
    const auto global_num_tokens_gpu = std::optional<at::Tensor>{};
    const auto request_metadata = xpool::transport::FfnRequestMetadata{
        .layer_ordinal = 0,
        .forward_mode = xpool::abi::XPoolForwardMode::Decode,
        .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
        .dp_padding_mode = xpool::abi::DpPaddingMode::None,
    };
    xpool::transport::launch_request_kernel(xpool::transport::TransportRequest{
        request_stream_, arena_.view(), hidden_states_, global_num_tokens_gpu,
        output_, request_metadata});
    ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  }

  cudaStream_t request_stream_ = nullptr;
  xpool::transport::TransportArena arena_;
  at::Tensor hidden_states_;
  at::Tensor output_;
};

__global__ void publish_shutdown(xpool::transport::TransportArenaView arena) {
  if (threadIdx.x == 0) {
    xpool::atomic::store_release(arena.state().shutdown, std::uint32_t{1});
  }
}

__global__ void publish_generation_failure(
    xpool::transport::TransportArenaView arena) {
  if (threadIdx.x == 0) {
    arena.publish_generation_failure(xpool::abi::FfnResultCode{
        xpool::abi::FfnResultCode::ProtocolMismatch});
  }
}

} // namespace

TEST_F(InstanceTransportCudaTest, RejectsBeforeStagingAfterShutdown) {
  publish_shutdown<<<1, 1, 0, request_stream_>>>(arena_.view());
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));

  ASSERT_NO_FATAL_FAILURE(enqueue_request());
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(request_stream_)));

  const auto output_cpu = output_.cpu();
  const auto *values = output_cpu.const_data_ptr<float>();
  for (auto index = std::size_t{0}; index < 4; ++index) {
    EXPECT_TRUE(std::isnan(values[index]));
  }
  EXPECT_EQ(arena_.read_trace().records.size(), 0U);
}

TEST_F(InstanceTransportCudaTest,
       RejectsBeforeStagingAfterCanonicalGenerationFailure) {
  publish_generation_failure<<<1, 1, 0, request_stream_>>>(arena_.view());
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));

  ASSERT_NO_FATAL_FAILURE(enqueue_request());
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(request_stream_)));

  const auto output_cpu = output_.cpu();
  const auto *values = output_cpu.const_data_ptr<float>();
  for (auto index = std::size_t{0}; index < 4; ++index) {
    EXPECT_TRUE(std::isnan(values[index]));
  }
  EXPECT_EQ(arena_.mailbox_status(),
            xpool::transport::MailboxStatus::Dormant);
  EXPECT_EQ(arena_.read_generation_failure(),
            xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(arena_.read_trace().records.size(), 0U);
}
