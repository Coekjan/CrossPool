#include <ATen/ATen.h>
#include <cuda/atomic>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <xpool/debug/options.hpp>
#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>
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

    auto debug_options = xpool::debug::Options{};
    debug_options.transport_observer = {true, 2};
    xpool::debug::configure(debug_options, 0);

    const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 2, 2, c10::ScalarType::Half);
    arena_ = xpool::transport::Arena::create(0, layout);
    ASSERT_TRUE(cuda_succeeded(cudaStreamCreateWithFlags(&request_stream_, cudaStreamNonBlocking)));

    const auto options = at::TensorOptions{}.device(at::kCUDA).dtype(at::kHalf);
    hidden_states_ = at::tensor({1.0F, 3.0F, 2.0F, 4.0F}).to(options).reshape({2, 2});
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
    const auto dp_rank_payload_rows = std::optional<at::Tensor>{};
    const auto request_metadata = xpool::transport::RequestMetadata{
        .layer_ordinal = 0,
        .forward_mode = xpool::ffn::ForwardMode::Decode,
        .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
        .dp_row_layout = xpool::ffn::DpRowLayout::None,
    };
    xpool::transport::launch_request_kernel(xpool::transport::Request{request_stream_, arena_.view(), hidden_states_,
                                                                      dp_rank_payload_rows, output_, request_metadata});
    ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  }

  cudaStream_t request_stream_ = nullptr;
  xpool::transport::Arena arena_;
  at::Tensor hidden_states_;
  at::Tensor output_;
};

XPOOL_KERNEL_FN void publish_shutdown(xpool::transport::ArenaView arena) {
  cuda::atomic_ref{arena.state().shutdown}.store(std::uint32_t{1}, cuda::memory_order_release);
}

XPOOL_KERNEL_FN void publish_generation_failure(xpool::transport::ArenaView arena) {
  arena.publish_generation_failure(xpool::ffn::ResultCode::ProtocolMismatch);
}

} // namespace

TEST_F(InstanceTransportCudaTest, RejectsBeforeStagingAfterShutdown) {
  publish_shutdown<<<1, 1, 0, request_stream_>>>(arena_.view());
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));

  ASSERT_NO_FATAL_FAILURE(enqueue_request());
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(request_stream_)));

  const auto output_cpu = output_.to(at::kFloat).cpu();
  const auto *values = output_cpu.const_data_ptr<float>();
  for (auto index = std::size_t{0}; index < 4; ++index) {
    EXPECT_TRUE(std::isnan(values[index]));
  }
}

TEST_F(InstanceTransportCudaTest, RejectsBeforeStagingAfterCanonicalGenerationFailure) {
  publish_generation_failure<<<1, 1, 0, request_stream_>>>(arena_.view());
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));

  ASSERT_NO_FATAL_FAILURE(enqueue_request());
  ASSERT_TRUE(cuda_succeeded(cudaStreamSynchronize(request_stream_)));

  const auto output_cpu = output_.to(at::kFloat).cpu();
  const auto *values = output_cpu.const_data_ptr<float>();
  for (auto index = std::size_t{0}; index < 4; ++index) {
    EXPECT_TRUE(std::isnan(values[index]));
  }
  EXPECT_EQ(arena_.mailbox_status(), xpool::transport::MailboxStatus::Dormant);
  EXPECT_EQ(arena_.read_generation_failure(), xpool::ffn::ResultCode::ProtocolMismatch);
}
