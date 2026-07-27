/// \file tests/suites/cext/transport/view_test.cu
/// \brief Device behavior tests for the Transport mailbox and failure cache.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/transport/arena.cuh>
#include <xpool/transport/protocol.cuh>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

class TransportArenaViewTest : public ::testing::Test {
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

__global__ void exercise_mailbox(xpool::transport::TransportArenaView arena,
                                 std::uint32_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }

  auto &mailbox = arena.mailbox();
  observed[0] = static_cast<std::uint32_t>(mailbox.observe());
  mailbox.open();
  observed[1] = static_cast<std::uint32_t>(mailbox.observe());
  observed[2] = mailbox.try_begin_staging() ? 1U : 0U;
  mailbox.payload_rows = 2;
  mailbox.request = {
      .layer_ordinal = 3,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
  mailbox.publish_request();
  observed[3] = static_cast<std::uint32_t>(mailbox.observe());
  mailbox.publish_result(xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok});
  observed[4] = static_cast<std::uint32_t>(mailbox.observe());
  observed[5] = mailbox.result_code;
  mailbox.acknowledge();
  observed[6] = static_cast<std::uint32_t>(mailbox.observe());
  observed[7] = mailbox.result_code;
  observed[8] = static_cast<std::uint32_t>(mailbox.payload_rows);
  observed[9] = mailbox.try_close_idle() ? 1U : 0U;
  observed[10] = static_cast<std::uint32_t>(mailbox.observe());
}

__global__ void publish_same_generation_failure(
    xpool::transport::TransportArenaView arena, std::uint32_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  const auto failure = xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::ProtocolMismatch};
  arena.publish_generation_failure(failure);
  arena.publish_generation_failure(failure);
  observed[0] = arena.generation_failure().value();
}

__global__ void close_staging_for_shutdown(
    xpool::transport::TransportArenaView arena, std::uint32_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed[0] = mailbox.try_begin_staging() ? 1U : 0U;
  mailbox.result_code = xpool::abi::FfnResultCode::Shutdown;
  mailbox.close_staging();
  observed[1] = static_cast<std::uint32_t>(mailbox.observe());
  observed[2] = mailbox.result_code;
}

__global__ void close_evaluated_for_generation_failure(
    xpool::transport::TransportArenaView arena, std::uint32_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed[0] = mailbox.try_begin_staging() ? 1U : 0U;
  mailbox.payload_rows = 1;
  mailbox.request = {
      .layer_ordinal = 0,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
  mailbox.publish_request();
  const auto failure = xpool::abi::FfnResultCode{
      xpool::abi::FfnResultCode::ProtocolMismatch};
  arena.publish_generation_failure(failure);
  mailbox.publish_result(failure);
  mailbox.close_evaluated();
  observed[1] = static_cast<std::uint32_t>(mailbox.observe());
  observed[2] = mailbox.result_code;
  observed[3] = arena.generation_failure().value();
}

__global__ void close_successful_evaluation_for_shutdown(
    xpool::transport::TransportArenaView arena, std::uint32_t *observed) {
  if (threadIdx.x != 0) {
    return;
  }
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed[0] = mailbox.try_begin_staging() ? 1U : 0U;
  mailbox.payload_rows = 1;
  mailbox.request = {
      .layer_ordinal = 0,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
  mailbox.publish_request();
  mailbox.publish_result(
      xpool::abi::FfnResultCode{xpool::abi::FfnResultCode::Ok});
  arena.publish_shutdown();
  mailbox.close_evaluated();
  observed[1] = static_cast<std::uint32_t>(mailbox.observe());
  observed[2] = mailbox.result_code;
  observed[3] = arena.shutdown_requested() ? 1U : 0U;
}

} // namespace

TEST_F(TransportArenaViewTest, ExecutesMailboxLifecycleAndResetsBody) {
  const auto layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 4, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
  auto arena = xpool::transport::TransportArena::create(0, layout);
  auto observed = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, 11 * sizeof(*observed))));

  exercise_mailbox<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  const std::array expected{
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Dormant),
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Idle),
      1U,
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Published),
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Evaluated),
      static_cast<std::uint32_t>(xpool::abi::FfnResultCode::Ok),
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Idle),
      static_cast<std::uint32_t>(xpool::abi::FfnResultCode::ProtocolMismatch),
      0U,
      1U,
      static_cast<std::uint32_t>(xpool::transport::MailboxStatus::Closed),
  };
  for (auto index = std::size_t{0}; index < expected.size(); ++index) {
    EXPECT_EQ(observed[index], expected[index]) << "observed[" << index << "]";
  }

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, KeepsRepeatedCanonicalGenerationFailure) {
  const auto layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 4, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
  auto arena = xpool::transport::TransportArena::create(0, layout);
  auto observed = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  publish_same_generation_failure<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(*observed, xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(arena.read_generation_failure(), xpool::abi::FfnResultCode::ProtocolMismatch);

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, LetsInstanceCloseStagingForShutdown) {
  const auto layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 4, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
  auto arena = xpool::transport::TransportArena::create(0, layout);
  auto observed = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(
      cuda_succeeded(cudaMallocManaged(&observed, 3 * sizeof(*observed))));

  close_staging_for_shutdown<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(observed[0], 1U);
  EXPECT_EQ(observed[1],
            static_cast<std::uint32_t>(
                xpool::transport::MailboxStatus::Closed));
  EXPECT_EQ(observed[2], xpool::abi::FfnResultCode::Shutdown);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest,
       LetsInstanceCloseEvaluatedCanonicalFailure) {
  const auto layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 4, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
  auto arena = xpool::transport::TransportArena::create(0, layout);
  auto observed = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(
      cuda_succeeded(cudaMallocManaged(&observed, 4 * sizeof(*observed))));

  close_evaluated_for_generation_failure<<<1, 1, 0, nullptr>>>(arena.view(),
                                                                observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(observed[0], 1U);
  EXPECT_EQ(observed[1],
            static_cast<std::uint32_t>(
                xpool::transport::MailboxStatus::Closed));
  EXPECT_EQ(observed[2], xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_EQ(observed[3], xpool::abi::FfnResultCode::ProtocolMismatch);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest,
       PreservesSuccessfulResultWhenInstanceClosesForShutdown) {
  const auto layout = xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, 1, 4, 2,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
  auto arena = xpool::transport::TransportArena::create(0, layout);
  auto observed = static_cast<std::uint32_t *>(nullptr);
  ASSERT_TRUE(
      cuda_succeeded(cudaMallocManaged(&observed, 4 * sizeof(*observed))));

  close_successful_evaluation_for_shutdown<<<1, 1, 0, nullptr>>>(arena.view(),
                                                                  observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(observed[0], 1U);
  EXPECT_EQ(observed[1],
            static_cast<std::uint32_t>(
                xpool::transport::MailboxStatus::Closed));
  EXPECT_EQ(observed[2], xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(observed[3], 1U);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}
