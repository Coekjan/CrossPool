#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>

#include <xpool/macros.hpp>
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

struct MailboxObservation {
  xpool::transport::MailboxStatus initial_status;
  xpool::transport::MailboxStatus opened_status;
  bool staging_acquired;
  xpool::transport::MailboxStatus published_status;
  xpool::transport::MailboxStatus evaluated_status;
  xpool::ffn::ResultCode evaluated_result;
  xpool::transport::MailboxStatus acknowledged_status;
  xpool::ffn::ResultCode acknowledged_result;
  std::size_t acknowledged_payload_rows;
  bool idle_closed;
  xpool::transport::MailboxStatus closed_status;
};

struct CloseObservation {
  bool staging_acquired;
  xpool::transport::MailboxStatus status;
  xpool::ffn::ResultCode result_code;
};

struct FailureCloseObservation {
  bool staging_acquired;
  xpool::transport::MailboxStatus status;
  xpool::ffn::ResultCode result_code;
  xpool::ffn::ResultCode generation_failure;
};

struct ShutdownCloseObservation {
  bool staging_acquired;
  xpool::transport::MailboxStatus status;
  xpool::ffn::ResultCode result_code;
  bool shutdown_requested;
};

XPOOL_KERNEL_FN void exercise_mailbox(xpool::transport::ArenaView arena, MailboxObservation *observed) {
  auto &mailbox = arena.mailbox();
  observed->initial_status = mailbox.observe();
  mailbox.open();
  observed->opened_status = mailbox.observe();
  observed->staging_acquired = mailbox.try_begin_staging();
  mailbox.payload_rows = 2;
  mailbox.request = {
      .layer_ordinal = 3,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_row_layout = xpool::ffn::DpRowLayout::None,
  };
  mailbox.publish_request();
  observed->published_status = mailbox.observe();
  mailbox.publish_result(xpool::ffn::ResultCode::Ok);
  observed->evaluated_status = mailbox.observe();
  observed->evaluated_result = mailbox.result_code;
  mailbox.acknowledge();
  observed->acknowledged_status = mailbox.observe();
  observed->acknowledged_result = mailbox.result_code;
  observed->acknowledged_payload_rows = mailbox.payload_rows;
  observed->idle_closed = mailbox.try_close_idle();
  observed->closed_status = mailbox.observe();
}

XPOOL_KERNEL_FN void publish_same_generation_failure(xpool::transport::ArenaView arena,
                                                     xpool::ffn::ResultCode *observed) {
  const auto failure = xpool::ffn::ResultCode::Timeout;
  arena.publish_generation_failure(failure);
  arena.publish_generation_failure(failure);
  *observed = arena.generation_failure();
}

XPOOL_KERNEL_FN void close_staging_for_shutdown(xpool::transport::ArenaView arena, CloseObservation *observed) {
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed->staging_acquired = mailbox.try_begin_staging();
  mailbox.result_code = xpool::ffn::ResultCode::Shutdown;
  mailbox.close_staging();
  observed->status = mailbox.observe();
  observed->result_code = mailbox.result_code;
}

XPOOL_KERNEL_FN void close_evaluated_for_generation_failure(xpool::transport::ArenaView arena,
                                                            FailureCloseObservation *observed) {
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed->staging_acquired = mailbox.try_begin_staging();
  mailbox.payload_rows = 1;
  mailbox.request = {
      .layer_ordinal = 0,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_row_layout = xpool::ffn::DpRowLayout::None,
  };
  mailbox.publish_request();
  const auto failure = xpool::ffn::ResultCode::ProtocolMismatch;
  arena.publish_generation_failure(failure);
  mailbox.publish_result(failure);
  mailbox.close_evaluated();
  observed->status = mailbox.observe();
  observed->result_code = mailbox.result_code;
  observed->generation_failure = arena.generation_failure();
}

XPOOL_KERNEL_FN void close_successful_evaluation_for_shutdown(xpool::transport::ArenaView arena,
                                                              ShutdownCloseObservation *observed) {
  auto &mailbox = arena.mailbox();
  mailbox.open();
  observed->staging_acquired = mailbox.try_begin_staging();
  mailbox.payload_rows = 1;
  mailbox.request = {
      .layer_ordinal = 0,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_row_layout = xpool::ffn::DpRowLayout::None,
  };
  mailbox.publish_request();
  mailbox.publish_result(xpool::ffn::ResultCode::Ok);
  arena.publish_shutdown();
  mailbox.close_evaluated();
  observed->status = mailbox.observe();
  observed->result_code = mailbox.result_code;
  observed->shutdown_requested = arena.shutdown_requested();
}

} // namespace

TEST_F(TransportArenaViewTest, ExecutesMailboxLifecycleAndResetsBody) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  auto arena = xpool::transport::Arena::create(0, layout);
  auto observed = static_cast<MailboxObservation *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  exercise_mailbox<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(observed->initial_status, xpool::transport::MailboxStatus::Dormant);
  EXPECT_EQ(observed->opened_status, xpool::transport::MailboxStatus::Idle);
  EXPECT_TRUE(observed->staging_acquired);
  EXPECT_EQ(observed->published_status, xpool::transport::MailboxStatus::Published);
  EXPECT_EQ(observed->evaluated_status, xpool::transport::MailboxStatus::Evaluated);
  EXPECT_EQ(observed->evaluated_result, xpool::ffn::ResultCode::Ok);
  EXPECT_EQ(observed->acknowledged_status, xpool::transport::MailboxStatus::Idle);
  EXPECT_EQ(observed->acknowledged_result, xpool::ffn::ResultCode::ProtocolMismatch);
  EXPECT_EQ(observed->acknowledged_payload_rows, 0);
  EXPECT_TRUE(observed->idle_closed);
  EXPECT_EQ(observed->closed_status, xpool::transport::MailboxStatus::Closed);

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, KeepsRepeatedCanonicalTimeout) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  auto arena = xpool::transport::Arena::create(0, layout);
  auto observed = static_cast<xpool::ffn::ResultCode *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  publish_same_generation_failure<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(*observed, xpool::ffn::ResultCode::Timeout);
  EXPECT_EQ(arena.read_generation_failure(), xpool::ffn::ResultCode::Timeout);

  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, LetsInstanceCloseStagingForShutdown) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  auto arena = xpool::transport::Arena::create(0, layout);
  auto observed = static_cast<CloseObservation *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  close_staging_for_shutdown<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_TRUE(observed->staging_acquired);
  EXPECT_EQ(observed->status, xpool::transport::MailboxStatus::Closed);
  EXPECT_EQ(observed->result_code, xpool::ffn::ResultCode::Shutdown);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, LetsInstanceCloseEvaluatedCanonicalFailure) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  auto arena = xpool::transport::Arena::create(0, layout);
  auto observed = static_cast<FailureCloseObservation *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  close_evaluated_for_generation_failure<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_TRUE(observed->staging_acquired);
  EXPECT_EQ(observed->status, xpool::transport::MailboxStatus::Closed);
  EXPECT_EQ(observed->result_code, xpool::ffn::ResultCode::ProtocolMismatch);
  EXPECT_EQ(observed->generation_failure, xpool::ffn::ResultCode::ProtocolMismatch);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}

TEST_F(TransportArenaViewTest, PreservesSuccessfulResultWhenInstanceClosesForShutdown) {
  const auto layout = xpool::transport::ArenaLayout::create(0, 0, 0, 1, 0, 1, 4, 2, c10::ScalarType::Half);
  auto arena = xpool::transport::Arena::create(0, layout);
  auto observed = static_cast<ShutdownCloseObservation *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&observed, sizeof(*observed))));

  close_successful_evaluation_for_shutdown<<<1, 1, 0, nullptr>>>(arena.view(), observed);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_TRUE(observed->staging_acquired);
  EXPECT_EQ(observed->status, xpool::transport::MailboxStatus::Closed);
  EXPECT_EQ(observed->result_code, xpool::ffn::ResultCode::Ok);
  EXPECT_TRUE(observed->shutdown_requested);
  EXPECT_TRUE(cuda_succeeded(cudaFree(observed)));
  EXPECT_NO_THROW(arena.destroy());
}
