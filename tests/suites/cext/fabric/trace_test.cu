/// \file tests/suites/cext/fabric/trace_test.cu
/// \brief Fabric trace state, fact-availability, and event tests.

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cstddef>

#include <xpool/abi.hpp>
#include <xpool/fabric/trace.cuh>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

__global__ void record_trace_states(xpool::fabric::FabricTraceRecord *records) {
  if (threadIdx.x != 0) {
    return;
  }

  const auto submission = xpool::fabric::FfnSubmission{
      .key = {.model_index = 1, .invocation_sequence = 2},
      .layer_ordinal = 3,
      .payload_rows = 8,
      .local_token_count = 4,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
  const auto decode_invocation = xpool::fabric::FfnInvocation{
      .key = submission.key,
      .layer_ordinal = submission.layer_ordinal,
      .payload_rows = submission.payload_rows,
      .input_pe = 0,
      .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
      .result_handoff = submission.result_handoff,
      .dp_padding_mode = submission.dp_padding_mode,
  };
  const auto admission = xpool::fabric::FfnExecutionAdmission{
      .key = submission.key,
      .executor_index = 1,
      .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
  };
  const auto result = xpool::fabric::FfnResult{
      .key = submission.key,
      .contribution = xpool::fabric::FfnResultContribution::Full,
  };

  records[0].begin_atnagent(1, submission, 4);

  records[1].begin_atnagent(2, submission, 4);
  records[1].decode_input_staged();
  records[1].submission_published();
  records[1].admission_observed(admission);
  records[1].result_observed(result);
  records[1].output_prepared();
  records[1].transport_evaluated_published();
  records[1].acknowledgement_published();

  records[2].begin_coordinator(3, decode_invocation, 4);
  records[2].fifo_enqueued(9);
  records[2].scheduled(1);
  records[2].admissions_published();
  records[2].invocations_published();
  records[2].completions_observed();
  records[2].results_published();
  records[2].acknowledgements_observed();
  records[2].scheduler_released();

  records[3].begin_execution(4, decode_invocation, 4, 1);
  records[3].decode_input_pull_started();
  records[3].decode_input_pull_completed();
  records[3].execution_started();
  records[3].execution_completed();
  records[3].completion_published();

  auto prefill_invocation = decode_invocation;
  prefill_invocation.execution_mode = xpool::fabric::FfnExecutionMode::Prefill;
  records[4].begin_execution(5, prefill_invocation, 4, 0);
  records[4].prefill_input_ready_observed();
  records[4].execution_started();
  records[4].execution_completed();
  records[4].completion_published();

  records[5].begin_atnagent(6, submission, 4);
  records[5].submission_published();
  records[5].admission_observed(admission);
  records[5].result_observed(result);
  records[5].output_prepared();
  records[5].transport_evaluated_published();
  records[5].acknowledgement_published();
}

class FabricTraceCudaTest : public ::testing::Test {
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

} // namespace

TEST_F(FabricTraceCudaTest, DistinguishesKindsAndEventEstablishedFacts) {
  constexpr auto record_count = std::size_t{6};
  auto *records = static_cast<xpool::fabric::FabricTraceRecord *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&records, record_count * sizeof(*records))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(records, 0, record_count * sizeof(*records))));

  record_trace_states<<<1, 1, 0, nullptr>>>(records);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(records[0].kind(), xpool::fabric::FabricTraceKind::AtnAgent);
  EXPECT_EQ(records[0].submission_payload_rows(), 8U);
  EXPECT_FALSE(records[0].executor_index().has_value());
  EXPECT_FALSE(records[0].result_contribution().has_value());

  EXPECT_EQ(records[1].kind(), xpool::fabric::FabricTraceKind::AtnAgent);
  EXPECT_EQ(records[1].executor_index(), 1U);
  EXPECT_EQ(records[1].execution_mode(), xpool::fabric::FfnExecutionMode::Decode);
  EXPECT_EQ(records[1].result_contribution(), xpool::fabric::FfnResultContribution::Full);
  EXPECT_TRUE(records[1].recorded(xpool::fabric::AtnAgentTraceEvent::AcknowledgementPublished));

  EXPECT_EQ(records[2].kind(), xpool::fabric::FabricTraceKind::Coordinator);
  EXPECT_EQ(records[2].invocation_payload_rows(), 8U);
  EXPECT_EQ(records[2].input_pe(), 0);
  EXPECT_EQ(records[2].ready_ticket(), 9U);
  EXPECT_TRUE(records[2].recorded(xpool::fabric::CoordinatorTraceEvent::SchedulerReleased));

  EXPECT_EQ(records[3].kind(), xpool::fabric::FabricTraceKind::Execution);
  EXPECT_EQ(records[3].execution_mode(), xpool::fabric::FfnExecutionMode::Decode);
  EXPECT_TRUE(records[3].recorded(xpool::fabric::ExecutionTraceEvent::DecodeInputPullCompleted));
  EXPECT_TRUE(records[3].recorded(xpool::fabric::ExecutionTraceEvent::CompletionPublished));

  EXPECT_EQ(records[4].kind(), xpool::fabric::FabricTraceKind::Execution);
  EXPECT_EQ(records[4].execution_mode(), xpool::fabric::FfnExecutionMode::Prefill);
  EXPECT_TRUE(records[4].recorded(xpool::fabric::ExecutionTraceEvent::PrefillInputReadyObserved));
  EXPECT_GT(records[4].timestamp(xpool::fabric::ExecutionTraceEvent::CompletionPublished), 0U);

  EXPECT_EQ(records[5].kind(), xpool::fabric::FabricTraceKind::AtnAgent);
  EXPECT_FALSE(records[5].recorded(xpool::fabric::AtnAgentTraceEvent::DecodeInputStaged));
  EXPECT_TRUE(records[5].recorded(xpool::fabric::AtnAgentTraceEvent::AcknowledgementPublished));

  EXPECT_TRUE(cuda_succeeded(cudaFree(records)));
}
