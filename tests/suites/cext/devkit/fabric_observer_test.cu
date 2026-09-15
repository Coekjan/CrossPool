#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <xpool/devkit/fabric_observer.cuh>
#include <xpool/ffn.hpp>
#include <xpool/macros.hpp>

namespace {

::testing::AssertionResult cuda_succeeded(cudaError_t error) {
  if (error == cudaSuccess) {
    return ::testing::AssertionSuccess();
  }
  return ::testing::AssertionFailure() << cudaGetErrorString(error);
}

XPOOL_KERNEL_FN void record_trace_states(xpool::devkit::fabric_observer::Record *records) {
  constexpr auto key = xpool::fabric::InvocationKey{.instance_index = 1, .invocation_sequence = 2};
  constexpr auto lease = std::uint64_t{7};
  const auto submission = xpool::fabric::Submission{
      .key = key,
      .layer_ordinal = 3,
      .payload_rows = 8,
      .dp_row_layout = xpool::ffn::DpRowLayout::PackedByRank,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_rank_payload_rows = 4,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
  };
  const auto invocation = xpool::fabric::Invocation{
      .key = key,
      .layer_ordinal = 3,
      .payload_rows = 8,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
  };
  const auto admission = xpool::fabric::Admission{
      .key = key,
      .executor_lane_index = 1,
      .executor_lease_sequence = lease,
  };
  const auto execution = xpool::fabric::LaneExecution{
      .key = key,
      .executor_lease_sequence = lease,
      .layer_ordinal = 3,
      .payload_rows = 8,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
  };

  records[0].begin_atnagent(1, submission);
  records[0].submission_published();
  records[0].admission_observed(admission);
  records[0].input_ready_published();
  records[0].output_commit_observed(xpool::fabric::OutputCommit{.key = key});
  records[0].output_acknowledgement_published();

  records[1].begin_coordinator(2, invocation);
  records[1].enqueued(9);
  records[1].scheduled(1, lease);
  records[1].admission_published();
  records[1].lane_execution_published();
  records[1].ffnagent_completions_observed();
  records[1].output_commit_published();
  records[1].output_acknowledgements_observed();
  records[1].lane_released();

  records[2].begin_ffnagent(3, execution, 1, 16, xpool::fabric::DeliveryVariant::DirectPartial);
  records[2].input_ready_observed();
  records[2].compute_started();
  records[2].compute_completed();
  records[2].partial_ready_published();
  records[2].completion_published();

  records[3].begin_ffnagent(4, execution, 1, 16, xpool::fabric::DeliveryVariant::SingleComplete);
  records[3].input_ready_observed();
  records[3].routing_metadata_observed();
  records[3].compute_started();
  records[3].compute_completed();
  records[3].peer_partials_ready_observed();
  records[3].completion_published();
}

class FabricObserverCudaTest : public ::testing::Test {
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

TEST_F(FabricObserverCudaTest, RetainsOnlyRoleEstablishedEvidence) {
  constexpr auto record_count = std::size_t{4};
  auto *records = static_cast<xpool::devkit::fabric_observer::Record *>(nullptr);
  ASSERT_TRUE(cuda_succeeded(cudaMallocManaged(&records, record_count * sizeof(*records))));
  ASSERT_TRUE(cuda_succeeded(cudaMemset(records, 0, record_count * sizeof(*records))));

  record_trace_states<<<1, 1, 0, nullptr>>>(records);
  ASSERT_TRUE(cuda_succeeded(cudaGetLastError()));
  ASSERT_TRUE(cuda_succeeded(cudaDeviceSynchronize()));

  EXPECT_EQ(records[0].kind(), xpool::devkit::fabric_observer::RecordKind::AtnAgent);
  EXPECT_EQ(records[0].dp_rank_payload_rows(), 4U);
  EXPECT_EQ(records[0].forward_mode(), xpool::ffn::ForwardMode::Decode);
  EXPECT_EQ(records[0].dp_row_layout(), xpool::ffn::DpRowLayout::PackedByRank);
  EXPECT_EQ(records[0].executor_lane_index(), 1U);
  EXPECT_EQ(records[0].executor_lease_sequence(), 7U);
  EXPECT_FALSE(records[0].ready_ticket().has_value());
  EXPECT_TRUE(records[0].recorded(xpool::hooks::FabricAtnAgentProtocolEvent::Kind::OutputAcknowledgementPublished));

  EXPECT_EQ(records[1].kind(), xpool::devkit::fabric_observer::RecordKind::Coordinator);
  EXPECT_EQ(records[1].ready_ticket(), 9U);
  EXPECT_FALSE(records[1].delivery().has_value());
  EXPECT_TRUE(records[1].recorded(xpool::hooks::FabricCoordinatorProtocolEvent::Kind::LaneReleased));

  EXPECT_EQ(records[2].kind(), xpool::devkit::fabric_observer::RecordKind::FfnAgent);
  EXPECT_EQ(records[2].payload_row_capacity(), 16U);
  EXPECT_EQ(records[2].delivery(), xpool::fabric::DeliveryVariant::DirectPartial);
  EXPECT_TRUE(records[2].recorded(xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PartialReadyPublished));
  EXPECT_FALSE(records[2].recorded(xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PeerPartialsReadyObserved));

  EXPECT_EQ(records[3].kind(), xpool::devkit::fabric_observer::RecordKind::FfnAgent);
  EXPECT_TRUE(records[3].recorded(xpool::hooks::FabricFfnAgentProtocolEvent::Kind::RoutingMetadataObserved));
  EXPECT_TRUE(records[3].recorded(xpool::hooks::FabricFfnAgentProtocolEvent::Kind::PeerPartialsReadyObserved));
  EXPECT_GT(records[3].timestamp(xpool::hooks::FabricFfnAgentProtocolEvent::Kind::CompletionPublished), 0U);

  EXPECT_TRUE(cuda_succeeded(cudaFree(records)));
}
