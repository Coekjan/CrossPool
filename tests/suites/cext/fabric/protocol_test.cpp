#include <cstdint>

#include <gtest/gtest.h>

#include <xpool/fabric/protocol.hpp>
#include <xpool/ffn.hpp>

namespace {

constexpr auto kKey = xpool::fabric::InvocationKey{
    .instance_index = 1,
    .invocation_sequence = 7,
};
constexpr auto kLease = std::uint64_t{11};

xpool::fabric::Submission submission() {
  return {
      .key = kKey,
      .layer_ordinal = 2,
      .payload_rows = 8,
      .dp_row_layout = xpool::ffn::DpRowLayout::PackedByRank,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_rank_payload_rows = 4,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
  };
}

} // namespace

TEST(FabricProtocolTest, ValidatesBusinessInvocationAndAllNineRecordFamilies) {
  const auto invocation = xpool::fabric::Invocation{
      .key = kKey,
      .layer_ordinal = 2,
      .payload_rows = 8,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
  };
  EXPECT_EQ(invocation.validate(), xpool::ffn::ResultCode::Ok);
  EXPECT_EQ(submission().validate(), xpool::ffn::ResultCode::Ok);
  EXPECT_EQ(
      (xpool::fabric::Admission{.key = kKey, .executor_lane_index = 1, .executor_lease_sequence = kLease}).validate(),
      xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::LaneExecution{.key = kKey,
                                          .executor_lease_sequence = kLease,
                                          .layer_ordinal = 2,
                                          .payload_rows = 8,
                                          .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete})
                .validate(),
            xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::InputReady{.key = kKey, .executor_lease_sequence = kLease}).validate(),
            xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::RoutingMetadataReady{.key = kKey, .executor_lease_sequence = kLease}).validate(),
            xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::PartialReady{.key = kKey, .executor_lease_sequence = kLease}).validate(),
            xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::FfnAgentCompletion{.key = kKey, .executor_lease_sequence = kLease}).validate(),
            xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::OutputCommit{.key = kKey}).validate(), xpool::ffn::ResultCode::Ok);
  EXPECT_EQ((xpool::fabric::OutputAcknowledgement{.key = kKey}).validate(), xpool::ffn::ResultCode::Ok);
}

TEST(FabricProtocolTest, UsesInvocationSequenceOnlyForInstanceScopedRecords) {
  EXPECT_EQ(submission().publication_sequence(), kKey.invocation_sequence);
  EXPECT_EQ((xpool::fabric::Admission{.key = kKey, .executor_lane_index = 1, .executor_lease_sequence = kLease})
                .publication_sequence(),
            kKey.invocation_sequence);
  EXPECT_EQ((xpool::fabric::OutputCommit{.key = kKey}).publication_sequence(), kKey.invocation_sequence);
  EXPECT_EQ((xpool::fabric::OutputAcknowledgement{.key = kKey}).publication_sequence(), kKey.invocation_sequence);

  EXPECT_EQ((xpool::fabric::LaneExecution{.key = kKey,
                                          .executor_lease_sequence = kLease,
                                          .layer_ordinal = 2,
                                          .payload_rows = 8,
                                          .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete})
                .publication_sequence(),
            kLease);
  EXPECT_EQ((xpool::fabric::PartialReady{.key = kKey, .executor_lease_sequence = kLease}).publication_sequence(),
            kLease);
}

TEST(FabricProtocolTest, RejectsMalformedIntrinsicFields) {
  auto invalid_submission = submission();
  invalid_submission.dp_rank_payload_rows = invalid_submission.payload_rows + 1;
  EXPECT_EQ(invalid_submission.validate(), xpool::ffn::ResultCode::ProtocolMismatch);
  invalid_submission = submission();
  invalid_submission.forward_mode = static_cast<xpool::ffn::ForwardMode>(3);
  EXPECT_EQ(invalid_submission.validate(), xpool::ffn::ResultCode::ProtocolMismatch);

  auto invalid_invocation = xpool::fabric::Invocation{
      .key = kKey,
      .layer_ordinal = 2,
      .payload_rows = 8,
      .output_requirement = static_cast<xpool::ffn::OutputRequirement>(99),
  };
  EXPECT_EQ(invalid_invocation.validate(), xpool::ffn::ResultCode::ProtocolMismatch);

  auto admission = xpool::fabric::Admission{
      .key = kKey,
      .executor_lane_index = 1,
      .executor_lease_sequence = 0,
  };
  EXPECT_EQ(admission.validate(), xpool::ffn::ResultCode::ProtocolMismatch);

  auto lane = xpool::fabric::LaneExecution{
      .key = kKey,
      .executor_lease_sequence = kLease,
      .layer_ordinal = 2,
      .payload_rows = 0,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
  };
  EXPECT_EQ(lane.validate(), xpool::ffn::ResultCode::ProtocolMismatch);

  auto completion = xpool::fabric::FfnAgentCompletion{
      .key = kKey,
      .executor_lease_sequence = 0,
  };
  EXPECT_EQ(completion.validate(), xpool::ffn::ResultCode::ProtocolMismatch);

  auto commit = xpool::fabric::OutputCommit{
      .key = {.instance_index = 1, .invocation_sequence = 0},
  };
  EXPECT_EQ(commit.validate(), xpool::ffn::ResultCode::ProtocolMismatch);
}
