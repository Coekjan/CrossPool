#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <limits>

#include <xpool/abi.hpp>
#include <xpool/fabric/protocol.hpp>

namespace {

constexpr auto kKey = xpool::fabric::FfnInvocationKey{
    .model_index = 1,
    .invocation_sequence = 7,
};

template <xpool::fabric::FabricRecord Record>
xpool::fabric::FabricPublication<Record> published(Record record) {
  return xpool::fabric::FabricPublication<Record>{
      .record = record,
      .sequence = record.key.invocation_sequence,
  };
}

xpool::fabric::FfnSubmission submission() {
  return {
      .key = kKey,
      .layer_ordinal = 2,
      .payload_rows = 8,
      .local_token_count = 4,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
}

xpool::fabric::FfnInvocation invocation() {
  return {
      .key = kKey,
      .layer_ordinal = 2,
      .payload_rows = 8,
      .input_pe = 0,
      .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = xpool::abi::DpPaddingMode::None,
  };
}

xpool::fabric::FabricArenaLayout arena_layout() {
  return xpool::fabric::FabricArenaLayout::create(2, 1, 1, 2, 6, 512, 512);
}

xpool::fabric::FabricModelLayout model_layout() {
  return {
      .dtype = xpool::abi::TensorDType::Fp32,
      .hidden_size = 4,
      .atn_tp_size = 2,
      .atn_dp_size = 1,
      .layer_begin = 3,
      .layer_count = 3,
      .decode_payload_offset = 256,
      .decode_payload_capacity_bytes = 256,
      .prefill_payload_capacity_bytes = 512,
  };
}

} // namespace

TEST(FabricProtocolTest, ValidatesEveryTypedPublication) {
  EXPECT_EQ(published(submission()).validate(), xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(invocation()).validate(), xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(xpool::fabric::FfnExecutionAdmission{
                          .key = kKey,
                          .executor_index = 1,
                          .execution_mode = xpool::fabric::FfnExecutionMode::Decode,
                      })
                .validate(),
            xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(xpool::fabric::FfnInputReady{.key = kKey}).validate(),
            xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(xpool::fabric::FfnAgentCompletion{.key = kKey}).validate(),
            xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(xpool::fabric::FfnResult{
                          .key = kKey,
                          .contribution = xpool::fabric::FfnResultContribution::Full,
                      })
                .validate(),
            xpool::abi::FfnResultCode::Ok);
  EXPECT_EQ(published(xpool::fabric::FfnResultAcknowledgement{.key = kKey}).validate(),
            xpool::abi::FfnResultCode::Ok);
}

TEST(FabricProtocolTest, AcceptsRankLocalIdleAndPrefillFacts) {
  auto idle = submission();
  idle.local_token_count = 0;
  idle.forward_mode = xpool::abi::XPoolForwardMode::Idle;
  EXPECT_EQ(idle.validate(), xpool::abi::FfnResultCode::Ok);

  auto prefill = invocation();
  prefill.execution_mode = xpool::fabric::FfnExecutionMode::Prefill;
  prefill.dp_padding_mode = xpool::abi::DpPaddingMode::SumLen;
  EXPECT_EQ(prefill.validate(), xpool::abi::FfnResultCode::Ok);
}

TEST(FabricProtocolTest, ValidatesInvocationAgainstModelPayloadGeometry) {
  const auto layout = arena_layout();
  const auto model = model_layout();
  auto request = invocation();

  EXPECT_EQ(request.validate(layout, model), xpool::abi::FfnResultCode::Ok);

  request.input_pe = 1;
  EXPECT_EQ(request.validate(layout, model),
            xpool::abi::FfnResultCode::ProtocolMismatch);

  request = invocation();
  request.payload_rows = 17;
  EXPECT_EQ(request.validate(layout, model),
            xpool::abi::FfnResultCode::ProtocolMismatch);

  request.execution_mode = xpool::fabric::FfnExecutionMode::Prefill;
  EXPECT_EQ(request.validate(layout, model), xpool::abi::FfnResultCode::Ok);

  request.payload_rows = 33;
  EXPECT_EQ(request.validate(layout, model),
            xpool::abi::FfnResultCode::ProtocolMismatch);

  request = invocation();
  request.layer_ordinal = model.layer_count;
  EXPECT_EQ(request.validate(layout, model),
            xpool::abi::FfnResultCode::ProtocolMismatch);

  request = invocation();
  request.payload_rows = std::numeric_limits<std::size_t>::max();
  EXPECT_EQ(request.validate(layout, model),
            xpool::abi::FfnResultCode::ProtocolMismatch);
}

TEST(FabricProtocolTest, RejectsMalformedRecordsAndPublicationIdentity) {
  auto invalid_submission = submission();
  invalid_submission.local_token_count = invalid_submission.payload_rows + 1;
  EXPECT_EQ(invalid_submission.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);

  invalid_submission = submission();
  invalid_submission.forward_mode = 3;
  EXPECT_EQ(invalid_submission.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);

  auto invalid_invocation = invocation();
  invalid_invocation.input_pe = -1;
  EXPECT_EQ(invalid_invocation.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);

  invalid_invocation = invocation();
  invalid_invocation.result_handoff = 99;
  EXPECT_EQ(invalid_invocation.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);

  auto publication = published(submission());
  ++publication.sequence;
  EXPECT_EQ(publication.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);

  publication = published(submission());
  publication.sequence = 0;
  EXPECT_EQ(publication.validate(), xpool::abi::FfnResultCode::ProtocolMismatch);
}
