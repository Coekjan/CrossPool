#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/fabric/protocol.hpp>
#include <xpool/fabric/runtime.hpp>

namespace {

xpool::fabric::FabricJoinMetadata valid_metadata() {
  return xpool::fabric::FabricJoinMetadata{
      .uid = xpool::fabric::FabricUid::decode(
          std::string(sizeof(nvshmemx_uniqueid_t) * 2, 'a')),
      .pe = 1,
      .atnagent_count = 2,
      .ffnagent_count = 2,
      .executor_count = 2,
      .scheduler_policy = xpool::fabric::FfnSchedulerPolicy::fifo(),
      .models =
          {
              xpool::fabric::FabricModelMetadata{
                  .max_decode_rows = 4,
                  .max_prefill_rows = 16,
                  .dtype = xpool::abi::TensorDType{xpool::abi::TensorDType::Bf16},
                  .hidden_size = 2048,
                  .atn_tp_size = 2,
                  .atn_dp_size = 1,
                  .layers =
                      {
                          xpool::fabric::FabricLayerMetadata{
                              .layer_id = 0,
                              .kind = xpool::fabric::FfnLayerKind::Dense,
                          },
                          xpool::fabric::FabricLayerMetadata{
                              .layer_id = 1,
                              .kind = xpool::fabric::FfnLayerKind::Sparse,
                          },
                      },
              },
          },
  };
}

} // namespace

TEST(FabricJoinMetadataTest, DerivesTopologyAndComparesExactInputs) {
  const auto metadata = valid_metadata();

  EXPECT_NO_THROW(metadata.validate());
  EXPECT_EQ(metadata.pe_count(), 4);
  EXPECT_EQ(metadata.coordinator_pe(), 2);
  EXPECT_EQ(metadata, valid_metadata());

  auto changed_rows = metadata;
  ++changed_rows.models[0].max_decode_rows;
  EXPECT_NE(changed_rows, metadata);
}

TEST(FabricLayerKindTest, ValidatesAndConstructsSemanticValues) {
  EXPECT_TRUE(xpool::fabric::FfnLayerKind::is_valid(xpool::fabric::FfnLayerKind::Dense));
  EXPECT_TRUE(xpool::fabric::FfnLayerKind::is_valid(std::uint64_t{2}));
  EXPECT_FALSE(xpool::fabric::FfnLayerKind::is_valid(std::int64_t{-1}));
  EXPECT_FALSE(xpool::fabric::FfnLayerKind::is_valid(std::numeric_limits<std::uint64_t>::max()));
  EXPECT_TRUE((std::is_convertible_v<xpool::fabric::FfnLayerKind::Type, xpool::fabric::FfnLayerKind>));
  EXPECT_FALSE((std::is_convertible_v<std::uint32_t, xpool::fabric::FfnLayerKind>));
  EXPECT_FALSE((std::is_constructible_v<xpool::fabric::FfnLayerKind, bool>));
  EXPECT_EQ(xpool::fabric::FfnLayerKind{std::uint32_t{2}}.value(), xpool::fabric::FfnLayerKind::Sparse);
  EXPECT_DEATH((void)xpool::fabric::FfnLayerKind{99}, "");
}

TEST(FfnSchedulerPolicyTest, ConstructsTypedVariants) {
  EXPECT_EQ(xpool::fabric::FfnSchedulerPolicy::fifo().type(),
            xpool::fabric::FfnSchedulerPolicy::Fifo);
  EXPECT_EQ(xpool::fabric::FfnSchedulerPolicy::random(7).type(),
            xpool::fabric::FfnSchedulerPolicy::Random);
  EXPECT_EQ(xpool::fabric::FfnSchedulerPolicy::fifo(),
            xpool::fabric::FfnSchedulerPolicy::fifo());
  EXPECT_NE(xpool::fabric::FfnSchedulerPolicy::random(7),
            xpool::fabric::FfnSchedulerPolicy::random(8));
}

TEST(FabricJoinMetadataTest, RejectsInvalidSemanticInputs) {
  auto metadata = valid_metadata();
  metadata.pe = metadata.pe_count();
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.executor_count = 0;
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models.clear();
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models[0].max_decode_rows = 0;
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models[0].hidden_size = 0;
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models[0].atn_tp_size = 1;
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models[0].layers.clear();
  EXPECT_THROW(metadata.validate(), c10::Error);

  metadata = valid_metadata();
  metadata.models[0].layers[1].layer_id = metadata.models[0].layers[0].layer_id;
  EXPECT_THROW(metadata.validate(), c10::Error);
}

TEST(FabricUidTest, EncodesDecodesAndOrdersOpaqueBytes) {
  const auto lower = xpool::fabric::FabricUid::decode(
      std::string(sizeof(nvshmemx_uniqueid_t) * 2, '0'));
  const auto higher = xpool::fabric::FabricUid::decode(
      std::string(sizeof(nvshmemx_uniqueid_t) * 2, '1'));

  EXPECT_EQ(lower.encode(), std::string(sizeof(nvshmemx_uniqueid_t) * 2, '0'));
  EXPECT_EQ(lower, xpool::fabric::FabricUid::decode(lower.encode()));
  EXPECT_LT(lower, higher);
  EXPECT_THROW(xpool::fabric::FabricUid::decode("ab"), c10::Error);
  EXPECT_THROW(
      xpool::fabric::FabricUid::decode(
          std::string(sizeof(nvshmemx_uniqueid_t) * 2, 'A')),
      c10::Error);
}
