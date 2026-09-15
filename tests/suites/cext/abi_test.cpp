#include <cstdint>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/ffn.hpp>

TEST(NativeAbiContractTest, PreservesSharedValues) {
  EXPECT_EQ(xpool::abi::kVersion, 81U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ForwardMode::Prefill), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ForwardMode::Decode), 2U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ForwardMode::Idle), 4U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::OutputRequirement::PerRankComplete), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::OutputRequirement::GroupSumComplete), 2U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::DpRowLayout::None), 0U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::DpRowLayout::UniformByRank), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::DpRowLayout::PackedByRank), 2U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ResultCode::Ok), 0U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ResultCode::Shutdown), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ResultCode::ProtocolMismatch), 2U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::ResultCode::Timeout), 3U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::LayerKind::Dense), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::ffn::LayerKind::Moe), 2U);
}

TEST(NativeAbiContractTest, ValidatesClosedSets) {
  EXPECT_TRUE(xpool::ffn::is_valid(xpool::ffn::ForwardMode::Decode));
  EXPECT_TRUE(xpool::ffn::is_valid(xpool::ffn::OutputRequirement::PerRankComplete));
  EXPECT_TRUE(xpool::ffn::is_valid(xpool::ffn::DpRowLayout::None));
  EXPECT_TRUE(xpool::ffn::is_valid(xpool::ffn::ResultCode::ProtocolMismatch));
  EXPECT_TRUE(xpool::ffn::is_valid(xpool::ffn::LayerKind::Moe));

  EXPECT_FALSE(xpool::ffn::is_valid(static_cast<xpool::ffn::ForwardMode>(99)));
  EXPECT_FALSE(xpool::ffn::is_valid(static_cast<xpool::ffn::OutputRequirement>(99)));
  EXPECT_FALSE(xpool::ffn::is_valid(static_cast<xpool::ffn::DpRowLayout>(99)));
  EXPECT_FALSE(xpool::ffn::is_valid(static_cast<xpool::ffn::ResultCode>(99)));
  EXPECT_FALSE(xpool::ffn::is_valid(static_cast<xpool::ffn::LayerKind>(99)));
}

TEST(NativeAbiContractTest, AdmitsSupportedPayloadElementTypes) {
  EXPECT_TRUE(xpool::ffn::is_supported_payload_dtype(c10::ScalarType::Half));
  EXPECT_TRUE(xpool::ffn::is_supported_payload_dtype(c10::ScalarType::BFloat16));
  EXPECT_FALSE(xpool::ffn::is_supported_payload_dtype(c10::ScalarType::Float));
}
