#include <gtest/gtest.h>

#include <xpool/ffn.hpp>
#include <xpool/transport/protocol.hpp>

namespace {

xpool::transport::RequestMetadata request(xpool::ffn::DpRowLayout padding) {
  return xpool::transport::RequestMetadata{
      .layer_ordinal = 0,
      .forward_mode = xpool::ffn::ForwardMode::Decode,
      .output_requirement = xpool::ffn::OutputRequirement::PerRankComplete,
      .dp_row_layout = padding,
  };
}

} // namespace

TEST(FfnRequestMetadataTest, ValidatesIntrinsicRequestSemantics) {
  const auto valid = request(xpool::ffn::DpRowLayout::None);
  EXPECT_TRUE(valid.valid());

  auto invalid = valid;
  invalid.forward_mode = static_cast<xpool::ffn::ForwardMode>(99);
  EXPECT_FALSE(invalid.valid());

  invalid = valid;
  invalid.output_requirement = static_cast<xpool::ffn::OutputRequirement>(99);
  EXPECT_FALSE(invalid.valid());

  invalid = valid;
  invalid.dp_row_layout = static_cast<xpool::ffn::DpRowLayout>(99);
  EXPECT_FALSE(invalid.valid());
}
