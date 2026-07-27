#include <gtest/gtest.h>

#include <c10/util/Exception.h>

#include <xpool/abi.hpp>
#include <xpool/transport/layout.hpp>
#include <xpool/transport/protocol.hpp>

namespace {

xpool::transport::TransportArenaLayout layout(std::size_t dp_size) {
  return xpool::transport::TransportArenaLayout::create(
      0, 0, 0, 1, 0, dp_size, 8, 4,
      xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32});
}

xpool::transport::FfnRequestMetadata request(xpool::abi::DpPaddingMode::Type padding) {
  return xpool::transport::FfnRequestMetadata{
      .layer_ordinal = 0,
      .forward_mode = xpool::abi::XPoolForwardMode::Decode,
      .result_handoff = xpool::abi::FfnResultHandoff::ReplicatedFull,
      .dp_padding_mode = padding,
  };
}

} // namespace

TEST(FfnRequestMetadataTest, ValidatesIntrinsicRequestSemantics) {
  const auto valid = request(xpool::abi::DpPaddingMode::None);
  EXPECT_TRUE(valid.valid());

  auto invalid = valid;
  invalid.forward_mode = 99;
  EXPECT_FALSE(invalid.valid());

  invalid = valid;
  invalid.result_handoff = 99;
  EXPECT_FALSE(invalid.valid());

  invalid = valid;
  invalid.dp_padding_mode = 99;
  EXPECT_FALSE(invalid.valid());
}

TEST(FfnRequestMetadataTest, ValidatesDpOneContext) {
  const auto value = request(xpool::abi::DpPaddingMode::None);
  const auto arena_layout = layout(1);

  EXPECT_NO_THROW(value.validate(arena_layout, 8, false));
  EXPECT_THROW(value.validate(arena_layout, 0, false), c10::Error);
  EXPECT_THROW(value.validate(arena_layout, 9, false), c10::Error);
  EXPECT_THROW(value.validate(arena_layout, 8, true), c10::Error);

  auto padded = value;
  padded.dp_padding_mode = xpool::abi::DpPaddingMode::MaxLen;
  EXPECT_THROW(padded.validate(arena_layout, 8, false), c10::Error);
}

TEST(FfnRequestMetadataTest, ValidatesDpWorldContext) {
  const auto arena_layout = layout(2);
  const auto max_len = request(xpool::abi::DpPaddingMode::MaxLen);
  const auto sum_len = request(xpool::abi::DpPaddingMode::SumLen);

  EXPECT_NO_THROW(max_len.validate(arena_layout, 8, true));
  EXPECT_NO_THROW(sum_len.validate(arena_layout, 8, true));
  EXPECT_THROW(max_len.validate(arena_layout, 8, false), c10::Error);
  EXPECT_THROW(request(xpool::abi::DpPaddingMode::None).validate(arena_layout, 8, true), c10::Error);
}
