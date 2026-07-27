#include <cstdint>
#include <limits>
#include <type_traits>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>

TEST(NativeAbiContractTest, PreservesSharedValues) {
  EXPECT_EQ(xpool::abi::kAbiVersion, 54U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::XPoolForwardMode::Extend), 1U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::XPoolForwardMode::Decode), 2U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::XPoolForwardMode::Idle), 4U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultHandoff::ReplicatedFull), 1U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultHandoff::ReduceScatterInput), 2U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::DpPaddingMode::None), 0U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::DpPaddingMode::MaxLen), 1U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::DpPaddingMode::SumLen), 2U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::TensorDType::Bf16), 1U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::TensorDType::Fp16), 2U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::TensorDType::Fp32), 3U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultCode::Ok), 0U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultCode::Shutdown), 1U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultCode::ProtocolMismatch), 2U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultCode::Timeout), 3U);
  EXPECT_EQ(static_cast<unsigned>(xpool::abi::FfnResultCode::NotImplemented), 4U);
  EXPECT_EQ(xpool::abi::TensorDType{xpool::abi::TensorDType::Bf16}.bytes(), 2U);
  EXPECT_EQ(xpool::abi::TensorDType{xpool::abi::TensorDType::Fp16}.bytes(), 2U);
  EXPECT_EQ(xpool::abi::TensorDType{xpool::abi::TensorDType::Fp32}.bytes(), 4U);
}

TEST(NativeAbiContractTest, ValidatesIntegralRepresentations) {
  EXPECT_TRUE(xpool::abi::XPoolForwardMode::is_valid(xpool::abi::XPoolForwardMode::Decode));
  EXPECT_TRUE(xpool::abi::FfnResultHandoff::is_valid(std::uint32_t{1}));
  EXPECT_TRUE(xpool::abi::DpPaddingMode::is_valid(std::int64_t{0}));
  EXPECT_TRUE(xpool::abi::TensorDType::is_valid(std::uint64_t{3}));
  EXPECT_TRUE(xpool::abi::FfnResultCode::is_valid(std::int32_t{2}));

  EXPECT_FALSE(xpool::abi::XPoolForwardMode::is_valid(std::int64_t{-1}));
  EXPECT_FALSE(xpool::abi::FfnResultHandoff::is_valid(std::numeric_limits<std::uint64_t>::max()));
  EXPECT_FALSE(xpool::abi::DpPaddingMode::is_valid(std::uint32_t{99}));
  EXPECT_FALSE(xpool::abi::TensorDType::is_valid(std::int32_t{-1}));
  EXPECT_FALSE(xpool::abi::FfnResultCode::is_valid(std::uint64_t{99}));
}

TEST(NativeAbiContractTest, ConstrainsImplicitConstructionByInputDomain) {
  EXPECT_TRUE((std::is_convertible_v<xpool::abi::TensorDType::Type, xpool::abi::TensorDType>));
  EXPECT_FALSE((std::is_convertible_v<std::uint32_t, xpool::abi::TensorDType>));
  EXPECT_FALSE((std::is_constructible_v<xpool::abi::TensorDType, bool>));
  EXPECT_FALSE((std::is_constructible_v<xpool::abi::TensorDType, xpool::abi::XPoolForwardMode::Type>));

  const auto dtype = xpool::abi::TensorDType{std::uint64_t{3}};
  EXPECT_EQ(dtype.value(), xpool::abi::TensorDType::Fp32);
}

TEST(NativeAbiContractTest, FailStopsInvalidSemanticConstruction) {
  EXPECT_DEATH((void)xpool::abi::XPoolForwardMode{99}, "");
  EXPECT_DEATH((void)xpool::abi::FfnResultHandoff{99}, "");
  EXPECT_DEATH((void)xpool::abi::DpPaddingMode{99}, "");
  EXPECT_DEATH((void)xpool::abi::TensorDType{99}, "");
  EXPECT_DEATH((void)xpool::abi::FfnResultCode{99}, "");
}
