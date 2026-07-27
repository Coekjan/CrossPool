#include <cstdint>
#include <limits>
#include <type_traits>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <xpool/runtime.hpp>

TEST(NativeRuntimeRoleTest, ParsesStableRoleValues) {
  EXPECT_FALSE(xpool::RuntimeRole::is_valid(0));
  EXPECT_FALSE(xpool::RuntimeRole::is_valid(std::int64_t{-1}));
  EXPECT_FALSE(xpool::RuntimeRole::is_valid(std::numeric_limits<std::uint64_t>::max()));
  EXPECT_TRUE(xpool::RuntimeRole::is_valid(xpool::RuntimeRole::Daemon));
  EXPECT_THROW(xpool::RuntimeRole::parse(0), c10::Error);
  EXPECT_EQ(xpool::RuntimeRole::parse(1).name(), "daemon");
  EXPECT_EQ(xpool::RuntimeRole::parse(2).name(), "instance");
  EXPECT_EQ(xpool::RuntimeRole::parse(3).name(), "atnagent");
  EXPECT_EQ(xpool::RuntimeRole::parse(4).name(), "ffnagent");
}

TEST(NativeRuntimeRoleTest, ConstrainsConstructionAndFailStopsInvalidValues) {
  EXPECT_TRUE((std::is_convertible_v<xpool::RuntimeRole::Type, xpool::RuntimeRole>));
  EXPECT_FALSE((std::is_convertible_v<std::uint32_t, xpool::RuntimeRole>));
  EXPECT_FALSE((std::is_constructible_v<xpool::RuntimeRole, bool>));
  EXPECT_EQ(xpool::RuntimeRole{std::uint32_t{2}}.value(), xpool::RuntimeRole::Instance);
  EXPECT_DEATH((void)xpool::RuntimeRole{99}, "");
}
