#include <cstdint>

#include <gtest/gtest.h>

#include <xpool/runtime.hpp>

TEST(NativeRuntimeRoleTest, PreservesClosedRoleValuesAndNames) {
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::RuntimeRole::Daemon), 1U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::RuntimeRole::Instance), 2U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::RuntimeRole::AtnAgent), 3U);
  EXPECT_EQ(static_cast<std::uint32_t>(xpool::RuntimeRole::FfnAgent), 4U);

  EXPECT_TRUE(xpool::is_valid(xpool::RuntimeRole::Daemon));
  EXPECT_FALSE(xpool::is_valid(static_cast<xpool::RuntimeRole>(0)));
  EXPECT_EQ(xpool::name(xpool::RuntimeRole::Daemon), "daemon");
  EXPECT_EQ(xpool::name(xpool::RuntimeRole::Instance), "instance");
  EXPECT_EQ(xpool::name(xpool::RuntimeRole::AtnAgent), "atnagent");
  EXPECT_EQ(xpool::name(xpool::RuntimeRole::FfnAgent), "ffnagent");
}
