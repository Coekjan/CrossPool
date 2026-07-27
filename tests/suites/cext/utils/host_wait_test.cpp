/// \file tests/suites/cext/utils/host_wait_test.cpp
/// \brief Host behavior tests for bounded polling precedence.

#include <gtest/gtest.h>

#include <chrono>

#include <xpool/utils/wait.hpp>

namespace {

using xpool::utils::wait::Result;

TEST(HostWaitTest, ReadinessPrecedesCancellationAndTimeout) {
  const auto expired = std::chrono::steady_clock::now();
  const auto interval = std::chrono::milliseconds{1};

  EXPECT_EQ(xpool::utils::wait::until(
                expired, [] { return true; }, [] { return true; }, interval),
            Result::Ready);
  EXPECT_EQ(xpool::utils::wait::until(
                expired, [] { return false; }, [] { return true; }, interval),
            Result::Cancelled);
  EXPECT_EQ(xpool::utils::wait::until(
                expired, [] { return false; }, [] { return false; }, interval),
            Result::TimedOut);
}

TEST(HostWaitTest, ReadinessOnlyOverloadReturnsItsTwoOutcomes) {
  const auto expired = std::chrono::steady_clock::now();
  const auto interval = std::chrono::milliseconds{1};

  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return true; }, interval),
            Result::Ready);
  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return false; }, interval),
            Result::TimedOut);
}

} // namespace
