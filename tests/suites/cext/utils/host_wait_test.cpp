#include <chrono>

#include <gtest/gtest.h>

#include <xpool/utils/wait.hpp>

namespace {

using xpool::utils::wait::Status;

TEST(HostWaitTest, ReadinessPrecedesCancellationAndTimeout) {
  const auto expired = std::chrono::steady_clock::now();
  const auto interval = std::chrono::milliseconds{1};

  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return true; }, [] { return true; }, interval), Status::Ready);
  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return false; }, [] { return true; }, interval), Status::Cancelled);
  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return false; }, [] { return false; }, interval), Status::TimedOut);
}

TEST(HostWaitTest, ReadinessOnlyOverloadReturnsItsTwoOutcomes) {
  const auto expired = std::chrono::steady_clock::now();
  const auto interval = std::chrono::milliseconds{1};

  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return true; }, interval), Status::Ready);
  EXPECT_EQ(xpool::utils::wait::until(expired, [] { return false; }, interval), Status::TimedOut);
}

} // namespace
