#include <gtest/gtest.h>

#include <xpool/debug/options.hpp>

TEST(DebugOptionsTest, DefaultsToDisabledWithoutConfiguration) {
  const xpool::debug::Options defaults{};
  EXPECT_FALSE(defaults.transport_observer.enable);
  EXPECT_EQ(defaults.transport_observer.record_capacity, 0U);
  EXPECT_FALSE(defaults.fabric_observer.enable);
  EXPECT_EQ(defaults.fabric_observer.record_capacity, 0U);
  EXPECT_FALSE(defaults.graph_observer.enable);
  EXPECT_FALSE(defaults.ffn_routing_observer.enable);
  EXPECT_EQ(defaults.ffn_routing_observer.record_capacity, 0U);
  EXPECT_EQ(xpool::debug::options(), defaults);
}
