#include <cstdint>
#include <optional>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <xpool/debug/options.hpp>
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

TEST(NativeRuntimeStateTest, RejectsInvalidIdentityBeforeInitialization) {
  auto &runtime = xpool::RuntimeState::singleton();
  EXPECT_THROW(runtime.initialize(static_cast<xpool::RuntimeRole>(0), c10::DeviceIndex{0}), c10::Error);
  EXPECT_THROW(runtime.role(), c10::Error);
  EXPECT_THROW(runtime.initialize(xpool::RuntimeRole::Instance, c10::DeviceIndex{-1}), c10::Error);
  EXPECT_THROW(runtime.initialize(xpool::RuntimeRole::Daemon, c10::DeviceIndex{0}), c10::Error);
  EXPECT_NO_THROW(runtime.initialize(xpool::RuntimeRole::Daemon, std::nullopt));
  EXPECT_NO_THROW(runtime.initialize(xpool::RuntimeRole::Daemon, std::nullopt));
  EXPECT_EQ(runtime.role(), xpool::RuntimeRole::Daemon);
}

TEST(NativeDebugOptionsTest, ValidatesCapacityBeforeFreezingConfiguration) {
  auto options = xpool::debug::Options{};
  options.transport_observer.enable = true;
  EXPECT_THROW(xpool::debug::configure(options, std::nullopt), c10::Error);
  options.transport_observer.enable = false;
  options.fabric_observer.enable = true;
  EXPECT_THROW(xpool::debug::configure(options, std::nullopt), c10::Error);
  options.fabric_observer.enable = false;
  options.ffn_routing_observer.enable = true;
  EXPECT_THROW(xpool::debug::configure(options, std::nullopt), c10::Error);
  options.ffn_routing_observer.enable = false;
  options.fabric_observer.record_capacity = 8;
  EXPECT_NO_THROW(xpool::debug::configure(options, std::nullopt));
  EXPECT_NO_THROW(xpool::debug::configure(options, std::nullopt));
  options.fabric_observer.enable = true;
  EXPECT_THROW(xpool::debug::configure(options, std::nullopt), c10::Error);
}
