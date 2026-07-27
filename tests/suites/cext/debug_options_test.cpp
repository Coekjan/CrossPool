#include <c10/util/Exception.h>

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>

#include <gtest/gtest.h>

#include <xpool/debug/options.hpp>

namespace {

void expect_parse_error(std::string_view json, std::string_view message) {
  try {
    static_cast<void>(xpool::debug::DebugOptions::parse(json));
    FAIL() << "expected native debug JSON parsing to fail";
  } catch (const c10::Error &error) {
    EXPECT_NE(std::string(error.what()).find(message), std::string::npos) << error.what();
  }
}

} // namespace

TEST(DebugOptionsTest, DefaultsToDisabledWithoutConfiguration) {
  const xpool::debug::DebugOptions defaults{};
  EXPECT_FALSE(defaults.loopback.enabled());
  EXPECT_EQ(defaults.loopback.site, xpool::debug::LoopbackSite::None);
  EXPECT_EQ(defaults.transport_observer.trace_capacity, 0U);
  EXPECT_EQ(defaults.transport_observer.capacity(), 0U);
  EXPECT_EQ(defaults.fabric_observer.trace_capacity, 0U);
  EXPECT_EQ(defaults.fabric_observer.capacity(), 0U);
  EXPECT_EQ(xpool::debug::options(), defaults);
}

TEST(DebugOptionsTest, ParsesCompleteNativeSnapshot) {
  const auto options = xpool::debug::DebugOptions::parse(R"({
    "loopback":{"enable":true,"site":"atnagent"},
    "transport_observer":{"enable":true,"trace_capacity":32},
    "fabric_observer":{"enable":false,"trace_capacity":64}
  })");

  EXPECT_TRUE(options.loopback.enabled());
  EXPECT_EQ(options.loopback.site, xpool::debug::LoopbackSite::AtnAgent);
  EXPECT_EQ(options.transport_observer.capacity(), 32U);
  EXPECT_EQ(options.transport_observer.trace_capacity, 32U);
  EXPECT_EQ(options.fabric_observer.capacity(), 0U);
  EXPECT_EQ(options.fabric_observer.trace_capacity, 64U);
}

TEST(DebugOptionsTest, ParsesDisabledLoopback) {
  const auto options = xpool::debug::DebugOptions::parse(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":8192},
    "fabric_observer":{"enable":false,"trace_capacity":8192}
  })");

  EXPECT_FALSE(options.loopback.enabled());
  EXPECT_EQ(options.loopback.site, xpool::debug::LoopbackSite::None);
}

TEST(DebugOptionsTest, RejectsMalformedOrWrongRootJson) {
  expect_parse_error("{", "native debug JSON is invalid");
  expect_parse_error("[]", "must be an object");
}

TEST(DebugOptionsTest, RejectsUnknownOrMissingStructure) {
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":8}
  })",
                     "exactly three sections");
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8},
    "unknown":0
  })",
                     "exactly three sections");
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null,"unknown":0},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "loopback section");
  expect_parse_error(R"({
    "loopback":{"enable":false,"unknown":null},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "native debug JSON is invalid");
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":8,"unknown":0},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "transport_observer section");
}

TEST(DebugOptionsTest, RejectsInvalidLoopbackValues) {
  expect_parse_error(R"({
    "loopback":{"enable":1,"site":"atnagent"},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "type must be boolean");
  expect_parse_error(R"({
    "loopback":{"enable":true,"site":"unknown"},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "loopback.site is invalid");
  expect_parse_error(R"({
    "loopback":{"enable":true,"site":false},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "loopback.site is invalid");
  expect_parse_error(R"({
    "loopback":{"enable":true,"site":2},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "loopback.site is invalid");
  expect_parse_error(R"({
    "loopback":{"enable":true,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":8},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "enable and site must agree");
}

TEST(DebugOptionsTest, RejectsInvalidTraceCapacities) {
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":1.5},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "trace_capacity must be an integer");
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":0},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "trace_capacity must be positive");
  expect_parse_error(R"({
    "loopback":{"enable":false,"site":null},
    "transport_observer":{"enable":false,"trace_capacity":-1},
    "fabric_observer":{"enable":false,"trace_capacity":8}
  })",
                     "trace_capacity must be positive");
}
