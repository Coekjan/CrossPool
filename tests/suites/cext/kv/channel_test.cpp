#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <xpool/kv/channel.hpp>

namespace {

TEST(KvCapacityChannel, PublishesCoherentRoleOwnedStateAcrossProcesses) {
  auto daemon = xpool::kv::DaemonCapacityChannel::create(2, 4, 4);
  auto atnagent = xpool::kv::AtnAgentCapacityChannel::attach(daemon.name(), 0, {0, 2});
  auto instance = xpool::kv::InstanceCapacityChannel::attach(daemon.name(), 0, 0, 2);

  EXPECT_FALSE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_device_memory(),
            (std::vector<std::optional<xpool::kv::KvDeviceMemoryReport>>{std::nullopt, std::nullopt}));
  EXPECT_EQ(instance.read_commands(),
            (std::vector<std::optional<xpool::kv::KvCapacityCommand>>{std::nullopt, std::nullopt}));

  atnagent.publish_device_memory(80, 32);
  instance.publish_capture_complete();
  instance.publish_backing_report({.prepared_sequence = 0, .backed_bundles = 1});
  daemon.publish_command(0, {.sequence = 1, .target_bundles = 3, .active_bundles = 1});
  daemon.publish_command(0, {.sequence = 1, .target_bundles = 3, .active_bundles = 3});

  EXPECT_EQ(daemon.read_device_memory()[0], (xpool::kv::KvDeviceMemoryReport{.total_bytes = 80, .free_bytes = 32}));
  EXPECT_EQ(daemon.read_backing_reports()[0],
            (xpool::kv::KvCapacityBackingReport{.prepared_sequence = 0, .backed_bundles = 1}));
  EXPECT_EQ(instance.read_commands()[0],
            (xpool::kv::KvCapacityCommand{.sequence = 1, .target_bundles = 3, .active_bundles = 3}));

  const auto child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      auto participant = xpool::kv::InstanceCapacityChannel::attach(daemon.name(), 2, 2, 2);
      participant.publish_capture_complete();
      participant.publish_backing_report({.prepared_sequence = 1, .backed_bundles = 3});
      participant.publish_pressure(3);
      participant.close();
      _exit(0);
    } catch (...) {
      _exit(1);
    }
  }

  auto status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_backing_reports()[2],
            (xpool::kv::KvCapacityBackingReport{.prepared_sequence = 1, .backed_bundles = 3}));
  EXPECT_EQ(daemon.read_pressure_reports()[2],
            (xpool::kv::KvCapacityPressureReport{.sequence = 1, .active_bundles = 3}));

  auto second_instance = xpool::kv::InstanceCapacityChannel::attach(daemon.name(), 2, 2, 2);
  second_instance.publish_pressure(std::nullopt);
  EXPECT_EQ(daemon.read_pressure_reports()[2],
            (xpool::kv::KvCapacityPressureReport{.sequence = 2, .active_bundles = std::nullopt}));
}

TEST(KvCapacityChannel, OwnerCloseUnlinksTheChannel) {
  auto daemon = xpool::kv::DaemonCapacityChannel::create(1, 1, 1);
  const auto name = daemon.name();
  daemon.close();
  EXPECT_ANY_THROW(static_cast<void>(xpool::kv::InstanceCapacityChannel::attach(name, 0, 0, 1)));
}

} // namespace
