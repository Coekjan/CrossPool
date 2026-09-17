#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <xpool/kv/channel.hpp>

namespace {

TEST(KvControlChannel, PublishesCoherentRoleOwnedStateAcrossProcesses) {
  auto daemon = xpool::kv::DaemonControlChannel::create(2, 2, 2);
  auto atnagent = xpool::kv::AtnAgentControlChannel::attach(daemon.name(), 0, {0, 1});
  auto leader = xpool::kv::InstanceControlChannel::attach(daemon.name(), 0, 0, 1);

  EXPECT_FALSE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_device_memory(),
            (std::vector<std::optional<xpool::kv::KvDeviceMemoryReport>>{std::nullopt, std::nullopt}));
  EXPECT_EQ(daemon.read_initial_backing(), (std::vector<std::optional<std::uint32_t>>{std::nullopt, std::nullopt}));
  EXPECT_EQ(leader.read_commands(), (std::vector<std::optional<xpool::kv::KvCapacityCommand>>{std::nullopt}));

  atnagent.publish_device_memory(80, 32);
  leader.publish_initial_backing(1);
  leader.publish_capture_complete();
  daemon.publish_service_ceiling(0, 4);
  daemon.publish_command(0, {.sequence = 1, .target_bundles = 3});

  const auto child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      auto participant = xpool::kv::InstanceControlChannel::attach(daemon.name(), 0, 1, 1);
      participant.publish_initial_backing(1);
      participant.publish_capture_complete();
      participant.publish_completion({.sequence = 1, .backed_bundles = 3});
      participant.close();
      _exit(0);
    } catch (...) {
    }
    _exit(1);
  }

  leader.publish_completion({.sequence = 1, .backed_bundles = 3});
  leader.publish_demand({.evaluated_sequence = 1, .requested_bundles = 4, .deadline_monotonic_ns = 900});

  auto status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_device_memory()[0], (xpool::kv::KvDeviceMemoryReport{.total_bytes = 80, .free_bytes = 32}));
  EXPECT_EQ(daemon.read_initial_backing(),
            (std::vector<std::optional<std::uint32_t>>{std::optional{1U}, std::optional{1U}}));
  EXPECT_EQ(leader.service_ceiling(), 4U);
  EXPECT_EQ(leader.read_commands()[0], (xpool::kv::KvCapacityCommand{.sequence = 1, .target_bundles = 3}));
  EXPECT_EQ(daemon.read_completions(), (std::vector<std::optional<xpool::kv::KvCapacityCompletion>>{
                                           xpool::kv::KvCapacityCompletion{.sequence = 1, .backed_bundles = 3},
                                           xpool::kv::KvCapacityCompletion{.sequence = 1, .backed_bundles = 3},
                                       }));
  EXPECT_EQ(daemon.read_demands()[0], (xpool::kv::KvCapacityDemand{
                                          .evaluated_sequence = 1,
                                          .requested_bundles = 4,
                                          .deadline_monotonic_ns = 900,
                                      }));

  auto participant = xpool::kv::InstanceControlChannel::attach(daemon.name(), 0, 1, 1);
  daemon.publish_command(0, {.sequence = 2, .target_bundles = 2});
  leader.publish_completion({.sequence = 2, .backed_bundles = 2});
  participant.publish_completion({.sequence = 2, .backed_bundles = 2});
  leader.publish_demand(
      {.evaluated_sequence = 2, .requested_bundles = std::nullopt, .deadline_monotonic_ns = std::nullopt});
  EXPECT_EQ(daemon.read_completions(), (std::vector<std::optional<xpool::kv::KvCapacityCompletion>>{
                                           xpool::kv::KvCapacityCompletion{.sequence = 2, .backed_bundles = 2},
                                           xpool::kv::KvCapacityCompletion{.sequence = 2, .backed_bundles = 2},
                                       }));
  EXPECT_EQ(daemon.read_demands()[0], (xpool::kv::KvCapacityDemand{
                                          .evaluated_sequence = 2,
                                          .requested_bundles = std::nullopt,
                                          .deadline_monotonic_ns = std::nullopt,
                                      }));
}

TEST(KvControlChannel, DemandReaderNeverCombinesConcurrentPublications) {
  auto daemon = xpool::kv::DaemonControlChannel::create(1, 1, 1);
  auto leader = xpool::kv::InstanceControlChannel::attach(daemon.name(), 0, 0, 1);
  auto reader_observed = std::atomic<bool>{false};
  auto done = std::atomic<bool>{false};
  auto writer_error = std::exception_ptr{};
  auto writer = std::thread{[&] {
    try {
      leader.publish_demand({.evaluated_sequence = 1, .requested_bundles = 2U, .deadline_monotonic_ns = 100U});
      while (!reader_observed.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      for (auto index = 1; index < 10000; ++index) {
        const auto first = (index & 1) == 0;
        leader.publish_demand({.evaluated_sequence = 1,
                               .requested_bundles = first ? 2U : 3U,
                               .deadline_monotonic_ns = first ? 100U : 200U});
      }
    } catch (...) {
      writer_error = std::current_exception();
    }
    done.store(true, std::memory_order_release);
  }};

  auto observed = std::size_t{0};
  auto mixed = false;
  while (!done.load(std::memory_order_acquire)) {
    const auto demand = daemon.read_demands()[0];
    if (!demand) {
      continue;
    }
    ++observed;
    reader_observed.store(true, std::memory_order_release);
    mixed |= !((demand->requested_bundles == 2U && demand->deadline_monotonic_ns == 100U) ||
               (demand->requested_bundles == 3U && demand->deadline_monotonic_ns == 200U));
  }
  writer.join();
  if (writer_error) {
    std::rethrow_exception(writer_error);
  }
  EXPECT_GT(observed, 0U);
  EXPECT_FALSE(mixed);
}

TEST(KvControlChannel, OwnerCloseUnlinksTheChannel) {
  auto daemon = xpool::kv::DaemonControlChannel::create(1, 1, 1);
  const auto name = daemon.name();
  daemon.close();
  EXPECT_ANY_THROW(static_cast<void>(xpool::kv::InstanceControlChannel::attach(name, 0, 0, 1)));
}

} // namespace
