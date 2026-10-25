#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <xpool/ecs/channel.hpp>

namespace {

TEST(ElasticContextStateQuotaChannel, PublishesAsymmetricBytePermissionAcrossProcesses) {
  auto daemon = xpool::ecs::DaemonControlChannel::create(2, {2});
  auto atnagent = xpool::ecs::AtnAgentControlChannel::attach(daemon.name(), 0, {0, 1});
  auto leader = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 0, 1);
  EXPECT_FALSE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_initial_backing(), (std::vector<std::optional<std::uint64_t>>{std::nullopt, std::nullopt}));
  EXPECT_FALSE(leader.service_ceiling());
  EXPECT_FALSE(leader.read_commands()[0]);
  atnagent.publish_device_memory(80, 32);
  leader.publish_initial_backing(0);
  EXPECT_FALSE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_initial_backing(), (std::vector<std::optional<std::uint64_t>>{0, std::nullopt}));
  daemon.publish_service_ceiling(0, {9, 11});
  daemon.publish_command(0, {.sequence = 1, .quota_bytes = {3, 5}});

  const auto child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    try {
      auto participant = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 1, 1);
      participant.publish_initial_backing(1);
      participant.publish_completion({.sequence = 1, .backed_bytes = 2});
      participant.close();
      _exit(0);
    } catch (...) {
    }
    _exit(1);
  }
  leader.publish_completion({.sequence = 1, .backed_bytes = 0});
  leader.publish_demand(
      {.evaluated_sequence = 1, .required_bytes = std::vector<std::uint64_t>{7, 8}, .deadline_monotonic_ns = 900});
  auto status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status));
  ASSERT_EQ(WEXITSTATUS(status), 0);
  EXPECT_TRUE(atnagent.captures_complete());
  EXPECT_EQ(daemon.read_device_memory()[0], (xpool::ecs::DeviceMemoryReport{.total_bytes = 80, .free_bytes = 32}));
  EXPECT_EQ(daemon.read_initial_backing(), (std::vector<std::optional<std::uint64_t>>{0, 1}));
  EXPECT_EQ(leader.service_ceiling(), (std::vector<std::uint64_t>{9, 11}));
  EXPECT_EQ(leader.read_commands()[0], (xpool::ecs::QuotaCommand{.sequence = 1, .quota_bytes = {3, 5}}));
  EXPECT_EQ(daemon.read_completions(), (std::vector<std::optional<xpool::ecs::QuotaCompletion>>{
                                           xpool::ecs::QuotaCompletion{.sequence = 1, .backed_bytes = 0},
                                           xpool::ecs::QuotaCompletion{.sequence = 1, .backed_bytes = 2},
                                       }));
  EXPECT_EQ(daemon.read_demands()[0]->required_bytes, (std::vector<std::uint64_t>{7, 8}));
  leader.publish_demand(
      {.evaluated_sequence = 1, .required_bytes = std::nullopt, .deadline_monotonic_ns = std::nullopt});
  EXPECT_FALSE(daemon.read_demands()[0]->required_bytes);
  EXPECT_FALSE(daemon.read_demands()[0]->deadline_monotonic_ns);
}

TEST(ElasticContextStateQuotaChannel, DirectoryPreservesDifferentInstanceTpDpRows) {
  auto daemon = xpool::ecs::DaemonControlChannel::create(4, {2, 2, 4});
  auto dp_one = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 1, 2, 2);
  auto second_model = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 2, 4, 1);
  daemon.publish_command(0, {.sequence = 1, .quota_bytes = {1, 2}});
  daemon.publish_command(1, {.sequence = 1, .quota_bytes = {3, 4}});
  daemon.publish_command(2, {.sequence = 1, .quota_bytes = {5, 6, 7, 8}});
  const auto first_commands = dp_one.read_commands();
  ASSERT_EQ(first_commands.size(), 2U);
  EXPECT_EQ(first_commands[0]->quota_bytes, (std::vector<std::uint64_t>{1, 2}));
  EXPECT_EQ(first_commands[1]->quota_bytes, (std::vector<std::uint64_t>{3, 4}));
  const auto second_commands = second_model.read_commands();
  ASSERT_EQ(second_commands.size(), 1U);
  EXPECT_EQ(second_commands[0]->quota_bytes, (std::vector<std::uint64_t>{5, 6, 7, 8}));
  EXPECT_THROW(static_cast<void>(xpool::ecs::InstanceControlChannel::attach(daemon.name(), 1, 1, 2)), c10::Error);
  EXPECT_THROW(static_cast<void>(xpool::ecs::InstanceControlChannel::attach(daemon.name(), 2, 4, 2)), c10::Error);
  EXPECT_THROW(static_cast<void>(xpool::ecs::DaemonControlChannel::create(1, {0})), c10::Error);
  EXPECT_THROW(
      static_cast<void>(xpool::ecs::DaemonControlChannel::create(1, {std::numeric_limits<std::size_t>::max(), 1})),
      c10::Error);
}

TEST(ElasticContextStateQuotaChannel, WireOwnershipExtentsAndImmutableOperationBoundaries) {
  auto daemon = xpool::ecs::DaemonControlChannel::create(2, {2});
  auto leader = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 0, 1);
  auto follower = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 1, 1);
  EXPECT_THROW(daemon.publish_command(0, {.sequence = 1, .quota_bytes = {1}}), c10::Error);
  EXPECT_THROW(daemon.publish_service_ceiling(0, {1}), c10::Error);
  daemon.publish_service_ceiling(0, {0, std::numeric_limits<std::uint64_t>::max()});
  EXPECT_EQ(leader.service_ceiling(), (std::vector<std::uint64_t>{0, std::numeric_limits<std::uint64_t>::max()}));
  EXPECT_THROW(daemon.publish_service_ceiling(0, {1, 2}), c10::Error);
  daemon.publish_command(0, {.sequence = 1, .quota_bytes = {0, 2}});
  daemon.publish_command(0, {.sequence = 1, .quota_bytes = {0, 2}});
  EXPECT_THROW(daemon.publish_command(0, {.sequence = 1, .quota_bytes = {1, 2}}), c10::Error);
  EXPECT_THROW(daemon.publish_command(0, {.sequence = 2, .quota_bytes = {0, 1}}), c10::Error);
  EXPECT_THROW(follower.publish_demand(
                   {.evaluated_sequence = 1, .required_bytes = std::nullopt, .deadline_monotonic_ns = std::nullopt}),
               c10::Error);
  EXPECT_THROW(
      leader.publish_demand(
          {.evaluated_sequence = 1, .required_bytes = std::vector<std::uint64_t>{1}, .deadline_monotonic_ns = 1}),
      c10::Error);
  leader.publish_completion({.sequence = 1, .backed_bytes = 0});
  EXPECT_THROW(leader.publish_completion({.sequence = 1, .backed_bytes = 1}), c10::Error);
  follower.publish_completion({.sequence = 1, .backed_bytes = 1});
  daemon.publish_command(0, {.sequence = 2, .quota_bytes = {0, 1}});
}

TEST(ElasticContextStateQuotaChannel, ReadersNeverCombineConcurrentMultiwordPublications) {
  auto daemon = xpool::ecs::DaemonControlChannel::create(2, {2});
  auto leader = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 0, 1);
  auto follower = xpool::ecs::InstanceControlChannel::attach(daemon.name(), 0, 1, 1);
  auto reader_observed = std::atomic<bool>{false};
  auto done = std::atomic<bool>{false};
  auto writer_error = std::exception_ptr{};
  const auto scale = std::uint64_t{1} << 32;
  auto writer = std::thread{[&] {
    try {
      for (auto sequence = std::uint32_t{1}; sequence <= 10000; ++sequence) {
        daemon.publish_command(0, {.sequence = sequence, .quota_bytes = {sequence * scale, sequence * scale + 7}});
        leader.publish_completion({.sequence = sequence, .backed_bytes = sequence * scale + 1});
        follower.publish_completion({.sequence = sequence, .backed_bytes = sequence * scale + 2});
        leader.publish_demand({.evaluated_sequence = sequence,
                               .required_bytes = std::vector<std::uint64_t>{sequence * scale + 3, sequence * scale + 4},
                               .deadline_monotonic_ns = sequence * scale + 5});
        if (sequence == 1) {
          while (!reader_observed.load(std::memory_order_acquire)) {
            std::this_thread::yield();
          }
        }
        leader.publish_demand(
            {.evaluated_sequence = sequence, .required_bytes = std::nullopt, .deadline_monotonic_ns = std::nullopt});
      }
    } catch (...) {
      writer_error = std::current_exception();
    }
    done.store(true, std::memory_order_release);
  }};
  auto observed = std::size_t{0};
  auto mixed = false;
  while (!done.load(std::memory_order_acquire)) {
    const auto command = leader.read_commands()[0];
    if (command) {
      mixed |=
          command->quota_bytes != std::vector<std::uint64_t>{command->sequence * scale, command->sequence * scale + 7};
    }
    const auto completions = daemon.read_completions();
    for (auto index = std::size_t{0}; index < completions.size(); ++index) {
      if (completions[index]) {
        mixed |= completions[index]->backed_bytes != completions[index]->sequence * scale + index + 1;
      }
    }
    const auto demand = daemon.read_demands()[0];
    if (demand) {
      ++observed;
      reader_observed.store(true, std::memory_order_release);
      if (demand->required_bytes) {
        mixed |= *demand->required_bytes != std::vector<std::uint64_t>{demand->evaluated_sequence * scale + 3,
                                                                       demand->evaluated_sequence * scale + 4} ||
                 demand->deadline_monotonic_ns != demand->evaluated_sequence * scale + 5;
      } else {
        mixed |= demand->deadline_monotonic_ns.has_value();
      }
    }
  }
  writer.join();
  if (writer_error) {
    std::rethrow_exception(writer_error);
  }
  EXPECT_GT(observed, 0U);
  EXPECT_FALSE(mixed);
}

TEST(ElasticContextStateQuotaChannel, OwnerCloseUnlinksTheChannel) {
  auto daemon = xpool::ecs::DaemonControlChannel::create(1, {1});
  const auto name = daemon.name();
  daemon.close();
  EXPECT_THROW(static_cast<void>(xpool::ecs::InstanceControlChannel::attach(name, 0, 0, 1)), c10::Error);
}

} // namespace
