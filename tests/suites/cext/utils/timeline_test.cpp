#include <cstdint>

#include <gtest/gtest.h>

#include <xpool/utils/trace.hpp>

namespace {

enum class TestEvent : std::uint32_t {
  Begin,
  End,
  Count,
};

} // namespace

TEST(TraceTimelineTest, RecordsDependencyGraph) {
  auto timeline = xpool::utils::trace::Timeline<TestEvent>{};

  timeline.record<TestEvent::Begin>(10);
  timeline.record<TestEvent::End, TestEvent::Begin>(20);

  EXPECT_TRUE(timeline.recorded(TestEvent::Begin));
  EXPECT_TRUE(timeline.recorded(TestEvent::End));
  EXPECT_EQ(timeline.timestamp(TestEvent::Begin), 10U);
  EXPECT_EQ(timeline.timestamp(TestEvent::End), 20U);
}

TEST(TraceTimelineTest, FailStopsOnMissingDependency) {
  EXPECT_DEATH(([] {
                 auto timeline = xpool::utils::trace::Timeline<TestEvent>{};
                 timeline.record<TestEvent::End, TestEvent::Begin>(20);
               }()),
               "");
}

TEST(TraceTimelineTest, FailStopsOnDuplicateEvent) {
  EXPECT_DEATH(([] {
                 auto timeline = xpool::utils::trace::Timeline<TestEvent>{};
                 timeline.record<TestEvent::Begin>(10);
                 timeline.record<TestEvent::Begin>(20);
               }()),
               "");
}
