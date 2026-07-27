#include <gtest/gtest.h>

#include <xpool/abort.hpp>

TEST(NativeAbortTest, ReturnsWhenConditionIsFalse) { xpool::abort_if(false); }

TEST(NativeAbortTest, AbortsWhenConditionIsTrue) { EXPECT_DEATH(xpool::abort_if(true), ""); }
