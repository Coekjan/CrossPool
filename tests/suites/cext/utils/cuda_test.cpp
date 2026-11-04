#include <cstdint>
#include <limits>

#include <c10/util/Exception.h>
#include <gtest/gtest.h>

#include <xpool/utils/cuda.hpp>

TEST(CudaAllocationTest, RejectsMalformedOrOverflowingObservedIntervals) {
  EXPECT_NO_THROW(xpool::utils::cuda::CaptureAllocation("source", 0x1000, 512));
  EXPECT_THROW(xpool::utils::cuda::CaptureAllocation("", 0x1000, 512), c10::Error);
  EXPECT_THROW(xpool::utils::cuda::CaptureAllocation("source", 0, 512), c10::Error);
  EXPECT_THROW(xpool::utils::cuda::CaptureAllocation("source", 0x1000, 0), c10::Error);
  EXPECT_THROW(xpool::utils::cuda::CaptureAllocation("source", std::numeric_limits<std::uintptr_t>::max(), 1),
               c10::Error);
}
