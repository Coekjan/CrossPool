#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <gtest/gtest.h>

#include <xpool/abi.hpp>
#include <xpool/utils/queue.hpp>

TEST(NativeAbiContractTest, PreservesWireValuesAndLayouts) {
  using namespace xpool::abi;

  EXPECT_EQ(kAbiVersion, 24U);
  EXPECT_EQ(sizeof(FfnRequestDescriptor), 96U);
  EXPECT_EQ(sizeof(FfnResultDescriptor), 24U);
  EXPECT_EQ(sizeof(DebugOptions), 8U);
  EXPECT_EQ(sizeof(TransportTraceRecord), 120U);
  EXPECT_EQ(sizeof(xpool::utils::queue::RingQueueState), 24U);

  EXPECT_TRUE(std::is_standard_layout_v<DebugOptions>);
  EXPECT_TRUE(std::is_trivially_copyable_v<DebugOptions>);
  EXPECT_TRUE(std::is_standard_layout_v<FfnRequestDescriptor>);
  EXPECT_TRUE(std::is_trivially_copyable_v<FfnRequestDescriptor>);
  EXPECT_TRUE(std::is_standard_layout_v<FfnResultDescriptor>);
  EXPECT_TRUE(std::is_trivially_copyable_v<FfnResultDescriptor>);
  EXPECT_TRUE(std::is_standard_layout_v<TransportTraceRecord>);
  EXPECT_TRUE(std::is_trivially_copyable_v<TransportTraceRecord>);

  const DebugOptions atn_options{
      static_cast<std::uint64_t>(DebugOption::kLoopback) |
      static_cast<std::uint64_t>(DebugLoopbackSite::kAtnagent)};
  EXPECT_TRUE(atn_options.enabled(DebugOption::kLoopback));
  EXPECT_EQ(atn_options.loopback_site(), DebugLoopbackSite::kAtnagent);

  EXPECT_EQ(static_cast<unsigned>(RuntimeRole::kInstance), 1U);
  EXPECT_EQ(static_cast<unsigned>(RuntimeRole::kAtnagent), 2U);
  EXPECT_EQ(static_cast<unsigned>(RuntimeRole::kFfnagent), 3U);
  EXPECT_EQ(static_cast<std::uint64_t>(DebugOption::kLoopback), 1ULL << 32);
  EXPECT_EQ(static_cast<std::uint64_t>(DebugOption::kTransportObserver),
            1ULL << 33);
  EXPECT_EQ(static_cast<unsigned>(DebugLoopbackSite::kNone), 0U);
  EXPECT_EQ(static_cast<unsigned>(DebugLoopbackSite::kInstance), 1U);
  EXPECT_EQ(static_cast<unsigned>(DebugLoopbackSite::kAtnagent), 2U);
  EXPECT_EQ(static_cast<unsigned>(DebugLoopbackSite::kFfnagent), 3U);
  EXPECT_EQ(static_cast<unsigned>(XPoolForwardMode::kExtend), 1U);
  EXPECT_EQ(static_cast<unsigned>(XPoolForwardMode::kDecode), 2U);
  EXPECT_EQ(static_cast<unsigned>(XPoolForwardMode::kIdle), 4U);
  EXPECT_EQ(static_cast<unsigned>(FfnCollectivePolicy::kFullReduced), 1U);
  EXPECT_EQ(static_cast<unsigned>(FfnCollectivePolicy::kAtnTpPartial), 2U);
  EXPECT_EQ(static_cast<unsigned>(DpPaddingMode::kNone), 0U);
  EXPECT_EQ(static_cast<unsigned>(DpPaddingMode::kMaxLen), 1U);
  EXPECT_EQ(static_cast<unsigned>(DpPaddingMode::kSumLen), 2U);
  EXPECT_EQ(static_cast<unsigned>(TensorDType::kBf16), 1U);
  EXPECT_EQ(static_cast<unsigned>(TensorDType::kFp16), 2U);
  EXPECT_EQ(static_cast<unsigned>(TensorDType::kFp32), 3U);
  EXPECT_EQ(static_cast<unsigned>(DescriptorStatus::kEmpty), 0U);
  EXPECT_EQ(static_cast<unsigned>(DescriptorStatus::kPublished), 1U);
  EXPECT_EQ(static_cast<unsigned>(DescriptorStatus::kGranted), 2U);
  EXPECT_EQ(static_cast<unsigned>(DescriptorStatus::kDone), 3U);
  EXPECT_EQ(static_cast<unsigned>(DescriptorStatus::kFailed), 4U);
  EXPECT_EQ(static_cast<unsigned>(FfnResultErrorCode::kOk), 0U);
  EXPECT_EQ(static_cast<unsigned>(FfnResultErrorCode::kShutdown), 1U);
  EXPECT_EQ(static_cast<unsigned>(FfnResultErrorCode::kNotImplemented), 2U);

  EXPECT_EQ(offsetof(xpool::utils::queue::RingQueueState, capacity), 0U);
  EXPECT_EQ(offsetof(xpool::utils::queue::RingQueueState, head), 8U);
  EXPECT_EQ(offsetof(xpool::utils::queue::RingQueueState, tail), 16U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, state) +
                offsetof(DescriptorState, status),
            4U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, state) +
                offsetof(DescriptorState, slot_id),
            8U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, request_metadata) +
                offsetof(FfnRequestMetadata, instance_index),
            12U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, tensor_metadata) +
                offsetof(FfnTensorMetadata, dtype),
            52U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, tensor_metadata) +
                offsetof(FfnTensorMetadata, num_tokens),
            56U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, offsets) +
                offsetof(FfnArenaOffsets, input_offset),
            64U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, offsets) +
                offsetof(FfnArenaOffsets, output_offset),
            72U);
  EXPECT_EQ(offsetof(FfnRequestDescriptor, trace_id), 88U);
  EXPECT_EQ(offsetof(FfnResultDescriptor, state) +
                offsetof(DescriptorState, status),
            4U);
  EXPECT_EQ(offsetof(FfnResultDescriptor, state) +
                offsetof(DescriptorState, slot_id),
            8U);
  EXPECT_EQ(offsetof(FfnResultDescriptor, output_offset), 16U);
  EXPECT_EQ(offsetof(TransportTraceRecord, trace_id), 0U);
  EXPECT_EQ(offsetof(TransportTraceRecord, request_begin), 24U);
  EXPECT_EQ(offsetof(TransportTraceRecord, slot_recycled), 112U);
}
