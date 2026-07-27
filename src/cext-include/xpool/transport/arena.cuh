#pragma once

/// \file xpool/transport/arena.cuh
/// \brief Device address access for one Transport arena mapping.

#include <cstddef>
#include <cstdint>

#include <xpool/abort.hpp>
#include <xpool/atomic.cuh>
#include <xpool/trace.cuh>
#include <xpool/transport/arena.hpp>
#include <xpool/transport/trace.cuh>

namespace xpool::transport {

template <typename T>
XPOOL_DEVICE_FN T *TransportArenaView::pointer_at(std::size_t offset, std::size_t index) const {
  const auto &arena_layout = layout();
  xpool::abort_if(offset == 0 || offset > arena_layout.header.total_bytes);
  const auto remaining = arena_layout.header.total_bytes - offset;
  xpool::abort_if(index > remaining / sizeof(T));
  const auto relative_offset = index * sizeof(T);
  xpool::abort_if(sizeof(T) > remaining - relative_offset);
  return reinterpret_cast<T *>(base_ + offset + relative_offset);
}

XPOOL_DEVICE_FN inline const TransportArenaLayout &TransportArenaView::layout() const {
  xpool::abort_if(base_ == nullptr);
  return *reinterpret_cast<const TransportArenaLayout *>(base_);
}

XPOOL_DEVICE_FN inline TransportArenaState &TransportArenaView::state() const {
  return *pointer_at<TransportArenaState>(layout().header.state_offset);
}

XPOOL_DEVICE_FN inline TransportMailbox &TransportArenaView::mailbox() const {
  return *pointer_at<TransportMailbox>(layout().mailbox_offset);
}

XPOOL_DEVICE_FN inline std::uint8_t *TransportArenaView::input_payload() const {
  return pointer_at<std::uint8_t>(layout().input_payload_offset);
}

XPOOL_DEVICE_FN inline std::uint8_t *TransportArenaView::output_payload() const {
  return pointer_at<std::uint8_t>(layout().output_payload_offset);
}

XPOOL_DEVICE_FN inline std::uint32_t *TransportArenaView::dp_token_counts() const {
  if (layout().dp_token_counts_offset == 0) {
    return nullptr;
  }
  return pointer_at<std::uint32_t>(layout().dp_token_counts_offset);
}

XPOOL_DEVICE_FN inline xpool::trace::Entry<TransportTraceRecord> TransportArenaView::reserve_trace() const {
  return xpool::trace::Buffer<TransportTraceRecord>{base_, layout().trace, state().trace}.reserve();
}

XPOOL_DEVICE_FN inline TransportTraceRecord *TransportArenaView::current_trace() const {
  const auto sequence = xpool::atomic::system_atomic(state().trace.sequence).load(cuda::memory_order_acquire);
  return xpool::trace::Buffer<TransportTraceRecord>{base_, layout().trace, state().trace}.find(sequence);
}

XPOOL_DEVICE_FN inline xpool::abi::FfnResultCode TransportArenaView::generation_failure() const {
  return xpool::abi::FfnResultCode{xpool::atomic::load_acquire(state().generation_failure_code)};
}

XPOOL_DEVICE_FN inline void
TransportArenaView::publish_generation_failure(xpool::abi::FfnResultCode result_code) const {
  xpool::abort_if(result_code != xpool::abi::FfnResultCode::ProtocolMismatch &&
                  result_code != xpool::abi::FfnResultCode::NotImplemented);
  const auto observed = xpool::atomic::compare_exchange_acq_rel(
      state().generation_failure_code, static_cast<std::uint32_t>(xpool::abi::FfnResultCode::Ok),
      result_code.value());
  xpool::abort_if(observed != xpool::abi::FfnResultCode::Ok && observed != result_code.value());
}

} // namespace xpool::transport
