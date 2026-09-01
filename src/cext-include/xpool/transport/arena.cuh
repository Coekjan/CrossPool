#pragma once

/// \file xpool/transport/arena.cuh
/// \brief Device address access for one Transport arena mapping.

#include <cstddef>
#include <cstdint>

#include <cuda/atomic>

#include <xpool/abort.hpp>
#include <xpool/transport/arena.hpp>

namespace xpool::transport {

template <typename T> XPOOL_DEVICE_FN T *ArenaView::pointer_at(std::size_t offset, std::size_t index) const {
  return reinterpret_cast<T *>(base_ + offset) + index;
}

XPOOL_DEVICE_FN inline const ArenaLayout &ArenaView::layout() const {
  return *reinterpret_cast<const ArenaLayout *>(base_);
}

XPOOL_DEVICE_FN inline ArenaState &ArenaView::state() const {
  return *pointer_at<ArenaState>(layout().header.state_offset);
}

XPOOL_DEVICE_FN inline Mailbox &ArenaView::mailbox() const { return *pointer_at<Mailbox>(layout().mailbox_offset); }

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> ArenaView::input_payload() const {
  const auto &arena_layout = layout();
  const auto bytes = arena_layout.payload_row_capacity * arena_layout.payload_row_bytes;
  return {pointer_at<std::uint8_t>(arena_layout.input_payload_offset), bytes};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint8_t> ArenaView::output_payload() const {
  const auto &arena_layout = layout();
  const auto bytes = arena_layout.payload_row_capacity * arena_layout.payload_row_bytes;
  return {pointer_at<std::uint8_t>(arena_layout.output_payload_offset), bytes};
}

XPOOL_DEVICE_FN inline cuda::std::span<std::uint32_t> ArenaView::dp_rank_payload_rows() const {
  const auto &arena_layout = layout();
  if (arena_layout.dp_rank_payload_rows_offset == 0) {
    return {};
  }
  return {pointer_at<std::uint32_t>(arena_layout.dp_rank_payload_rows_offset), arena_layout.atn_dp_size};
}

XPOOL_DEVICE_FN inline xpool::ffn::ResultCode ArenaView::generation_failure() const {
  return cuda::atomic_ref{state().generation_failure_code}.load(cuda::memory_order_acquire);
}

XPOOL_DEVICE_FN inline void ArenaView::publish_generation_failure(xpool::ffn::ResultCode result_code) const {
  xpool::abort_if(result_code != xpool::ffn::ResultCode::ProtocolMismatch &&
                  result_code != xpool::ffn::ResultCode::Timeout);
  auto observed = xpool::ffn::ResultCode::Ok;
  static_cast<void>(cuda::atomic_ref{state().generation_failure_code}.compare_exchange_strong(
      observed, result_code, cuda::memory_order_acq_rel, cuda::memory_order_acquire));
  xpool::abort_if(observed != xpool::ffn::ResultCode::Ok && observed != result_code);
}

} // namespace xpool::transport
