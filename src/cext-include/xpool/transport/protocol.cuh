#pragma once

/// \file xpool/transport/protocol.cuh
/// \brief Device-side transport arena protocol helpers.

#include <cstdint>

#include <xpool/abi.hpp>
#include <xpool/transport.hpp>
#include <xpool/utils/arith.cuh>
#include <xpool/utils/device.cuh>
#include <xpool/utils/queue.cuh>

namespace xpool::transport {

__device__ inline const TransportArenaLayout &
TransportArena::arena_layout() const {
  xpool::utils::device::trap_if(base == nullptr);
  return *reinterpret_cast<const TransportArenaLayout *>(base);
}

__device__ inline std::uint64_t TransportArena::slot_count() const {
  return xpool::utils::arith::checked_nonnegative(arena_layout().slot_count);
}

__device__ inline void TransportArena::check_range(std::uint64_t offset,
                                                   std::uint64_t bytes) const {
  const std::uint64_t arena_bytes_count =
      xpool::utils::arith::checked_nonnegative(arena_layout().arena_bytes);
  xpool::utils::device::trap_if(offset > arena_bytes_count ||
                                bytes > arena_bytes_count - offset);
}

__device__ inline void TransportArena::check_slot(std::uint32_t slot) const {
  xpool::utils::device::trap_if(static_cast<std::uint64_t>(slot) >=
                                slot_count());
}

template <typename value_t>
__device__ value_t *TransportArena::pointer_at(std::uint64_t offset,
                                               std::uint64_t bytes) const {
  check_range(offset, bytes);
  return reinterpret_cast<value_t *>(base + offset);
}

__device__ inline std::uint64_t
TransportArena::slot_offset(std::int64_t base_offset, std::uint32_t slot,
                            std::int64_t stride_bytes,
                            std::uint64_t bytes) const {
  check_slot(slot);
  const std::uint64_t offset = xpool::utils::arith::checked_add(
      xpool::utils::arith::checked_nonnegative(base_offset),
      xpool::utils::arith::checked_mul(
          static_cast<std::uint64_t>(slot),
          xpool::utils::arith::checked_nonnegative(stride_bytes)));
  check_range(offset, bytes);
  return offset;
}

__device__ inline std::uint64_t
TransportArena::slot_offset(std::int64_t base_offset, std::uint32_t slot,
                            std::int64_t stride_bytes) const {
  return slot_offset(base_offset, slot, stride_bytes,
                     xpool::utils::arith::checked_nonnegative(stride_bytes));
}

template <typename value_t>
__device__ value_t &TransportArena::array_at(std::int64_t base_offset,
                                             std::uint32_t slot) const {
  return *pointer_at<value_t>(
      slot_offset(base_offset, slot, static_cast<std::int64_t>(sizeof(value_t)),
                  static_cast<std::uint64_t>(sizeof(value_t))));
}

template <typename value_t>
__device__ value_t *
TransportArena::slot_buffer_at(std::int64_t base_offset, std::uint32_t slot,
                               std::int64_t stride_bytes) const {
  return pointer_at<value_t>(
      slot_offset(base_offset, slot, stride_bytes,
                  xpool::utils::arith::checked_nonnegative(stride_bytes)),
      xpool::utils::arith::checked_nonnegative(stride_bytes));
}

template <typename value_t>
__device__ std::uint64_t
TransportArena::tensor_bytes(const xpool::abi::FfnTensorMetadata &metadata) {
  return xpool::utils::arith::checked_mul(
      xpool::utils::arith::checked_mul(
          static_cast<std::uint64_t>(metadata.num_tokens),
          static_cast<std::uint64_t>(metadata.hidden_size)),
      static_cast<std::uint64_t>(sizeof(value_t)));
}

__device__ inline std::uint32_t &TransportArena::shutdown() const {
  return *pointer_at<std::uint32_t>(
      xpool::utils::arith::checked_nonnegative(arena_layout().shutdown_offset));
}

__device__ inline std::uint32_t &TransportArena::error_code() const {
  return *pointer_at<std::uint32_t>(xpool::utils::arith::checked_nonnegative(
      arena_layout().error_code_offset));
}

__device__ inline xpool::abi::FfnRequestDescriptor &
TransportArena::request(std::uint32_t slot) const {
  return array_at<xpool::abi::FfnRequestDescriptor>(
      arena_layout().request_base_offset, slot);
}

__device__ inline xpool::abi::FfnResultDescriptor &
TransportArena::result(std::uint32_t slot) const {
  return array_at<xpool::abi::FfnResultDescriptor>(
      arena_layout().result_base_offset, slot);
}

/// Read the nanosecond device-global timer shared by CUDA contexts on this GPU.
/// \return Global timer timestamp in nanoseconds.
__device__ inline std::uint64_t transport_global_timer() {
  std::uint64_t value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}

__device__ inline xpool::abi::TransportTraceRecord *
TransportArena::begin_trace(std::uint32_t num_tokens) const {
  const TransportArenaLayout &layout = arena_layout();
  if (layout.trace_capacity == 0) {
    return nullptr;
  }
  auto *sequence = pointer_at<unsigned long long>(
      xpool::utils::arith::checked_nonnegative(layout.trace_sequence_offset));
  const unsigned long long trace_id = atomicAdd(sequence, 1ULL) + 1ULL;
  if (trace_id > static_cast<unsigned long long>(layout.trace_capacity)) {
    auto *dropped = pointer_at<unsigned long long>(
        xpool::utils::arith::checked_nonnegative(layout.trace_dropped_offset));
    atomicAdd(dropped, 1ULL);
  }
  auto *records = pointer_at<xpool::abi::TransportTraceRecord>(
      xpool::utils::arith::checked_nonnegative(layout.trace_records_offset),
      xpool::utils::arith::checked_mul(
          static_cast<std::uint64_t>(layout.trace_capacity),
          sizeof(xpool::abi::TransportTraceRecord)));
  xpool::abi::TransportTraceRecord *record =
      &records[(trace_id - 1ULL) %
               static_cast<unsigned long long>(layout.trace_capacity)];
  *record = xpool::abi::TransportTraceRecord{};
  record->trace_id = trace_id;
  record->num_tokens = num_tokens;
  record->request_begin = transport_global_timer();
  return record;
}

__device__ inline xpool::abi::TransportTraceRecord *
TransportArena::trace(std::uint64_t trace_id) const {
  const TransportArenaLayout &layout = arena_layout();
  if (trace_id == 0 || layout.trace_capacity == 0) {
    return nullptr;
  }
  auto *records = pointer_at<xpool::abi::TransportTraceRecord>(
      xpool::utils::arith::checked_nonnegative(layout.trace_records_offset),
      xpool::utils::arith::checked_mul(
          static_cast<std::uint64_t>(layout.trace_capacity),
          sizeof(xpool::abi::TransportTraceRecord)));
  xpool::abi::TransportTraceRecord *record =
      &records[(trace_id - 1ULL) %
               static_cast<std::uint64_t>(layout.trace_capacity)];
  return record->trace_id == trace_id ? record : nullptr;
}

template <typename value_t>
__device__ inline value_t *
TransportArena::input_buffer(std::uint32_t slot) const {
  const TransportArenaLayout &layout = arena_layout();
  return slot_buffer_at<value_t>(layout.input_base_offset, slot,
                                 layout.slot_stride_bytes);
}

template <typename value_t>
__device__ inline value_t *
TransportArena::output_buffer(std::uint32_t slot) const {
  const TransportArenaLayout &layout = arena_layout();
  return slot_buffer_at<value_t>(layout.output_base_offset, slot,
                                 layout.slot_stride_bytes);
}

__device__ inline std::uint32_t *
TransportArena::dp_token_counts_buffer(std::uint32_t slot) const {
  const TransportArenaLayout &layout = arena_layout();
  return slot_buffer_at<std::uint32_t>(layout.dp_token_counts_base_offset, slot,
                                       layout.dp_token_counts_stride_bytes);
}

__device__ inline xpool::utils::queue::RingQueue<std::uint32_t>
TransportArena::free_queue() const {
  const xpool::utils::queue::RingQueueLayout &queue = arena_layout().free_queue;
  auto *state = pointer_at<xpool::utils::queue::RingQueueState>(
      xpool::utils::arith::checked_nonnegative(queue.state_offset));
  auto *cells = pointer_at<xpool::utils::queue::RingQueueCell<std::uint32_t>>(
      xpool::utils::arith::checked_nonnegative(queue.cells_offset),
      xpool::utils::arith::checked_mul(
          slot_count(),
          static_cast<std::uint64_t>(
              sizeof(xpool::utils::queue::RingQueueCell<std::uint32_t>))));
  return xpool::utils::queue::RingQueue<std::uint32_t>{state, cells};
}

__device__ inline xpool::utils::queue::RingQueue<std::uint32_t>
TransportArena::used_queue() const {
  const xpool::utils::queue::RingQueueLayout &queue = arena_layout().used_queue;
  auto *state = pointer_at<xpool::utils::queue::RingQueueState>(
      xpool::utils::arith::checked_nonnegative(queue.state_offset));
  auto *cells = pointer_at<xpool::utils::queue::RingQueueCell<std::uint32_t>>(
      xpool::utils::arith::checked_nonnegative(queue.cells_offset),
      xpool::utils::arith::checked_mul(
          slot_count(),
          static_cast<std::uint64_t>(
              sizeof(xpool::utils::queue::RingQueueCell<std::uint32_t>))));
  return xpool::utils::queue::RingQueue<std::uint32_t>{state, cells};
}

__device__ inline xpool::abi::FfnArenaOffsets
TransportArena::ffn_offsets(std::uint32_t slot,
                            bool has_dp_token_counts) const {
  const TransportArenaLayout &layout = arena_layout();
  return xpool::abi::FfnArenaOffsets{
      slot_offset(layout.input_base_offset, slot, layout.slot_stride_bytes),
      slot_offset(layout.output_base_offset, slot, layout.slot_stride_bytes),
      has_dp_token_counts
          ? slot_offset(layout.dp_token_counts_base_offset, slot,
                        layout.dp_token_counts_stride_bytes)
          : 0,
  };
}

template <typename value_t>
__device__ inline const value_t *TransportArena::request_input(
    const xpool::abi::FfnRequestDescriptor &request) const {
  return pointer_at<value_t>(request.offsets.input_offset,
                             tensor_bytes<value_t>(request.tensor_metadata));
}

template <typename value_t>
__device__ inline value_t *TransportArena::request_output(
    const xpool::abi::FfnRequestDescriptor &request) const {
  return pointer_at<value_t>(request.offsets.output_offset,
                             tensor_bytes<value_t>(request.tensor_metadata));
}

} // namespace xpool::transport
