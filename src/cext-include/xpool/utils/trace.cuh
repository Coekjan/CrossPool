#pragma once

/// \file xpool/utils/trace.cuh
/// \brief Device reservation over one process-local observer buffer.

#include <cuda/atomic>
#include <cuda/std/optional>
#include <cuda/std/span>

#include <cstddef>
#include <cstdint>

#include <xpool/macros.hpp>
#include <xpool/utils/trace.hpp>

namespace xpool::utils::trace {

/// Non-owning device view over one fixed-capacity process-local trace buffer.
/// Reservations and record writes use relaxed device ordering; Host snapshots
/// are valid only after the related device execution is quiesced or synchronized.
template <typename Record> class BufferView {
public:
  static_assert(std::is_trivially_copyable_v<Record>);

  /// Construct a view over caller-owned records and counters.
  /// \param records Fixed-capacity record storage.
  /// \param state Shared reservation and overflow counters.
  XPOOL_DEVICE_FN BufferView(cuda::std::span<Record> records, BufferState &state) : records_(records), state_(state) {}

  /// Reserve the next record without wrapping or overwriting evidence.
  /// \return A writable entry, or no value after capacity is exhausted.
  XPOOL_DEVICE_FN cuda::std::optional<Entry<Record>> reserve() const {
    if (records_.empty()) {
      return cuda::std::nullopt;
    }
    const auto sequence = cuda::atomic_ref{state_.sequence}.fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed) + 1;
    if (sequence > records_.size()) {
      cuda::atomic_ref{state_.dropped}.fetch_add(std::uint64_t{1}, cuda::memory_order_relaxed);
      return cuda::std::nullopt;
    }
    auto *record = &records_[static_cast<std::size_t>(sequence - 1)];
    *record = Record{};
    return Entry<Record>{record, sequence};
  }

  /// Find a retained record by its one-based reservation sequence.
  /// \param sequence Positive process-local reservation sequence.
  /// \return The retained record, or null when outside buffer capacity.
  XPOOL_DEVICE_FN Record *find(std::uint64_t sequence) const {
    if (sequence == 0 || sequence > records_.size()) {
      return nullptr;
    }
    return &records_[static_cast<std::size_t>(sequence - 1)];
  }

private:
  cuda::std::span<Record> records_;
  BufferState &state_;
};

} // namespace xpool::utils::trace
